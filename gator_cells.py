"""Exactly 10,000-parameter ring of ten connected 980-parameter Gator cells."""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gtr_a_experiments import GTRFixedSlots, fixed_code, relative_code
from parallel_prefix_benchmark import evaluate, make_prefix_batch


class GatorCellRing(nn.Module):
    def __init__(self, n_cells=10, relays=4, neighbors=2):
        super().__init__()
        if neighbors not in {2, 4, 6, 8}:
            raise ValueError("neighbors must be one of 2, 4, 6, 8")
        self.n_cells, self.relays, self.neighbors = n_cells, relays, neighbors
        self.cells = nn.ModuleList(GTRFixedSlots("gtrarw", relays, 8) for _ in range(n_cells))
        self.left_projection = nn.Linear(8, 8)
        self.right_projection = nn.Linear(8, 8)
        self.coupling_norm = nn.LayerNorm(8)
        self.neighbor_gates = nn.Parameter(torch.zeros(n_cells, 2))
        self.self_gates = nn.Parameter(torch.ones(n_cells))
        self.output_votes = nn.Parameter(torch.zeros(n_cells))
        assert self.parameter_count == 10_000, self.parameter_count

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _initialize(self, cell, observed, output_slots):
        batch, observed_length = observed.shape
        length = observed_length + output_slots
        addresses = fixed_code(length, 4, observed.device)
        address_state = cell.address_to_state(addresses)
        mask = cell.token_embedding.weight.mean(0).view(1, 1, 8).expand(batch, output_slots, -1)
        content = torch.cat([cell.token_embedding(observed), mask], 1)
        h = cell.norm(content + address_state[None])
        q = F.softmax(cell.start_logits / 0.35, -1).view(1, 1, 4).expand(batch, length, 4)
        return h, q, addresses, address_state

    def _local_relay(self, cell, h, q, addresses, address_state, relay, observed_length, temperature):
        batch, length, _ = h.shape
        write = torch.ones(length, device=h.device, dtype=h.dtype)
        write[observed_length:] = relay / max(self.relays - 1, 1)
        rel_bias = torch.einsum(
            "ijc,nc->ijn", relative_code(length, h.device), cell.relative_node_bias
        )
        scores = q.clamp_min(1e-9).log()[:, None, :, :] + rel_bias[None]
        scores = scores + write.clamp_min(1e-9).log()[None, None, :, None]
        alpha = scores.softmax(2)
        node_memory = torch.einsum("bijn,bjd->bind", alpha, h)
        context = (q.unsqueeze(-1) * node_memory).sum(2)
        occupancy = q.mean(1, keepdim=True).expand_as(q)
        edge_logits = cell.router(torch.cat([h, context, occupancy], -1))
        edge_logits = edge_logits + cell.address_to_edges(addresses)[None]
        transition = F.softmax(
            edge_logits.view(batch, length, 4, 4).masked_fill(
                ~cell.adjacency.view(1, 1, 4, 4), -1e9
            ) / temperature,
            -1,
        )
        q_next = torch.einsum("bts,btsd->btd", q, transition)
        step_state = cell.step_projection(fixed_code(self.relays, 4, h.device)[relay])
        base = cell.norm(h + context) + address_state[None] + step_state
        destinations = torch.stack([
            torch.tanh(transform(base + cell.node_embedding.weight[node]))
            for node, transform in enumerate(cell.node_transforms)
        ], 2)
        h_next = cell.norm(h + (q_next.unsqueeze(-1) * destinations).sum(2))
        return h_next, q_next

    def forward(self, observed, *, output_slots=7, temperature=0.35):
        observed_length = observed.shape[1]
        states = [self._initialize(cell, observed, output_slots) for cell in self.cells]
        hs = [state[0] for state in states]
        qs = [state[1] for state in states]
        addresses = [state[2] for state in states]
        address_states = [state[3] for state in states]
        for relay in range(self.relays):
            local = [
                self._local_relay(cell, hs[c], qs[c], addresses[c], address_states[c],
                                  relay, observed_length, temperature)
                for c, cell in enumerate(self.cells)
            ]
            local_h = [item[0] for item in local]
            qs = [item[1] for item in local]
            coupled = []
            radius = self.neighbors // 2
            for c in range(self.n_cells):
                left_state = torch.stack([
                    local_h[(c - distance) % self.n_cells] for distance in range(1, radius + 1)
                ]).mean(0)
                right_state = torch.stack([
                    local_h[(c + distance) % self.n_cells] for distance in range(1, radius + 1)
                ]).mean(0)
                left = self.left_projection(left_state)
                right = self.right_projection(right_state)
                gates = torch.sigmoid(self.neighbor_gates[c])
                mixed = self.self_gates[c] * local_h[c] + gates[0] * left + gates[1] * right
                coupled.append(self.coupling_norm(mixed))
            hs = coupled
        logits = []
        for c, cell in enumerate(self.cells):
            out_h = hs[c][:, observed_length:] + cell.address_to_readout(addresses[c][observed_length:])[None]
            logits.append(out_h @ cell.token_embedding.weight.t())
        votes = F.softmax(self.output_votes, 0)
        return sum(votes[c] * logits[c] for c in range(self.n_cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatorCellRing(neighbors=args.neighbors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    print(f"device={device} parameters={model.parameter_count} cells=10 "
          f"topology=ring neighbors={args.neighbors}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(args.steps):
        observed, targets = make_prefix_batch(args.batch_size, 7, device)
        temperature = max(0.35, 1.25 * (0.997**step))
        logits = model(observed, output_slots=7, temperature=temperature)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 1000 == 0:
            result = evaluate(model, 7, device, batch_size=1024)
            print(f"step={step+1} correct={result['correct']:.2f}/7 "
                  f"acc={result['accuracy']:.3f} loss={result['loss']:.4f}", flush=True)
            torch.save({"step": step + 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, "gator_10cell_ring_checkpoint.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = evaluate(model, 7, device)
    print(f"final correct={result['correct']:.2f}/7 acc={result['accuracy']:.3f} "
          f"exact={result['exact']:.3f} loss={result['loss']:.4f}")
    print(f"runtime_seconds={elapsed:.2f} steps_per_second={args.steps / elapsed:.2f}")


if __name__ == "__main__":
    main()
