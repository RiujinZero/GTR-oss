"""Sparse 40-node Gator preserving roughly four nodes per 1k parameters."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gtr_a_experiments import fixed_code, relative_code
from parallel_prefix_benchmark import evaluate, make_prefix_batch


class SparseGator(nn.Module):
    def __init__(
        self,
        n_nodes=40,
        d_model=7,
        degree=3,
        relays=4,
        *,
        broad_start=False,
    ):
        super().__init__()
        self.n_nodes, self.d_model, self.degree, self.relays = n_nodes, d_model, degree, relays
        self.token_embedding = nn.Embedding(16, d_model)
        self.address_to_state = nn.Linear(4, d_model, bias=False)
        self.address_to_edges = nn.Linear(4, n_nodes * degree, bias=False)
        self.address_to_readout = nn.Linear(4, d_model, bias=False)
        self.node_embedding = nn.Embedding(n_nodes, d_model)
        self.step_projection = nn.Linear(4, d_model, bias=False)
        # Vectorized node transforms: one d x d matrix and bias per node.
        self.node_weight = nn.Parameter(torch.empty(n_nodes, d_model, d_model))
        self.node_bias = nn.Parameter(torch.empty(n_nodes, d_model))
        self.router = nn.Linear(2 * d_model + n_nodes, n_nodes * degree)
        self.norm = nn.LayerNorm(d_model)
        self.relative_node_bias = nn.Parameter(torch.zeros(n_nodes, 4))
        if broad_start:
            initial_start = torch.zeros(n_nodes)
        else:
            initial_start = torch.cat(
                [
                    torch.tensor([4.0]),
                    torch.full((n_nodes - 1,), -4.0),
                ]
            )
        self.start_logits = nn.Parameter(initial_start)
        self.broad_start = broad_start
        nn.init.kaiming_uniform_(self.node_weight, a=5**0.5)
        bound = 1 / d_model**0.5
        nn.init.uniform_(self.node_bias, -bound, bound)
        source = torch.arange(n_nodes)[:, None]
        offsets = torch.tensor([0, 1, -1])[:degree][None]
        self.register_buffer("destinations", (source + offsets).remainder(n_nodes))
        assert self.parameter_count == 10_010, self.parameter_count

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, observed, *, output_slots=7, temperature=0.35):
        batch, observed_length = observed.shape
        length = observed_length + output_slots
        addresses = fixed_code(length, 4, observed.device)
        address_state = self.address_to_state(addresses)
        address_edges = self.address_to_edges(addresses)
        address_out = self.address_to_readout(addresses)
        mask = self.token_embedding.weight.mean(0).view(1, 1, -1).expand(batch, output_slots, -1)
        h = torch.cat([self.token_embedding(observed), mask], 1)
        h = self.norm(h + address_state[None])
        q = F.softmax(self.start_logits / temperature, -1).view(1, 1, -1).expand(batch, length, -1)

        for relay in range(self.relays):
            write = torch.ones(length, device=h.device, dtype=h.dtype)
            write[observed_length:] = relay / max(self.relays - 1, 1)
            rel_bias = torch.einsum(
                "ijc,nc->ijn", relative_code(length, h.device), self.relative_node_bias
            )
            scores = q.clamp_min(1e-9).log()[:, None, :, :] + rel_bias[None]
            scores = scores + write.clamp_min(1e-9).log()[None, None, :, None]
            alpha = scores.softmax(2)
            node_memory = torch.einsum("bijn,bjd->bind", alpha, h)
            context = (q.unsqueeze(-1) * node_memory).sum(2)
            occupancy = q.mean(1, keepdim=True).expand_as(q)

            edge_logits = self.router(torch.cat([h, context, occupancy], -1))
            edge_logits = (edge_logits + address_edges[None]).view(
                batch, length, self.n_nodes, self.degree
            )
            transition = F.softmax(edge_logits / temperature, -1)
            contributions = (q.unsqueeze(-1) * transition).reshape(batch, length, -1)
            destinations = self.destinations.reshape(-1).view(1, 1, -1).expand(batch, length, -1)
            q_next = torch.zeros_like(q).scatter_add(2, destinations, contributions)

            step_state = self.step_projection(fixed_code(self.relays, 4, h.device)[relay])
            base = self.norm(h + context) + address_state[None] + step_state
            node_inputs = base.unsqueeze(2) + self.node_embedding.weight[None, None]
            destinations_h = torch.tanh(
                torch.einsum("btnd,nde->btne", node_inputs, self.node_weight)
                + self.node_bias[None, None]
            )
            h = self.norm(h + (q_next.unsqueeze(-1) * destinations_h).sum(2))
            q = q_next
        read_h = h[:, observed_length:] + address_out[observed_length:][None]
        return read_h @ self.token_embedding.weight.t()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    parser.add_argument(
        "--broad-start",
        action="store_true",
        help="initialize location mass uniformly across all graph nodes",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.eval_batch_size <= 0
        or args.log_every <= 0
    ):
        parser.error("step, batch, evaluation, and logging values must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseGator(
        n_nodes=args.nodes,
        d_model=args.width,
        degree=args.degree,
        relays=args.relays,
        broad_start=args.broad_start,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    checkpoint = args.checkpoint or Path(
        f"gator_sparse_n{args.nodes}_w{args.width}_seed{args.seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} parameters={model.parameter_count} "
        f"cells=1 nodes={args.nodes} width={args.width} "
        f"degree={args.degree} relays={args.relays} "
        f"broad_start={args.broad_start} seed={args.seed}",
        flush=True,
    )
    started = time.perf_counter()
    interval_started = started
    for step in range(args.steps):
        observed, targets = make_prefix_batch(
            args.batch_size,
            7,
            device,
        )
        temperature = max(0.35, 1.25 * (0.997**step))
        logits = model(observed, output_slots=7, temperature=temperature)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed % args.log_every == 0:
            result = evaluate(
                model,
                7,
                device,
                batch_size=args.eval_batch_size,
            )
            now = time.perf_counter()
            speed = args.log_every / (now - interval_started)
            interval_started = now
            print(
                f"step={completed} steps_per_second={speed:.2f} "
                f"correct={result['correct']:.2f}/7 "
                f"acc={result['accuracy']:.3f} exact={result['exact']:.3f} "
                f"loss={result['loss']:.4f}",
                flush=True,
            )
            torch.save(
                {
                    "step": completed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                checkpoint,
            )
    result = evaluate(
        model,
        7,
        device,
        batch_size=max(args.eval_batch_size, 4096),
    )
    runtime = time.perf_counter() - started
    print(
        f"final correct={result['correct']:.2f}/7 "
        f"acc={result['accuracy']:.3f} exact={result['exact']:.3f} "
        f"loss={result['loss']:.4f}",
        flush=True,
    )
    print(
        f"runtime_seconds={runtime:.2f} "
        f"steps_per_second={args.steps / runtime:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
