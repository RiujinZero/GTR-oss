"""GTR-A fixed-slot experiment suite; every main model has exactly 980 parameters."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_code(length: int, dims: int, device, *, norm_length: int | None = None) -> torch.Tensor:
    x = torch.arange(length, device=device, dtype=torch.float32)
    scale = max((length if norm_length is None else norm_length) - 1, 1)
    angle = x / scale * math.pi
    parts = [torch.sin(angle), torch.cos(angle), torch.sin(2 * angle), torch.cos(2 * angle)]
    if dims == 5:
        parts.append(2 * x / scale - 1)
    return torch.stack(parts[:dims], dim=-1)


def relative_code(length: int, device, *, norm_length: int | None = None) -> torch.Tensor:
    delta = torch.arange(length, device=device)[:, None] - torch.arange(length, device=device)[None, :]
    scale = max((length if norm_length is None else norm_length) - 1, 1)
    delta = delta.float() / scale
    return torch.stack([delta, delta.abs(), torch.sin(math.pi * delta), torch.cos(math.pi * delta)], -1)


def make_full_batch(batch_size: int, device, *, exhaustive=False):
    if exhaustive:
        rule = torch.arange(4, device=device).repeat_interleave(12)
        start = torch.arange(4, 16, device=device).repeat(4)
    else:
        rule = torch.randint(0, 4, (batch_size,), device=device)
        start = 4 + torch.randint(0, 12, (batch_size,), device=device)
    full = torch.empty(rule.numel(), 16, dtype=torch.long, device=device)
    full[:, 0], full[:, 1] = rule, start
    for position in range(2, 16):
        full[:, position] = 4 + (full[:, position - 1] - 4 + rule + 1).remainder(12)
    return full[:, :2], full[:, 2:], rule


class GTRFixedSlots(nn.Module):
    def __init__(
        self,
        variant="gtr0",
        relays=4,
        d_model=8,
        n_nodes=4,
        routing_dim=None,
    ):
        super().__init__()
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least two")
        if routing_dim is not None and (
            isinstance(routing_dim, bool)
            or not isinstance(routing_dim, int)
            or routing_dim < 1
        ):
            raise ValueError("routing_dim must be a positive integer or None")
        self.variant = variant.lower()
        self.relays = relays
        self.d_model = d_model
        self.n_nodes = n_nodes
        valid = {"gtr0", "gtra", "gtraw", "gtrar", "gtrarw"}
        if self.variant not in valid:
            raise ValueError(f"unknown GTR variant: {self.variant}")
        self.anchored = self.variant in {"gtra", "gtraw", "gtrar", "gtrarw"}
        self.controlled_writing = self.variant in {"gtraw", "gtrarw"}
        self.relation_aware = self.variant in {"gtrar", "gtrarw"}
        self.token_embedding = nn.Embedding(16, d_model)
        if self.anchored:
            self.address_to_state = nn.Linear(4, d_model, bias=False)
            self.address_to_edges = nn.Linear(
                4,
                n_nodes * n_nodes,
                bias=False,
            )
            self.address_to_readout = nn.Linear(4, d_model, bias=False)
        else:
            self.position_embedding = nn.Embedding(16, d_model)
        self.node_embedding = nn.Embedding(n_nodes, d_model)
        self.step_projection = nn.Linear(4, d_model, bias=False)
        self.node_transforms = nn.ModuleList(
            nn.Linear(d_model, d_model)
            for _ in range(n_nodes)
        )
        # routing_dim decouples the router's decision bandwidth from
        # d_model: without it, a wider d_model widens the router's input
        # for free even though its output (n_nodes * n_nodes edges) never
        # grows, so d_model sweeps can't isolate how much routing capacity
        # is actually needed. routing_dim=None preserves the original
        # direct router (input width 2*d_model+n_nodes) exactly.
        self.routing_dim = routing_dim
        router_input_dim = 2 * d_model + n_nodes
        if routing_dim is not None:
            self.routing_bottleneck = nn.Linear(router_input_dim, routing_dim)
            router_input_dim = routing_dim
        else:
            self.routing_bottleneck = None
        self.router = nn.Linear(
            router_input_dim,
            n_nodes * n_nodes,
        )
        self.norm = nn.LayerNorm(d_model)
        if self.relation_aware:
            self.relative_node_bias = nn.Parameter(
                torch.zeros(n_nodes, 4)
            )
            self.register_parameter("lm_bias", None)
        else:
            self.lm_bias = nn.Parameter(torch.zeros(16))
        self.start_logits = nn.Parameter(
            torch.cat(
                (
                    torch.tensor([4.0]),
                    torch.full((n_nodes - 1,), -4.0),
                )
            )
        )
        if n_nodes == 4:
            adjacency = torch.tensor(
                [
                    [1, 1, 0, 0],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                ],
                dtype=torch.bool,
            )
        else:
            source = torch.arange(n_nodes)[:, None]
            destination = torch.arange(n_nodes)[None, :]
            adjacency = (
                destination.eq(source)
                | destination.eq((source - 1).remainder(n_nodes))
                | destination.eq((source + 1).remainder(n_nodes))
            )
        self.register_buffer("adjacency", adjacency)
        if d_model == 8 and n_nodes == 4 and routing_dim is None:
            assert self.parameter_count == 980, (self.variant, self.parameter_count)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _memory(self, h, q, relay, observed_length):
        batch, length, _ = h.shape
        write = torch.ones(length, device=h.device, dtype=h.dtype)
        if self.controlled_writing:
            fraction = relay / max(self.relays - 1, 1)
            write[observed_length:] = fraction
        occupancy = q.mean(1, keepdim=True).expand_as(q)
        if self.relation_aware:
            rel_bias = torch.einsum("ijc,nc->ijn", relative_code(length, h.device), self.relative_node_bias)
            scores = q.clamp_min(1e-9).log()[:, None, :, :] + rel_bias[None]
            scores = scores + write.clamp_min(1e-9).log()[None, None, :, None]
            alpha = scores.softmax(dim=2)
            return torch.einsum("bijn,bjd->bind", alpha, h), occupancy
        weighted = q.unsqueeze(-1) * h.unsqueeze(2) * write[None, :, None, None]
        state_sum = weighted.sum(1, keepdim=True).expand(-1, length, -1, -1)
        mass = (q * write[None, :, None]).sum(1, keepdim=True).unsqueeze(-1)
        return state_sum / mass.clamp_min(1e-6), occupancy

    def forward(self, observed, *, output_slots=14, temperature=0.35, return_diagnostics=False):
        batch, observed_length = observed.shape
        length = observed_length + output_slots
        if not self.anchored and length > 16:
            raise ValueError("Unanchored learned positions support at most 16 total slots")
        mask = self.token_embedding.weight.mean(0).view(1, 1, self.d_model).expand(batch, output_slots, -1)
        content = torch.cat([self.token_embedding(observed), mask], 1)
        addresses = fixed_code(length, 4, observed.device)
        if self.anchored:
            address_state = self.address_to_state(addresses)
            address_edges = self.address_to_edges(addresses)
            address_out = self.address_to_readout(addresses)
            h = self.norm(content + address_state[None])
        else:
            h = self.norm(content + self.position_embedding.weight[:length][None])
        q = F.softmax(
            self.start_logits / temperature,
            -1,
        ).view(1, 1, self.n_nodes).expand(
            batch,
            length,
            self.n_nodes,
        )
        traces = [q]
        for relay in range(self.relays):
            node_memory, occupancy = self._memory(h, q, relay, observed_length)
            context = (q.unsqueeze(-1) * node_memory).sum(2)
            router_input = torch.cat([h, context, occupancy], -1)
            if self.routing_bottleneck is not None:
                router_input = torch.tanh(self.routing_bottleneck(router_input))
            edge_logits = self.router(router_input)
            if self.anchored:
                edge_logits = edge_logits + address_edges[None]
            edge_logits = edge_logits.view(
                batch,
                length,
                self.n_nodes,
                self.n_nodes,
            ).masked_fill(
                ~self.adjacency.view(
                    1,
                    1,
                    self.n_nodes,
                    self.n_nodes,
                ),
                -1e9,
            )
            transition = F.softmax(edge_logits / temperature, -1)
            q_next = torch.einsum("bts,btsd->btd", q, transition)
            step_state = self.step_projection(fixed_code(self.relays, 4, h.device)[relay])
            base = self.norm(h + context)
            if self.anchored:
                base = base + address_state[None]
            base = base + step_state
            destinations = torch.stack([
                torch.tanh(transform(base + self.node_embedding.weight[node]))
                for node, transform in enumerate(self.node_transforms)
            ], 2)
            h = self.norm(h + (q_next.unsqueeze(-1) * destinations).sum(2))
            q = q_next
            traces.append(q)
        read_h = h + address_out[None] if self.anchored else h
        logits = read_h[:, observed_length:] @ self.token_embedding.weight.t()
        if self.lm_bias is not None:
            logits = logits + self.lm_bias
        if return_diagnostics:
            return logits, {"locations": torch.stack(traces, 1), "hidden": h[:, observed_length:]}
        return logits


class IndependentSlots(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(16, 8)
        self.condition_type = nn.Embedding(2, 8)
        self.fc1 = nn.Linear(21, 27)
        self.fc2 = nn.Linear(27, 8)
        self.hidden_scale = nn.Parameter(torch.ones(()))
        self.readout_scale = nn.Parameter(torch.ones(()))
        self.lm_bias = nn.Parameter(torch.zeros(16))
        assert self.parameter_count == 980

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, observed):
        batch = observed.shape[0]
        condition = self.token_embedding(observed) + self.condition_type.weight[None]
        condition = condition.reshape(batch, 1, 16).expand(-1, 14, -1)
        address = fixed_code(16, 5, observed.device)[2:].unsqueeze(0).expand(batch, -1, -1)
        hidden = self.hidden_scale * F.gelu(self.fc1(torch.cat([condition, address], -1)))
        hidden = self.readout_scale * self.fc2(hidden)
        return hidden @ self.token_embedding.weight.t() + self.lm_bias


class OneShotTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(16, 6)
        self.position_embedding = nn.Embedding(16, 6)
        self.attention = nn.MultiheadAttention(6, 2, batch_first=True)
        self.norm1 = nn.LayerNorm(6)
        self.ff1, self.ff2 = nn.Linear(6, 44), nn.Linear(44, 6)
        self.norm2 = nn.LayerNorm(6)
        self.attention_scale = nn.Parameter(torch.ones(()))
        self.ff_scale = nn.Parameter(torch.ones(()))
        self.lm_bias = nn.Parameter(torch.zeros(16))
        assert sum(p.numel() for p in self.parameters()) == 980

    def forward(self, observed):
        batch = observed.shape[0]
        mask = self.token_embedding.weight.mean(0).view(1, 1, 6).expand(batch, 14, -1)
        h = torch.cat([self.token_embedding(observed), mask], 1) + self.position_embedding.weight[None]
        attended, _ = self.attention(h, h, h, need_weights=False)
        h = self.norm1(h + self.attention_scale * attended)
        h = self.norm2(h + self.ff_scale * self.ff2(F.gelu(self.ff1(h))))
        return h[:, 2:] @ self.token_embedding.weight.t() + self.lm_bias


def mutual_information(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.flatten().cpu(), y.flatten().cpu()
    total, result = x.numel(), 0.0
    for xv in x.unique():
        px = (x == xv).float().mean().item()
        for yv in y.unique():
            joint = ((x == xv) & (y == yv)).float().mean().item()
            if joint:
                py = (y == yv).float().mean().item()
                result += joint * math.log(joint / (px * py) + 1e-12, 2)
    return result


@torch.no_grad()
def evaluate(model, device):
    observed, targets, rules = make_full_batch(48, device, exhaustive=True)
    if isinstance(model, GTRFixedSlots):
        logits, diag = model(observed, return_diagnostics=True)
    else:
        logits, diag = model(observed), None
    pred = logits.argmax(-1)
    correct = pred == targets
    step = rules + 1
    step_consistency = (((pred[:, 1:] - pred[:, :-1]).remainder(12)) == step[:, None]).float().mean()
    phase_scores = []
    for b in range(48):
        period = 12 // math.gcd(int(step[b]), 12)
        phase_scores.append(max(
            (pred[b] == targets[b].roll(-shift)).float().mean().item() for shift in range(period)
        ))
    result = {
        "token_accuracy": correct.float().mean().item(),
        "correct_out_of_14": correct.sum(1).float().mean().item(),
        "exact_sequence": correct.all(1).float().mean().item(),
        "step_consistency": step_consistency.item(),
        "phase_aligned_accuracy": sum(phase_scores) / len(phase_scores),
        "loss": F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1)).item(),
        "accuracy_by_position": correct.float().mean(0).cpu().tolist(),
    }
    if diag is not None:
        paths = diag["locations"][:, :, 2:].argmax(-1).permute(0, 2, 1)
        path_id = sum(paths[:, :, r] * (4**r) for r in range(paths.shape[-1]))
        positions = torch.arange(14, device=device)[None].expand(48, -1)
        target_flat = targets
        rule_flat = rules[:, None].expand(-1, 14)
        norm_h = F.normalize(diag["hidden"], dim=-1)
        similarity = torch.einsum("btd,bsd->bts", norm_h, norm_h)
        offdiag = ~torch.eye(14, dtype=torch.bool, device=device)
        result.update({
            "mi_route_rule": mutual_information(path_id, rule_flat),
            "mi_route_position": mutual_information(path_id, positions),
            "mi_route_target": mutual_information(path_id, target_flat),
            "slot_cosine_similarity": similarity[:, offdiag].mean().item(),
        })
    return result


def train_model(model, steps, batch_size, seed, device):
    torch.manual_seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    for step in range(steps):
        observed, targets, _ = make_full_batch(batch_size, device)
        temperature = max(0.35, 1.25 * (0.997**step))
        if isinstance(model, GTRFixedSlots):
            logits = model(observed, temperature=temperature)
        else:
            logits = model(observed)
        # Headline experiments deliberately use cross-entropy only.
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return evaluate(model, device)


def build(name, relays=4):
    key = name.lower()
    if key == "independent": return IndependentSlots()
    if key == "transformer": return OneShotTransformer()
    return GTRFixedSlots(key, relays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gtr0", "gtra", "gtraw", "gtrar", "gtrarw"])
    parser.add_argument("--relays", nargs="+", type=int, default=[4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="gtr_a_results.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in args.models:
        relay_values = args.relays if name.startswith("gtr") else [0]
        for relays in relay_values:
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = build(name, relays or 4)
                metrics = train_model(model, args.steps, args.batch_size, seed, device)
                row = {"model": name, "relays": relays, "seed": seed, "steps": args.steps, **metrics}
                rows.append(row)
                print(f"{name} R={relays} seed={seed} correct={metrics['correct_out_of_14']:.2f}/14 "
                      f"exact={metrics['exact_sequence']:.3f} step={metrics['step_consistency']:.3f} "
                      f"phase={metrics['phase_aligned_accuracy']:.3f}", flush=True)
    path = Path(args.output)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"results={path.resolve()}")


if __name__ == "__main__":
    main()
