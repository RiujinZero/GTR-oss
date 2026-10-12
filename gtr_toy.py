"""A 980-parameter Graph Trace Relay (GTR) language-model toy.

Run:
    python gtr_toy.py --steps 1000

The model is parallel across batch and token positions. It is recurrent only across
relay hops, exactly as a token moving through a graph must be.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GTRConfig:
    vocab_size: int = 16
    d_model: int = 8
    n_nodes: int = 4
    max_len: int = 16
    relay_steps: int = 4


class GraphTraceRelayLM(nn.Module):
    """Small recurrent graph router for next-token prediction.

    State at relay hop r:
      h[b, t, d]  = token representation
      q[b, t, n]  = probability/mass of token t being at graph node n

    At every hop, all token positions update together. The router creates a
    token-specific transition matrix P[source_node, destination_node], then
    moves q through it and mixes destination-node transforms into h.
    """

    def __init__(self, cfg: GTRConfig = GTRConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_len, cfg.d_model)
        self.node_embedding = nn.Embedding(cfg.n_nodes, cfg.d_model)
        self.step_embedding = nn.Embedding(cfg.relay_steps, cfg.d_model)

        # A tiny transform/expert at each graph node.
        self.node_transforms = nn.ModuleList(
            nn.Linear(cfg.d_model, cfg.d_model) for _ in range(cfg.n_nodes)
        )

        # Inputs: current token state, cross-token context, graph occupancy.
        # Outputs: a complete N x N transition matrix per token.
        router_in = 2 * cfg.d_model + cfg.n_nodes
        self.router = nn.Linear(router_in, cfg.n_nodes * cfg.n_nodes)

        self.norm = nn.LayerNorm(cfg.d_model)

        # Output projection is weight-tied to token_embedding, so only a bias is new.
        self.lm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        # Nearly all route mass starts at node 0, but the starting point is learnable.
        self.start_logits = nn.Parameter(
            torch.tensor([4.0] + [-4.0] * (cfg.n_nodes - 1))
        )

        # Fixed supergraph. The weights on its allowed edges are dynamic.
        adjacency = torch.tensor(
            [
                [1, 1, 0, 0],  # node 0 -> {0, 1}
                [0, 1, 1, 1],  # node 1 -> {1, 2, 3}
                [1, 0, 1, 1],  # node 2 -> {0, 2, 3}
                [1, 1, 0, 1],  # node 3 -> {0, 1, 3}
            ],
            dtype=torch.bool,
        )
        if adjacency.shape != (cfg.n_nodes, cfg.n_nodes):
            raise ValueError("The default adjacency matrix requires n_nodes=4.")
        self.register_buffer("adjacency", adjacency)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _straight_through_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
        """One-hot forward pass, softmax gradient backward pass."""
        indices = probabilities.argmax(dim=-1)
        hard = F.one_hot(indices, num_classes=probabilities.size(-1)).to(
            probabilities.dtype
        )
        return hard + probabilities - probabilities.detach()

    def _initial_locations(
        self, batch: int, length: int, temperature: float, hard: bool
    ) -> torch.Tensor:
        soft = F.softmax(self.start_logits / temperature, dim=-1)
        locations = self._straight_through_one_hot(soft) if hard else soft
        return locations.view(1, 1, -1).expand(batch, length, -1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        temperature: float = 1.0,
        hard_routes: bool = False,
        return_trace: bool = False,
        causal: bool = True,
    ):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence_length]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        batch, length = tokens.shape
        cfg = self.cfg
        if length > cfg.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len={cfg.max_len}")

        positions = torch.arange(length, device=tokens.device)
        h = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        h = self.norm(h)

        q = self._initial_locations(batch, length, temperature, hard_routes)
        location_trace = [q]
        transition_trace = []

        prefix_denominator = torch.arange(
            1, length + 1, device=tokens.device, dtype=h.dtype
        ).view(1, length, 1)

        for relay_step in range(cfg.relay_steps):
            # Coupling among token trajectories. This uses only tensor operations;
            # there is no loop over token positions.
            # Tokens influence one another through node-local memory: tokens that
            # occupy the same node contribute more strongly to one another.
            weighted_states = q.unsqueeze(-1) * h.unsqueeze(2)
            if causal:
                node_state_sum = weighted_states.cumsum(dim=1)
                node_mass = q.cumsum(dim=1).unsqueeze(-1)
                occupancy = q.cumsum(dim=1) / prefix_denominator
            else:
                node_state_sum = weighted_states.sum(dim=1, keepdim=True).expand(
                    -1, length, -1, -1
                )
                node_mass = q.sum(dim=1, keepdim=True).unsqueeze(-1).expand(
                    -1, length, -1, -1
                )
                occupancy = q.mean(dim=1, keepdim=True).expand_as(q)

            node_memory = node_state_sum / node_mass.clamp_min(1e-6)
            context = (q.unsqueeze(-1) * node_memory).sum(dim=2)

            router_input = torch.cat([h, context, occupancy], dim=-1)
            edge_logits = self.router(router_input).view(
                batch, length, cfg.n_nodes, cfg.n_nodes
            )
            edge_logits = edge_logits.masked_fill(
                ~self.adjacency.view(1, 1, cfg.n_nodes, cfg.n_nodes), -1e9
            )

            transition = F.softmax(edge_logits / temperature, dim=-1)
            if hard_routes:
                transition = self._straight_through_one_hot(transition)

            # Move every token's node mass through its dynamic transition matrix.
            # q_next[d] = sum_s q[s] * transition[s, d]
            q_next = torch.einsum("bts,btsd->btd", q, transition)

            base = self.norm(
                h + context + self.step_embedding.weight[relay_step]
            )
            destination_outputs = torch.stack(
                [
                    torch.tanh(
                        transform(base + self.node_embedding.weight[node])
                    )
                    for node, transform in enumerate(self.node_transforms)
                ],
                dim=2,
            )  # [batch, length, node, d_model]

            update = (q_next.unsqueeze(-1) * destination_outputs).sum(dim=2)
            h = self.norm(h + update)
            q = q_next

            location_trace.append(q)
            transition_trace.append(transition)

        logits = h @ self.token_embedding.weight.t() + self.lm_bias
        if not return_trace:
            return logits

        trace = {
            "locations": torch.stack(location_trace, dim=1),
            # shape: [batch, relay_step + 1, token_position, node]
            "transitions": torch.stack(transition_trace, dim=1),
            # shape: [batch, relay_step, token_position, source_node, dest_node]
        }
        return logits, trace


def make_rule_batch(
    batch_size: int, seq_len: int = 16, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic next-token task requiring information from an earlier token.

    The first token (0..3) selects a modular stepping rule. Data tokens are
    4..15. The first data token is random, then the selected rule determines
    the remainder of the sequence.
    """
    if seq_len < 3:
        raise ValueError("seq_len must be at least 3")

    rule = torch.randint(0, 4, (batch_size,), device=device)
    sequence = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    sequence[:, 0] = rule
    sequence[:, 1] = 4 + torch.randint(0, 12, (batch_size,), device=device)

    for position in range(2, seq_len):
        step_size = rule + 1
        sequence[:, position] = 4 + (
            (sequence[:, position - 1] - 4 + step_size) % 12
        )

    return sequence[:, :-1], sequence[:, 1:]


def train_demo(
    *, steps: int = 1000, batch_size: int = 128, seed: int = 0
) -> GraphTraceRelayLM:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTraceRelayLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)

    print(f"device={device}  trainable_parameters={model.parameter_count}")

    for training_step in range(steps + 1):
        model.train()
        inputs, targets = make_rule_batch(batch_size, device=device)
        temperature = max(0.35, 1.25 * (0.997**training_step))

        logits, trace = model(
            inputs,
            temperature=temperature,
            hard_routes=False,  # soft first; discretize or fine-tune later
            return_trace=True,
            causal=True,
        )

        # Position 0 predicts a deliberately random first data token, so omit it.
        lm_loss = F.cross_entropy(
            logits[:, 1:].reshape(-1, model.cfg.vocab_size),
            targets[:, 1:].reshape(-1),
        )

        locations = trace["locations"][:, 1:]  # omit starting locations
        mean_load = locations.mean(dim=(0, 1, 2))
        balance_loss = ((mean_load - 1.0 / model.cfg.n_nodes) ** 2).mean()
        route_entropy = -(
            locations.clamp_min(1e-9) * locations.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()

        # Balance keeps every node alive. A tiny late entropy penalty sharpens paths.
        sharpen_weight = 0.0 if training_step < steps // 2 else 2e-3
        loss = lm_loss + 0.05 * balance_loss + sharpen_weight * route_entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if training_step % 100 == 0 or training_step == steps:
            with torch.no_grad():
                predictions = logits[:, 1:].argmax(dim=-1)
                accuracy = (predictions == targets[:, 1:]).float().mean().item()
            loads = mean_load.detach().cpu().numpy().round(2)
            print(
                f"step={training_step:4d}  lm_loss={lm_loss.item():.4f}  "
                f"accuracy={accuracy:.3f}  temperature={temperature:.3f}  "
                f"node_load={loads}"
            )

    # Inspect a soft run and turn its distributions into an argmax trace.
    model.eval()
    inputs, targets = make_rule_batch(1, device=device)
    with torch.no_grad():
        logits, trace = model(
            inputs,
            temperature=0.35,
            hard_routes=False,
            return_trace=True,
            causal=True,
        )
        predictions = logits.argmax(dim=-1)
        paths = trace["locations"].argmax(dim=-1)[0].transpose(0, 1)

    print("\ninput: ", inputs[0].tolist())
    print("target:", targets[0].tolist())
    print("pred:  ", predictions[0].tolist())
    print("\nargmax node trace: rows=tokens, columns=start + relay hops")
    print(paths.cpu())

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_demo(steps=args.steps, batch_size=args.batch_size, seed=args.seed)
