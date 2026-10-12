"""Fixed-slot, simultaneous-sequence Graph Trace Relay proof of concept."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from gtr_toy import GraphTraceRelayLM, make_rule_batch


def make_full_batch(batch_size: int, device: torch.device):
    inputs, targets = make_rule_batch(batch_size, seq_len=16, device=device)
    full_sequence = torch.cat([inputs[:, :1], targets], dim=1)
    # Only these two tokens are revealed; positions 2..15 are fixed output slots.
    observed = full_sequence[:, :2]
    return observed, full_sequence[:, 2:]


def simultaneous_forward(
    model: GraphTraceRelayLM,
    observed: torch.Tensor,
    *,
    output_slots: int = 14,
    temperature: float = 1.0,
    return_trace: bool = False,
):
    """Predict every output slot together; no target token is fed to the model."""
    batch, observed_length = observed.shape
    total_length = observed_length + output_slots
    cfg = model.cfg
    positions = torch.arange(total_length, device=observed.device)

    observed_h = model.token_embedding(observed)
    # Parameter-free shared MASK state, deliberately identical across output slots
    # before position information is added.
    mask_h = model.token_embedding.weight.mean(dim=0).view(1, 1, -1)
    mask_h = mask_h.expand(batch, output_slots, -1)
    h = torch.cat([observed_h, mask_h], dim=1)
    h = model.norm(h + model.position_embedding(positions)[None])

    q = model._initial_locations(batch, total_length, temperature, False)
    location_trace = [q]
    for relay_step in range(cfg.relay_steps):
        weighted_states = q.unsqueeze(-1) * h.unsqueeze(2)
        # Fully synchronous formation: every slot reads the previous state of the
        # entire flock. There is no causal prefix and no within-hop update order.
        node_state_sum = weighted_states.sum(dim=1, keepdim=True).expand(
            -1, total_length, -1, -1
        )
        node_mass = q.sum(dim=1, keepdim=True).unsqueeze(-1).expand(
            -1, total_length, -1, -1
        )
        occupancy = q.mean(dim=1, keepdim=True).expand_as(q)
        node_memory = node_state_sum / node_mass.clamp_min(1e-6)
        context = (q.unsqueeze(-1) * node_memory).sum(dim=2)

        router_input = torch.cat([h, context, occupancy], dim=-1)
        edge_logits = model.router(router_input).view(
            batch, total_length, cfg.n_nodes, cfg.n_nodes
        )
        edge_logits = edge_logits.masked_fill(
            ~model.adjacency.view(1, 1, cfg.n_nodes, cfg.n_nodes), -1e9
        )
        transition = F.softmax(edge_logits / temperature, dim=-1)
        q_next = torch.einsum("bts,btsd->btd", q, transition)

        base = model.norm(h + context + model.step_embedding.weight[relay_step])
        destination_outputs = torch.stack(
            [
                torch.tanh(transform(base + model.node_embedding.weight[node]))
                for node, transform in enumerate(model.node_transforms)
            ],
            dim=2,
        )
        update = (q_next.unsqueeze(-1) * destination_outputs).sum(dim=2)
        h = model.norm(h + update)
        q = q_next
        location_trace.append(q)

    logits = h[:, observed_length:] @ model.token_embedding.weight.t() + model.lm_bias
    if return_trace:
        return logits, torch.stack(location_trace, dim=1)
    return logits


def train_demo(steps=2000, batch_size=128, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphTraceRelayLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    print(f"device={device} trainable_parameters={model.parameter_count}")

    for step in range(steps + 1):
        model.train()
        observed, targets = make_full_batch(batch_size, device)
        temperature = max(0.35, 1.25 * (0.997**step))
        logits, locations = simultaneous_forward(
            model, observed, temperature=temperature, return_trace=True
        )
        sequence_loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        routed = locations[:, 1:]
        mean_load = routed.mean(dim=(0, 1, 2))
        balance_loss = ((mean_load - 0.25) ** 2).mean()
        route_entropy = -(
            routed.clamp_min(1e-9) * routed.clamp_min(1e-9).log()
        ).sum(-1).mean()
        sharpen = 0.0 if step < steps // 2 else 0.002
        loss = sequence_loss + 0.05 * balance_loss + sharpen * route_entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            slot_correct = logits.argmax(-1) == targets
            accuracy = slot_correct.float().mean().item()
            correct_out_of_14 = slot_correct.sum(dim=1).float().mean().item()
            exact = slot_correct.all(dim=1).float().mean().item()
            print(
                f"step={step:4d} loss={sequence_loss.item():.4f} "
                f"correct={correct_out_of_14:.2f}/14 token_acc={accuracy:.3f} "
                f"exact_sequence={exact:.3f} "
                f"temperature={temperature:.3f}",
                flush=True,
            )

    model.eval()
    observed, targets = make_full_batch(1, device)
    with torch.no_grad():
        logits, locations = simultaneous_forward(
            model, observed, temperature=0.35, return_trace=True
        )
    predictions = logits.argmax(-1)
    sample_correct = (predictions == targets).sum().item()
    paths = locations.argmax(-1)[0].transpose(0, 1)
    print("\nobserved:", observed[0].tolist())
    print("target:  ", targets[0].tolist())
    print("parallel:", predictions[0].tolist())
    print(f"correct:  {sample_correct}/14 (token and position must both match)")
    print("\nnode paths (2 observed tokens followed by 14 simultaneous slots):")
    print(paths.cpu())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_demo(args.steps, args.batch_size, args.seed)
