"""10k-step, matched-parameter GTR vs Transformer curriculum sanity test."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gtr_toy import GraphTraceRelayLM
from transformer_toy import TinyTransformerLM


def make_curriculum_batch(batch_size, order, *, seq_len=16, device="cpu"):
    """Order-k modular recurrence; the first k data values are random context."""
    rule = torch.randint(0, 4, (batch_size,), device=device)
    sequence = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    sequence[:, 0] = rule
    sequence[:, 1 : order + 1] = 4 + torch.randint(
        0, 12, (batch_size, order), device=device
    )
    for position in range(order + 1, seq_len):
        history = sequence[:, position - order : position] - 4
        weights = torch.arange(1, order + 1, device=device)
        value = (history * weights).sum(dim=1) + rule + 1
        sequence[:, position] = 4 + value.remainder(12)
    # Predictable logits begin at index `order` (target sequence position order+1).
    return sequence[:, :-1], sequence[:, 1:], order


def predictive_loss(logits, targets, start):
    return F.cross_entropy(
        logits[:, start:].reshape(-1, 16), targets[:, start:].reshape(-1)
    )


@torch.no_grad()
def evaluate(model, kind, order, device, seed, batch_size=512):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed + order)
    inputs, targets, start = make_curriculum_batch(batch_size, order, device=device)
    if kind == "gtr":
        logits = model(inputs, temperature=0.35, causal=True)
    else:
        logits = model(inputs)
    loss = predictive_loss(logits, targets, start)
    accuracy = (logits[:, start:].argmax(-1) == targets[:, start:]).float().mean()
    torch.random.set_rng_state(state)
    return loss.item(), accuracy.item()


def main(steps=10_000, stage_steps=1_000, batch_size=128, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtr = GraphTraceRelayLM().to(device)
    transformer = TinyTransformerLM().to(device)
    assert gtr.parameter_count == transformer.parameter_count == 980
    models = {"gtr": gtr, "transformer": transformer}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
        for name, model in models.items()
    }
    out_path = Path(__file__).with_name("curriculum_results.csv")
    rows = []
    started = time.perf_counter()
    print(f"device={device} parameters=980 steps={steps} stage_steps={stage_steps}")

    for step in range(steps):
        order = min(10, step // stage_steps + 1)
        inputs, targets, start = make_curriculum_batch(batch_size, order, device=device)

        # GTR objective, preserving its route regularization.
        temperature = max(0.35, 1.25 * (0.997**step))
        g_logits, trace = gtr(
            inputs, temperature=temperature, return_trace=True, causal=True
        )
        g_lm = predictive_loss(g_logits, targets, start)
        locations = trace["locations"][:, 1:]
        mean_load = locations.mean(dim=(0, 1, 2))
        balance = ((mean_load - 0.25) ** 2).mean()
        entropy = -(locations.clamp_min(1e-9) * locations.clamp_min(1e-9).log()).sum(-1).mean()
        g_loss = g_lm + 0.05 * balance + (0.002 if step >= steps // 2 else 0.0) * entropy
        optimizers["gtr"].zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gtr.parameters(), 1.0)
        optimizers["gtr"].step()

        t_logits = transformer(inputs)
        t_loss = predictive_loss(t_logits, targets, start)
        optimizers["transformer"].zero_grad(set_to_none=True)
        t_loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizers["transformer"].step()

        if (step + 1) % stage_steps == 0 or step + 1 == steps:
            elapsed = time.perf_counter() - started
            for eval_order in range(1, order + 1):
                for name, model in models.items():
                    model.eval()
                    loss, accuracy = evaluate(model, name, eval_order, device, seed + 50_000)
                    rows.append(
                        {"step": step + 1, "trained_order": order, "eval_order": eval_order,
                         "model": name, "loss": loss, "accuracy": accuracy,
                         "elapsed_seconds": elapsed}
                    )
                    model.train()
            current = [r for r in rows if r["step"] == step + 1 and r["eval_order"] == order]
            summary = "  ".join(
                f"{r['model']} loss={r['loss']:.4f} acc={r['accuracy']:.3f}" for r in current
            )
            print(f"step={step+1:5d} order={order:2d}  {summary}  elapsed={elapsed:.1f}s", flush=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"results={out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--stage-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.steps, args.stage_steps, args.batch_size, args.seed)
