"""Find the smallest one-shot Transformer matching GTR-ARW on length-7 prefixes."""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from parallel_prefix_benchmark import PrefixTransformer980, evaluate, make_prefix_batch


CONFIGS = [
    (4, 1, 8),
    (6, 2, 24),
    (8, 2, 32),
    (12, 3, 48),
    (14, 2, 56),
    (16, 4, 64),
    (20, 4, 80),
    (24, 4, 96),
    (28, 4, 112),
    (32, 4, 128),
    (40, 5, 160),
    (48, 6, 192),
    (64, 8, 256),
    (80, 8, 320),
]


def train(config, *, layers=4, steps=5_000, batch_size=128, seed=0, device="cpu"):
    d_model, heads, ff_width = config
    torch.manual_seed(seed)
    model = PrefixTransformer980(d_model, heads, ff_width, n_layers=layers, require_980=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    torch.manual_seed(seed)
    for _ in range(steps):
        observed, targets = make_prefix_batch(batch_size, 7, device)
        logits = model(observed, output_slots=7)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    result = evaluate(model, 7, torch.device(device))
    result.update({"d_model": d_model, "heads": heads, "ff_width": ff_width,
                   "parameters": model.parameter_count, "layers": layers,
                   "steps": steps, "seed": seed})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--max-parameters", type=int, default=100_000)
    parser.add_argument("--min-parameters", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.22)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    path = Path(f"transformer_prefix_scaling_{args.layers}layer.csv")
    for config in CONFIGS:
        torch.manual_seed(0)
        probe = PrefixTransformer980(*config, n_layers=args.layers, require_980=False)
        if not args.min_parameters <= probe.parameter_count <= args.max_parameters:
            continue
        result = train(config, layers=args.layers, steps=args.steps, device=device)
        rows.append(result)
        print(f"params={result['parameters']:6d} d={config[0]:2d} ff={config[2]:3d} "
              f"correct={result['correct']:.2f}/7 acc={result['accuracy']:.3f} "
              f"loss={result['loss']:.4f}", flush=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        if result["accuracy"] >= args.target:
            break
    print(f"results={path.resolve()}")


if __name__ == "__main__":
    main()
