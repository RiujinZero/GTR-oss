"""Train a configurable one-shot Transformer on parallel prefix formation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gator import OptimizationEngine
from gator.cli import add_optimization_arguments, optimization_config_from_args
from parallel_prefix_benchmark import PrefixTransformer980, evaluate, make_prefix_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=144)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-width", type=int, default=576)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--portable-batches",
        action="store_true",
        help=(
            "use the same dedicated CPU batch stream as Convergent Gator"
        ),
    )
    parser.add_argument("--checkpoint-dir", default="experimental_checkpoints")
    add_optimization_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrefixTransformer980(
        args.d_model,
        args.heads,
        args.ff_width,
        n_layers=args.layers,
        require_980=False,
    ).to(device)
    train_model = torch.compile(model) if args.compile else model
    optimization = optimization_config_from_args(args)
    engine = OptimizationEngine(model, optimization)
    sampling_device = (
        torch.device("cpu") if args.portable_batches else device
    )
    batch_generator = torch.Generator(
        device=sampling_device
    ).manual_seed(args.seed + 1_000_003)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"transformer_l{args.layers}_d{args.d_model}_p{model.parameter_count}_seed{args.seed}"
    latest_path = checkpoint_dir / f"{stem}_latest.pt"
    best_path = checkpoint_dir / f"{stem}_best.pt"
    best_accuracy = -1.0

    print(
        f"device={device} parameters={model.parameter_count} layers={args.layers} "
        f"d_model={args.d_model} heads={args.heads} ff_width={args.ff_width} "
        f"compiled={args.compile} portable_batches={args.portable_batches}"
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    interval_started = started
    interval_step = 0

    for step in range(args.steps):
        observed, targets = make_prefix_batch(
            args.batch_size,
            7,
            device,
            generator=batch_generator,
            sampling_device=sampling_device,
        )
        logits = train_model(observed, output_slots=7)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        step_result = engine.step(loss)

        if (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            speed = (
                (step + 1) - interval_step
            ) / (now - interval_started)
            result = evaluate(
                model,
                7,
                device,
                batch_size=args.eval_batch_size,
                sampling_device=sampling_device,
            )
            state = {
                "step": step + 1,
                "model": model.state_dict(),
                "training": engine.state_dict(),
                "architecture": {
                    "kind": "one_shot_transformer",
                    "d_model": args.d_model,
                    "heads": args.heads,
                    "ff_width": args.ff_width,
                    "layers": args.layers,
                    "parameters": model.parameter_count,
                },
                "metrics": result,
                "batch_generator_state": batch_generator.get_state(),
            }
            torch.save(state, latest_path)
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                torch.save(state, best_path)
            print(
                f"step={step+1} steps_per_second={speed:.2f} "
                f"correct={result['correct']:.2f}/7 "
                f"acc={result['accuracy']:.3f} exact={result['exact']:.3f} "
                f"loss={result['loss']:.4f} lr={step_result.learning_rate:.6g} "
                f"grad_norm={step_result.gradient_norm:.4f} best={best_accuracy:.3f}",
                flush=True,
            )
            interval_started = time.perf_counter()
            interval_step = step + 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = evaluate(
        model,
        7,
        device,
        sampling_device=sampling_device,
    )
    print(
        f"final correct={final['correct']:.2f}/7 acc={final['accuracy']:.3f} "
        f"exact={final['exact']:.3f} loss={final['loss']:.4f}"
    )
    print(f"runtime_seconds={elapsed:.2f} steps_per_second={args.steps / elapsed:.2f}")
    print(f"best_checkpoint={best_path}")
    print(f"latest_checkpoint={latest_path}")


if __name__ == "__main__":
    main()
