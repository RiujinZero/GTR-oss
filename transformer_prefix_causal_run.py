"""Train a configurable AUTOREGRESSIVE (causal, teacher-forced) Transformer
on parallel prefix formation -- decodes output slots one token at a time,
in contrast to transformer_prefix_run.py's one-shot simultaneous
prediction of all slots. Reports both teacher-forced metrics (diagnostic:
how well the model predicts each next token given the TRUE previous
tokens) and free-running generated metrics (the real test: how well the
model does when only fed its own previous predictions, the way it would
actually be used)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gator import OptimizationEngine
from gator.cli import add_optimization_arguments, optimization_config_from_args
from parallel_prefix_benchmark import PrefixTransformerCausal, make_prefix_batch


def evaluate_causal(model, length, device, seed=12345, batch_size=4096, sampling_device=None):
    sampling_device = torch.device(sampling_device or device)
    generator = torch.Generator(device=sampling_device).manual_seed(seed + length)
    observed, targets = make_prefix_batch(
        batch_size, length, device, generator=generator, sampling_device=sampling_device,
    )

    # Teacher-forced: predicts each position from the TRUE previous tokens.
    tf_logits = model(observed, targets)
    tf_correct = tf_logits.argmax(-1) == targets

    # Free-running: predicts each position from its OWN previous predictions
    # -- this is the real test of autoregressive generation.
    generated = model.generate(observed, length)
    gen_correct = generated == targets

    return {
        "length": length,
        "teacher_forced_loss": F.cross_entropy(tf_logits.reshape(-1, 16), targets.reshape(-1)).item(),
        "teacher_forced_correct": tf_correct.sum(1).float().mean().item(),
        "teacher_forced_accuracy": tf_correct.float().mean().item(),
        "teacher_forced_exact": tf_correct.all(1).float().mean().item(),
        "generated_correct": gen_correct.sum(1).float().mean().item(),
        "generated_accuracy": gen_correct.float().mean().item(),
        "generated_exact": gen_correct.all(1).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=14)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--ff-width", type=int, default=52)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--portable-batches",
        action="store_true",
        help="use the same dedicated CPU batch stream as Convergent Gator",
    )
    parser.add_argument("--checkpoint-dir", default="experimental_checkpoints")
    add_optimization_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrefixTransformerCausal(
        args.d_model, args.heads, args.ff_width, n_layers=args.layers,
    ).to(device)
    optimization = optimization_config_from_args(args)
    engine = OptimizationEngine(model, optimization)
    sampling_device = torch.device("cpu") if args.portable_batches else device
    batch_generator = torch.Generator(device=sampling_device).manual_seed(args.seed + 1_000_003)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"transformer_causal_l{args.layers}_d{args.d_model}_p{model.parameter_count}_seed{args.seed}"
    latest_path = checkpoint_dir / f"{stem}_latest.pt"
    best_path = checkpoint_dir / f"{stem}_best.pt"
    best_generated_accuracy = -1.0

    print(
        f"device={device} parameters={model.parameter_count} layers={args.layers} "
        f"d_model={args.d_model} heads={args.heads} ff_width={args.ff_width} "
        f"portable_batches={args.portable_batches}"
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    interval_started = started
    interval_step = 0

    for step in range(args.steps):
        observed, targets = make_prefix_batch(
            args.batch_size, 7, device, generator=batch_generator, sampling_device=sampling_device,
        )
        logits = model(observed, targets)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        step_result = engine.step(loss)

        if (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            speed = ((step + 1) - interval_step) / (now - interval_started)
            result = evaluate_causal(
                model, 7, device, batch_size=args.eval_batch_size, sampling_device=sampling_device,
            )
            state = {
                "step": step + 1,
                "model": model.state_dict(),
                "training": engine.state_dict(),
                "architecture": {
                    "kind": "causal_teacher_forced_transformer",
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
            if result["generated_accuracy"] > best_generated_accuracy:
                best_generated_accuracy = result["generated_accuracy"]
                torch.save(state, best_path)
            print(
                f"step={step+1} steps_per_second={speed:.2f} "
                f"tf_correct={result['teacher_forced_correct']:.2f}/7 "
                f"tf_acc={result['teacher_forced_accuracy']:.3f} "
                f"tf_exact={result['teacher_forced_exact']:.3f} "
                f"tf_loss={result['teacher_forced_loss']:.4f} "
                f"gen_correct={result['generated_correct']:.2f}/7 "
                f"gen_acc={result['generated_accuracy']:.3f} "
                f"gen_exact={result['generated_exact']:.3f} "
                f"lr={step_result.learning_rate:.6g} "
                f"grad_norm={step_result.gradient_norm:.4f} "
                f"best_gen_acc={best_generated_accuracy:.3f}",
                flush=True,
            )
            interval_started = time.perf_counter()
            interval_step = step + 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = evaluate_causal(model, 7, device, sampling_device=sampling_device)
    print(
        f"final tf_correct={final['teacher_forced_correct']:.2f}/7 "
        f"tf_acc={final['teacher_forced_accuracy']:.3f} tf_exact={final['teacher_forced_exact']:.3f} "
        f"tf_loss={final['teacher_forced_loss']:.4f} "
        f"gen_correct={final['generated_correct']:.2f}/7 "
        f"gen_acc={final['generated_accuracy']:.3f} gen_exact={final['generated_exact']:.3f}"
    )
    print(f"runtime_seconds={elapsed:.2f} steps_per_second={args.steps / elapsed:.2f}")
    print(f"best_checkpoint={best_path}")
    print(f"latest_checkpoint={latest_path}")


if __name__ == "__main__":
    main()
