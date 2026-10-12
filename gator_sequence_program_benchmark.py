"""Simultaneous execution of compositional sequence-transformation programs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from gator import OptimizationEngine
from gator.cli import (
    add_optimization_arguments,
    optimization_config_from_args,
)
from gator_convergent import ConvergentGator


VALUE_COUNT = 8
OP_TOKEN_OFFSET = 8
OP_NAMES = (
    "noop",
    "reverse",
    "rotate_left",
    "rotate_right",
    "swap_pairs",
    "complement",
    "prefix_xor",
    "sort",
)


@dataclass(frozen=True)
class ProgramBatch:
    observed: torch.Tensor
    targets: torch.Tensor
    values: torch.Tensor
    operations: torch.Tensor


def apply_operation(values: torch.Tensor, operation: torch.Tensor) -> torch.Tensor:
    """Apply one independently selected operation to every batch row."""

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, length]")
    if operation.shape != values.shape[:1]:
        raise ValueError("operation must have shape [batch]")
    batch, length = values.shape
    if length < 2:
        raise ValueError("sequence length must be at least two")

    pair_indices = torch.arange(length, device=values.device)
    paired = pair_indices.clone()
    even_limit = length - (length % 2)
    paired[:even_limit:2] += 1
    paired[1:even_limit:2] -= 1

    prefix_xor = values.clone()
    for index in range(1, length):
        prefix_xor[:, index] = (
            prefix_xor[:, index - 1] ^ values[:, index]
        )

    candidates = torch.stack(
        (
            values,
            values.flip(1),
            values.roll(-1, dims=1),
            values.roll(1, dims=1),
            values[:, paired],
            (VALUE_COUNT - 1) - values,
            prefix_xor,
            values.sort(dim=1).values,
        ),
        dim=1,
    )
    rows = torch.arange(batch, device=values.device)
    return candidates[rows, operation]


def execute_program(
    values: torch.Tensor,
    operations: torch.Tensor,
) -> torch.Tensor:
    """Execute locally selected instructions without storing route templates."""

    if operations.ndim != 2 or operations.shape[0] != values.shape[0]:
        raise ValueError("operations must have shape [batch, program_length]")
    result = values
    for column in range(operations.shape[1]):
        result = apply_operation(result, operations[:, column])
    return result


def make_program_batch(
    batch_size: int,
    sequence_length: int,
    program_length: int,
    active_operations: int | torch.Tensor,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None,
    sampling_device: torch.device | str | None = None,
) -> ProgramBatch:
    """Generate fresh values and a fixed-width curriculum program."""

    if min(batch_size, sequence_length, program_length) <= 0:
        raise ValueError("batch, sequence, and program lengths must be positive")
    device = torch.device(device)
    sampling_device = torch.device(sampling_device or device)
    if isinstance(active_operations, int):
        if not 1 <= active_operations <= program_length:
            raise ValueError(
                "active_operations must be within the program length"
            )
        active_counts = torch.full(
            (batch_size,),
            active_operations,
            dtype=torch.long,
            device=sampling_device,
        )
    else:
        if active_operations.shape != (batch_size,):
            raise ValueError(
                "active_operations tensor must have shape [batch]"
            )
        active_counts = active_operations.to(
            device=sampling_device,
            dtype=torch.long,
        )
        if (
            active_counts.lt(1).any()
            or active_counts.gt(program_length).any()
        ):
            raise ValueError(
                "active_operations values must be within the program length"
            )
    values = torch.randint(
        0,
        VALUE_COUNT,
        (batch_size, sequence_length),
        generator=generator,
        device=sampling_device,
    )
    sampled_operations = torch.randint(
        1,
        len(OP_NAMES),
        (batch_size, program_length),
        generator=generator,
        device=sampling_device,
    )
    positions = torch.arange(
        program_length,
        device=sampling_device,
    )[None]
    operations = torch.where(
        positions < active_counts[:, None],
        sampled_operations,
        torch.zeros_like(sampled_operations),
    )
    targets = execute_program(values, operations)
    observed = torch.cat(
        (values, operations + OP_TOKEN_OFFSET),
        dim=1,
    )
    return ProgramBatch(
        observed=observed.to(device),
        targets=targets.to(device),
        values=values.to(device),
        operations=operations.to(device),
    )


@torch.no_grad()
def evaluate(
    model: ConvergentGator,
    *,
    sequence_length: int,
    program_length: int,
    active_operations: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    sampling_device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device=sampling_device).manual_seed(
        seed + 1009 * active_operations
    )
    batch = make_program_batch(
        batch_size,
        sequence_length,
        program_length,
        active_operations,
        device,
        generator=generator,
        sampling_device=sampling_device,
    )
    logits = model(batch.observed, output_slots=sequence_length)
    predictions = logits.argmax(-1)
    correct = predictions.eq(batch.targets)
    return {
        "loss": F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.targets.reshape(-1),
        ).item(),
        "accuracy": correct.float().mean().item(),
        "correct": correct.sum(1).float().mean().item(),
        "exact": correct.all(1).float().mean().item(),
    }


def active_operations_for_step(
    step: int,
    total_steps: int,
    program_length: int,
    curriculum_end: float,
) -> int:
    if program_length == 1:
        return 1
    curriculum_steps = max(1, int(total_steps * curriculum_end))
    stage_width = curriculum_steps / program_length
    return min(program_length, 1 + int(step / stage_width))


def sample_replay_depths(
    batch_size: int,
    current_depth: int,
    *,
    replay_fraction: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Keep the current depth dominant while replaying all earlier depths."""

    if batch_size <= 0 or current_depth <= 0:
        raise ValueError("batch size and current depth must be positive")
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("replay_fraction must be in [0, 1)")
    device = torch.device(device)
    depths = torch.full(
        (batch_size,),
        current_depth,
        dtype=torch.long,
        device=device,
    )
    if current_depth == 1 or replay_fraction == 0.0:
        return depths
    replay = torch.rand(
        batch_size,
        device=device,
        generator=generator,
    ).lt(replay_fraction)
    earlier = torch.randint(
        1,
        current_depth,
        (batch_size,),
        device=device,
        generator=generator,
    )
    return torch.where(replay, earlier, depths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--program-length", type=int, default=3)
    parser.add_argument("--curriculum-end", type=float, default=0.60)
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.0,
        help=(
            "fraction of rows replayed uniformly across earlier program "
            "depths after each curriculum transition"
        ),
    )
    parser.add_argument(
        "--mastery-accuracy",
        type=float,
        default=0.99,
        help=(
            "stop when held-out full-program token accuracy reaches this "
            "value; set 1.0 to require perfect token accuracy"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--relays", type=int, default=2)
    parser.add_argument("--lanes", type=int, default=10)
    parser.add_argument(
        "--connection-layout",
        choices=("ring", "small_world", "dense"),
        default="small_world",
    )
    parser.add_argument("--connection-seed", type=int, default=0)
    parser.add_argument("--connection-rewire", type=float, default=0.50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--portable-batches", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    add_optimization_arguments(parser)
    parser.set_defaults(learning_rate=0.008, weight_decay=1e-4)
    args = parser.parse_args()

    if min(
        args.steps,
        args.batch_size,
        args.eval_batch_size,
        args.log_every,
        args.sequence_length,
        args.program_length,
    ) <= 0:
        parser.error("training, evaluation, and sequence sizes must be positive")
    if not 0.0 < args.curriculum_end <= 1.0:
        parser.error("--curriculum-end must be in (0, 1]")
    if not 0.0 <= args.replay_fraction < 1.0:
        parser.error("--replay-fraction must be in [0, 1)")
    if not 0.0 < args.mastery_accuracy <= 1.0:
        parser.error("--mastery-accuracy must be in (0, 1]")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_device = (
        torch.device("cpu") if args.portable_batches else device
    )
    generator = torch.Generator(device=sampling_device).manual_seed(
        args.seed + 1_000_003
    )
    model = ConvergentGator(
        n_nodes=args.nodes,
        d_model=args.width,
        degree=args.degree,
        relays=args.relays,
        lanes=args.lanes,
        connection_layout=args.connection_layout,
        connection_seed=args.connection_seed,
        connection_rewire=args.connection_rewire,
    ).to(device)
    train_model = torch.compile(model) if args.compile else model
    engine = OptimizationEngine(
        model,
        optimization_config_from_args(args),
    )
    checkpoint = args.checkpoint or Path(
        "experimental_checkpoints/sequence_program/"
        f"gator_sequence_program_seed{args.seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_exact = -1.0
    started = interval_started = time.perf_counter()
    interval_step = 0
    last_stage = None

    print(
        f"device={device} parameters={model.parameter_count} "
        f"sequence={args.sequence_length} program={args.program_length} "
        f"operations={len(OP_NAMES)} nodes={args.nodes} width={args.width} "
        f"lanes={args.lanes} relays={args.relays} degree={args.degree} "
        f"layout={args.connection_layout} rewire={args.connection_rewire:g} "
        f"replay_fraction={args.replay_fraction:g} "
        f"mastery_accuracy={args.mastery_accuracy:g}",
        flush=True,
    )

    executed_steps = 0
    mastery_reached = False
    for step in range(args.steps):
        current_depth = active_operations_for_step(
            step,
            args.steps,
            args.program_length,
            args.curriculum_end,
        )
        if current_depth != last_stage:
            print(
                f"curriculum step={step + 1} "
                f"current_depth={current_depth}",
                flush=True,
            )
            last_stage = current_depth
        active_depths = sample_replay_depths(
            args.batch_size,
            current_depth,
            replay_fraction=args.replay_fraction,
            device=sampling_device,
            generator=generator,
        )
        batch = make_program_batch(
            args.batch_size,
            args.sequence_length,
            args.program_length,
            active_depths,
            device,
            generator=generator,
            sampling_device=sampling_device,
        )
        temperature = max(0.35, 1.25 * (0.997**step))
        logits = train_model(
            batch.observed,
            output_slots=args.sequence_length,
            temperature=temperature,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.targets.reshape(-1),
        )
        step_result = engine.step(loss)
        completed = step + 1
        executed_steps = completed
        if completed % args.log_every:
            continue

        now = time.perf_counter()
        speed = (completed - interval_step) / (now - interval_started)
        stage_results = {
            stage: evaluate(
                model,
                sequence_length=args.sequence_length,
                program_length=args.program_length,
                active_operations=stage,
                batch_size=args.eval_batch_size,
                device=device,
                seed=args.seed + 20_000,
                sampling_device=sampling_device,
            )
            for stage in range(1, args.program_length + 1)
        }
        hardest = stage_results[args.program_length]
        print(
            f"step={completed} steps_per_second={speed:.2f} "
            f"current_depth={current_depth} "
            f"train_depth_mean="
            f"{batch.operations.ne(0).sum(1).float().mean().item():.2f} "
            f"op1_acc={stage_results[1]['accuracy']:.3f} "
            f"op2_acc={stage_results.get(2, hardest)['accuracy']:.3f} "
            f"op3_acc={hardest['accuracy']:.3f} "
            f"op3_correct={hardest['correct']:.2f}/{args.sequence_length} "
            f"op3_exact={hardest['exact']:.3f} "
            f"op3_loss={hardest['loss']:.4f} "
            f"grad_norm={step_result.gradient_norm:.4f}",
            flush=True,
        )
        state = {
            "step": completed,
            "model": model.state_dict(),
            "training": engine.state_dict(),
            "batch_generator_state": generator.get_state(),
            "best_exact": max(best_exact, hardest["exact"]),
            "config": vars(args),
            "operations": OP_NAMES,
            "results": stage_results,
        }
        torch.save(state, checkpoint)
        if hardest["exact"] > best_exact:
            best_exact = hardest["exact"]
            torch.save(
                state,
                checkpoint.with_name(
                    f"{checkpoint.stem}.best{checkpoint.suffix}"
                ),
            )
        if hardest["accuracy"] >= args.mastery_accuracy:
            mastery_reached = True
            print(
                f"mastery_reached step={completed} "
                f"op3_accuracy={hardest['accuracy']:.4f} "
                f"threshold={args.mastery_accuracy:.4f}",
                flush=True,
            )
            break
        interval_started = now
        interval_step = completed

    elapsed = time.perf_counter() - started
    print(
        f"runtime_seconds={elapsed:.2f} "
        f"steps_per_second={executed_steps / max(elapsed, 1e-9):.2f} "
        f"completed_steps={executed_steps} "
        f"mastery_reached={mastery_reached} "
        f"best_op3_exact={best_exact:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
