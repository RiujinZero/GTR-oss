"""Train Gator-Path on simultaneous modular parallel-prefix formation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from gator_path.prefix import GatorPathPrefixConfig, GatorPathPrefixModel
from parallel_prefix_benchmark import make_prefix_batch


def _temperature(
    step: int,
    *,
    start: float,
    minimum: float,
    decay: float,
    hold_steps: int = 0,
) -> float:
    effective_step = max(step - hold_steps, 0)
    return max(minimum, start * (decay**effective_step))


@torch.no_grad()
def evaluate_path_prefix(
    model: GatorPathPrefixModel,
    *,
    length: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    temperature: float,
    hard_ingress: bool = False,
    hard_handoffs: bool = False,
) -> dict[str, float]:
    """Evaluate on a deterministic fresh generated batch."""

    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + length)
    observed, targets = make_prefix_batch(
        batch_size,
        length,
        device,
        generator=generator,
    )
    if model.config.frontier_policy:
        ingress_frontier = model.initial_frontier(
            observed,
            output_slots=length,
            temperature=temperature,
            hard_routes=hard_ingress,
        )
        ingress_cell_probability = ingress_frontier.mass.sum(dim=3)
        ingress_load = ingress_cell_probability.mean(dim=(0, 1))
        ingress_confidence = ingress_cell_probability.max(dim=-1).values.mean()
    else:
        ingress_state, _ = model.initial_state(
            observed,
            output_slots=length,
            temperature=temperature,
        )
        ingress_load = torch.bincount(
            ingress_state.cell_id.reshape(-1),
            minlength=model.config.num_cells,
        ).to(model.token_embedding.weight.dtype)
        ingress_load = ingress_load / ingress_load.sum().clamp_min(1.0)
        ingress_confidence = torch.tensor(
            1.0,
            device=device,
        )
    positive_ingress = ingress_load > 0
    ingress_entropy = -(
        ingress_load[positive_ingress]
        * ingress_load[positive_ingress].log()
    ).sum()
    logits, diagnostics = model(
        observed,
        output_slots=length,
        temperature=temperature,
        hard_ingress=hard_ingress,
        hard_handoffs=hard_handoffs,
        return_diagnostics=True,
    )
    correct = logits.argmax(dim=-1).eq(targets)
    route_entropy_values: list[Tensor] = []
    movement_values: list[Tensor] = []
    occupied_rows = 0.0
    for relay_result in diagnostics.relay_results:
        if model.config.frontier_policy:
            route_entropy_values.append(relay_result.route_entropy)
            movement_values.append(relay_result.move_fraction)
            occupied_rows += float(relay_result.occupied_locations)
        else:
            probabilities = relay_result.route.joint_probabilities.clamp_min(
                torch.finfo(logits.dtype).tiny
            )
            route_entropy_values.append(
                -(probabilities * probabilities.log()).sum(
                    dim=(-2, -1)
                ).mean()
            )
            moved = relay_result.route.destination_cell[..., 0].ne(
                relay_result.source_cell
            ) | relay_result.route.destination_node[..., 0].ne(
                relay_result.source_node
            )
            movement_values.append(moved.float().mean())
            occupied_rows += relay_result.mailbox.occupied_count
    metrics = {
        "loss": float(
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
            )
        ),
        "correct": float(correct.sum(dim=1).float().mean()),
        "accuracy": float(correct.float().mean()),
        "exact": float(correct.all(dim=1).float().mean()),
        "route_entropy": float(torch.stack(route_entropy_values).mean()),
        "route_move_fraction": float(torch.stack(movement_values).mean()),
        "occupied_rows": occupied_rows / diagnostics.relay_steps,
        "ingress_cells": float(positive_ingress.sum()),
        "ingress_entropy": float(ingress_entropy),
        "ingress_effective_cells": float(ingress_entropy.exp()),
        "ingress_confidence": float(ingress_confidence),
    }
    model.train(was_training)
    return metrics


def _checkpoint_payload(
    *,
    model: GatorPathPrefixModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    best_accuracy: float,
    architecture: GatorPathPrefixConfig,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": "gator-path-prefix-local-frontier-v3",
        "step": step,
        "best_accuracy": best_accuracy,
        "architecture": asdict(architecture),
        "arguments": vars(arguments),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=10_000)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=2048)
    parser.add_argument("--length", type=_positive_int, default=7)
    parser.add_argument("--cells", type=_positive_int, default=1)
    parser.add_argument("--payload-width", type=_positive_int, default=8)
    parser.add_argument("--address-width", type=_positive_int, default=4)
    parser.add_argument("--relays", type=_positive_int, default=4)
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=10,
        help=(
            "number of bounded local alternatives exposed to the current "
            "relay's surrogate gradient"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=8.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--minimum-lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature-start", type=float, default=1.25)
    parser.add_argument("--temperature-min", type=float, default=0.35)
    parser.add_argument("--temperature-decay", type=float, default=0.997)
    parser.add_argument(
        "--temperature-hold-steps",
        type=int,
        default=0,
    )
    parser.add_argument("--hard-routes", action="store_true")
    parser.add_argument(
        "--hard-ingress",
        action="store_true",
        help=(
            "use a straight-through top-1 first steward while leaving "
            "downstream handoffs independently configurable"
        ),
    )
    parser.add_argument(
        "--hard-handoffs",
        action="store_true",
        help=(
            "use straight-through top-1 local relay handoffs while leaving "
            "ingress independently configurable"
        ),
    )
    parser.add_argument(
        "--local-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "commit one route per relay, discard alternatives, and use "
            "their soft mixture only as a training surrogate"
        ),
    )
    parser.add_argument(
        "--learned-ingress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="learn the first steward cell/node instead of starting in cell 0",
    )
    parser.add_argument(
        "--frontier-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "carry a genuine soft current packet frontier early in training "
            "and let repeated local cell policies anneal it toward one path"
        ),
    )
    parser.add_argument(
        "--private-cell-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "give each frontier cell its own tied token input/readout table; "
            "disable only for the earlier shared-I/O ablation"
        ),
    )
    parser.add_argument(
        "--relation-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use receiver-specific relative-position reads inside each local "
            "cell/node mailbox"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=12_345)
    parser.add_argument("--log-every", type=_positive_int, default=500)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the training forward pass with torch.compile",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "learning_rate",
        "gradient_clip",
        "temperature_start",
        "temperature_min",
        "temperature_decay",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        parser.error("--weight-decay must be finite and non-negative")
    if args.minimum_lr < 0 or not math.isfinite(args.minimum_lr):
        parser.error("--minimum-lr must be finite and non-negative")
    if args.temperature_min > args.temperature_start:
        parser.error("--temperature-min cannot exceed --temperature-start")
    if args.temperature_hold_steps < 0:
        parser.error("--temperature-hold-steps cannot be negative")

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    architecture = GatorPathPrefixConfig(
        num_cells=args.cells,
        payload_width=args.payload_width,
        address_width=args.address_width,
        relay_steps=args.relays,
        top_k=args.top_k,
        straight_through_hard=args.hard_routes,
        local_policy=args.local_policy,
        learned_ingress=args.learned_ingress,
        frontier_policy=args.frontier_policy,
        private_cell_io=args.private_cell_io,
        relation_aware=args.relation_aware,
    )
    model = GatorPathPrefixModel(architecture).to(device)
    train_model = torch.compile(model) if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.steps,
            eta_min=args.minimum_lr,
        )
    else:
        scheduler = None

    start_step = 0
    best_accuracy = -1.0
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        saved_architecture = checkpoint.get("architecture")
        if saved_architecture != asdict(architecture):
            parser.error(
                "--resume architecture does not match the requested model"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        if start_step > args.steps:
            parser.error(
                f"checkpoint step {start_step} exceeds --steps {args.steps}"
            )
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if (
            device.type == "cuda"
            and checkpoint.get("cuda_rng_state") is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    gpu = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "none"
    )
    print(
        f"device={device.type} gpu={gpu!r} parameters={model.parameter_count} "
        f"cells={args.cells} relays={args.relays} top_k={args.top_k} "
        f"length={args.length} hard_routes={args.hard_routes} "
        f"hard_ingress={args.hard_ingress} "
        f"hard_handoffs={args.hard_handoffs} "
        f"local_policy={args.local_policy} "
        f"learned_ingress={args.learned_ingress} "
        f"frontier_policy={args.frontier_policy} "
        f"private_cell_io={args.private_cell_io} "
        f"relation_aware={args.relation_aware} "
        f"compiled={args.compile} optimizer=adamw",
        flush=True,
    )
    initial_temperature = _temperature(
        start_step,
        start=args.temperature_start,
        minimum=args.temperature_min,
        decay=args.temperature_decay,
        hold_steps=args.temperature_hold_steps,
    )
    initial = evaluate_path_prefix(
        model,
        length=args.length,
        device=device,
        batch_size=args.eval_batch_size,
        seed=args.eval_seed,
        temperature=initial_temperature,
        hard_ingress=args.hard_routes or args.hard_ingress,
        hard_handoffs=args.hard_routes or args.hard_handoffs,
    )
    print(
        f"step={start_step} correct={initial['correct']:.2f}/{args.length} "
        f"acc={initial['accuracy']:.3f} exact={initial['exact']:.3f} "
        f"loss={initial['loss']:.4f} route_entropy={initial['route_entropy']:.3f} "
        f"ingress_cells={initial['ingress_cells']:.0f} "
        f"ingress_entropy={initial['ingress_entropy']:.3f} "
        f"ingress_effective={initial['ingress_effective_cells']:.2f} "
        f"ingress_conf={initial['ingress_confidence']:.3f} "
        f"occupied={initial['occupied_rows']:.1f}",
        flush=True,
    )

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    interval_started = started
    interval_step = start_step
    latest_gradient_norm = torch.tensor(float("nan"), device=device)

    for step in range(start_step, args.steps):
        observed, targets = make_prefix_batch(
            args.batch_size,
            args.length,
            device,
        )
        temperature = _temperature(
            step,
            start=args.temperature_start,
            minimum=args.temperature_min,
            decay=args.temperature_decay,
            hold_steps=args.temperature_hold_steps,
        )
        logits = train_model(
            observed,
            output_slots=args.length,
            temperature=temperature,
            hard_ingress=args.hard_routes or args.hard_ingress,
            hard_handoffs=args.hard_routes or args.hard_handoffs,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, architecture.vocab_size),
            targets.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        latest_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        completed = step + 1
        if completed % args.log_every != 0 and completed != args.steps:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        interval_rate = (completed - interval_step) / (
            now - interval_started
        )
        metrics = evaluate_path_prefix(
            model,
            length=args.length,
            device=device,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed,
            temperature=temperature,
            hard_ingress=args.hard_routes or args.hard_ingress,
            hard_handoffs=args.hard_routes or args.hard_handoffs,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"step={completed} steps_per_second={interval_rate:.2f} "
            f"correct={metrics['correct']:.2f}/{args.length} "
            f"acc={metrics['accuracy']:.3f} exact={metrics['exact']:.3f} "
            f"loss={metrics['loss']:.4f} train_loss={float(loss.detach()):.4f} "
            f"lr={current_lr:.6g} temperature={temperature:.3f} "
            f"grad_norm={float(latest_gradient_norm):.4f} "
            f"route_entropy={metrics['route_entropy']:.3f} "
            f"route_move={metrics['route_move_fraction']:.3f} "
            f"ingress_cells={metrics['ingress_cells']:.0f} "
            f"ingress_entropy={metrics['ingress_entropy']:.3f} "
            f"ingress_effective={metrics['ingress_effective_cells']:.2f} "
            f"ingress_conf={metrics['ingress_confidence']:.3f} "
            f"occupied={metrics['occupied_rows']:.1f}",
            flush=True,
        )
        interval_started = time.perf_counter()
        interval_step = completed

        if args.checkpoint_dir is not None:
            improved = metrics["accuracy"] > best_accuracy
            best_accuracy = max(best_accuracy, metrics["accuracy"])
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed,
                best_accuracy=best_accuracy,
                architecture=architecture,
                arguments=args,
            )
            _save_checkpoint(
                payload,
                args.checkpoint_dir / "gator_path_prefix_latest.pt",
            )
            if improved:
                _save_checkpoint(
                    payload,
                    args.checkpoint_dir / "gator_path_prefix_best.pt",
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    final_temperature = _temperature(
        max(args.steps - 1, 0),
        start=args.temperature_start,
        minimum=args.temperature_min,
        decay=args.temperature_decay,
        hold_steps=args.temperature_hold_steps,
    )
    final = evaluate_path_prefix(
        model,
        length=args.length,
        device=device,
        batch_size=args.eval_batch_size,
        seed=args.eval_seed,
        temperature=final_temperature,
        hard_ingress=args.hard_routes or args.hard_ingress,
        hard_handoffs=args.hard_routes or args.hard_handoffs,
    )
    print(
        f"final correct={final['correct']:.2f}/{args.length} "
        f"acc={final['accuracy']:.3f} exact={final['exact']:.3f} "
        f"loss={final['loss']:.4f} runtime_seconds={runtime:.2f} "
        f"steps_per_second={(args.steps - start_step) / max(runtime, 1e-9):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
