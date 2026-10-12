"""Train GTR-L on simultaneous modular parallel-prefix formation.

This is the first validation gate for the linear-scaling design: it must
escape chance at a comparable step to the existing Gator-Path model, reach at
least 99% token accuracy, keep every stream and node alive, and scale
approximately linearly from 40 to 160 nodes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from collections import deque
from dataclasses import asdict
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.nn import functional as F

from gator.diagnostics import AdamWWeightTracer, ProcoimWeightTracer
from gator.optimizers.procoim import (
    ProcoimAnchor,
    add_procoim_arguments,
    procoim_config_from_args,
)
from gator.optimizers.slot_loss import (
    SlotLossResult,
    batch_centered_cross_entropy_slot_loss,
    cross_entropy_slot_loss,
)
from gator.training import (
    ProcoimConsolidationResult,
    ProcoimRetentionWindow,
    ProcoimTrainingEngine,
)
from gator.continual.replay_reservoir import ReplayReservoir
from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.procoim_s_retention import ProcoimSRetentionWindow
from gator_linear.core import GatorLinearConfig
from gator_linear.prefix import GatorLinearPrefixConfig, GatorLinearPrefixModel
from parallel_prefix_benchmark import (
    make_balanced_prefix_batch,
    make_prefix_batch,
)


def _temperature(
    step: int,
    *,
    start: float,
    minimum: float,
    decay: float,
) -> float:
    return max(minimum, start * (decay**step))


@torch.no_grad()
def evaluate_linear_prefix(
    model: GatorLinearPrefixModel,
    *,
    length: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    """Evaluate on a deterministic fresh batch and report health diagnostics."""

    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + length)
    observed, targets = make_prefix_batch(
        batch_size, length, device, generator=generator
    )
    logits, diagnostics = model(
        observed,
        output_slots=length,
        temperature=temperature,
        return_diagnostics=True,
    )
    correct = logits.argmax(dim=-1).eq(targets)
    valid_correct = (
        logits[..., 4:].argmax(dim=-1).add(4).eq(targets)
    )
    per_position_loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction="none",
    ).mean(dim=0)
    valid_vocabulary_loss = F.cross_entropy(
        logits[..., 4:].transpose(1, 2),
        targets - 4,
    )

    # Health metrics that the acceptance criteria depend on: a stream or node
    # that stops receiving mass early is the documented failure mode.
    fusion = diagnostics["fusion_weights"].mean(dim=(0, 1))
    occupancy = diagnostics["relays"][-1]["occupancy"]
    node_load = occupancy.mean(dim=(0, 1, 2))
    node_entropy = -(
        occupancy.clamp_min(1e-9).log() * occupancy
    ).sum(-1).mean()
    route_entropy = torch.stack(
        [
            -(
                record["edge_probability"].clamp_min(1e-9).log()
                * record["edge_probability"]
            ).sum(-1).mean()
            for record in diagnostics["relays"]
        ]
    ).mean()
    bus_gate = torch.stack(
        [record["bus_gate"].mean() for record in diagnostics["relays"]]
    ).mean()

    # Coalition health: whether the streams remain complementary voters or
    # have collapsed into one voter duplicated S times. The reference solves
    # this task only as a coalition -- no single trace exceeds ~33% -- so
    # tracking this is as informative as accuracy while accuracy is flat.
    per_stream, stream_weights = model.stream_logits(
        observed, output_slots=length, temperature=temperature
    )
    predictions = per_stream.argmax(-1)
    streams = predictions.shape[2]
    off_diagonal = ~torch.eye(
        streams, dtype=torch.bool, device=predictions.device
    )
    if streams == 1:
        agreement = logits.new_tensor(1.0)
    else:
        agreement = (
            predictions.unsqueeze(2) == predictions.unsqueeze(3)
        ).float()[..., off_diagonal].mean()
    # Per-stream accuracy. The diagnostic is the *spread*, not the level:
    # streams that all score the same are one voter duplicated S times, which
    # is the collapse signature whatever that shared score happens to be.
    # (The reference's ~33% is what its solution happened to settle on, not a
    # target.)
    per_stream_accuracy = (
        predictions.eq(targets.unsqueeze(-1)).float().mean(dim=(0, 1))
    )
    best_single = float(per_stream_accuracy.max())
    worst_single = float(per_stream_accuracy.min())
    stream_spread = float(per_stream_accuracy.std(unbiased=False))

    metrics = {
        "valid_vocabulary_loss": float(valid_vocabulary_loss),
        "loss": float(
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
            )
        ),
        "stream_agreement": float(agreement),
        "best_single_stream": best_single,
        "worst_single_stream": worst_single,
        "stream_accuracy_spread": stream_spread,
        "correct": float(correct.sum(dim=1).float().mean()),
        "accuracy": float(correct.float().mean()),
        "valid_accuracy": float(valid_correct.float().mean()),
        "valid_exact": float(
            valid_correct.all(dim=1).float().mean()
        ),
        "valid_by_position": (
            valid_correct.float().mean(dim=0).tolist()
        ),
        "by_position": correct.float().mean(dim=0).tolist(),
        "loss_by_position": per_position_loss.tolist(),
        "exact": float(correct.all(dim=1).float().mean()),
        "route_entropy": float(route_entropy),
        "node_entropy": float(node_entropy),
        "min_node_load": float(node_load.min()),
        "max_node_load": float(node_load.max()),
        "min_stream_weight": float(fusion.min()),
        "max_stream_weight": float(fusion.max()),
        "bus_gate": float(bus_gate),
    }
    model.train(was_training)
    return metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _select_anchor_sentinels(
    anchors: Sequence[ProcoimAnchor],
    *,
    count: int,
    episode: int,
    exclude_id: Any | None = None,
) -> tuple[ProcoimAnchor, ...]:
    """Choose rotating sentinels, excluding the current example itself."""

    if count <= 0 or not anchors:
        return ()
    # Pending anchors follow permanent anchors and therefore replace an
    # older record with the same stable ID inside a provisional window.
    # Anonymous legacy anchors remain distinct by object identity.
    unique: dict[Any, ProcoimAnchor] = {}
    for anchor in anchors:
        if anchor.anchor_id == exclude_id:
            continue
        key = (
            ("anchor_id", anchor.anchor_id)
            if anchor.anchor_id is not None
            else ("object_id", id(anchor))
        )
        unique[key] = anchor
    bank = tuple(unique.values())
    if not bank:
        return ()
    selected = min(count, len(bank))
    start = (episode * selected) % len(bank)
    return tuple(bank[(start + offset) % len(bank)] for offset in range(selected))


def _upsert_anchor(
    anchors: deque[ProcoimAnchor],
    new_anchor: ProcoimAnchor,
) -> None:
    """Insert one representative anchor or strengthen its existing record."""

    for existing in anchors:
        if existing.anchor_id != new_anchor.anchor_id:
            continue
        if not existing.slotwise or not new_anchor.slotwise:
            if (
                new_anchor.best_aggregate is not None
                and (
                    existing.best_aggregate is None
                    or new_anchor.best_aggregate < existing.best_aggregate
                )
            ):
                existing.best_aggregate = new_anchor.best_aggregate
            return

        assert existing.start_losses is not None
        assert existing.best_losses is not None
        assert existing.valid_mask is not None
        assert existing.protected_mask is not None
        assert existing.hard_mask is not None
        assert new_anchor.start_losses is not None
        assert new_anchor.best_losses is not None
        assert new_anchor.valid_mask is not None
        assert new_anchor.protected_mask is not None
        assert new_anchor.hard_mask is not None
        if existing.best_losses.shape != new_anchor.best_losses.shape:
            raise ValueError("repeated anchor changed slot shape")
        newly_protected = (
            new_anchor.protected_mask & ~existing.protected_mask
        )
        existing.start_losses[newly_protected] = (
            new_anchor.start_losses[newly_protected]
        )
        existing.best_losses[newly_protected] = (
            new_anchor.best_losses[newly_protected]
        )
        shared = existing.protected_mask & new_anchor.protected_mask
        existing.best_losses[shared] = torch.minimum(
            existing.best_losses[shared],
            new_anchor.best_losses[shared],
        )
        existing.valid_mask |= new_anchor.valid_mask
        existing.protected_mask |= new_anchor.protected_mask
        existing.hard_mask |= new_anchor.hard_mask
        existing.best_aggregate = float(
            existing.best_losses[existing.valid_mask].mean()
        )
        return
    anchors.append(new_anchor)


def prefix_slot_loss_provider(
    model: GatorLinearPrefixModel,
    batch: tuple[torch.Tensor, torch.Tensor],
    forward_kwargs: dict[str, Any],
) -> SlotLossResult:
    """Evaluate the generated prefix task as independent output-slot CE."""

    observed, targets = batch
    model_kwargs = dict(forward_kwargs)
    valid_vocabulary_only = bool(
        model_kwargs.pop("_prefix_valid_vocabulary_only", False)
    )
    center_batch_logits = bool(
        model_kwargs.pop("_prefix_center_batch_logits", False)
    )
    logits = model(observed, **model_kwargs)
    if valid_vocabulary_only:
        logits = logits[..., 4:]
        targets = targets - 4
    if center_batch_logits:
        return batch_centered_cross_entropy_slot_loss(logits, targets)
    return cross_entropy_slot_loss(logits, targets)


@torch.no_grad()
def evaluate_prefix_batch(
    model: GatorLinearPrefixModel,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    temperature: float,
) -> dict[str, float]:
    """Evaluate a fixed ladder batch without altering module training modes."""

    was_training = model.training
    model.eval()
    observed, targets = batch
    logits = model(
        observed,
        output_slots=targets.shape[1],
        temperature=temperature,
    )
    correct = logits.argmax(dim=-1).eq(targets)
    result = {
        "loss": float(
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
            )
        ),
        "accuracy": float(correct.float().mean()),
        "exact": float(correct.all(dim=1).float().mean()),
    }
    model.train(was_training)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=10_000)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=2048)
    parser.add_argument("--length", type=_positive_int, default=7)
    parser.add_argument("--streams", type=_positive_int, default=10)
    parser.add_argument("--nodes", type=_positive_int, default=40)
    parser.add_argument("--width", type=_positive_int, default=8)
    parser.add_argument("--relays", type=_positive_int, default=4)
    parser.add_argument("--degree", type=_positive_int, default=6)
    parser.add_argument("--buses", type=_positive_int, default=4)
    parser.add_argument("--mailbox-channels", type=_positive_int, default=4)
    parser.add_argument("--mailbox-width", type=_positive_int, default=8)
    parser.add_argument("--basis-rank", type=_positive_int, default=4)
    parser.add_argument("--address-width", type=_positive_int, default=4)
    parser.add_argument("--router-rank", type=_positive_int, default=8)
    parser.add_argument("--conv-kernel", type=_positive_int, default=3)
    parser.add_argument(
        "--no-private-stream-io",
        dest="private_stream_io",
        action="store_false",
        help=(
            "share every relay projection across streams. Streams then apply "
            "the same function to near-identical inputs and converge, so the "
            "terminal coalition degenerates into one voter duplicated S "
            "times; the reference avoids this with per-cell private I/O"
        ),
    )
    parser.set_defaults(private_stream_io=True)
    parser.add_argument("--private-stream-rank", type=_positive_int, default=4)
    parser.add_argument(
        "--update-gate-bias", type=float, default=0.0,
        help=(
            "initial bias on each relay's residual gate. 0 admits half "
            "the context update per relay and LayerNorm renormalises "
            "after, diluting input influence once per relay; raise it to "
            "let the input path outweigh a constant readout"
        ),
    )
    parser.add_argument(
        "--hard-routing", action="store_true",
        help=(
            "commit to the argmax node in the forward pass (straight-through, "
            "soft gradient) instead of blending over every node -- the "
            "mechanism the reference architecture uses at every relay. "
            "Soft blending lets 'average toward the input-independent "
            "marginal' become a real, cheap forward-pass output; hard "
            "commit removes that point from the range entirely"
        ),
    )
    parser.add_argument(
        "--router-occupancy", action="store_true",
        help=(
            "feed each node's current load (mean occupancy) into the router "
            "as an extra input, the way the reference architecture's router "
            "sees occupancy directly -- without it, GTR-L's router is "
            "structurally blind to how crowded a node already is and can "
            "only balance load indirectly, via --route-balance-loss-weight"
        ),
    )
    parser.add_argument(
        "--no-readout-bias", dest="readout_bias", action="store_false",
        help=(
            "drop the vocabulary bias so the target marginal cannot be "
            "expressed for free, forcing it through the fused input path"
        ),
    )
    parser.set_defaults(readout_bias=True)
    parser.add_argument(
        "--heterogeneous-streams",
        action="store_true",
        help=(
            "give streams different active widths so they cannot converge "
            "to the same function; equal-width streams share a function "
            "space and were measured collapsing into it"
        ),
    )
    parser.add_argument(
        "--stream-width-min-fraction", type=float, default=0.25,
        help="narrowest stream as a fraction of --width",
    )
    parser.add_argument(
        "--no-scan-normalise",
        dest="scan_normalise",
        action="store_false",
        help=(
            "leave the decayed scans unnormalised so they carry running "
            "totals rather than running means; the normaliser grows as "
            "(l+1) whenever 1/(1-decay) exceeds the sequence length, which "
            "turns every long-range channel into a mean and leaves no "
            "channel able to express a long-range sum"
        ),
    )
    parser.set_defaults(scan_normalise=True)
    parser.add_argument(
        "--graph-layout",
        choices=("ring", "small_world", "dense"),
        default="small_world",
    )
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--graph-rewire", type=float, default=0.5)
    parser.add_argument("--ingress-temperature", type=float, default=4.0)
    parser.add_argument(
        "--route-entropy-loss-weight",
        type=float,
        default=0.0,
        help=(
            "penalise the entropy of each slot/stream node distribution, "
            "sharpening addressing without hard top-1 routing; pair with "
            "--route-balance-loss-weight or nodes will starve"
        ),
    )
    parser.add_argument(
        "--route-balance-loss-weight",
        type=float,
        default=0.0,
        help=(
            "penalise deviation of mean node load from uniform so every node "
            "stays alive while individual slots route sharply"
        ),
    )
    parser.add_argument(
        "--optimizer", choices=("adamw", "procoim"), default="adamw"
    )
    parser.add_argument(
        "--balanced-prefix-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "stratify training microbatches so every target token occurs "
            "equally often in every slot; batch size must be a multiple of "
            "12. Evaluation remains independently random"
        ),
    )
    parser.add_argument(
        "--prefix-episode-mode",
        choices=("fresh", "fixed", "two"),
        default="fresh",
        help=(
            "fresh samples every episode; fixed repeatedly optimizes one "
            "microbatch; two alternates two fixed microbatches for the "
            "v0.2 Procoim retention test ladder"
        ),
    )
    parser.add_argument(
        "--prefix-loss-vocabulary",
        choices=("full", "valid"),
        default="full",
        help=(
            "full trains over all 16 benchmark symbols; valid renormalizes "
            "CE over the 12 symbols the modular-prefix task can emit, "
            "removing invalid-token suppression as a search shortcut"
        ),
    )
    parser.add_argument(
        "--prefix-center-batch-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "for Procoim on balanced microbatches, subtract each physical "
            "slot's batch-mean token logits before candidate CE scoring; "
            "this removes the constant target-marginal shortcut while "
            "leaving ordinary validation logits unchanged"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=8.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--scheduler", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--minimum-lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature-start", type=float, default=1.25)
    parser.add_argument("--temperature-min", type=float, default=0.35)
    parser.add_argument("--temperature-decay", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=12_345)
    parser.add_argument("--log-every", type=_positive_int, default=500)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=("none", "bfloat16", "float16"), default="none",
        help=(
            "Autocast mixed-precision dtype for the forward pass and loss. "
            "bfloat16 needs no loss scaling (no dynamic-range overflow risk, "
            "unlike float16) and is the recommended choice on Ampere+ GPUs; "
            "float16 is supported for older hardware and uses GradScaler."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--adamw-weight-trace-every",
        type=_nonnegative_int,
        default=0,
        help=(
            "record AdamW weight/update geometry at this step interval; "
            "zero disables tracing"
        ),
    )
    parser.add_argument(
        "--adamw-weight-trace-dir",
        type=Path,
        default=None,
        help=(
            "trace output directory; when omitted, an enabled trace uses "
            "CHECKPOINT_DIR/adamw_weight_trace"
        ),
    )
    parser.add_argument(
        "--adamw-weight-trace-save-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "save full weights, interval deltas, clipped-gradient summaries, "
            "per-step update RMS, and Adam moments at every trace interval"
        ),
    )
    parser.add_argument(
        "--adamw-weight-trace-step-statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also aggregate exact gradients and update-path geometry after "
            "every optimizer step; disabled by default because the Python "
            "tensor walk can roughly halve CPU training throughput"
        ),
    )
    parser.add_argument(
        "--procoim-weight-trace-every",
        type=_nonnegative_int,
        default=0,
        help=(
            "record Procoim's exact parameter-space displacement at this "
            "forward-call interval, in the same interval_delta format "
            "AdamWWeightTracer uses so the two are directly comparable; "
            "zero disables tracing"
        ),
    )
    parser.add_argument(
        "--procoim-weight-trace-dir",
        type=Path,
        default=None,
        help=(
            "trace output directory; when omitted, an enabled trace uses "
            "CHECKPOINT_DIR/procoim_weight_trace"
        ),
    )
    parser.add_argument(
        "--procoim-weight-trace-save-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save full weights and interval deltas at every trace interval",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    add_procoim_arguments(parser)
    parser.add_argument(
        "--procoim-consolidation-mode",
        choices=("anchor", "utility"),
        default="anchor",
        help=(
            "consolidation criterion for --procoim-consolidation-interval-examples windows: "
            "'anchor' is the existing hard per-anchor audit (ProcoimRetentionWindow, unchanged "
            "default); 'utility' uses Procoim-S's aggregate-utility scorer against a stratified "
            "replay reservoir instead (ProcoimSRetentionWindow) -- the mechanism found to accept "
            "updates the anchor AND-gate rejects"
        ),
    )
    parser.add_argument(
        "--procoim-s-replay-capacity",
        type=_positive_int,
        default=256,
        help="ReplayReservoir capacity for --procoim-consolidation-mode utility",
    )
    parser.add_argument(
        "--procoim-s-replay-sample-size",
        type=_nonnegative_int,
        default=4,
        help=(
            "replay items sampled per consolidation alpha for "
            "--procoim-consolidation-mode utility"
        ),
    )
    parser.add_argument(
        "--procoim-s-replay-soft-tolerance",
        type=float,
        default=0.02,
        help="ProcoimSConfig.replay_soft_tolerance for --procoim-consolidation-mode utility",
    )
    parser.add_argument(
        "--procoim-s-replay-catastrophic-tolerance",
        type=float,
        default=0.20,
        help="ProcoimSConfig.replay_catastrophic_tolerance for --procoim-consolidation-mode utility",
    )
    parser.add_argument(
        "--procoim-s-minimum-utility",
        type=float,
        default=0.0,
        help="ProcoimSConfig.minimum_utility for --procoim-consolidation-mode utility",
    )
    parser.add_argument(
        "--procoim-distribution-sentinel-batch-size",
        type=_nonnegative_int,
        default=0,
        help=(
            "always replay one fixed balanced prefix batch after Gate 1; "
            "zero disables this broad cross-example non-regression guard"
        ),
    )
    parser.add_argument(
        "--procoim-distribution-sentinel-hard-tolerance-fraction",
        type=float,
        default=0.002,
        help=(
            "maximum fractional CE regression permitted on the balanced "
            "distribution sentinel; soft penalty begins at zero"
        ),
    )
    parser.add_argument(
        "--procoim-reference-guard-batch-size",
        type=_nonnegative_int,
        default=0,
        help=(
            "fixed independently generated eval-like batch that a locally "
            "eligible proposal must also improve; zero disables the gate"
        ),
    )
    parser.add_argument(
        "--procoim-reference-guard-seed",
        type=int,
        default=None,
        help=(
            "seed for the optimizer's fixed reference batch; defaults to a "
            "seed distinct from the untouched reporting evaluation"
        ),
    )
    parser.add_argument(
        "--procoim-past-reference-buffer-size",
        type=_nonnegative_int,
        default=0,
        help=(
            "reservoir of prior episode batches; one uniformly selected "
            "past partner gates each fresh episode after the buffer is seeded"
        ),
    )
    parser.add_argument(
        "--procoim-past-reference-policy",
        choices=("improve", "nonregress"),
        default="improve",
        help=(
            "improve requires the recalled batch to clear the reference "
            "gain target; nonregress permits only the configured tolerance"
        ),
    )
    parser.add_argument(
        "--procoim-past-reference-tolerance-fraction",
        type=float,
        default=0.0,
        help=(
            "maximum relative CE regression under the nonregress policy; "
            "zero requires no recalled-batch regression"
        ),
    )
    parser.add_argument(
        "--procoim-past-reference-seed",
        type=int,
        default=None,
        help=(
            "dedicated partner-selection and reservoir seed; defaults to a "
            "deterministic value derived from --seed"
        ),
    )
    parser.add_argument(
        "--procoim-past-reference-rotation-forwards",
        type=_nonnegative_int,
        default=1024,
        help=(
            "replace the recalled partner at the first candidate-bank "
            "boundary after this many counted episode forwards; zero keeps "
            "one partner for the full episode"
        ),
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
        "ingress_temperature",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.balanced_prefix_batches and args.batch_size % 12 != 0:
        parser.error(
            "--balanced-prefix-batches requires --batch-size to be a multiple of 12"
        )
    if args.prefix_center_batch_logits and args.optimizer != "procoim":
        parser.error("--prefix-center-batch-logits currently requires Procoim")
    if (
        args.prefix_center_batch_logits
        and not args.balanced_prefix_batches
    ):
        parser.error(
            "--prefix-center-batch-logits requires --balanced-prefix-batches"
        )
    if (
        args.prefix_center_batch_logits
        and args.procoim_global_regression_weight <= 0
    ):
        parser.error(
            "--prefix-center-batch-logits requires a positive "
            "--procoim-global-regression-weight"
        )
    if args.temperature_min > args.temperature_start:
        parser.error("--temperature-min cannot exceed --temperature-start")
    if args.adamw_weight_trace_every > 0:
        if args.optimizer != "adamw":
            parser.error("--adamw-weight-trace-every requires AdamW")
        if args.adamw_weight_trace_dir is None:
            if args.checkpoint_dir is None:
                parser.error(
                    "--adamw-weight-trace-every requires either "
                    "--adamw-weight-trace-dir or --checkpoint-dir"
                )
            args.adamw_weight_trace_dir = (
                args.checkpoint_dir / "adamw_weight_trace"
            )
    elif args.adamw_weight_trace_dir is not None:
        parser.error(
            "--adamw-weight-trace-dir requires a positive "
            "--adamw-weight-trace-every"
        )
    if args.procoim_weight_trace_every > 0:
        if args.optimizer != "procoim":
            parser.error("--procoim-weight-trace-every requires Procoim")
        if args.procoim_weight_trace_dir is None:
            if args.checkpoint_dir is None:
                parser.error(
                    "--procoim-weight-trace-every requires either "
                    "--procoim-weight-trace-dir or --checkpoint-dir"
                )
            args.procoim_weight_trace_dir = (
                args.checkpoint_dir / "procoim_weight_trace"
            )
    elif args.procoim_weight_trace_dir is not None:
        parser.error(
            "--procoim-weight-trace-dir requires a positive "
            "--procoim-weight-trace-every"
        )
    procoim_config = None
    if args.optimizer == "procoim":
        if args.procoim_anchor_buffer_size < 0:
            parser.error("--procoim-anchor-buffer-size must be nonnegative")
        if (
            args.procoim_distribution_sentinel_batch_size > 0
            and args.procoim_distribution_sentinel_batch_size % 12 != 0
        ):
            parser.error(
                "--procoim-distribution-sentinel-batch-size must be a multiple of 12"
            )
        if (
            not math.isfinite(
                args.procoim_distribution_sentinel_hard_tolerance_fraction
            )
            or args.procoim_distribution_sentinel_hard_tolerance_fraction < 0
        ):
            parser.error(
                "--procoim-distribution-sentinel-hard-tolerance-fraction "
                "must be finite and nonnegative"
            )
        if (
            args.procoim_reference_guard_seed is not None
            and args.procoim_reference_guard_batch_size == 0
        ):
            parser.error(
                "--procoim-reference-guard-seed requires a positive "
                "--procoim-reference-guard-batch-size"
            )
        if (
            args.procoim_reference_guard_batch_size > 0
            and args.procoim_past_reference_buffer_size > 0
        ):
            parser.error(
                "fixed and past-partner reference gates are mutually exclusive"
            )
        if (
            args.procoim_past_reference_seed is not None
            and args.procoim_past_reference_buffer_size == 0
        ):
            parser.error(
                "--procoim-past-reference-seed requires a positive "
                "--procoim-past-reference-buffer-size"
            )
        if (
            not math.isfinite(
                args.procoim_past_reference_tolerance_fraction
            )
            or not 0
            <= args.procoim_past_reference_tolerance_fraction
            <= 1
        ):
            parser.error(
                "--procoim-past-reference-tolerance-fraction must be "
                "finite and in [0, 1]"
            )
        if (
            args.procoim_past_reference_buffer_size > 0
            and args.prefix_episode_mode != "fresh"
        ):
            parser.error(
                "past-partner references require --prefix-episode-mode fresh"
            )
        if (
            args.procoim_directional_accumulation_batches > 1
            and args.prefix_episode_mode != "fresh"
        ):
            parser.error(
                "directional accumulation currently requires "
                "--prefix-episode-mode fresh"
            )
        if args.procoim_anchor_sentinel_count < 0:
            parser.error("--procoim-anchor-sentinel-count must be nonnegative")
        if args.procoim_consolidation_interval_examples < 0:
            parser.error(
                "--procoim-consolidation-interval-examples must be nonnegative"
            )
        if args.procoim_full_audit_anchor_count < 0:
            parser.error(
                "--procoim-full-audit-anchor-count must be nonnegative"
            )
        alphas = args.procoim_consolidation_alphas
        if (
            not alphas
            or any(not math.isfinite(alpha) or not 0 < alpha <= 1 for alpha in alphas)
            or any(left <= right for left, right in zip(alphas, alphas[1:]))
        ):
            parser.error(
                "--procoim-consolidation-alphas must be strictly descending "
                "values in (0, 1]"
            )
        if (
            not math.isfinite(args.procoim_provisional_shrink_factor)
            or not 0 < args.procoim_provisional_shrink_factor <= 1
        ):
            parser.error("--procoim-provisional-shrink-factor must be in (0, 1]")
        if (
            not math.isfinite(args.procoim_s_replay_soft_tolerance)
            or args.procoim_s_replay_soft_tolerance < 0
        ):
            parser.error("--procoim-s-replay-soft-tolerance must be nonnegative")
        if (
            not math.isfinite(args.procoim_s_replay_catastrophic_tolerance)
            or args.procoim_s_replay_catastrophic_tolerance < 0
        ):
            parser.error(
                "--procoim-s-replay-catastrophic-tolerance must be nonnegative"
            )
        try:
            procoim_config = procoim_config_from_args(args)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        if args.compile:
            parser.error("Procoim MVP does not support --compile")
        if args.amp_dtype != "none":
            parser.error("Procoim MVP requires --amp-dtype none (FP32 weights)")
        if args.scheduler != "constant":
            parser.error("Procoim MVP does not use a PyTorch LR scheduler")
        if (
            args.route_entropy_loss_weight != 0.0
            or args.route_balance_loss_weight != 0.0
        ):
            parser.error(
                "Procoim prefix MVP requires zero route auxiliary losses; "
                "its protected objective is slotwise cross-entropy"
            )

    # Every flag, fully resolved (including derived defaults set during
    # validation above), as the first line of every log -- so a run's exact
    # reproduction recipe is never dependent on a separately-kept launch
    # script surviving. See Papers/GRANT_EVIDENCE_STANDARDS.md rule 2.
    print(
        "config="
        + json.dumps(
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in sorted(vars(args).items())
            },
            sort_keys=True,
        ),
        flush=True,
    )

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    core = GatorLinearConfig(
        vocab_size=16,
        streams=args.streams,
        nodes=args.nodes,
        width=args.width,
        relays=args.relays,
        degree=args.degree,
        buses=args.buses,
        mailbox_channels=args.mailbox_channels,
        mailbox_width=args.mailbox_width,
        basis_rank=args.basis_rank,
        address_width=args.address_width,
        router_rank=args.router_rank,
        conv_kernel=args.conv_kernel,
        scan_normalise=args.scan_normalise,
        private_stream_io=args.private_stream_io,
        private_stream_rank=args.private_stream_rank,
        update_gate_bias=args.update_gate_bias,
        readout_bias=args.readout_bias,
        hard_routing=args.hard_routing,
        router_occupancy=args.router_occupancy,
        heterogeneous_streams=args.heterogeneous_streams,
        stream_width_min_fraction=args.stream_width_min_fraction,
        # Total slots the mixer sees: the observed block plus the formed
        # block. Scan timescales are derived from this, so they track the
        # task instead of needing to be retuned per length.
        context_length=2 * args.length + 1,
        graph_layout=args.graph_layout,
        graph_seed=args.graph_seed,
        graph_rewire=args.graph_rewire,
        ingress_temperature=args.ingress_temperature,
    )
    architecture = GatorLinearPrefixConfig(core=core)
    model = GatorLinearPrefixModel(architecture).to(device)
    train_model = torch.compile(model) if args.compile else model
    amp_dtype = {
        "none": None, "bfloat16": torch.bfloat16, "float16": torch.float16,
    }[args.amp_dtype]
    amp_enabled = amp_dtype is not None and device.type == "cuda"
    # Only float16 needs loss scaling: its narrower exponent range can
    # underflow small gradients to zero, which bfloat16 (same exponent
    # range as float32, just less mantissa) does not risk.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype is torch.float16)
    optimizer = None
    scheduler = None
    procoim_engine = None
    adamw_tracer = None
    procoim_tracer = None
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.steps, eta_min=args.minimum_lr
            )
            if args.scheduler == "cosine"
            else None
        )
        if args.adamw_weight_trace_every > 0:
            assert args.adamw_weight_trace_dir is not None
            adamw_tracer = AdamWWeightTracer(
                model,
                optimizer,
                args.adamw_weight_trace_dir,
                interval=args.adamw_weight_trace_every,
                save_tensors=args.adamw_weight_trace_save_tensors,
                step_statistics=args.adamw_weight_trace_step_statistics,
            )
    else:
        assert procoim_config is not None
        procoim_engine = ProcoimTrainingEngine(model, procoim_config)
        if args.procoim_weight_trace_every > 0:
            assert args.procoim_weight_trace_dir is not None
            procoim_tracer = ProcoimWeightTracer(
                model,
                args.procoim_weight_trace_dir,
                interval=args.procoim_weight_trace_every,
                save_tensors=args.procoim_weight_trace_save_tensors,
            )
    procoim_anchors: deque[ProcoimAnchor] = deque(
        maxlen=args.procoim_anchor_buffer_size
    )
    pending_procoim_anchors: deque[ProcoimAnchor] = deque()
    retention_window = None
    if (
        procoim_engine is not None
        and args.procoim_consolidation_interval_examples > 0
    ):
        if args.procoim_consolidation_mode == "utility":
            retention_window = ProcoimSRetentionWindow(
                procoim_engine,
                interval_examples=(
                    args.procoim_consolidation_interval_examples
                ),
                commit_alphas=tuple(args.procoim_consolidation_alphas),
                replay_reservoir=ReplayReservoir(
                    capacity=args.procoim_s_replay_capacity
                ),
                config=ProcoimSConfig(
                    minimum_utility=args.procoim_s_minimum_utility,
                    replay_soft_tolerance=args.procoim_s_replay_soft_tolerance,
                    replay_catastrophic_tolerance=(
                        args.procoim_s_replay_catastrophic_tolerance
                    ),
                ),
                replay_sample_size=args.procoim_s_replay_sample_size,
            )
        else:
            retention_window = ProcoimRetentionWindow(
                procoim_engine,
                interval_examples=(
                    args.procoim_consolidation_interval_examples
                ),
                commit_alphas=tuple(args.procoim_consolidation_alphas),
                shrink_factor=args.procoim_provisional_shrink_factor,
            )
    last_consolidation_result: ProcoimConsolidationResult | None = None

    gpu = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    )
    print(
        f"device={device.type} gpu={gpu!r} parameters={model.parameter_count} "
        f"streams={args.streams} nodes={args.nodes} width={args.width} "
        f"relays={args.relays} degree={args.degree} buses={args.buses} "
        f"channels={args.mailbox_channels} basis_rank={args.basis_rank} "
        f"length={args.length} optimizer={args.optimizer} "
        f"episode_mode={args.prefix_episode_mode} compiled={args.compile} "
        f"loss_vocabulary={args.prefix_loss_vocabulary} "
        f"center_batch_logits={args.prefix_center_batch_logits} "
        f"directional_accumulation="
        f"{args.procoim_directional_accumulation_batches} "
        f"reference_guard_batch="
        f"{args.procoim_reference_guard_batch_size} "
        f"past_reference_buffer="
        f"{args.procoim_past_reference_buffer_size} "
        f"past_reference_policy={args.procoim_past_reference_policy} "
        f"past_reference_rotation_forwards="
        f"{args.procoim_past_reference_rotation_forwards} "
        f"amp_dtype={args.amp_dtype}",
        flush=True,
    )

    initial = evaluate_linear_prefix(
        model,
        length=args.length,
        device=device,
        batch_size=args.eval_batch_size,
        seed=args.eval_seed,
        temperature=args.temperature_start,
    )
    print(
        f"step=0 acc={initial['accuracy']:.3f} exact={initial['exact']:.3f} "
        f"valid_acc={initial['valid_accuracy']:.3f} "
        f"valid_exact={initial['valid_exact']:.3f} "
        f"loss={initial['loss']:.4f} node_entropy={initial['node_entropy']:.3f} "
        f"min_node={initial['min_node_load']:.5f} "
        f"min_stream={initial['min_stream_weight']:.4f} "
        f"by_pos=[{' '.join(f'{value:.3f}' for value in initial['by_position'])}] "
        f"valid_by_pos=[{' '.join(f'{value:.3f}' for value in initial['valid_by_position'])}] "
        f"by_pos_ce=[{' '.join(f'{value:.3f}' for value in initial['loss_by_position'])}] "
        f"valid_ce={initial['valid_vocabulary_loss']:.4f}",
        flush=True,
    )
    if adamw_tracer is not None:
        initial_trace = adamw_tracer.maybe_snapshot(
            0,
            force=True,
            context={
                "evaluation_accuracy": float(initial["accuracy"]),
                "evaluation_exact": float(initial["exact"]),
                "evaluation_loss": float(initial["loss"]),
                "valid_vocabulary_loss": float(
                    initial["valid_vocabulary_loss"]
                ),
            },
        )
        assert initial_trace is not None
        print(initial_trace.log_line(), flush=True)
    if procoim_tracer is not None:
        initial_procoim_trace = procoim_tracer.maybe_snapshot(
            0,
            force=True,
            context={
                "evaluation_accuracy": float(initial["accuracy"]),
                "evaluation_exact": float(initial["exact"]),
                "evaluation_loss": float(initial["loss"]),
                "valid_vocabulary_loss": float(
                    initial["valid_vocabulary_loss"]
                ),
            },
        )
        assert initial_procoim_trace is not None
        print(initial_procoim_trace.log_line(), flush=True)

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    interval_started = started
    interval_step = 0
    best_accuracy = float(initial["accuracy"])
    escaped_chance_at: int | None = None

    batch_factory = (
        make_balanced_prefix_batch
        if args.balanced_prefix_batches
        else make_prefix_batch
    )
    reference_guard_batch: tuple[
        torch.Tensor, torch.Tensor
    ] | None = None
    if (
        procoim_engine is not None
        and args.procoim_reference_guard_batch_size > 0
    ):
        reference_seed = (
            args.procoim_reference_guard_seed
            if args.procoim_reference_guard_seed is not None
            else args.eval_seed + args.length + 3_000_049
        )
        reference_generator = torch.Generator(device=device).manual_seed(
            reference_seed
        )
        reference_guard_batch = make_prefix_batch(
            args.procoim_reference_guard_batch_size,
            args.length,
            device,
            generator=reference_generator,
        )
    past_reference_buffer: list[
        tuple[torch.Tensor, torch.Tensor]
    ] = []
    past_reference_seen = 0
    past_reference_rng = random.Random(
        (
            args.procoim_past_reference_seed
            if args.procoim_past_reference_seed is not None
            else args.seed + 4_000_063
        )
    )
    last_reference_kind = "fixed" if reference_guard_batch is not None else "none"
    last_past_reference_index = -1
    distribution_sentinel: ProcoimAnchor | None = None
    if (
        procoim_engine is not None
        and args.procoim_distribution_sentinel_batch_size > 0
    ):
        sentinel_generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + args.length + 2_000_033
        )
        sentinel_observed, sentinel_targets = make_balanced_prefix_batch(
            args.procoim_distribution_sentinel_batch_size,
            args.length,
            device,
            generator=sentinel_generator,
            sampling_device=device,
        )
        distribution_sentinel = ProcoimAnchor(
            batch=(sentinel_observed, sentinel_targets),
            forward_kwargs={
                "output_slots": args.length,
                "temperature": args.temperature_start,
                "_prefix_valid_vocabulary_only": (
                    args.prefix_loss_vocabulary == "valid"
                ),
                # The broad guard watches ordinary predictive CE, not the
                # centered search surrogate.
                "_prefix_center_batch_logits": False,
            },
            anchor_id=("distribution_sentinel", args.seed, args.length),
            soft_regression_fraction=0.0,
            hard_regression_fraction=(
                args.procoim_distribution_sentinel_hard_tolerance_fraction
            ),
            absolute_tolerance=1e-6,
        )
    fixed_episode_batches: tuple[
        tuple[torch.Tensor, torch.Tensor], ...
    ] = ()
    if args.prefix_episode_mode != "fresh":
        fixed_generator = torch.Generator(device=device).manual_seed(
            args.seed + 1_000_003
        )
        fixed_count = 1 if args.prefix_episode_mode == "fixed" else 2
        fixed_episode_batches = tuple(
            batch_factory(
                args.batch_size,
                args.length,
                device,
                generator=fixed_generator,
                sampling_device=device,
            )
            for _ in range(fixed_count)
        )

    training_step = 0
    episode = 0
    next_log_step = args.log_every
    last_procoim_result = None
    while training_step < args.steps:
        if fixed_episode_batches:
            fixed_index = episode % len(fixed_episode_batches)
            observed, targets = fixed_episode_batches[fixed_index]
            example_key: Any = ("fixed", fixed_index)
        else:
            observed, targets = batch_factory(
                args.batch_size, args.length, device
            )
            example_key = ("fresh", episode)
        accumulation_batches = (
            tuple(
                batch_factory(args.batch_size, args.length, device)
                for _ in range(
                    args.procoim_directional_accumulation_batches - 1
                )
            )
            if procoim_engine is not None
            else ()
        )
        temperature = _temperature(
            training_step,
            start=args.temperature_start,
            minimum=args.temperature_min,
            decay=args.temperature_decay,
        )
        if args.optimizer == "procoim":
            assert procoim_engine is not None
            # Provisional gains are protected immediately. They are still
            # discarded if the enclosing theta_base + delta_work window
            # fails consolidation.
            rotating_sentinels = _select_anchor_sentinels(
                (*procoim_anchors, *pending_procoim_anchors),
                count=args.procoim_anchor_sentinel_count,
                episode=episode,
                exclude_id=example_key,
            )
            sentinels = (
                (() if distribution_sentinel is None else (distribution_sentinel,))
                + rotating_sentinels
            )
            episode_reference_batch = reference_guard_batch
            episode_reference_provider = None
            episode_reference_reduction: float | None = None
            last_reference_kind = (
                "fixed" if reference_guard_batch is not None else "none"
            )
            last_past_reference_index = -1
            if (
                args.procoim_past_reference_buffer_size > 0
                and past_reference_buffer
            ):
                last_past_reference_index = past_reference_rng.randrange(
                    len(past_reference_buffer)
                )
                episode_reference_batch = past_reference_buffer[
                    last_past_reference_index
                ]
                last_reference_kind = "past"

                if (
                    args.procoim_past_reference_rotation_forwards > 0
                    and len(past_reference_buffer) > 1
                ):
                    def next_past_reference_batch():
                        nonlocal last_past_reference_index
                        candidate_index = past_reference_rng.randrange(
                            len(past_reference_buffer) - 1
                        )
                        if candidate_index >= last_past_reference_index:
                            candidate_index += 1
                        last_past_reference_index = candidate_index
                        return past_reference_buffer[candidate_index]

                    episode_reference_provider = (
                        next_past_reference_batch
                    )

                if args.procoim_past_reference_policy == "improve":
                    episode_reference_reduction = (
                        procoim_config.reference_reduction_fraction
                        if (
                            procoim_config.reference_reduction_fraction
                            is not None
                        )
                        else procoim_config.reduction_fraction
                    )
                else:
                    episode_reference_reduction = -(
                        args.procoim_past_reference_tolerance_fraction
                    )
            minimum_episode_forwards = (
                procoim_engine.optimizer.minimum_episode_forward_calls(
                    anchor_count=len(sentinels),
                    reference_enabled=episode_reference_batch is not None,
                )
            )
            if args.steps - training_step < minimum_episode_forwards:
                break
            last_procoim_result = procoim_engine.run_episode(
                (observed, targets),
                prefix_slot_loss_provider,
                forward_kwargs={
                    "output_slots": args.length,
                    "temperature": temperature,
                    "_prefix_valid_vocabulary_only": (
                        args.prefix_loss_vocabulary == "valid"
                    ),
                    "_prefix_center_batch_logits": (
                        args.prefix_center_batch_logits
                    ),
                },
                example_id=example_key,
                anchors=sentinels,
                accumulation_batches=accumulation_batches,
                reference_batch=episode_reference_batch,
                reference_forward_kwargs=(
                    None
                    if episode_reference_batch is None
                    else {
                        "output_slots": args.length,
                        "temperature": temperature,
                        "_prefix_valid_vocabulary_only": (
                            args.prefix_loss_vocabulary == "valid"
                        ),
                        # Keep the population gate on ordinary predictive
                        # CE even when the local search uses centered logits.
                        "_prefix_center_batch_logits": False,
                    }
                ),
                reference_reduction_fraction=episode_reference_reduction,
                reference_batch_provider=episode_reference_provider,
                reference_rotation_forwards=(
                    args.procoim_past_reference_rotation_forwards
                    if episode_reference_provider is not None
                    else None
                ),
                max_forward_calls=args.steps - training_step,
            )
            training_step = procoim_engine.step_number
            task_loss = torch.tensor(
                last_procoim_result.end_aggregate_loss,
                device=device,
            )
            if procoim_tracer is not None:
                procoim_trace = procoim_tracer.maybe_snapshot(
                    training_step,
                    context={
                        "training_loss": float(task_loss.detach()),
                        "temperature": float(temperature),
                        "accepted_total": (
                            procoim_engine.optimizer.total_accepted_updates
                        ),
                    },
                )
                if procoim_trace is not None:
                    print(procoim_trace.log_line(), flush=True)
            gradient_norm = math.nan
            if args.procoim_anchor_buffer_size > 0:
                new_anchor = ProcoimAnchor.from_episode(
                    batch=(
                        observed.detach().clone(),
                        targets.detach().clone(),
                    ),
                    state=last_procoim_result.state,
                    forward_kwargs={
                        "output_slots": args.length,
                        "temperature": temperature,
                        "_prefix_valid_vocabulary_only": (
                            args.prefix_loss_vocabulary == "valid"
                        ),
                        "_prefix_center_batch_logits": (
                            args.prefix_center_batch_logits
                        ),
                    },
                    anchor_id=example_key,
                )
                assert new_anchor.protected_mask is not None
                # An episode that earned no protected slot gain contributes
                # no replay constraint. In particular, padded/empty slots
                # can never create an anchor write.
                if bool(new_anchor.protected_mask.any()):
                    _upsert_anchor(
                        (
                            pending_procoim_anchors
                            if retention_window is not None
                            else procoim_anchors
                        ),
                        new_anchor,
                    )
        else:
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                logits = train_model(
                    observed, output_slots=args.length, temperature=temperature
                )
                if args.prefix_loss_vocabulary == "valid":
                    task_loss = F.cross_entropy(
                        logits[..., 4:].reshape(-1, 12),
                        (targets - 4).reshape(-1),
                    )
                else:
                    task_loss = F.cross_entropy(
                        logits.reshape(-1, architecture.vocab_size),
                        targets.reshape(-1),
                    )
                statistics = model.latest_route_statistics
                route_entropy_term = statistics["route_entropy"]
                route_balance_term = statistics["route_balance"]
                loss = (
                    task_loss
                    + args.route_entropy_loss_weight * route_entropy_term
                    + args.route_balance_loss_weight * route_balance_term
                )
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Gradient clipping needs true-scale gradients, so unscale before
            # measuring the norm -- a no-op when the scaler is disabled (the
            # bfloat16 and no-AMP paths).
            scaler.unscale_(optimizer)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip,
                    error_if_nonfinite=True,
                )
            )
            if adamw_tracer is not None:
                adamw_tracer.before_optimizer_step(
                    step=training_step + 1,
                    loss=float(loss.detach()),
                    preclip_gradient_norm=gradient_norm,
                    gradient_clip=args.gradient_clip,
                    force_snapshot=(training_step + 1 == args.steps),
                )
            scaler.step(optimizer)
            if adamw_tracer is not None:
                adamw_tracer.after_optimizer_step()
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            training_step += 1
            if adamw_tracer is not None:
                trace = adamw_tracer.maybe_snapshot(
                    training_step,
                    context={
                        "learning_rates": [
                            float(group["lr"])
                            for group in optimizer.param_groups
                        ],
                        "training_loss": float(task_loss.detach()),
                        "temperature": float(temperature),
                    },
                )
                if trace is not None:
                    print(trace.log_line(), flush=True)

        if (
            procoim_engine is not None
            and args.procoim_past_reference_buffer_size > 0
        ):
            past_reference_seen += 1
            remembered_batch = (
                observed.detach().clone(),
                targets.detach().clone(),
            )
            if (
                len(past_reference_buffer)
                < args.procoim_past_reference_buffer_size
            ):
                past_reference_buffer.append(remembered_batch)
            else:
                replacement_index = past_reference_rng.randrange(
                    past_reference_seen
                )
                if (
                    replacement_index
                    < args.procoim_past_reference_buffer_size
                ):
                    past_reference_buffer[replacement_index] = (
                        remembered_batch
                    )
        episode += 1
        if retention_window is not None and isinstance(
            retention_window, ProcoimSRetentionWindow
        ):
            episode_forward_kwargs = {
                "output_slots": args.length,
                "temperature": temperature,
                "_prefix_valid_vocabulary_only": (
                    args.prefix_loss_vocabulary == "valid"
                ),
                "_prefix_center_batch_logits": (
                    args.prefix_center_batch_logits
                ),
            }
            retention_window.record_episode(
                batch=(observed, targets),
                forward_kwargs=episode_forward_kwargs,
                example_id=str(example_key),
            )
            if retention_window.due:
                def _scalar_prefix_loss(
                    consolidation_model, consolidation_batch, consolidation_forward_kwargs
                ):
                    return float(
                        prefix_slot_loss_provider(
                            consolidation_model,
                            consolidation_batch,
                            consolidation_forward_kwargs,
                        ).aggregate
                    )

                last_consolidation_result = retention_window.consolidate(
                    scalar_loss_fn=_scalar_prefix_loss
                )
                training_step += last_consolidation_result.audit_forwards
        elif retention_window is not None:
            retention_window.record_episode()
            if retention_window.due:
                permanent = tuple(procoim_anchors)
                audit_count = min(
                    args.procoim_full_audit_anchor_count,
                    len(permanent),
                )
                audit_anchors = (
                    permanent[-audit_count:]
                    if audit_count > 0
                    else ()
                ) + tuple(pending_procoim_anchors)
                if distribution_sentinel is not None:
                    audit_anchors = (distribution_sentinel,) + audit_anchors
                remaining = args.steps - training_step
                if not audit_anchors or remaining >= len(audit_anchors):
                    last_consolidation_result = retention_window.consolidate(
                        audit_anchors,
                        prefix_slot_loss_provider,
                        max_forward_calls=remaining,
                    )
                    assert procoim_engine is not None
                    training_step = procoim_engine.step_number
                    if procoim_tracer is not None:
                        consolidation_trace = procoim_tracer.maybe_snapshot(
                            training_step,
                            context={
                                "consolidation_alpha": float(
                                    last_consolidation_result.committed_alpha
                                ),
                            },
                        )
                        if consolidation_trace is not None:
                            print(
                                consolidation_trace.log_line(), flush=True
                            )
                    if last_consolidation_result.committed_alpha > 0:
                        for anchor in pending_procoim_anchors:
                            _upsert_anchor(procoim_anchors, anchor)
                else:
                    # A partial window is never silently promoted merely
                    # because the global forward budget cannot afford its
                    # audit.
                    retention_window.rollback_unconsolidated()
                    last_consolidation_result = ProcoimConsolidationResult(
                        committed_alpha=0.0,
                        audit_forwards=0,
                        audit_penalty=0.0,
                        hard_violation=False,
                        rolled_back=True,
                    )
                pending_procoim_anchors.clear()
        completed = training_step
        if completed < next_log_step and completed < args.steps:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        rate = (completed - interval_step) / (now - interval_started)
        metrics = evaluate_linear_prefix(
            model,
            length=args.length,
            device=device,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed,
            temperature=temperature,
        )
        ladder_metrics = None
        if fixed_episode_batches:
            ladder_metrics = evaluate_prefix_batch(
                model,
                (
                    torch.cat(
                        [batch[0] for batch in fixed_episode_batches],
                        dim=0,
                    ),
                    torch.cat(
                        [batch[1] for batch in fixed_episode_batches],
                        dim=0,
                    ),
                ),
                temperature=temperature,
            )
        # Historical "escaped" means clearing twice the 1/12 chance rate,
        # not merely landing one noisy validation estimate above chance.
        if escaped_chance_at is None and metrics["accuracy"] > 2.0 / 12.0:
            escaped_chance_at = completed
        best_accuracy = max(best_accuracy, metrics["accuracy"])
        if args.optimizer == "procoim":
            assert last_procoim_result is not None
            group_counts = ",".join(
                f"{name}:{count}"
                for name, count in (
                    last_procoim_result.parameter_group_acceptance_counts.items()
                )
            )
            group_scales = ",".join(
                f"{name}:{scale:.2e}"
                for name, scale in (
                    last_procoim_result.parameter_group_probe_scales.items()
                )
            )
            group_k = ",".join(
                f"{name}:{k}"
                for name, k in (
                    last_procoim_result.parameter_group_candidate_directions.items()
                )
            )
            group_gains = ",".join(
                f"{name}:{gain:.2e}"
                for name, gain in (
                    last_procoim_result.parameter_group_gain_per_forward.items()
                )
            )
            group_basis_epochs = ",".join(
                f"{name}:{epoch}"
                for name, epoch in (
                    last_procoim_result.parameter_group_basis_epochs.items()
                )
            )
            optimizer_details = (
                f"optimizer=procoim episode={episode} "
                f"episode_forwards={last_procoim_result.total_forward_calls} "
                f"baseline_forwards={last_procoim_result.baseline_forwards} "
                f"candidate_forwards={last_procoim_result.candidate_forwards} "
                f"anchor_forwards={last_procoim_result.anchor_forwards} "
                f"reference_forwards={last_procoim_result.reference_forwards} "
                f"reference_vetoes={last_procoim_result.reference_vetoes} "
                f"reference_vetoes_total="
                f"{procoim_engine.optimizer.total_reference_vetoes} "
                f"reference_rotations="
                f"{last_procoim_result.reference_rotations} "
                f"reference_kind={last_reference_kind} "
                f"past_reference_index={last_past_reference_index} "
                f"past_reference_buffer={len(past_reference_buffer)} "
                f"reference_start_loss="
                f"{math.nan if last_procoim_result.reference_start_loss is None else last_procoim_result.reference_start_loss:.4f} "
                f"reference_end_loss="
                f"{math.nan if last_procoim_result.reference_end_loss is None else last_procoim_result.reference_end_loss:.4f} "
                f"reference_reduction="
                f"{math.nan if last_procoim_result.reference_reduction is None else last_procoim_result.reference_reduction:.6f} "
                f"verification_forwards="
                f"{last_procoim_result.verification_forwards} "
                f"stability_forwards={last_procoim_result.stability_forwards} "
                f"accepted={last_procoim_result.accepted_updates} "
                f"accepted_total={procoim_engine.accepted_updates} "
                f"accepted_per_1k="
                f"{1000.0 * procoim_engine.accepted_updates / max(completed, 1):.3f} "
                f"start_loss={last_procoim_result.start_aggregate_loss:.4f} "
                f"end_loss={last_procoim_result.end_aggregate_loss:.4f} "
                f"reduction={last_procoim_result.aggregate_reduction:.4f} "
                f"active={last_procoim_result.active_slots} "
                f"soft={last_procoim_result.soft_frozen_slots} "
                f"protected={last_procoim_result.protected_slots} "
                f"target_hit={last_procoim_result.slot_target_hit_rate:.3f} "
                f"accept_rate={last_procoim_result.acceptance_rate:.4f} "
                f"hard_veto_rate={last_procoim_result.hard_veto_rate:.4f} "
                f"anchor_vetoes={last_procoim_result.anchor_vetoes} "
                f"anchor_vetoes_total={last_procoim_result.total_anchor_vetoes} "
                f"max_protected_regression="
                f"{last_procoim_result.max_protected_regression:.4g} "
                f"mean_protected_regression="
                f"{last_procoim_result.mean_protected_regression:.4g} "
                f"stalls={last_procoim_result.stall_count} "
                f"probe_scale={last_procoim_result.current_probe_scale:.3e} "
                f"reduction_fraction="
                f"{last_procoim_result.effective_reduction_fraction:.6f} "
                f"completed={last_procoim_result.completed} "
                f"completion={last_procoim_result.completion_reason} "
                f"episode_seconds={last_procoim_result.wall_clock_seconds:.3f} "
                f"reuse_accepts={last_procoim_result.reused_direction_accepts} "
                f"reuse_accepts_total="
                f"{last_procoim_result.total_reused_direction_accepts} "
                f"estimate_accepts="
                f"{last_procoim_result.antithetic_estimate_accepts} "
                f"estimate_accepts_total="
                f"{last_procoim_result.total_antithetic_estimate_accepts} "
                f"directional_adam_accepts="
                f"{last_procoim_result.directional_adam_accepts} "
                f"directional_adam_accepts_total="
                f"{last_procoim_result.total_directional_adam_accepts} "
                f"group_accepts=[{group_counts}] "
                f"group_scales=[{group_scales}] "
                f"group_k=[{group_k}] "
                f"group_gain_per_forward=[{group_gains}] "
                f"group_basis_epochs=[{group_basis_epochs}]"
            )
            if retention_window is not None:
                consolidation = last_consolidation_result
                audit_retained = (
                    math.nan
                    if (
                        consolidation is None
                        or consolidation.audit_retained_reduction is None
                    )
                    else consolidation.audit_retained_reduction
                )
                optimizer_details += (
                    f" window_examples={retention_window.examples_in_window}"
                    f" windows_committed={retention_window.windows_committed}"
                    f" windows_rolled_back="
                    f"{retention_window.windows_rolled_back}"
                    f" consolidation_alpha="
                    f"{math.nan if consolidation is None else consolidation.committed_alpha:.3f}"
                    f" consolidation_forwards="
                    f"{0 if consolidation is None else consolidation.audit_forwards}"
                    f" consolidation_hard_veto="
                    f"{False if consolidation is None else consolidation.hard_violation}"
                    f" consolidation_penalty="
                    f"{0.0 if consolidation is None else consolidation.audit_penalty:.4g}"
                    f" consolidation_retained_reduction="
                    f"{audit_retained:.6g}"
                )
        else:
            optimizer_details = (
                f"optimizer=adamw grad_norm={gradient_norm:.4f} "
                f"route_entropy_loss={float(route_entropy_term.detach()):.4f} "
                f"route_balance_loss={float(route_balance_term.detach()):.4f}"
            )
        ladder_details = (
            ""
            if ladder_metrics is None
            else (
                f" ladder_loss={ladder_metrics['loss']:.4f}"
                f" ladder_acc={ladder_metrics['accuracy']:.3f}"
                f" ladder_exact={ladder_metrics['exact']:.3f}"
            )
        )
        print(
            f"step={completed} steps_per_second={rate:.2f} "
            f"acc={metrics['accuracy']:.3f} exact={metrics['exact']:.3f} "
            f"valid_acc={metrics['valid_accuracy']:.3f} "
            f"valid_exact={metrics['valid_exact']:.3f} "
            f"loss={metrics['loss']:.4f} train_loss={float(task_loss.detach()):.4f} "
            f"temperature={temperature:.3f} "
            f"route_entropy={metrics['route_entropy']:.3f} "
            f"node_entropy={metrics['node_entropy']:.3f} "
            f"min_node={metrics['min_node_load']:.5f} "
            f"max_node={metrics['max_node_load']:.5f} "
            f"min_stream={metrics['min_stream_weight']:.4f} "
            f"max_stream={metrics['max_stream_weight']:.4f} "
            f"bus_gate={metrics['bus_gate']:.3f} "
            f"stream_agreement={metrics['stream_agreement']:.4f} "
            f"stream_acc={metrics['worst_single_stream']:.3f}"
            f"..{metrics['best_single_stream']:.3f} "
            f"stream_spread={metrics['stream_accuracy_spread']:.4f} "
            f"by_pos=[{' '.join(f'{value:.3f}' for value in metrics['by_position'])}] "
            f"valid_by_pos=[{' '.join(f'{value:.3f}' for value in metrics['valid_by_position'])}] "
            f"by_pos_ce=[{' '.join(f'{value:.3f}' for value in metrics['loss_by_position'])}] "
            f"valid_ce={metrics['valid_vocabulary_loss']:.4f} "
            f"{optimizer_details}{ladder_details}",
            flush=True,
        )
        interval_started = time.perf_counter()
        interval_step = completed
        next_log_step = (
            completed // args.log_every + 1
        ) * args.log_every

        if args.checkpoint_dir is not None:
            payload: dict[str, Any] = {
                "format": "gator-linear-prefix-v1",
                "step": completed,
                "best_accuracy": best_accuracy,
                "architecture": asdict(architecture.core),
                "arguments": {
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                },
                "model": model.state_dict(),
            }
            if procoim_engine is not None:
                payload["procoim"] = procoim_engine.state_dict()
                payload["procoim_anchors"] = [
                    {
                        "batch": tuple(
                            value.detach().cpu()
                            for value in anchor.batch
                        ),
                        "forward_kwargs": dict(anchor.forward_kwargs or {}),
                        "best_aggregate": anchor.best_aggregate,
                        "anchor_id": anchor.anchor_id,
                        "start_losses": (
                            None if anchor.start_losses is None
                            else anchor.start_losses.detach().cpu()
                        ),
                        "best_losses": (
                            None if anchor.best_losses is None
                            else anchor.best_losses.detach().cpu()
                        ),
                        "valid_mask": (
                            None if anchor.valid_mask is None
                            else anchor.valid_mask.detach().cpu()
                        ),
                        "protected_mask": (
                            None if anchor.protected_mask is None
                            else anchor.protected_mask.detach().cpu()
                        ),
                        "hard_mask": (
                            None if anchor.hard_mask is None
                            else anchor.hard_mask.detach().cpu()
                        ),
                    }
                    for anchor in procoim_anchors
                ]
                if past_reference_buffer:
                    payload["procoim_past_references"] = {
                        "batches": [
                            tuple(value.detach().cpu() for value in batch)
                            for batch in past_reference_buffer
                        ],
                        "seen": past_reference_seen,
                        "rng_state": past_reference_rng.getstate(),
                    }
                if distribution_sentinel is not None:
                    payload["procoim_distribution_sentinel"] = {
                        "batch": tuple(
                            value.detach().cpu()
                            for value in distribution_sentinel.batch
                        ),
                        "forward_kwargs": dict(
                            distribution_sentinel.forward_kwargs or {}
                        ),
                        "best_aggregate": (
                            distribution_sentinel.best_aggregate
                        ),
                        "anchor_id": distribution_sentinel.anchor_id,
                    }
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                payload, args.checkpoint_dir / "gator_linear_prefix_latest.pt"
            )

    if retention_window is not None and retention_window.has_provisional_updates:
        # The step budget is exhausted, so no uncounted final audit is
        # allowed. Restore the last trusted theta_base.
        retention_window.rollback_unconsolidated()
        pending_procoim_anchors.clear()

    runtime = time.perf_counter() - started
    final = evaluate_linear_prefix(
        model,
        length=args.length,
        device=device,
        batch_size=args.eval_batch_size,
        seed=args.eval_seed,
        temperature=args.temperature_min,
    )
    best_accuracy = max(best_accuracy, float(final["accuracy"]))
    if adamw_tracer is not None:
        final_trace = adamw_tracer.maybe_snapshot(
            training_step,
            force=True,
            context={
                "final_evaluation_accuracy": float(final["accuracy"]),
                "final_evaluation_exact": float(final["exact"]),
                "final_evaluation_loss": float(final["loss"]),
                "final_valid_vocabulary_loss": float(
                    final["valid_vocabulary_loss"]
                ),
            },
        )
        if final_trace is not None:
            print(final_trace.log_line(), flush=True)
    if procoim_tracer is not None:
        final_procoim_trace = procoim_tracer.maybe_snapshot(
            training_step,
            force=True,
            context={
                "final_evaluation_accuracy": float(final["accuracy"]),
                "final_evaluation_exact": float(final["exact"]),
                "final_evaluation_loss": float(final["loss"]),
                "final_valid_vocabulary_loss": float(
                    final["valid_vocabulary_loss"]
                ),
            },
        )
        if final_procoim_trace is not None:
            print(final_procoim_trace.log_line(), flush=True)
    print(
        f"final acc={final['accuracy']:.4f} exact={final['exact']:.4f} "
        f"valid_acc={final['valid_accuracy']:.4f} "
        f"valid_exact={final['valid_exact']:.4f} "
        f"loss={final['loss']:.4f} best_acc={best_accuracy:.4f} "
        f"escaped_chance_at={escaped_chance_at} "
        f"min_node={final['min_node_load']:.5f} "
        f"min_stream={final['min_stream_weight']:.4f} "
        f"optimizer={args.optimizer} episodes={episode} steps={training_step} "
        f"runtime_seconds={runtime:.2f} "
        f"steps_per_second={training_step / max(runtime, 1e-9):.2f} "
        f"by_pos=[{' '.join(f'{value:.3f}' for value in final['by_position'])}] "
        f"valid_by_pos=[{' '.join(f'{value:.3f}' for value in final['valid_by_position'])}] "
        f"by_pos_ce=[{' '.join(f'{value:.3f}' for value in final['loss_by_position'])}] "
        f"valid_ce={final['valid_vocabulary_loss']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
