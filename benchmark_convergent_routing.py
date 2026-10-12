"""Benchmark Convergent Gator route and node execution.

The benchmark deliberately uses synthetic tensors created directly on the
target device.  It does not include tokenization, dataset iteration,
validation, logging, or checkpoint I/O.  Three scopes are timed separately:

* ``forward``: inference-only model execution;
* ``forward_backward``: forward, cross entropy, and backward; and
* ``optimizer``: AdamW ``step()`` with already-materialized gradients;
* ``training_step``: a complete forward, loss, backward, and AdamW update.

All four run by default for compatibility.  Repeat ``--phase`` to select an
isolated subset.  In particular, ``--phase training_step`` starts that timing
from a fresh process/model without the earlier diagnostic phases mutating it.

The dense passport remains the execution oracle.  Once Convergent Gator's
packed route backend is present, ``--route-backend both`` runs the same seeded
architecture and synthetic batch through both implementations.

Representative Blackwell comparison::

    python benchmark_convergent_routing.py \
      --route-backend both --device cuda --dtype float32 --compile \
      --batch-size 6 --context-tokens 512 --output-slots 128 \
      --nodes 154 --d-model 768 --lanes 10 --relays 2 --degree 3 \
      --active-node-top-k 32 --router-rank 96 \
      --decode-workload teacher_forced --gradient-clip 1.0 \
      --warmup-steps 20 --measured-steps 100 \
      --json-output convergent-routing-n154.json

This script reports measurements; it does not decide that an optimization is
safe.  Execution-only replacements still require forward and gradient
equivalence tests against the dense implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F

from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)


RouteBackend = Literal["dense", "packed"]
TimedCallable = Callable[[], torch.Tensor | None]
PHASE_NAMES = (
    "forward",
    "forward_backward",
    "optimizer",
    "training_step",
)


@dataclass(frozen=True)
class PhaseResult:
    """One isolated benchmark scope."""

    name: str
    milliseconds_per_step: float
    steps_per_second: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    peak_allocated_delta_bytes: int | None
    peak_reserved_delta_bytes: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route-backend",
        choices=("dense", "packed", "both"),
        default="dense",
        help="Route-state implementation to benchmark.",
    )
    parser.add_argument("--nodes", type=int, default=66)
    parser.add_argument("--d-model", "--width", dest="d_model", type=int, default=20)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Defaults to --d-model so width sweeps keep the lexical shell aligned.",
    )
    parser.add_argument("--lanes", type=int, default=10)
    parser.add_argument("--relays", type=int, default=2)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--active-node-top-k", type=int, default=4)
    parser.add_argument("--router-rank", type=int, default=16)
    parser.add_argument(
        "--packed-memory-backend",
        choices=("matrix", "convolution", "fft", "fft_opaque"),
        default="matrix",
    )
    parser.add_argument(
        "--packed-memory-reader-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--packed-memory-feature-chunk",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--packed-memory-compact-nodes",
        action="store_true",
    )
    parser.add_argument("--adaptive-route-min-k", type=int, default=None)
    parser.add_argument(
        "--adaptive-route-mode",
        choices=("entropy", "margin", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--adaptive-route-entropy-mix",
        type=float,
        default=0.5,
    )
    parser.add_argument("--compact-node-dispatch", action="store_true")
    parser.add_argument(
        "--node-compute-mode",
        choices=("full_private", "shared_adapter"),
        default="full_private",
    )
    parser.add_argument("--node-adapter-rank", type=int, default=8)
    parser.add_argument("--ingress-groups", type=int, default=None)
    parser.add_argument("--ingress-top-groups", type=int, default=2)
    parser.add_argument(
        "--no-ingress-learned-groups",
        action="store_false",
        dest="ingress_learned_groups",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--output-slots", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument(
        "--phase",
        "--phases",
        action="append",
        choices=PHASE_NAMES,
        dest="phases",
        default=None,
        help=(
            "repeat to select isolated scopes; defaults to every phase in "
            "historical order"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help=(
            "optional max gradient norm inside training_step; use 1.0 to "
            "match the archived 140M trainer"
        ),
    )
    parser.add_argument(
        "--decode-workload",
        choices=("parallel", "teacher_forced"),
        default="parallel",
        help=(
            "parallel uses blank simultaneous slots; teacher_forced uses a "
            "shifted response prefix with causal receiver memory"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or an indexed CUDA device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model before warmup; compilation is excluded from timing.",
    )
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--cudnn-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional file receiving the complete JSON result.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress the human-readable table and print only JSON.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.embedding_dim is None:
        args.embedding_dim = args.d_model
    args.phases = tuple(dict.fromkeys(args.phases or PHASE_NAMES))
    return args


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    positive = (
        "nodes",
        "d_model",
        "lanes",
        "relays",
        "degree",
        "active_node_top_k",
        "router_rank",
        "packed_memory_feature_chunk",
        "node_adapter_rank",
        "ingress_top_groups",
        "batch_size",
        "context_tokens",
        "output_slots",
        "vocab_size",
        "measured_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.embedding_dim is not None and args.embedding_dim <= 0:
        parser.error("--embedding-dim must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.degree > args.nodes:
        parser.error("--degree cannot exceed --nodes")
    if args.lanes > args.nodes:
        parser.error("--lanes cannot exceed --nodes")
    if args.active_node_top_k > args.nodes:
        parser.error("--active-node-top-k cannot exceed --nodes")
    if args.vocab_size <= 4:
        parser.error("--vocab-size must leave room above the reserved token IDs")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.gradient_clip is not None and args.gradient_clip <= 0:
        parser.error("--gradient-clip must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.packed_memory_compact_nodes and args.compile:
        parser.error("--packed-memory-compact-nodes is eager-only")
    if args.packed_memory_compact_nodes and (
        args.route_backend != "packed"
        or args.packed_memory_backend == "matrix"
    ):
        parser.error(
            "--packed-memory-compact-nodes requires one packed "
            "convolution/fft backend"
        )
    if (
        args.packed_memory_reader_backend == "triton"
        and args.packed_memory_backend == "matrix"
    ):
        parser.error(
            "the Triton packed reader requires convolution, fft, or "
            "fft_opaque memory"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and (
            args.route_backend not in {"packed", "both"}
            or args.packed_memory_backend != "fft_opaque"
        )
    ):
        parser.error(
            "the Triton packed writer currently requires a packed route "
            "benchmark and --packed-memory-backend fft_opaque"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and args.packed_memory_feature_chunk > 128
    ):
        parser.error(
            "the Triton packed writer supports feature chunks up to 128"
        )
    if args.compile and args.packed_memory_backend == "fft":
        parser.error(
            "raw FFT exposes complex tensors to Inductor; use fft_opaque "
            "for compiled benchmarking"
        )
    if args.compact_node_dispatch and args.compile:
        parser.error("--compact-node-dispatch is eager-only")
    if args.compact_node_dispatch and args.route_backend != "packed":
        parser.error("--compact-node-dispatch requires --route-backend packed")
    if args.node_compute_mode == "shared_adapter" and args.route_backend != "packed":
        parser.error("shared_adapter node compute requires --route-backend packed")
    if args.ingress_groups is not None:
        if not 1 <= args.ingress_groups <= args.nodes:
            parser.error("--ingress-groups must be in [1, --nodes]")
        if args.route_backend != "packed":
            parser.error("hierarchical ingress requires --route-backend packed")
        if args.ingress_top_groups > args.ingress_groups:
            parser.error("--ingress-top-groups cannot exceed --ingress-groups")
    if not 0.0 <= args.adaptive_route_entropy_mix <= 1.0:
        parser.error("--adaptive-route-entropy-mix must be in [0, 1]")
    if args.adaptive_route_min_k is not None:
        if args.route_backend != "packed":
            parser.error(
                "adaptive K is an architecture A/B and requires one packed "
                "backend, not --route-backend both"
            )
        if not 1 <= args.adaptive_route_min_k <= args.active_node_top_k:
            parser.error(
                "--adaptive-route-min-k must be in [1, --active-node-top-k]"
            )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = mapping[value]
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 benchmark execution requires CUDA")
    return dtype


def configured_backends(value: str) -> tuple[RouteBackend, ...]:
    if value == "both":
        return ("dense", "packed")
    if value in {"dense", "packed"}:
        return (value,)  # type: ignore[return-value]
    raise ValueError(f"unknown route backend: {value}")


def route_backend_is_supported() -> bool:
    """Permit the benchmark to import before the backend lands.

    Older frozen source bundles do not have ``route_backend`` in the config.
    Dense measurements remain useful there; packed requests fail with a clear
    message instead of an opaque dataclass constructor error.
    """

    return "route_backend" in {
        field.name for field in fields(ConvergentLanguageConfig)
    }


def route_shape_estimates(
    *,
    batch_size: int,
    total_tokens: int,
    nodes: int,
    width: int,
    lanes: int,
    relays: int,
    top_k: int,
    degree: int,
    value_element_size: int,
) -> dict[str, int]:
    """Return allocation-independent logical route-shape estimates."""

    packet_rows = batch_size * total_tokens * lanes
    dense_frontier_values = packet_rows * nodes
    packed_frontier_values = packet_rows * top_k
    return {
        "packet_rows": packet_rows,
        "dense_frontier_values": dense_frontier_values,
        "dense_frontier_bytes": dense_frontier_values * value_element_size,
        "dense_route_history_bytes": (
            dense_frontier_values * (relays + 1) * value_element_size
        ),
        "packed_frontier_entries": packed_frontier_values,
        # Node IDs are conservatively estimated as int64. An implementation
        # may use narrower IDs when its node population permits it.
        "packed_frontier_bytes_int64_ids": packed_frontier_values
        * (torch.int64.itemsize + value_element_size),
        "packed_route_history_bytes_int64_ids": packed_frontier_values
        * (relays + 1)
        * (torch.int64.itemsize + value_element_size),
        "dense_edge_candidates_per_relay": dense_frontier_values * degree,
        "packed_edge_candidates_per_relay": packed_frontier_values * degree,
        "dense_node_memory_values_per_relay": (
            batch_size * total_tokens * nodes * width
        ),
        "packed_requested_memory_values_per_relay": (
            packed_frontier_values * width
        ),
        "dense_node_output_values_per_relay": (
            packet_rows * nodes * width
        ),
        "packed_node_output_values_per_relay": (
            packed_frontier_values * width
        ),
    }


def _synthetic_batch(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed + 17)
    context_ids = torch.randint(
        4,
        args.vocab_size,
        (args.batch_size, args.context_tokens),
        device=device,
        generator=generator,
    )
    context_mask = torch.ones_like(context_ids, dtype=torch.bool)
    targets = torch.randint(
        4,
        args.vocab_size,
        (args.batch_size, args.output_slots),
        device=device,
        generator=generator,
    )
    # Match LanguageTrainer's teacher-forced contract when selected: BLANK
    # in slot zero followed by the shifted reference response.
    output_token_ids = torch.full_like(targets, 1)
    if args.output_slots > 1:
        output_token_ids[:, 1:] = targets[:, :-1]
    output_mask = torch.ones_like(targets, dtype=torch.bool)
    return (
        context_ids,
        context_mask,
        targets,
        output_token_ids,
        output_mask,
    )


def _build_model(
    args: argparse.Namespace,
    *,
    backend: RouteBackend,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, ConvergentLanguageModel]:
    supports_backend = route_backend_is_supported()
    if backend == "packed" and not supports_backend:
        raise RuntimeError(
            "packed route execution was requested, but this source bundle's "
            "ConvergentLanguageConfig has no route_backend field"
        )
    effective_node_mode = (
        args.node_compute_mode if backend == "packed" else "full_private"
    )
    if args.compile and effective_node_mode == "full_private":
        grouped_private_supported = (
            device.type == "cuda"
            and hasattr(torch, "_grouped_mm")
            and torch.cuda.get_device_capability(device) >= (8, 0)
            and (args.d_model * torch.bfloat16.itemsize) % 16 == 0
        )
        if not grouped_private_supported:
            raise RuntimeError(
                "compiled full-private sparse dispatch requires an aligned "
                "CUDA grouped-MM device; benchmark eager or packed "
                "shared_adapter on this target"
            )
    config_kwargs: dict[str, Any] = {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "embedding_dim": args.embedding_dim,
        "n_nodes": args.nodes,
        "degree": args.degree,
        "router_mode": "shared_low_rank",
        "router_rank": args.router_rank,
        "ingress_mode": "learned",
        "active_node_top_k": args.active_node_top_k,
        "relays": args.relays,
        "lanes": args.lanes,
        "terminal_vote_mode": "slot",
        "blank_token_id": 1,
        "mask_token_id": 3,
    }
    if supports_backend:
        config_kwargs["route_backend"] = backend
        config_kwargs["packed_memory_backend"] = (
            args.packed_memory_backend if backend == "packed" else "matrix"
        )
        config_kwargs["packed_memory_reader_backend"] = (
            args.packed_memory_reader_backend if backend == "packed" else "torch"
        )
        config_kwargs["packed_memory_writer_backend"] = (
            args.packed_memory_writer_backend if backend == "packed" else "torch"
        )
        config_kwargs["packed_memory_feature_chunk"] = (
            args.packed_memory_feature_chunk
        )
        config_kwargs["packed_memory_compact_nodes"] = (
            args.packed_memory_compact_nodes if backend == "packed" else False
        )
        config_kwargs["adaptive_route_min_k"] = (
            args.adaptive_route_min_k if backend == "packed" else None
        )
        config_kwargs["adaptive_route_mode"] = args.adaptive_route_mode
        config_kwargs["adaptive_route_entropy_mix"] = (
            args.adaptive_route_entropy_mix
        )
        config_kwargs["compact_node_dispatch"] = (
            args.compact_node_dispatch if backend == "packed" else False
        )
        config_kwargs["node_compute_mode"] = (
            args.node_compute_mode if backend == "packed" else "full_private"
        )
        config_kwargs["node_adapter_rank"] = args.node_adapter_rank
        config_kwargs["ingress_groups"] = (
            args.ingress_groups if backend == "packed" else None
        )
        config_kwargs["ingress_top_groups"] = args.ingress_top_groups
        config_kwargs["ingress_learned_groups"] = (
            args.ingress_learned_groups
        )
    # Reset before every backend so --route-backend both is a seeded A/B.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    eager = ConvergentLanguageModel(
        ConvergentLanguageConfig(**config_kwargs)
    ).to(device=device, dtype=dtype)
    measured: torch.nn.Module = eager
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires a PyTorch build with torch.compile")
        measured = torch.compile(eager)
    return measured, eager


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_phase(
    name: str,
    function: TimedCallable,
    *,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
) -> PhaseResult:
    observable: torch.Tensor | None = None
    for _ in range(warmup_steps):
        observable = function()
    _sync(device)

    if device.type == "cuda":
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        for _ in range(measured_steps):
            observable = function()
        finished.record()
        _sync(device)
        elapsed_ms = float(started.elapsed_time(finished))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        allocated_delta = max(peak_allocated - baseline_allocated, 0)
        reserved_delta = max(peak_reserved - baseline_reserved, 0)
    else:
        started_at = time.perf_counter()
        for _ in range(measured_steps):
            observable = function()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        peak_allocated = None
        peak_reserved = None
        allocated_delta = None
        reserved_delta = None

    # Keep a tensor result visible until timing has ended. This also catches a
    # benchmark callback that accidentally stops performing useful work.
    if observable is None or not isinstance(observable, torch.Tensor):
        raise RuntimeError(f"{name} benchmark did not produce a tensor")
    if not bool(torch.isfinite(observable.detach()).all()):
        raise FloatingPointError(f"{name} benchmark produced a nonfinite value")
    milliseconds = elapsed_ms / measured_steps
    return PhaseResult(
        name=name,
        milliseconds_per_step=milliseconds,
        steps_per_second=1000.0 / milliseconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        peak_allocated_delta_bytes=allocated_delta,
        peak_reserved_delta_bytes=reserved_delta,
    )


def _run_backend(
    args: argparse.Namespace,
    *,
    backend: RouteBackend,
    device: torch.device,
    dtype: torch.dtype,
    batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> dict[str, Any]:
    measured, eager = _build_model(
        args,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    (
        context_ids,
        context_mask,
        targets,
        output_token_ids,
        output_mask,
    ) = batch
    decode_kwargs: dict[str, Any] = {}
    if args.decode_workload == "teacher_forced":
        decode_kwargs = {
            "output_token_ids": output_token_ids,
            "output_mask": output_mask,
            "causal": True,
        }
    # A device scalar keeps scheduled routing temperatures inside one compiled
    # graph rather than specializing a graph per Python float.
    temperature = torch.tensor(
        args.temperature,
        device=device,
        dtype=dtype,
    )

    def forward() -> torch.Tensor:
        with torch.inference_mode():
            return measured(
                context_ids,
                context_mask,
                output_slots=args.output_slots,
                temperature=temperature,
                **decode_kwargs,
            )

    def forward_backward() -> torch.Tensor:
        eager.zero_grad(set_to_none=True)
        logits = measured(
            context_ids,
            context_mask,
            output_slots=args.output_slots,
            temperature=temperature,
            **decode_kwargs,
        )
        loss = F.cross_entropy(
            logits.float().reshape(-1, args.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        return loss.detach()

    phase_results: list[PhaseResult] = []
    if "forward" in args.phases:
        phase_results.append(
            _measure_phase(
                "forward",
                forward,
                device=device,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
        )
    if "forward_backward" in args.phases:
        phase_results.append(
            _measure_phase(
                "forward_backward",
                forward_backward,
                device=device,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
        )

    if "optimizer" in args.phases:
        optimizer = torch.optim.AdamW(
            eager.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        # Materialize gradients and initialize AdamW state before timing.
        # Reusing those gradients isolates optimizer math and memory traffic
        # from forward, backward, gradient allocation, and zeroing.
        forward_backward()
        optimizer.step()

        def optimizer_step() -> torch.Tensor:
            optimizer.step()
            first_parameter = next(eager.parameters())
            return first_parameter.reshape(-1)[0]

        phase_results.append(
            _measure_phase(
                "optimizer",
                optimizer_step,
                device=device,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
        )
        del optimizer_step
        del optimizer

    if "training_step" in args.phases:
        training_optimizer = torch.optim.AdamW(
            eager.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        def training_step() -> torch.Tensor:
            training_optimizer.zero_grad(set_to_none=True)
            logits = measured(
                context_ids,
                context_mask,
                output_slots=args.output_slots,
                temperature=temperature,
                **decode_kwargs,
            )
            loss = F.cross_entropy(
                logits.float().reshape(-1, args.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            if args.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    eager.parameters(),
                    args.gradient_clip,
                )
            training_optimizer.step()
            return loss.detach()

        phase_results.append(
            _measure_phase(
                "training_step",
                training_step,
                device=device,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
        )

    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in eager.parameters()
    )
    if not finite_gradients:
        raise FloatingPointError("model has a nonfinite gradient")
    return {
        "route_backend": backend,
        "parameters": int(sum(p.numel() for p in eager.parameters())),
        "trainable_parameters": int(
            sum(p.numel() for p in eager.parameters() if p.requires_grad)
        ),
        "phases": [asdict(result) for result in phase_results],
        "decode_workload": args.decode_workload,
        "causal": args.decode_workload == "teacher_forced",
        "gradient_clip": args.gradient_clip,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    torch.backends.cudnn.allow_tf32 = bool(args.cudnn_tf32)
    batch = _synthetic_batch(args, device)
    estimates = route_shape_estimates(
        batch_size=args.batch_size,
        total_tokens=args.context_tokens + args.output_slots,
        nodes=args.nodes,
        width=args.d_model,
        lanes=args.lanes,
        relays=args.relays,
        top_k=args.active_node_top_k,
        degree=args.degree,
        value_element_size=torch.empty((), dtype=dtype).element_size(),
    )
    backend_results = []
    for backend in configured_backends(args.route_backend):
        backend_results.append(
            _run_backend(
                args,
                backend=backend,
                device=device,
                dtype=dtype,
                batch=batch,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hardware: dict[str, Any] = {
        "device": str(device),
        "dtype": args.dtype,
        "compile": bool(args.compile),
        "phases": list(args.phases),
        "decode_workload": args.decode_workload,
        "causal": args.decode_workload == "teacher_forced",
        "gradient_clip": args.gradient_clip,
        "float32_matmul_precision": args.float32_matmul_precision,
        "cudnn_tf32": bool(args.cudnn_tf32),
    }
    if device.type == "cuda":
        hardware.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "total_vram_bytes": int(
                    torch.cuda.get_device_properties(device).total_memory
                ),
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "device_resident_synthetic_inputs": True,
            "data_pipeline_included": False,
            "forward_includes_backward": False,
            "forward_backward_includes_optimizer": False,
            "optimizer_reuses_materialized_gradients": True,
            "training_step_includes_optimizer": True,
            "training_step_includes_data_pipeline": False,
        },
        "hardware": hardware,
        "shape": {
            "batch_size": args.batch_size,
            "context_tokens": args.context_tokens,
            "output_slots": args.output_slots,
            "total_tokens": args.context_tokens + args.output_slots,
            "vocab_size": args.vocab_size,
            "embedding_dim": args.embedding_dim,
            "d_model": args.d_model,
            "nodes": args.nodes,
            "lanes": args.lanes,
            "relays": args.relays,
            "degree": args.degree,
            "router_rank": args.router_rank,
            "active_node_top_k": args.active_node_top_k,
            "packed_memory_backend": args.packed_memory_backend,
            "packed_memory_reader_backend": (
                args.packed_memory_reader_backend
            ),
            "packed_memory_writer_backend": (
                args.packed_memory_writer_backend
            ),
            "packed_memory_feature_chunk": args.packed_memory_feature_chunk,
            "packed_memory_compact_nodes": args.packed_memory_compact_nodes,
            "adaptive_route_min_k": args.adaptive_route_min_k,
            "adaptive_route_mode": args.adaptive_route_mode,
            "adaptive_route_entropy_mix": args.adaptive_route_entropy_mix,
            "compact_node_dispatch": args.compact_node_dispatch,
            "node_compute_mode": args.node_compute_mode,
            "node_adapter_rank": args.node_adapter_rank,
            "ingress_groups": args.ingress_groups,
            "ingress_top_groups": args.ingress_top_groups,
            "ingress_learned_groups": args.ingress_learned_groups,
        },
        "timing": {
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
        },
        "route_shape_estimates": estimates,
        "backends": backend_results,
    }


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024**2):.2f} MiB"


def print_console(result: dict[str, Any]) -> None:
    hardware = result["hardware"]
    shape = result["shape"]
    print(
        f"device={hardware['device']} dtype={hardware['dtype']} "
        f"compile={hardware['compile']} "
        f"matmul={hardware.get('float32_matmul_precision', 'unknown')} "
        f"cudnn_tf32={hardware.get('cudnn_tf32', 'unknown')} "
        f"batch={shape['batch_size']} context={shape['context_tokens']} "
        f"output={shape['output_slots']} width={shape['d_model']} "
        f"nodes={shape['nodes']} lanes={shape['lanes']} "
        f"relays={shape['relays']} degree={shape['degree']} "
        f"rank={shape['router_rank']} top_k={shape['active_node_top_k']} "
        f"memory={shape.get('packed_memory_backend', 'matrix')} "
        f"reader={shape.get('packed_memory_reader_backend', 'torch')} "
        f"writer={shape.get('packed_memory_writer_backend', 'torch')} "
        f"memory_chunk={shape.get('packed_memory_feature_chunk', 32)} "
        f"node_compute={shape.get('node_compute_mode', 'full_private')} "
        f"ingress_groups={shape.get('ingress_groups')}"
    )
    if "gpu" in hardware:
        print(
            f"gpu={hardware['gpu']!r} capability="
            f"{'.'.join(map(str, hardware['cuda_capability']))} "
            f"total_vram={_format_bytes(hardware['total_vram_bytes'])}"
        )
    estimates = result["route_shape_estimates"]
    print(
        "route_estimate "
        f"dense_frontier={_format_bytes(estimates['dense_frontier_bytes'])} "
        "packed_frontier_int64_ids="
        f"{_format_bytes(estimates['packed_frontier_bytes_int64_ids'])} "
        f"dense_candidates={estimates['dense_edge_candidates_per_relay']} "
        f"packed_candidates={estimates['packed_edge_candidates_per_relay']}"
    )
    for backend in result["backends"]:
        print(
            f"backend={backend['route_backend']} "
            f"parameters={backend['parameters']:,}"
        )
        for phase in backend["phases"]:
            print(
                f"  phase={phase['name']} "
                f"ms={phase['milliseconds_per_step']:.4f} "
                f"steps_per_second={phase['steps_per_second']:.3f} "
                f"peak_allocated={_format_bytes(phase['peak_allocated_bytes'])} "
                f"peak_reserved={_format_bytes(phase['peak_reserved_bytes'])} "
                "peak_allocated_delta="
                f"{_format_bytes(phase['peak_allocated_delta_bytes'])} "
                "peak_reserved_delta="
                f"{_format_bytes(phase['peak_reserved_delta_bytes'])}"
            )


def emit_json(
    result: dict[str, Any],
    *,
    output: Path | None,
    print_payload: bool,
) -> str:
    payload = json.dumps(result, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if print_payload:
        print(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    if not args.json_only:
        print_console(result)
    emit_json(
        result,
        output=args.json_output,
        print_payload=args.json_only,
    )
    if args.json_output is not None and not args.json_only:
        print(f"json_output={args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
