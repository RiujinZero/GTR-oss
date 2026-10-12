"""Run the fused FFT/Triton Blackwell release gate as isolated processes.

The gate stops on the first failed stage.  Every subprocess streams to the
terminal and to its own local log, while ``summary.json`` is rewritten after
each stage so an interrupted or failed run still leaves machine-readable
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Sequence, TextIO

import torch

from gator.models.triton_packed_memory import (
    triton_packed_reader_available,
)
from gator.models.triton_packed_writer import (
    triton_packed_writer_available,
)


DEFAULT_CUDA_TESTS = (
    "tests.test_triton_packed_memory",
    "tests.test_triton_packed_writer",
    "tests.test_opaque_fft_memory",
)
REPOSITORY_ROOT = Path(__file__).resolve().parent
CUDA_UNIT_GATE_SCRIPT = REPOSITORY_ROOT / "run_cuda_unit_gate.py"
RECEIVER_VALIDATOR_SCRIPT = REPOSITORY_ROOT / "validate_fused_fft_backend.py"
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "benchmark_convergent_routing.py"
FULL_MODEL_EQUIVALENCE_SCRIPT = REPOSITORY_ROOT / (
    "validate_convergent_fft_backend.py"
)
TRAINING_RESUME_SCRIPT = REPOSITORY_ROOT / (
    "validate_convergent_fft_training.py"
)


@dataclass(frozen=True)
class ValidatorShape:
    batch: int
    tokens: int
    lanes: int
    routes: int
    nodes: int
    width: int
    chunk: int
    warmup: int
    iterations: int


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    continue_on_failure: bool = False


@dataclass(frozen=True)
class StageResult:
    name: str
    command: tuple[str, ...]
    log_path: str
    started_utc: str
    completed_utc: str
    duration_seconds: float
    return_code: int
    status: str
    launch_error: str | None = None


StageRunner = Callable[[StageSpec, Path, TextIO], StageResult]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("GTR_SOURCE_REVISION") or None,
        help=(
            "full 40-character Git commit SHA; defaults to "
            "GTR_SOURCE_REVISION"
        ),
    )
    parser.add_argument(
        "--bundle-sha256",
        default=os.environ.get("GTR_BUNDLE_SHA256") or None,
        help=(
            "64-character source-bundle SHA-256; defaults to "
            "GTR_BUNDLE_SHA256"
        ),
    )
    parser.add_argument(
        "--allow-missing-identity",
        action="store_true",
        help=(
            "development only: permit a run without source revision and "
            "bundle SHA-256; release evidence requires both"
        ),
    )
    parser.add_argument(
        "--validation-seed",
        "--seed",
        action="append",
        dest="validation_seeds",
        type=int,
        default=None,
        help="repeat for each correctness seed; defaults to 0 and 1",
    )
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument(
        "--causal-mode",
        choices=("both", "causal", "noncausal"),
        default="both",
        help="receiver/full-model correctness modes to launch",
    )
    parser.add_argument(
        "--causal",
        action="store_const",
        const="causal",
        dest="causal_mode",
        help="alias for --causal-mode causal",
    )
    parser.add_argument(
        "--no-causal",
        action="store_const",
        const="noncausal",
        dest="causal_mode",
        help="alias for --causal-mode noncausal",
    )
    parser.add_argument(
        "--allow-non-blackwell",
        action="store_true",
        help="allow CUDA capability below 10.x for development only",
    )
    parser.add_argument(
        "--cuda-test-module",
        action="append",
        default=None,
        help="repeat to replace the default focused CUDA test modules",
    )

    _add_validator_shape_args(
        parser,
        "quick",
        ValidatorShape(1, 17, 2, 3, 9, 8, 8, 3, 10),
    )

    parser.add_argument(
        "--full-model-equivalence",
        choices=("auto", "required", "skip"),
        default="required",
        help=(
            "required (default) fails if the full-model oracle is absent; "
            "auto includes it when present; skip explicitly omits it"
        ),
    )
    parser.add_argument("--equivalence-batch-size", type=int, default=1)
    parser.add_argument("--equivalence-context-tokens", type=int, default=7)
    parser.add_argument("--equivalence-output-slots", type=int, default=5)
    parser.add_argument("--equivalence-nodes", type=int, default=9)
    parser.add_argument("--equivalence-width", type=int, default=8)
    parser.add_argument("--equivalence-embedding-dim", type=int, default=8)
    parser.add_argument("--equivalence-lanes", type=int, default=2)
    parser.add_argument("--equivalence-routes", type=int, default=3)
    parser.add_argument("--equivalence-degree", type=int, default=3)
    parser.add_argument("--equivalence-relays", type=int, default=2)
    parser.add_argument("--equivalence-router-rank", type=int, default=6)
    parser.add_argument("--equivalence-vocab-size", type=int, default=128)
    parser.add_argument("--equivalence-chunk", type=int, default=4)
    parser.add_argument(
        "--training-resume",
        choices=("auto", "required", "skip"),
        default="required",
        help=(
            "required (default) runs the compiled candidate training and "
            "checkpoint/resume oracle; auto includes it when present; skip "
            "explicitly omits it"
        ),
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=100,
        help="finite compiled candidate steps before checkpoint/resume",
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--training-causal-mode",
        choices=("both", "causal", "noncausal"),
        default="both",
        help=(
            "causality modes for the training/resume oracle; its model shape "
            "uses the equivalence-* arguments"
        ),
    )
    _add_validator_shape_args(
        parser,
        "production",
        ValidatorShape(1, 640, 10, 32, 154, 768, 32, 3, 10),
    )
    parser.add_argument(
        "--production-causal-grad-mass-atol",
        type=float,
        default=1e-3,
        help=(
            "production-causal-only grad_mass absolute tolerance; frozen "
            "from repeated Blackwell FP32 recomputation diagnostics"
        ),
    )
    parser.add_argument(
        "--production-causal-grad-mass-rtol",
        type=float,
        default=1.5e-3,
        help=(
            "production-causal-only grad_mass relative tolerance; every "
            "other receiver tensor retains the validator defaults"
        ),
    )
    parser.add_argument(
        "--production-causal-grad-mass-max-relative-l2",
        type=float,
        default=2e-5,
        help=(
            "independent relative-L2 ceiling for the fused-writer "
            "production-causal tolerance exception"
        ),
    )
    parser.add_argument(
        "--production-causal-grad-mass-max-absolute",
        type=float,
        default=2.5e-2,
        help=(
            "independent maximum-absolute-drift ceiling for the "
            "fused-writer production-causal tolerance exception"
        ),
    )
    parser.add_argument(
        "--production-causal-grad-mass-max-outliers",
        type=int,
        default=4,
        help="maximum sparse elementwise outliers in that one tensor",
    )
    parser.add_argument(
        "--production-causal-grad-mass-max-outlier-fraction",
        type=float,
        default=2e-5,
        help="maximum sparse elementwise outlier fraction in that one tensor",
    )

    parser.add_argument(
        "--full-model-benchmark",
        "--performance-ab",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "run fresh-process dense, packed-Torch, and packed-Triton "
            "performance evidence after correctness passes"
        ),
    )
    parser.add_argument("--benchmark-batch-size", type=int, default=6)
    parser.add_argument("--benchmark-context-tokens", type=int, default=512)
    parser.add_argument("--benchmark-output-slots", type=int, default=128)
    parser.add_argument("--benchmark-nodes", type=int, default=154)
    parser.add_argument("--benchmark-width", type=int, default=768)
    parser.add_argument("--benchmark-embedding-dim", type=int, default=768)
    parser.add_argument("--benchmark-lanes", type=int, default=10)
    parser.add_argument("--benchmark-routes", type=int, default=32)
    parser.add_argument("--benchmark-degree", type=int, default=3)
    parser.add_argument("--benchmark-relays", type=int, default=2)
    parser.add_argument("--benchmark-router-rank", type=int, default=96)
    parser.add_argument("--benchmark-vocab-size", type=int, default=32768)
    parser.add_argument("--benchmark-chunk", type=int, default=32)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    return parser


def _add_validator_shape_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    defaults: ValidatorShape,
) -> None:
    for name in (
        "batch",
        "tokens",
        "lanes",
        "routes",
        "nodes",
        "width",
        "chunk",
        "warmup",
        "iterations",
    ):
        parser.add_argument(
            f"--{prefix}-{name.replace('_', '-')}",
            type=int,
            default=getattr(defaults, name),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.source_revision is not None:
        args.source_revision = args.source_revision.lower()
    if args.bundle_sha256 is not None:
        args.bundle_sha256 = args.bundle_sha256.lower()
    return args


def validator_shape(args: argparse.Namespace, prefix: str) -> ValidatorShape:
    return ValidatorShape(
        **{
            name: getattr(args, f"{prefix}_{name}")
            for name in ValidatorShape.__dataclass_fields__
        }
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.source_revision is not None and re.fullmatch(
        r"[0-9a-fA-F]{40}", args.source_revision
    ) is None:
        raise ValueError(
            "--source-revision must be a full 40-character Git commit SHA"
        )
    if args.bundle_sha256 is not None and re.fullmatch(
        r"[0-9a-fA-F]{64}", args.bundle_sha256
    ) is None:
        raise ValueError(
            "--bundle-sha256 must be exactly 64 hexadecimal characters"
        )
    if not str(args.device).startswith("cuda"):
        raise ValueError("the Blackwell gate requires a CUDA device")
    for prefix in ("quick", "production"):
        shape = validator_shape(args, prefix)
        for name, value in asdict(shape).items():
            if value <= 0:
                raise ValueError(f"--{prefix}-{name} must be positive")
        if shape.routes > shape.nodes:
            raise ValueError(f"--{prefix}-routes cannot exceed nodes")
    benchmark_positive = (
        "benchmark_batch_size",
        "benchmark_context_tokens",
        "benchmark_output_slots",
        "benchmark_nodes",
        "benchmark_width",
        "benchmark_embedding_dim",
        "benchmark_lanes",
        "benchmark_routes",
        "benchmark_degree",
        "benchmark_relays",
        "benchmark_router_rank",
        "benchmark_vocab_size",
        "benchmark_chunk",
        "benchmark_iterations",
    )
    for name in benchmark_positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.benchmark_warmup < 0:
        raise ValueError("--benchmark-warmup must be nonnegative")
    if args.benchmark_routes > args.benchmark_nodes:
        raise ValueError("--benchmark-routes cannot exceed nodes")
    if not validation_seeds(args):
        raise ValueError("at least one validation seed is required")
    equivalence_positive = (
        "equivalence_batch_size",
        "equivalence_context_tokens",
        "equivalence_output_slots",
        "equivalence_nodes",
        "equivalence_width",
        "equivalence_embedding_dim",
        "equivalence_lanes",
        "equivalence_routes",
        "equivalence_degree",
        "equivalence_relays",
        "equivalence_router_rank",
        "equivalence_vocab_size",
        "equivalence_chunk",
    )
    for name in equivalence_positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.equivalence_routes > args.equivalence_nodes:
        raise ValueError("--equivalence-routes cannot exceed nodes")
    if args.training_steps <= 0:
        raise ValueError("--training-steps must be positive")
    if args.training_seed < 0:
        raise ValueError("--training-seed must be nonnegative")
    for name in (
        "production_causal_grad_mass_atol",
        "production_causal_grad_mass_rtol",
        "production_causal_grad_mass_max_relative_l2",
        "production_causal_grad_mass_max_absolute",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and nonnegative"
            )
    if args.production_causal_grad_mass_max_outliers < 0:
        raise ValueError(
            "--production-causal-grad-mass-max-outliers must be nonnegative"
        )
    fraction = args.production_causal_grad_mass_max_outlier_fraction
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(
            "--production-causal-grad-mass-max-outlier-fraction must be "
            "finite and between 0 and 1"
        )


def validation_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    values = args.validation_seeds
    return tuple(values) if values is not None else (0, 1)


def validation_causal_modes(args: argparse.Namespace) -> tuple[bool, ...]:
    if args.causal_mode == "both":
        return (False, True)
    if args.causal_mode == "causal":
        return (True,)
    if args.causal_mode == "noncausal":
        return (False,)
    raise ValueError(f"unknown causal mode: {args.causal_mode}")


def require_target_environment(
    device_value: str,
    *,
    require_blackwell: bool,
) -> dict[str, Any]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("Blackwell release gate requires torch.compile")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Blackwell release gate requires CUDA; "
            "torch.cuda.is_available() is false"
        )
    device = torch.device(device_value)
    if device.type != "cuda":
        raise RuntimeError("Blackwell release gate requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    capability = tuple(torch.cuda.get_device_capability(device))
    if require_blackwell and capability[0] < 10:
        raise RuntimeError(
            "Blackwell release gate requires CUDA capability 10.x or newer; "
            f"found {capability[0]}.{capability[1]}"
        )
    if not triton_packed_reader_available():
        raise RuntimeError(
            "Blackwell release gate requires the Triton packed reader plus "
            "torch.library.triton_op/wrap_triton support"
        )
    if not triton_packed_writer_available():
        raise RuntimeError(
            "Blackwell release gate requires the Triton packed writer plus "
            "torch.library.triton_op/wrap_triton support"
        )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_version": torch.version.cuda,
        "triton_version": package_version("triton"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_capability": list(capability),
        "total_vram_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "triton_reader_available": True,
        "triton_writer_available": True,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _validator_command(
    python: str,
    device: str,
    shape: ValidatorShape,
    *,
    seed: int,
    causal: bool,
    grad_mass_atol: float | None = None,
    grad_mass_rtol: float | None = None,
    grad_mass_max_relative_l2: float | None = None,
    grad_mass_max_absolute: float | None = None,
    grad_mass_max_outliers: int | None = None,
    grad_mass_max_outlier_fraction: float | None = None,
) -> tuple[str, ...]:
    command = [
        python,
        "-u",
        str(RECEIVER_VALIDATOR_SCRIPT),
        "--device",
        device,
        "--batch-size",
        str(shape.batch),
        "--tokens",
        str(shape.tokens),
        "--lanes",
        str(shape.lanes),
        "--route-width",
        str(shape.routes),
        "--nodes",
        str(shape.nodes),
        "--width",
        str(shape.width),
        "--feature-chunk",
        str(shape.chunk),
        "--warmup-steps",
        str(shape.warmup),
        "--measured-steps",
        str(shape.iterations),
        "--seed",
        str(seed),
        "--compile",
        "--packed-memory-writer-backend",
        "triton",
    ]
    for flag, value in (
        ("--grad-mass-atol", grad_mass_atol),
        ("--grad-mass-rtol", grad_mass_rtol),
        ("--grad-mass-max-relative-l2", grad_mass_max_relative_l2),
        ("--grad-mass-max-absolute", grad_mass_max_absolute),
        ("--grad-mass-max-outliers", grad_mass_max_outliers),
        ("--grad-mass-max-outlier-fraction", grad_mass_max_outlier_fraction),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    command.append("--causal" if causal else "--no-causal")
    return tuple(command)


def _benchmark_command(
    args: argparse.Namespace,
    *,
    route_backend: str,
    memory_backend: str,
    reader_backend: str,
    writer_backend: str,
    json_output: Path,
) -> tuple[str, ...]:
    return (
        args.python,
        "-u",
        str(BENCHMARK_SCRIPT),
        "--route-backend",
        route_backend,
        "--packed-memory-backend",
        memory_backend,
        "--packed-memory-reader-backend",
        reader_backend,
        "--packed-memory-writer-backend",
        writer_backend,
        "--packed-memory-feature-chunk",
        str(args.benchmark_chunk),
        "--vocab-size",
        str(args.benchmark_vocab_size),
        "--embedding-dim",
        str(args.benchmark_embedding_dim),
        "--d-model",
        str(args.benchmark_width),
        "--nodes",
        str(args.benchmark_nodes),
        "--lanes",
        str(args.benchmark_lanes),
        "--degree",
        str(args.benchmark_degree),
        "--relays",
        str(args.benchmark_relays),
        "--router-rank",
        str(args.benchmark_router_rank),
        "--active-node-top-k",
        str(args.benchmark_routes),
        "--batch-size",
        str(args.benchmark_batch_size),
        "--context-tokens",
        str(args.benchmark_context_tokens),
        "--output-slots",
        str(args.benchmark_output_slots),
        "--device",
        args.device,
        "--dtype",
        "float32",
        "--compile",
        "--phase",
        "training_step",
        "--decode-workload",
        "teacher_forced",
        "--gradient-clip",
        "1.0",
        "--warmup-steps",
        str(args.benchmark_warmup),
        "--measured-steps",
        str(args.benchmark_iterations),
        "--seed",
        str(args.benchmark_seed),
        "--json-output",
        str(json_output),
    )


def full_model_equivalence_available() -> bool:
    return FULL_MODEL_EQUIVALENCE_SCRIPT.is_file()


def training_resume_available() -> bool:
    return TRAINING_RESUME_SCRIPT.is_file()


def _full_model_equivalence_command(
    args: argparse.Namespace,
    *,
    seed: int,
    causal: bool,
) -> tuple[str, ...]:
    command = [
        args.python,
        "-u",
        str(FULL_MODEL_EQUIVALENCE_SCRIPT),
        "--device",
        args.device,
        "--batch-size",
        str(args.equivalence_batch_size),
        "--context-tokens",
        str(args.equivalence_context_tokens),
        "--output-slots",
        str(args.equivalence_output_slots),
        "--nodes",
        str(args.equivalence_nodes),
        "--d-model",
        str(args.equivalence_width),
        "--embedding-dim",
        str(args.equivalence_embedding_dim),
        "--lanes",
        str(args.equivalence_lanes),
        "--active-node-top-k",
        str(args.equivalence_routes),
        "--degree",
        str(args.equivalence_degree),
        "--relays",
        str(args.equivalence_relays),
        "--router-rank",
        str(args.equivalence_router_rank),
        "--vocab-size",
        str(args.equivalence_vocab_size),
        "--packed-memory-feature-chunk",
        str(args.equivalence_chunk),
        "--seed",
        str(seed),
        "--compile",
        "--packed-memory-writer-backend",
        "triton",
    ]
    command.append("--causal" if causal else "--no-causal")
    return tuple(command)


def _training_resume_command(
    args: argparse.Namespace,
    *,
    causal: bool,
    checkpoint_path: Path,
) -> tuple[str, ...]:
    command = [
        args.python,
        "-u",
        str(TRAINING_RESUME_SCRIPT),
        "--device",
        args.device,
        "--steps",
        str(args.training_steps),
        "--batch-size",
        str(args.equivalence_batch_size),
        "--context-tokens",
        str(args.equivalence_context_tokens),
        "--output-slots",
        str(args.equivalence_output_slots),
        "--nodes",
        str(args.equivalence_nodes),
        "--d-model",
        str(args.equivalence_width),
        "--embedding-dim",
        str(args.equivalence_embedding_dim),
        "--lanes",
        str(args.equivalence_lanes),
        "--active-node-top-k",
        str(args.equivalence_routes),
        "--degree",
        str(args.equivalence_degree),
        "--relays",
        str(args.equivalence_relays),
        "--router-rank",
        str(args.equivalence_router_rank),
        "--vocab-size",
        str(args.equivalence_vocab_size),
        "--packed-memory-feature-chunk",
        str(args.equivalence_chunk),
        "--seed",
        str(args.training_seed),
        "--checkpoint-path",
        str(checkpoint_path),
        "--compile",
        "--packed-memory-writer-backend",
        "triton",
    ]
    command.append("--causal" if causal else "--no-causal")
    return tuple(command)


def build_stages(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[StageSpec, ...]:
    test_modules = args.cuda_test_module or list(DEFAULT_CUDA_TESTS)
    stages = [
        StageSpec(
            name="cuda_unit_tests",
            command=(
                args.python,
                "-u",
                str(CUDA_UNIT_GATE_SCRIPT),
                *test_modules,
            ),
        )
    ]
    seeds = validation_seeds(args)
    causal_modes = validation_causal_modes(args)
    for shape_name in ("quick", "production"):
        shape = validator_shape(args, shape_name)
        for seed in seeds:
            for causal in causal_modes:
                mode = "causal" if causal else "noncausal"
                stages.append(
                    StageSpec(
                        name=(
                            f"{shape_name}_validator_seed{seed}_{mode}"
                        ),
                        command=_validator_command(
                            args.python,
                            args.device,
                            shape,
                            seed=seed,
                            causal=causal,
                            grad_mass_atol=(
                                args.production_causal_grad_mass_atol
                                if shape_name == "production" and causal
                                else None
                            ),
                            grad_mass_rtol=(
                                args.production_causal_grad_mass_rtol
                                if shape_name == "production" and causal
                                else None
                            ),
                            grad_mass_max_relative_l2=(
                                args.production_causal_grad_mass_max_relative_l2
                                if shape_name == "production" and causal
                                else None
                            ),
                            grad_mass_max_absolute=(
                                args.production_causal_grad_mass_max_absolute
                                if shape_name == "production" and causal
                                else None
                            ),
                            grad_mass_max_outliers=(
                                args.production_causal_grad_mass_max_outliers
                                if shape_name == "production" and causal
                                else None
                            ),
                            grad_mass_max_outlier_fraction=(
                                args.production_causal_grad_mass_max_outlier_fraction
                                if shape_name == "production" and causal
                                else None
                            ),
                        ),
                    )
                )

    include_equivalence = (
        args.full_model_equivalence == "required"
        or (
            args.full_model_equivalence == "auto"
            and full_model_equivalence_available()
        )
    )
    if (
        args.full_model_equivalence == "required"
        and not full_model_equivalence_available()
    ):
        raise FileNotFoundError(
            f"required full-model equivalence script is missing: "
            f"{FULL_MODEL_EQUIVALENCE_SCRIPT}"
        )
    if include_equivalence:
        for seed in seeds:
            for causal in causal_modes:
                mode = "causal" if causal else "noncausal"
                stages.append(
                    StageSpec(
                        name=f"full_model_equivalence_seed{seed}_{mode}",
                        command=_full_model_equivalence_command(
                            args,
                            seed=seed,
                            causal=causal,
                        ),
                    )
                )
    include_training_resume = (
        args.training_resume == "required"
        or (
            args.training_resume == "auto"
            and training_resume_available()
        )
    )
    if args.training_resume == "required" and not training_resume_available():
        raise FileNotFoundError(
            f"required training/resume script is missing: "
            f"{TRAINING_RESUME_SCRIPT}"
        )
    if include_training_resume:
        training_modes = (
            (False, True)
            if args.training_causal_mode == "both"
            else (args.training_causal_mode == "causal",)
        )
        for causal in training_modes:
            mode = "causal" if causal else "noncausal"
            stages.append(
                StageSpec(
                    name=(
                        f"candidate_training_resume_seed"
                        f"{args.training_seed}_{mode}"
                    ),
                    command=_training_resume_command(
                        args,
                        causal=causal,
                        checkpoint_path=(
                            output_dir
                            / f"candidate_training_resume_{mode}.pt"
                        ),
                    ),
                )
            )
    if args.full_model_benchmark:
        performance_cases = (
            (
                "dense_matrix_torch",
                "dense",
                "matrix",
                "torch",
                "torch",
            ),
            (
                "packed_fft_opaque_torch",
                "packed",
                "fft_opaque",
                "torch",
                "torch",
            ),
            (
                "packed_fft_opaque_triton_reader_torch_writer",
                "packed",
                "fft_opaque",
                "triton",
                "torch",
            ),
            (
                "packed_fft_opaque_triton_reader_triton_writer",
                "packed",
                "fft_opaque",
                "triton",
                "triton",
            ),
        )
        for (
            name,
            route_backend,
            memory_backend,
            reader_backend,
            writer_backend,
        ) in performance_cases:
            stages.append(
                StageSpec(
                    name=f"benchmark_{name}",
                    command=_benchmark_command(
                        args,
                        route_backend=route_backend,
                        memory_backend=memory_backend,
                        reader_backend=reader_backend,
                        writer_backend=writer_backend,
                        json_output=output_dir / f"benchmark_{name}.json",
                    ),
                    # Performance evidence is only meaningful as a complete
                    # matrix. Attempt every isolated backend even when one
                    # fails, while the final gate status remains failed.
                    continue_on_failure=True,
                )
            )
    return tuple(stages)


def _write_both(combined: TextIO, text: str) -> None:
    print(text, end="", flush=True)
    combined.write(text)
    combined.flush()


def run_stage(spec: StageSpec, output_dir: Path, combined: TextIO) -> StageResult:
    log_path = output_dir / f"{spec.name}.log"
    started_utc = utc_now()
    started = time.perf_counter()
    header = (
        f"\n{'=' * 100}\n"
        f"stage={spec.name}\n"
        f"command={' '.join(spec.command)}\n"
        f"{'=' * 100}\n"
    )
    _write_both(combined, header)
    return_code = 127
    launch_error = None
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stage_log:
            stage_log.write(header)
            process = subprocess.Popen(
                spec.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=REPOSITORY_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=os.name != "nt",
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    combined.write(line)
                    combined.flush()
                    stage_log.write(line)
            except BaseException:
                terminate_process_tree(process)
                raise
            finally:
                process.stdout.close()
            return_code = process.wait()
    except (OSError, subprocess.SubprocessError) as error:
        launch_error = f"{type(error).__name__}: {error}"
        _write_both(combined, f"launch_error={launch_error}\n")
    duration = time.perf_counter() - started
    status = "passed" if return_code == 0 and launch_error is None else "failed"
    _write_both(
        combined,
        f"stage={spec.name} status={status} return_code={return_code} "
        f"duration_seconds={duration:.3f}\n",
    )
    return StageResult(
        name=spec.name,
        command=spec.command,
        log_path=str(log_path),
        started_utc=started_utc,
        completed_utc=utc_now(),
        duration_seconds=duration,
        return_code=return_code,
        status=status,
        launch_error=launch_error,
    )


def execute_stages(
    specs: Sequence[StageSpec],
    output_dir: Path,
    combined: TextIO,
    *,
    runner: StageRunner = run_stage,
    on_result: Callable[[StageResult], None] | None = None,
) -> tuple[StageResult, ...]:
    results = []
    for spec in specs:
        result = runner(spec, output_dir, combined)
        results.append(result)
        if on_result is not None:
            on_result(result)
        if result.status != "passed" and not spec.continue_on_failure:
            break
    return tuple(results)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop an interrupted stage and its POSIX compiler subprocess group."""

    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        if process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        else:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.SubprocessError:
            pass


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("blackwell_release_gate") / timestamp


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (args.output_dir or default_output_dir()).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        "output_dir": str(output_dir),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": None,
        "evidence_identity": {
            "source_revision": args.source_revision,
            "bundle_sha256": args.bundle_sha256,
        },
        "stages": [],
        "failed_stage": None,
    }
    summary["configuration"]["effective_validation_seeds"] = list(
        validation_seeds(args)
    )
    summary["configuration"]["effective_causal_modes"] = [
        "causal" if causal else "noncausal"
        for causal in validation_causal_modes(args)
    ]
    write_summary(summary_path, summary)
    combined_path = output_dir / "release_gate.log"
    with combined_path.open("w", encoding="utf-8", buffering=1) as combined:
        try:
            if not args.allow_missing_identity and (
                args.source_revision is None or args.bundle_sha256 is None
            ):
                raise RuntimeError(
                    "release evidence requires --source-revision and "
                    "--bundle-sha256 (or their GTR_* environment values)"
                )
            summary["environment"] = require_target_environment(
                args.device,
                require_blackwell=not args.allow_non_blackwell,
            )
            summary["environment"].update(summary["evidence_identity"])
            _write_both(
                combined,
                "evidence "
                f"source_revision={args.source_revision or 'UNSET'} "
                f"bundle_sha256={args.bundle_sha256 or 'UNSET'}\n",
            )
            _write_both(
                combined,
                "runtime "
                f"python={summary['environment']['python_version']} "
                f"torch={summary['environment']['torch_version']} "
                f"torch_git={summary['environment']['torch_git_version']} "
                f"cuda={summary['environment']['torch_cuda_version']} "
                f"triton={summary['environment']['triton_version']}\n",
            )
            write_summary(summary_path, summary)
            specs = build_stages(args, output_dir=output_dir)

            def record(result: StageResult) -> None:
                summary["stages"].append(asdict(result))
                if result.status != "passed":
                    summary["failed_stage"] = result.name
                write_summary(summary_path, summary)

            results = execute_stages(
                specs,
                output_dir,
                combined,
                on_result=record,
            )
            passed = len(results) == len(specs) and all(
                result.status == "passed" for result in results
            )
            summary["status"] = "passed" if passed else "failed"
        except BaseException as error:
            summary["status"] = "failed"
            summary["preflight_error"] = f"{type(error).__name__}: {error}"
            _write_both(combined, f"release_gate_error={summary['preflight_error']}\n")
        finally:
            summary["completed_utc"] = utc_now()
            write_summary(summary_path, summary)
    print(
        f"release_gate={summary['status']} summary={summary_path} "
        f"log={combined_path}",
        flush=True,
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
