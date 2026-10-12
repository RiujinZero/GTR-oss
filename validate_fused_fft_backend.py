"""Validate the compile-safe FFT receiver and fused Triton kernels on CUDA.

This is a target-GPU release gate, not a CPU smoke.  It compares the existing
raw eager FFT receiver with its Torch writer/reader against the compiled
opaque FFT boundary with the fused Triton writer/reader.  Both paths receive
the same unique packed routes, real float32 tensors, and gradient probe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import math
import time
from typing import Callable, Sequence

import torch

from gator.models.opaque_fft_memory import opaque_packed_fft_node_context
from gator.models.packed_convolution_memory import (
    packed_convolution_node_context,
)
from gator.models.triton_packed_memory import (
    triton_packed_reader_available,
)
from gator.models.triton_packed_writer import (
    triton_packed_writer_available,
)


TensorTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
Receiver = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Shape:
    batch: int
    tokens: int
    lanes: int
    routes: int
    nodes: int
    width: int
    feature_chunk: int


@dataclass(frozen=True)
class Timing:
    milliseconds: float
    peak_allocated: int
    peak_reserved: int
    peak_allocated_delta: int
    peak_reserved_delta: int


@dataclass(frozen=True)
class Drift:
    maximum_absolute: float
    maximum_relative: float
    relative_l2: float
    mismatched: int
    nonfinite: int
    total: int
    worst_tolerance_ratio: float
    worst_index: tuple[int, ...]
    worst_actual: float
    worst_expected: float
    worst_allowed: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", "--batch", dest="batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument(
        "--routes",
        "--top-k",
        "--route-width",
        "--active-node-top-k",
        dest="routes",
        type=int,
        default=3,
    )
    parser.add_argument("--nodes", type=int, default=9)
    parser.add_argument("--width", "--d-model", dest="width", type=int, default=8)
    parser.add_argument(
        "--feature-chunk",
        "--chunk",
        "--packed-memory-feature-chunk",
        dest="feature_chunk",
        type=int,
        default=8,
    )
    parser.add_argument("--warmup", "--warmup-steps", type=int, default=3)
    parser.add_argument(
        "--iterations",
        "--measured-steps",
        type=int,
        default=10,
    )
    parser.add_argument("--seed", type=int, default=937)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--route-backend",
        choices=("packed",),
        default="packed",
        help="fixed validator backend; accepted for trainer-command parity",
    )
    parser.add_argument(
        "--packed-memory-backend",
        choices=("fft_opaque",),
        default="fft_opaque",
        help="fixed validator backend; accepted for trainer-command parity",
    )
    parser.add_argument(
        "--packed-memory-reader-backend",
        choices=("triton",),
        default="triton",
        help="fixed validator backend; accepted for trainer-command parity",
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=("triton",),
        default="triton",
        help="fixed validator backend; accepted for trainer-command parity",
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument(
        "--grad-mass-atol",
        type=float,
        default=None,
        help="optional elementwise absolute tolerance override for grad_mass",
    )
    parser.add_argument(
        "--grad-mass-rtol",
        type=float,
        default=None,
        help="optional elementwise relative tolerance override for grad_mass",
    )
    parser.add_argument(
        "--grad-mass-max-relative-l2",
        type=float,
        default=None,
        help="optional independent relative-L2 ceiling for grad_mass",
    )
    parser.add_argument(
        "--grad-mass-max-absolute",
        type=float,
        default=None,
        help="optional independent maximum-absolute-drift ceiling for grad_mass",
    )
    parser.add_argument(
        "--grad-mass-max-outliers",
        type=int,
        default=None,
        help=(
            "optional count ceiling for sparse grad_mass elementwise "
            "outliers; requires the matching fraction and aggregate guards"
        ),
    )
    parser.add_argument(
        "--grad-mass-max-outlier-fraction",
        type=float,
        default=None,
        help=(
            "optional fraction ceiling for sparse grad_mass elementwise "
            "outliers; requires the matching count and aggregate guards"
        ),
    )
    parser.add_argument(
        "--diagnose-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "compare raw Torch autograd, raw functional VJP, raw Triton, "
            "compiled opaque Torch, and the compiled target; diagnostic "
            "mode exits before timing so live comparison tensors cannot "
            "contaminate peak-memory evidence"
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="required; accepted for parity with repository benchmark commands",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch": args.batch,
        "tokens": args.tokens,
        "lanes": args.lanes,
        "routes": args.routes,
        "nodes": args.nodes,
        "width": args.width,
        "feature_chunk": args.feature_chunk,
        "iterations": args.iterations,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(
            "these arguments must be positive: " + ", ".join(invalid)
        )
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.routes > args.nodes:
        raise ValueError("routes cannot exceed nodes when routes must be unique")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol):
        raise ValueError("tolerances must be finite")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("tolerances must be nonnegative")
    for name in (
        "grad_mass_atol",
        "grad_mass_rtol",
        "grad_mass_max_relative_l2",
        "grad_mass_max_absolute",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(
                f"{name.replace('_', '-')} must be finite and nonnegative"
            )
    if (
        args.grad_mass_max_outliers is not None
        and args.grad_mass_max_outliers < 0
    ):
        raise ValueError("grad-mass-max-outliers must be nonnegative")
    if args.grad_mass_max_outlier_fraction is not None and (
        not math.isfinite(args.grad_mass_max_outlier_fraction)
        or not 0 <= args.grad_mass_max_outlier_fraction <= 1
    ):
        raise ValueError(
            "grad-mass-max-outlier-fraction must be finite and between 0 and 1"
        )
    sparse_values = (
        args.grad_mass_max_outliers,
        args.grad_mass_max_outlier_fraction,
    )
    if (sparse_values[0] is None) != (sparse_values[1] is None):
        raise ValueError(
            "grad-mass sparse outlier count and fraction must be set together"
        )
    if sparse_values[0] is not None and (
        args.grad_mass_max_relative_l2 is None
        or args.grad_mass_max_absolute is None
    ):
        raise ValueError(
            "grad-mass sparse outliers require relative-L2 and maximum-absolute guards"
        )
    if not str(args.device).startswith("cuda"):
        raise ValueError("this target-GPU validator requires a CUDA device")
    if not args.compile:
        raise ValueError("this validator requires --compile")


def require_cuda_triton() -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("fused FFT validation requires torch.compile")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "fused FFT validation requires CUDA; torch.cuda.is_available() "
            "is false"
        )
    if not triton_packed_reader_available():
        raise RuntimeError(
            "fused FFT validation requires the Triton packed reader plus "
            "torch.library.triton_op/wrap_triton support"
        )
    if not triton_packed_writer_available():
        raise RuntimeError(
            "fused FFT validation requires the Triton packed writer plus "
            "torch.library.triton_op/wrap_triton support"
        )


def shape_from_args(args: argparse.Namespace) -> Shape:
    return Shape(
        batch=args.batch,
        tokens=args.tokens,
        lanes=args.lanes,
        routes=args.routes,
        nodes=args.nodes,
        width=args.width,
        feature_chunk=args.feature_chunk,
    )


def build_unique_inputs(
    shape: Shape,
    *,
    device: torch.device | str,
    seed: int,
) -> TensorTuple:
    """Build normalized unique packed routes and differentiable FP32 state."""

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden = torch.randn(
        shape.batch,
        shape.tokens,
        shape.lanes,
        shape.width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).requires_grad_(True)
    # topk returns distinct source indices, giving every packet a collision-
    # merged frontier without a Python loop or CPU staging.
    route_scores = torch.rand(
        shape.batch,
        shape.tokens,
        shape.lanes,
        shape.nodes,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    nodes = route_scores.topk(shape.routes, dim=-1).indices.contiguous()
    mass = torch.softmax(
        torch.randn(
            shape.batch,
            shape.tokens,
            shape.lanes,
            shape.routes,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    ).detach().requires_grad_(True)
    valid = torch.ones_like(nodes, dtype=torch.bool)
    write = (
        0.25
        + 0.75
        * torch.rand(
            shape.batch,
            shape.tokens,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    ).requires_grad_(True)
    bias = (
        0.1
        * torch.randn(
            shape.nodes,
            4,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    ).requires_grad_(True)
    return hidden, nodes, mass, valid, write, bias


def clone_inputs(inputs: TensorTuple) -> TensorTuple:
    hidden, nodes, mass, valid, write, bias = inputs
    return (
        hidden.detach().clone().requires_grad_(True),
        nodes.clone(),
        mass.detach().clone().requires_grad_(True),
        valid.clone(),
        write.detach().clone().requires_grad_(True),
        bias.detach().clone().requires_grad_(True),
    )


def differentiable(inputs: TensorTuple) -> tuple[torch.Tensor, ...]:
    return inputs[0], inputs[2], inputs[4], inputs[5]


def evaluate(
    receiver: Receiver,
    inputs: TensorTuple,
    probe: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    output = receiver(*inputs)
    gradients = torch.autograd.grad(
        (output * probe).sum(),
        differentiable(inputs),
    )
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def evaluate_vjp(
    receiver: Receiver,
    inputs: TensorTuple,
    probe: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Evaluate the receiver through the same functional VJP as opaque FFT."""

    hidden, nodes, mass, valid, write, bias = inputs

    def differentiable_receiver(
        receiver_hidden: torch.Tensor,
        receiver_mass: torch.Tensor,
        receiver_write: torch.Tensor,
        receiver_bias: torch.Tensor,
    ) -> torch.Tensor:
        return receiver(
            receiver_hidden,
            nodes,
            receiver_mass,
            valid,
            receiver_write,
            receiver_bias,
        )

    output, vjp = torch.func.vjp(
        differentiable_receiver,
        hidden,
        mass,
        write,
        bias,
    )
    gradients = vjp(probe)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _unravel_index(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
    coordinates: list[int] = []
    remaining = flat_index
    for size in reversed(shape):
        coordinates.append(remaining % size)
        remaining //= size
    return tuple(reversed(coordinates))


def drift(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Drift:
    actual_float = actual.float()
    expected_float = expected.float()
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    safe_actual = torch.where(finite, actual_float, torch.zeros_like(actual_float))
    safe_expected = torch.where(
        finite,
        expected_float,
        torch.zeros_like(expected_float),
    )
    finite_difference = (safe_actual - safe_expected).abs()
    difference = torch.where(
        finite,
        finite_difference,
        torch.full_like(finite_difference, float("inf")),
    )
    scale = torch.maximum(safe_actual.abs(), safe_expected.abs()).clamp_min(1e-7)
    allowed = atol + rtol * safe_expected.abs()
    tolerance_ratio = difference / allowed.clamp_min(1e-30)
    flat_ratio = tolerance_ratio.reshape(-1)
    worst_flat_index = int(flat_ratio.argmax().item())
    flat_actual = actual_float.reshape(-1)
    flat_expected = expected_float.reshape(-1)
    flat_allowed = allowed.reshape(-1)
    comparison_norm = torch.maximum(
        torch.linalg.vector_norm(safe_actual),
        torch.linalg.vector_norm(safe_expected),
    ).clamp_min(1e-30)
    relative_l2 = torch.linalg.vector_norm(finite_difference) / comparison_norm
    if not bool(finite.all()):
        relative_l2 = torch.full_like(relative_l2, float("inf"))
    return Drift(
        maximum_absolute=float(difference.max()),
        maximum_relative=float((difference / scale).max()),
        relative_l2=float(relative_l2),
        mismatched=int(difference.gt(allowed).sum().item()),
        nonfinite=int((~finite).sum().item()),
        total=difference.numel(),
        worst_tolerance_ratio=float(flat_ratio[worst_flat_index].item()),
        worst_index=_unravel_index(worst_flat_index, actual.shape),
        worst_actual=float(flat_actual[worst_flat_index].item()),
        worst_expected=float(flat_expected[worst_flat_index].item()),
        worst_allowed=float(flat_allowed[worst_flat_index].item()),
    )


def compare_values(
    comparison: str,
    labels: tuple[str, ...],
    actual_values: tuple[torch.Tensor, ...],
    expected_values: tuple[torch.Tensor, ...],
    *,
    atol: float,
    rtol: float,
    tolerance_overrides: dict[str, tuple[float, float]] | None = None,
    relative_l2_limits: dict[str, float] | None = None,
    maximum_absolute_limits: dict[str, float] | None = None,
    sparse_outlier_limits: dict[str, tuple[int, float]] | None = None,
) -> list[str]:
    """Report every tensor and collect failures instead of stopping early."""

    if len(labels) != len(actual_values) or len(labels) != len(expected_values):
        raise ValueError(
            "comparison labels, actual values, and expected values must "
            "have equal lengths"
        )
    failures: list[str] = []
    for label, actual, expected in zip(labels, actual_values, expected_values):
        if actual.shape != expected.shape:
            message = (
                f"{comparison}/{label}: shape mismatch "
                f"actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
            )
            print(f"drift comparison={comparison} tensor={label} {message}", flush=True)
            failures.append(message)
            continue
        metadata_mismatches: list[str] = []
        if actual.dtype != expected.dtype:
            metadata_mismatches.append(
                f"dtype actual={actual.dtype} expected={expected.dtype}"
            )
        if actual.device != expected.device:
            metadata_mismatches.append(
                f"device actual={actual.device} expected={expected.device}"
            )
        if actual.layout != expected.layout:
            metadata_mismatches.append(
                f"layout actual={actual.layout} expected={expected.layout}"
            )
        if metadata_mismatches:
            message = (
                f"{comparison}/{label}: metadata mismatch "
                + "; ".join(metadata_mismatches)
            )
            print(
                f"drift comparison={comparison} tensor={label} {message}",
                flush=True,
            )
            failures.append(message)
            continue
        tensor_atol, tensor_rtol = (
            (tolerance_overrides or {}).get(label, (atol, rtol))
        )
        result = drift(
            actual,
            expected,
            atol=tensor_atol,
            rtol=tensor_rtol,
        )
        sparse_limit = (sparse_outlier_limits or {}).get(label)
        if sparse_limit is not None and (
            label not in (relative_l2_limits or {})
            or label not in (maximum_absolute_limits or {})
        ):
            raise ValueError(
                f"{label} sparse outlier policy requires relative-L2 and "
                "maximum-absolute guards"
            )
        outlier_fraction = result.mismatched / result.total
        print(
            f"drift comparison={comparison} tensor={label} "
            f"atol={tensor_atol:.9g} rtol={tensor_rtol:.9g} "
            f"max_abs={result.maximum_absolute:.9g} "
            f"max_rel={result.maximum_relative:.9g} "
            f"rel_l2={result.relative_l2:.9g} "
            f"mismatched={result.mismatched}/{result.total} "
            f"outlier_fraction={outlier_fraction:.9g} "
            f"nonfinite={result.nonfinite}/{result.total} "
            f"elementwise_policy={'sparse' if sparse_limit else 'strict'} "
            f"worst_tolerance_ratio={result.worst_tolerance_ratio:.9g} "
            f"worst_index={result.worst_index} "
            f"worst_actual={result.worst_actual:.9g} "
            f"worst_expected={result.worst_expected:.9g} "
            f"worst_allowed={result.worst_allowed:.9g}",
            flush=True,
        )
        accepted_sparse_outliers = False
        tensor_failure_count = len(failures)
        if sparse_limit is None:
            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=tensor_atol,
                    rtol=tensor_rtol,
                    msg=lambda message, name=label, axis=comparison: (
                        f"{axis}/{name}: {message}"
                    ),
                )
            except AssertionError as error:
                failures.append(str(error))
        else:
            maximum_outliers, maximum_fraction = sparse_limit
            if (
                result.mismatched > maximum_outliers
                or outlier_fraction > maximum_fraction
            ):
                failures.append(
                    f"{comparison}/{label}: sparse elementwise outliers "
                    f"{result.mismatched}/{result.total} "
                    f"({outlier_fraction:.9g}) exceed count={maximum_outliers} "
                    f"or fraction={maximum_fraction:.9g}"
                )
            elif result.mismatched:
                accepted_sparse_outliers = True
        if result.nonfinite:
            failures.append(
                f"{comparison}/{label}: nonfinite={result.nonfinite}/"
                f"{result.total}; all compared values must be finite"
            )
        relative_l2_limit = (relative_l2_limits or {}).get(label)
        if (
            relative_l2_limit is not None
            and result.relative_l2 > relative_l2_limit
        ):
            failures.append(
                f"{comparison}/{label}: relative_l2={result.relative_l2:.9g} "
                f"exceeds {relative_l2_limit:.9g}"
            )
        maximum_absolute_limit = (maximum_absolute_limits or {}).get(label)
        if (
            maximum_absolute_limit is not None
            and result.maximum_absolute > maximum_absolute_limit
        ):
            failures.append(
                f"{comparison}/{label}: max_abs={result.maximum_absolute:.9g} "
                f"exceeds {maximum_absolute_limit:.9g}"
            )
        if accepted_sparse_outliers and len(failures) == tensor_failure_count:
            print(
                f"elementwise_acceptance comparison={comparison} tensor={label} "
                f"accepted_sparse_outliers={result.mismatched}/{result.total} "
                f"count_limit={maximum_outliers} "
                f"fraction_limit={maximum_fraction:.9g}",
                flush=True,
            )
    return failures


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def measure(
    receiver: Receiver,
    inputs: TensorTuple,
    probe: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    backward: bool,
) -> Timing:
    def invoke() -> torch.Tensor:
        output = receiver(*inputs)
        if not backward:
            return output
        gradients = torch.autograd.grad(
            (output * probe).sum(),
            differentiable(inputs),
        )
        return gradients[0]

    observable = None
    for _ in range(warmup):
        observable = invoke()
    _synchronize(device)
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(iterations):
        observable = invoke()
    finished.record()
    finished.synchronize()
    if observable is None or not bool(torch.isfinite(observable).all()):
        raise FloatingPointError("timed receiver produced a non-finite result")
    return Timing(
        milliseconds=float(started.elapsed_time(finished)) / iterations,
        peak_allocated=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved=int(torch.cuda.max_memory_reserved(device)),
        peak_allocated_delta=max(
            int(torch.cuda.max_memory_allocated(device)) - baseline_allocated,
            0,
        ),
        peak_reserved_delta=max(
            int(torch.cuda.max_memory_reserved(device)) - baseline_reserved,
            0,
        ),
    )


def print_timing(label: str, phase: str, result: Timing) -> None:
    print(
        f"timing backend={label} phase={phase} ms={result.milliseconds:.4f} "
        f"steps_per_second={1000.0 / result.milliseconds:.3f} "
        f"peak_allocated_bytes={result.peak_allocated} "
        f"peak_reserved_bytes={result.peak_reserved} "
        f"peak_allocated_delta_bytes={result.peak_allocated_delta} "
        f"peak_reserved_delta_bytes={result.peak_reserved_delta}",
        flush=True,
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require_cuda_triton()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    shape = shape_from_args(args)
    print(
        f"runtime torch={torch.__version__} cuda={torch.version.cuda} "
        f"triton={package_version('triton')} device={device} "
        f"gpu={torch.cuda.get_device_name(device)!r} "
        f"capability={torch.cuda.get_device_capability(device)} "
        f"total_vram_bytes={torch.cuda.get_device_properties(device).total_memory}",
        flush=True,
    )
    print(
        f"shape batch={shape.batch} tokens={shape.tokens} lanes={shape.lanes} "
        f"routes={shape.routes} nodes={shape.nodes} width={shape.width} "
        f"feature_chunk={shape.feature_chunk} causal={args.causal} "
        f"warmup={args.warmup} iterations={args.iterations} "
        f"route_backend={args.route_backend} "
        f"memory_backend={args.packed_memory_backend} "
        f"reader_backend={args.packed_memory_reader_backend} "
        f"writer_backend={args.packed_memory_writer_backend}",
        flush=True,
    )

    source = build_unique_inputs(shape, device=device, seed=args.seed)
    eager_inputs = clone_inputs(source)
    compiled_inputs = clone_inputs(source)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    probe = torch.randn(
        source[0].shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    def eager_fft(
        hidden: torch.Tensor,
        nodes: torch.Tensor,
        mass: torch.Tensor,
        valid: torch.Tensor,
        write: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        result = packed_convolution_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            causal=args.causal,
            epsilon=args.epsilon,
            feature_chunk_size=shape.feature_chunk,
            compact_active_nodes=False,
            convolution_backend="fft",
            reader_backend="torch",
            assume_unique_routes=True,
        )
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("eager FFT unexpectedly returned diagnostics")
        return result

    def eager_fft_triton(
        hidden: torch.Tensor,
        nodes: torch.Tensor,
        mass: torch.Tensor,
        valid: torch.Tensor,
        write: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        result = packed_convolution_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            causal=args.causal,
            epsilon=args.epsilon,
            feature_chunk_size=shape.feature_chunk,
            compact_active_nodes=False,
            convolution_backend="fft",
            reader_backend="triton",
            assume_unique_routes=True,
        )
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("eager FFT unexpectedly returned diagnostics")
        return result

    def opaque_torch(
        hidden: torch.Tensor,
        nodes: torch.Tensor,
        mass: torch.Tensor,
        valid: torch.Tensor,
        write: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return opaque_packed_fft_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            causal=args.causal,
            epsilon=args.epsilon,
            feature_chunk_size=shape.feature_chunk,
            compact_active_nodes=False,
            assume_unique_routes=True,
            reader_backend="torch",
            writer_backend="torch",
        )

    def opaque_triton_reader_torch_writer(
        hidden: torch.Tensor,
        nodes: torch.Tensor,
        mass: torch.Tensor,
        valid: torch.Tensor,
        write: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return opaque_packed_fft_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            causal=args.causal,
            epsilon=args.epsilon,
            feature_chunk_size=shape.feature_chunk,
            compact_active_nodes=False,
            assume_unique_routes=True,
            reader_backend="triton",
            writer_backend="torch",
        )

    def opaque_triton(
        hidden: torch.Tensor,
        nodes: torch.Tensor,
        mass: torch.Tensor,
        valid: torch.Tensor,
        write: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return opaque_packed_fft_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            causal=args.causal,
            epsilon=args.epsilon,
            feature_chunk_size=shape.feature_chunk,
            compact_active_nodes=False,
            assume_unique_routes=True,
            reader_backend="triton",
            writer_backend="triton",
        )

    reference_output, reference_gradients = evaluate(
        eager_fft, eager_inputs, probe
    )
    wrapper_started = time.perf_counter()
    compiled_receiver = torch.compile(opaque_triton, fullgraph=True)
    wrapper_seconds = time.perf_counter() - wrapper_started
    first_started = time.perf_counter()
    candidate_output, candidate_gradients = evaluate(
        compiled_receiver, compiled_inputs, probe
    )
    _synchronize(device)
    first_train_seconds = time.perf_counter() - first_started
    print(
        f"compile wrapper_seconds={wrapper_seconds:.6f} "
        f"first_forward_backward_seconds={first_train_seconds:.6f}",
        flush=True,
    )

    labels = ("output", "grad_hidden", "grad_mass", "grad_write", "grad_bias")
    expected_values = (reference_output, *reference_gradients)
    actual_values = (candidate_output, *candidate_gradients)
    tolerance_overrides: dict[str, tuple[float, float]] = {}
    if args.grad_mass_atol is not None or args.grad_mass_rtol is not None:
        tolerance_overrides["grad_mass"] = (
            args.atol if args.grad_mass_atol is None else args.grad_mass_atol,
            args.rtol if args.grad_mass_rtol is None else args.grad_mass_rtol,
        )
    relative_l2_limits = (
        {"grad_mass": args.grad_mass_max_relative_l2}
        if args.grad_mass_max_relative_l2 is not None
        else {}
    )
    maximum_absolute_limits = (
        {"grad_mass": args.grad_mass_max_absolute}
        if args.grad_mass_max_absolute is not None
        else {}
    )
    sparse_outlier_limits = (
        {
            "grad_mass": (
                args.grad_mass_max_outliers,
                args.grad_mass_max_outlier_fraction,
            )
        }
        if args.grad_mass_max_outliers is not None
        else {}
    )
    failures: list[str] = []

    if args.diagnose_components:
        functional_inputs = clone_inputs(source)
        functional_output, functional_gradients = evaluate_vjp(
            eager_fft,
            functional_inputs,
            probe,
        )
        functional_values = (functional_output, *functional_gradients)

        raw_triton_inputs = clone_inputs(source)
        raw_triton_output, raw_triton_gradients = evaluate(
            eager_fft_triton,
            raw_triton_inputs,
            probe,
        )
        raw_triton_values = (raw_triton_output, *raw_triton_gradients)

        opaque_torch_inputs = clone_inputs(source)
        opaque_torch_started = time.perf_counter()
        compiled_opaque_torch = torch.compile(opaque_torch, fullgraph=True)
        opaque_torch_wrapper_seconds = time.perf_counter() - opaque_torch_started
        opaque_torch_output, opaque_torch_gradients = evaluate(
            compiled_opaque_torch,
            opaque_torch_inputs,
            probe,
        )
        _synchronize(device)
        opaque_torch_values = (opaque_torch_output, *opaque_torch_gradients)
        print(
            "diagnostic "
            f"opaque_torch_wrapper_seconds={opaque_torch_wrapper_seconds:.6f}",
            flush=True,
        )

        opaque_mixed_inputs = clone_inputs(source)
        opaque_mixed_started = time.perf_counter()
        compiled_opaque_mixed = torch.compile(
            opaque_triton_reader_torch_writer,
            fullgraph=True,
        )
        opaque_mixed_wrapper_seconds = (
            time.perf_counter() - opaque_mixed_started
        )
        opaque_mixed_output, opaque_mixed_gradients = evaluate(
            compiled_opaque_mixed,
            opaque_mixed_inputs,
            probe,
        )
        _synchronize(device)
        opaque_mixed_values = (
            opaque_mixed_output,
            *opaque_mixed_gradients,
        )
        print(
            "diagnostic "
            f"opaque_mixed_wrapper_seconds={opaque_mixed_wrapper_seconds:.6f}",
            flush=True,
        )

        for comparison, diagnostic_actual, diagnostic_expected in (
            (
                "functional_vjp_vs_eager_autograd",
                functional_values,
                expected_values,
            ),
            ("raw_triton_vs_raw_torch", raw_triton_values, expected_values),
            (
                "compiled_opaque_torch_vs_raw_torch",
                opaque_torch_values,
                expected_values,
            ),
            (
                "compiled_opaque_torch_vs_functional_vjp",
                opaque_torch_values,
                functional_values,
            ),
            (
                "compiled_opaque_triton_reader_torch_writer_vs_torch",
                opaque_mixed_values,
                opaque_torch_values,
            ),
            (
                "target_vs_compiled_opaque_triton_reader_torch_writer",
                actual_values,
                opaque_mixed_values,
            ),
        ):
            failures.extend(
                compare_values(
                    comparison,
                    labels,
                    diagnostic_actual,
                    diagnostic_expected,
                    atol=args.atol,
                    rtol=args.rtol,
                    tolerance_overrides=tolerance_overrides,
                    relative_l2_limits=relative_l2_limits,
                    maximum_absolute_limits=maximum_absolute_limits,
                    sparse_outlier_limits=sparse_outlier_limits,
                )
            )

    failures.extend(
        compare_values(
            "end_to_end_target_vs_raw_torch",
            labels,
            actual_values,
            expected_values,
            atol=args.atol,
            rtol=args.rtol,
            tolerance_overrides=tolerance_overrides,
            relative_l2_limits=relative_l2_limits,
            maximum_absolute_limits=maximum_absolute_limits,
            sparse_outlier_limits=sparse_outlier_limits,
        )
    )
    if failures:
        print(f"validation=FAIL comparisons={len(failures)}", flush=True)
        raise AssertionError("\n\n".join(failures))
    if args.diagnose_components:
        print("diagnostic=PASS", flush=True)
        return

    for label, receiver, inputs in (
        ("eager_fft_torch", eager_fft, eager_inputs),
        (
            "compiled_fft_opaque_triton_reader_triton_writer",
            compiled_receiver,
            compiled_inputs,
        ),
    ):
        print_timing(
            label,
            "forward",
            measure(
                receiver,
                inputs,
                probe,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                backward=False,
            ),
        )
        print_timing(
            label,
            "forward_backward",
            measure(
                receiver,
                inputs,
                probe,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                backward=True,
            ),
        )
    print("validation=PASS", flush=True)


if __name__ == "__main__":
    main()
