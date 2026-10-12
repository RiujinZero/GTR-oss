"""Benchmark the dense Arrow/Gator relay kernels.

The benchmark isolates the three dense compute choices used by the language
model:

* per-cell graph router: semantic einsum versus grouped batched GEMM;
* per-cell/per-node transform: expanded node inputs versus an exact folded
  wide projection;
* shared cell-handoff projection: ordinary F.linear dimensions versus
  parameter-free padding to a multiple of eight.

CUDA measurements use events and report incremental peak allocated memory.
CPU measurements use ``time.perf_counter`` and omit allocator memory.
Compilation, when requested, is warmed up before measurements begin.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch

from gator.models.dense_kernels import (
    cell_linear,
    cell_node_linear,
    fold_cell_node_parameters,
    padded_linear,
)


TensorKernel = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class BenchmarkResult:
    component: str
    implementation: str
    milliseconds: float
    peak_mebibytes: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark representative dense Gator relay kernels."
    )
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("--cells", type=int, default=62)
    parser.add_argument("--tokens", type=int, default=104)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument(
        "--edges",
        type=int,
        default=5,
        help="number of shared inter-cell handoff actions/edges",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or an indexed CUDA device such as 'cuda:0'",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile each isolated kernel before warmup/timing",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in (
        "batch",
        "cells",
        "tokens",
        "nodes",
        "d_model",
        "edges",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if device.index is None:
            device = torch.device("cuda:0")
    return device


def _maybe_compile(kernel: TensorKernel, enabled: bool) -> TensorKernel:
    if not enabled:
        return kernel
    if not hasattr(torch, "compile"):
        raise RuntimeError("--compile requires a PyTorch build with torch.compile")
    return torch.compile(kernel, fullgraph=True)


def _measure(
    component: str,
    implementation: str,
    kernel: TensorKernel,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    output: torch.Tensor | None = None
    with torch.inference_mode():
        for _ in range(warmup):
            output = kernel()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            output = None
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                output = kernel()
            end.record()
            torch.cuda.synchronize(device)
            milliseconds = start.elapsed_time(end) / iterations
            peak = torch.cuda.max_memory_allocated(device)
            peak_mebibytes = max(peak - baseline, 0) / (1024**2)
        else:
            start_time = time.perf_counter()
            for _ in range(iterations):
                output = kernel()
            milliseconds = (
                (time.perf_counter() - start_time) * 1000.0 / iterations
            )
            peak_mebibytes = None

    # Keep the result observable until after timing so compiled kernels cannot
    # discard the output as dead work.
    if output is None:
        raise RuntimeError("benchmark kernel did not produce an output")
    _ = output.shape
    return BenchmarkResult(
        component=component,
        implementation=implementation,
        milliseconds=milliseconds,
        peak_mebibytes=peak_mebibytes,
    )


def _assert_close(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    maximum_error = (actual - expected).abs().max().item()
    print(f"  {label:<42} PASS  max_abs_error={maximum_error:.3e}")
    return maximum_error


def _print_results(results: list[BenchmarkResult]) -> None:
    print("\nTimings (forward kernel only; lower is better)")
    print(
        f"  {'component':<12} {'implementation':<38} "
        f"{'ms/iter':>11} {'peak_delta_MiB':>16} {'relative':>10}"
    )
    baselines: dict[str, float] = {}
    for result in results:
        baseline = baselines.setdefault(result.component, result.milliseconds)
        speedup = baseline / result.milliseconds
        peak = (
            "n/a"
            if result.peak_mebibytes is None
            else f"{result.peak_mebibytes:.2f}"
        )
        print(
            f"  {result.component:<12} {result.implementation:<38} "
            f"{result.milliseconds:>11.4f} {peak:>16} {speedup:>9.2f}x"
        )


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    batch = args.batch
    cells = args.cells
    tokens = args.tokens
    nodes = args.nodes
    width = args.d_model
    handoff_edges = args.edges
    router_features = 2 * width + nodes
    router_edges = nodes * nodes

    print(
        f"device={device} compile={args.compile} dtype=float32 "
        f"batch={batch} cells={cells} tokens={tokens} nodes={nodes} "
        f"d_model={width} router_edges={router_edges} "
        f"handoff_edges={handoff_edges} warmup={args.warmup} "
        f"iterations={args.iterations}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    router_input = torch.randn(
        batch,
        cells,
        tokens,
        router_features,
        device=device,
        generator=generator,
    )
    router_weight = torch.randn(
        cells,
        router_edges,
        router_features,
        device=device,
        generator=generator,
    )
    router_bias = torch.randn(
        cells,
        router_edges,
        device=device,
        generator=generator,
    )

    base = torch.randn(
        batch,
        cells,
        tokens,
        width,
        device=device,
        generator=generator,
    )
    node_embedding = torch.randn(
        cells,
        nodes,
        width,
        device=device,
        generator=generator,
    )
    node_weight = torch.randn(
        cells,
        nodes,
        width,
        width,
        device=device,
        generator=generator,
    )
    node_bias = torch.randn(
        cells,
        nodes,
        width,
        device=device,
        generator=generator,
    )
    folded_weight, folded_offset = fold_cell_node_parameters(
        node_embedding,
        node_weight,
        node_bias,
    )

    handoff_input = torch.randn(
        batch,
        cells,
        tokens,
        router_features,
        device=device,
        generator=generator,
    )
    handoff_weight = torch.randn(
        handoff_edges,
        router_features,
        device=device,
        generator=generator,
    )
    handoff_bias = torch.randn(
        handoff_edges,
        device=device,
        generator=generator,
    )

    def router_einsum() -> torch.Tensor:
        return cell_linear(
            router_input,
            router_weight,
            router_bias,
            backend="einsum",
        )

    def router_grouped_bmm() -> torch.Tensor:
        return cell_linear(
            router_input,
            router_weight,
            router_bias,
            backend="grouped_bmm",
            pad_to_multiple=1,
        )

    def node_expanded() -> torch.Tensor:
        expanded = base.unsqueeze(3) + node_embedding[None, :, None]
        return cell_node_linear(
            expanded,
            node_weight,
            node_bias,
            backend="einsum",
        )

    def node_folded() -> torch.Tensor:
        projected = cell_linear(
            base,
            folded_weight,
            backend="grouped_bmm",
            pad_to_multiple=1,
        )
        return projected.view(
            batch,
            cells,
            tokens,
            nodes,
            width,
        ) + folded_offset[None, :, None]

    def handoff_padding_1() -> torch.Tensor:
        return padded_linear(
            handoff_input,
            handoff_weight,
            handoff_bias,
            pad_to_multiple=1,
        )

    def handoff_padding_8() -> torch.Tensor:
        return padded_linear(
            handoff_input,
            handoff_weight,
            handoff_bias,
            pad_to_multiple=8,
        )

    print("\nCorrectness")
    with torch.inference_mode():
        _assert_close(
            "router grouped_bmm versus einsum",
            router_grouped_bmm(),
            router_einsum(),
        )
        _assert_close(
            "folded node path versus expanded path",
            node_folded(),
            node_expanded(),
        )
        _assert_close(
            "handoff padding=8 versus padding=1",
            handoff_padding_8(),
            handoff_padding_1(),
        )

    theoretical_node_input_mib = (
        batch * cells * tokens * nodes * width * base.element_size() / (1024**2)
    )
    print(
        "\nFolded node setup is hoisted once per model forward. Its relay hot path "
        "avoids materializing the expanded [B,C,T,N,D] node input "
        f"({theoretical_node_input_mib:.2f} MiB at this shape)."
    )
    if args.compile:
        print("Compilation occurs during warmup and is excluded from timings.")

    kernels: list[tuple[str, str, TensorKernel]] = [
        ("router", "einsum", router_einsum),
        ("router", "grouped_bmm padding=1", router_grouped_bmm),
        ("node", "expanded cell-node einsum", node_expanded),
        ("node", "folded grouped_bmm (pre-folded)", node_folded),
        ("handoff", "F.linear padding=1", handoff_padding_1),
        ("handoff", "F.linear padding=8", handoff_padding_8),
    ]
    results = [
        _measure(
            component,
            implementation,
            _maybe_compile(kernel, args.compile),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for component, implementation, kernel in kernels
    ]
    _print_results(results)


if __name__ == "__main__":
    main()
