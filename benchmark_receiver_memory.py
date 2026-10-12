"""Benchmark exact Arrow receiver-memory backends on representative shapes."""

from __future__ import annotations

import argparse
import time

import torch

from gator.models.memory import relative_receiver_context


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    backend: str,
    *,
    states: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    bias: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = relative_receiver_context(
            states, nodes, mass, bias, backend=backend
        )
    synchronize(states.device)
    started = time.perf_counter()
    for _ in range(iterations):
        output = relative_receiver_context(
            states, nodes, mass, bias, backend=backend
        )
    synchronize(states.device)
    assert output is not None
    return (time.perf_counter() - started) / iterations, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cells", type=int, default=62)
    parser.add_argument("--tokens", type=int, default=104)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    shape = (args.batch_size, args.cells, args.tokens)
    states = torch.randn(*shape, args.hidden, device=device, dtype=dtype)
    nodes = torch.randn(
        *shape, args.nodes, device=device, dtype=dtype
    ).softmax(-1)
    mass = torch.rand(*shape, device=device, dtype=dtype)
    bias = torch.randn(
        args.cells, args.nodes, 4, device=device, dtype=dtype
    ) * 0.1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    attention_time, attention = benchmark(
        "attention",
        states=states,
        nodes=nodes,
        mass=mass,
        bias=bias,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    attention_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    convolution_time, convolution = benchmark(
        "convolution",
        states=states,
        nodes=nodes,
        mass=mass,
        bias=bias,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    convolution_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    difference = float((attention.float() - convolution.float()).abs().max())
    score_values = (
        args.batch_size
        * args.cells
        * args.tokens
        * args.tokens
        * args.nodes
    )
    print(
        f"device={device} dtype={dtype} shape={shape} "
        f"hidden={args.hidden} nodes={args.nodes}"
    )
    print(
        f"quadratic_score_values={score_values} "
        f"quadratic_score_bytes={score_values * states.element_size()}"
    )
    print(
        f"attention_ms={attention_time * 1e3:.3f} "
        f"attention_peak_bytes={attention_peak}"
    )
    print(
        f"convolution_ms={convolution_time * 1e3:.3f} "
        f"convolution_peak_bytes={convolution_peak} "
        f"speedup={attention_time / convolution_time:.3f} "
        f"max_abs_difference={difference:.8g}"
    )


if __name__ == "__main__":
    main()
