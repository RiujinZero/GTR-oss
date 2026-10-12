"""Isolated throughput check: eager vs CUDA-graphed broadcast-mix cell
coupling at production shape (batch=2, cells=8, tokens=2064, d_model=1312,
neighbors=4 -> radius=2). Forward+backward timed with cuda events after
warmup, matching the methodology used for the fp8_te node-compute/readout
throughput scripts earlier in this investigation.
"""

import copy
import time

import torch
import torch.nn as nn

from gator.cuda_graph_broadcast_mix import cuda_graph_broadcast_mix

torch.manual_seed(0)
batch, cells, tokens, d_model, radius = 2, 8, 2064, 1312, 2

left = nn.Linear(d_model, d_model, device="cuda", dtype=torch.bfloat16)
right = nn.Linear(d_model, d_model, device="cuda", dtype=torch.bfloat16)
norm = nn.LayerNorm(d_model, device="cuda", dtype=torch.bfloat16)
neighbor_gates = nn.Parameter(torch.randn(cells, 2, device="cuda", dtype=torch.bfloat16))
self_gates = nn.Parameter(torch.randn(cells, device="cuda", dtype=torch.bfloat16))


def eager_call(local):
    left_mix = torch.stack(
        [torch.roll(local, d, 1) for d in range(1, radius + 1)]
    ).mean(0)
    right_mix = torch.stack(
        [torch.roll(local, -d, 1) for d in range(1, radius + 1)]
    ).mean(0)
    gates = torch.sigmoid(neighbor_gates)
    mixed = self_gates[None, :, None, None] * local
    mixed = mixed + gates[None, :, 0, None, None] * left(left_mix)
    mixed = mixed + gates[None, :, 1, None, None] * right(right_mix)
    return norm(mixed)


def graphed_call(local, slot):
    return cuda_graph_broadcast_mix(
        local,
        left_projection=left, right_projection=right, coupling_norm=norm,
        neighbor_gates=neighbor_gates, self_gates=self_gates, radius=radius,
        graph_slot=slot,
    )


def bench(fn, iters=30, warmup=10):
    local = torch.randn(
        batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    for _ in range(warmup):
        out = fn(local)
        out.sum().backward()
        local.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(local)
        out.sum().backward()
        local.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    eager_ms = bench(eager_call)
    graphed_ms = bench(lambda x: graphed_call(x, 0))

print(f"eager:   {eager_ms:.3f} ms/iter")
print(f"graphed: {graphed_ms:.3f} ms/iter")
print(f"speedup: {eager_ms / graphed_ms:.3f}x")
