"""Fair benchmark of the tanh-fusion question at production shape
(B=2, C=8, T=2048, N=4, D=1312): unlike _weighted_nodes4_ab.py (which
took `destination` as a pre-existing input on both sides, so neither
side paid for or saved the tanh step), this one starts from
`node_affine` (pre-tanh) on every version, so the comparison actually
tests whether fusing tanh into the weighted-sum kernel beats
torch.compile's own fusion of the plain tanh-then-sum chain.

1. baseline: torch.tanh(node_affine), then broadcast+sum + h
2. custom Triton, tanh fused into the kernel (destination never materialized)
"""

import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.weighted_nodes4 import weighted_nodes4_tanh

B, C, T, N, D = 2, 8, 2048, 4, 1312
DEVICE = "cuda"
DTYPE = torch.bfloat16


def bench(fn, *args, warmup=5, iters=20, **kwargs):
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        out.float().sum().backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args, **kwargs)
        out.float().sum().backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


torch.manual_seed(0)
q = torch.randn(B, C, T, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
node_affine = torch.randn(B, C, T, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
h = torch.randn(B, C, T, D, device=DEVICE, dtype=DTYPE, requires_grad=True)


def baseline_with_tanh(q, node_affine, h):
    destination = torch.tanh(node_affine)
    return (q.unsqueeze(-1) * destination).sum(-2) + h


def triton_tanh_fused(q, node_affine, h):
    return weighted_nodes4_tanh(q, node_affine, h)


versions = {
    "1. tanh + broadcast+sum (baseline)": baseline_with_tanh,
    "2. triton, tanh fused into kernel": triton_tanh_fused,
}

print("=== EAGER ===")
for name, fn in versions.items():
    t = bench(fn, q, node_affine, h)
    print(f"{name}: {t:.3f} ms/iter")

print()
print("=== torch.compile(mode='max-autotune-no-cudagraphs') ===")
compiled_times = {}
for name, fn in versions.items():
    compiled_fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")
    t = bench(compiled_fn, q, node_affine, h, warmup=3, iters=10)
    compiled_times[name] = t
    print(f"{name}: {t:.3f} ms/iter")

print()
print("=== Summary (compiled, speedup vs baseline) ===")
base = compiled_times["1. tanh + broadcast+sum (baseline)"]
for name, t in compiled_times.items():
    print(f"{name}: {t:.3f} ms  ({base/t:.2f}x)")
