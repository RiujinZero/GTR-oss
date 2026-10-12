"""Four-way benchmark at the real production relay-transform shape
(B=2, C=8, T=2048, N=4, D=1312), eager and under
torch.compile(mode="max-autotune-no-cudagraphs") (the mode production
actually uses):

1. broadcast multiply + sum   (q.unsqueeze(-1) * destination).sum(-2) + h  -- current code
2. torch.einsum                einsum("bctn,bctnd->bctd", q, destination) + h
3. custom N=4 Triton, no residual fusion (separate + h afterward)
4. custom N=4 Triton, residual fused into the same kernel
"""

import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.weighted_nodes4 import weighted_nodes4

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
destination = torch.randn(B, C, T, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
h = torch.randn(B, C, T, D, device=DEVICE, dtype=DTYPE, requires_grad=True)


def broadcast_sum(q, destination, h):
    return (q.unsqueeze(-1) * destination).sum(-2) + h


def einsum_version(q, destination, h):
    return torch.einsum("bctn,bctnd->bctd", q, destination) + h


def triton_no_residual(q, destination, h):
    return weighted_nodes4(q, destination) + h


def triton_with_residual(q, destination, h):
    return weighted_nodes4(q, destination, h)


versions = {
    "1. broadcast+sum (baseline)": broadcast_sum,
    "2. einsum": einsum_version,
    "3. triton, no residual fusion": triton_no_residual,
    "4. triton, residual fused": triton_with_residual,
}

print("=== EAGER ===")
eager_times = {}
for name, fn in versions.items():
    t = bench(fn, q, destination, h)
    eager_times[name] = t
    print(f"{name}: {t:.3f} ms/iter")

print()
print("=== torch.compile(mode='max-autotune-no-cudagraphs') ===")
compiled_times = {}
for name, fn in versions.items():
    compiled_fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")
    t = bench(compiled_fn, q, destination, h, warmup=3, iters=10)
    compiled_times[name] = t
    baseline = compiled_times["1. broadcast+sum (baseline)"] if "1." in compiled_times else None
    suffix = f"   speedup vs baseline: {baseline/t:.2f}x" if baseline else ""
    print(f"{name}: {t:.3f} ms/iter{suffix}")

print()
print("=== Summary (compiled, speedup vs baseline) ===")
base = compiled_times["1. broadcast+sum (baseline)"]
for name, t in compiled_times.items():
    print(f"{name}: {t:.3f} ms  ({base/t:.2f}x)")
