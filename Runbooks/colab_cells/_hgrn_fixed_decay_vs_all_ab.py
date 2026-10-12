"""Three-way speed comparison at production receiver-memory shape
(rows=2626, length=2032, channels=32), eager and under
torch.compile(mode="max-autotune-no-cudagraphs"): plain decayed_scan,
Phase 1.5's compiled_chunk_hgrn (general kernel + materialized
[B,T,D] gate), and Phase 2's fixed_decay_chunk_hgrn (specialized
kernel, [D]-shaped log_decay, no gate/gc/dg materialization).
"""

import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator_linear.core import decayed_scan
from gator_linear._hgrn_compiled import compiled_chunk_hgrn
from gator_linear._hgrn_fixed_decay import fixed_decay_chunk_hgrn

ROWS, LENGTH, CHANNELS = 2626, 2032, 32
DEVICE = "cuda"
DTYPE = torch.bfloat16


def bench(fn, *args, warmup=5, iters=20, **kwargs):
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        loss = out[0].float().sum() if isinstance(out, tuple) else out.float().sum()
        loss.backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args, **kwargs)
        loss = out[0].float().sum() if isinstance(out, tuple) else out.float().sum()
        loss.backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


torch.manual_seed(0)
values = torch.randn(ROWS, LENGTH, CHANNELS, device=DEVICE, dtype=DTYPE, requires_grad=True)
decay_param = (torch.rand(CHANNELS, device=DEVICE, dtype=DTYPE) * 0.9 + 0.05).requires_grad_(True)
initial_state = torch.zeros(ROWS, CHANNELS, device=DEVICE, dtype=DTYPE)


def call_decayed_scan(values, decay):
    return decayed_scan(values, decay, initial_state=initial_state, return_final_state=True, chunk_size=64)


def call_chunk_hgrn(values, decay):
    return compiled_chunk_hgrn(values, decay.log()[None, None, :].expand(ROWS, LENGTH, CHANNELS).contiguous(), initial_state, output_final_state=True)


def call_fixed_decay(values, decay):
    return fixed_decay_chunk_hgrn(values, decay.log(), initial_state, output_final_state=True)


print("=== EAGER ===")
t_decayed = bench(call_decayed_scan, values, decay_param)
t_hgrn = bench(call_chunk_hgrn, values, decay_param)
t_fixed = bench(call_fixed_decay, values, decay_param)
print(f"decayed_scan:          {t_decayed:.2f} ms/iter")
print(f"compiled_chunk_hgrn:   {t_hgrn:.2f} ms/iter")
print(f"fixed_decay_chunk_hgrn:{t_fixed:.2f} ms/iter")

print()
print("=== torch.compile(mode='max-autotune-no-cudagraphs') ===")
compiled_decayed = torch.compile(call_decayed_scan, mode="max-autotune-no-cudagraphs")
compiled_hgrn = torch.compile(call_chunk_hgrn, mode="max-autotune-no-cudagraphs")
compiled_fixed = torch.compile(call_fixed_decay, mode="max-autotune-no-cudagraphs")

t_decayed_c = bench(compiled_decayed, values, decay_param, warmup=3, iters=10)
t_hgrn_c = bench(compiled_hgrn, values, decay_param, warmup=3, iters=10)
t_fixed_c = bench(compiled_fixed, values, decay_param, warmup=3, iters=10)
print(f"decayed_scan:           {t_decayed_c:.2f} ms/iter")
print(f"compiled_chunk_hgrn:    {t_hgrn_c:.2f} ms/iter   speedup vs decayed_scan: {t_decayed_c/t_hgrn_c:.2f}x")
print(f"fixed_decay_chunk_hgrn: {t_fixed_c:.2f} ms/iter   speedup vs decayed_scan: {t_decayed_c/t_fixed_c:.2f}x   speedup vs compiled_chunk_hgrn: {t_hgrn_c/t_fixed_c:.2f}x")
