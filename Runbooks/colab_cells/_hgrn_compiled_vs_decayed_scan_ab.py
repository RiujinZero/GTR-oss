"""Apples-to-apples speed comparison at the real production receiver-
memory shape (rows=2626, length=2032, channels=32), both paths under
torch.compile(mode="max-autotune-no-cudagraphs") -- the same mode the
300-step production run uses. The original 5.48x number
(Runbooks/colab_cells/_hgrn_receiver_ab.py) compared EAGER decayed_scan
vs EAGER chunk_hgrn. That gap may not hold once compiled: decayed_scan
is plain PyTorch ops Inductor can fuse/autotune, while
compiled_chunk_hgrn is an opaque torch.library.triton_op Inductor
cannot fuse anything around. If that's the real explanation for flat
production throughput despite zero graph breaks/recompiles, this
isolated test should show it directly.
"""

import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator_linear.core import chunk_hgrn_decayed_scan, decayed_scan

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
    return chunk_hgrn_decayed_scan(values, decay, initial_state=initial_state, return_final_state=True)


print("=== EAGER ===")
t_decayed_eager = bench(call_decayed_scan, values, decay_param)
t_hgrn_eager = bench(call_chunk_hgrn, values, decay_param)
print(f"decayed_scan (eager):        {t_decayed_eager:.2f} ms/iter")
print(f"compiled_chunk_hgrn (eager): {t_hgrn_eager:.2f} ms/iter")
print(f"speedup: {t_decayed_eager / t_hgrn_eager:.2f}x")

print()
print("=== torch.compile(mode='max-autotune-no-cudagraphs') ===")
compiled_decayed = torch.compile(call_decayed_scan, mode="max-autotune-no-cudagraphs")
compiled_hgrn = torch.compile(call_chunk_hgrn, mode="max-autotune-no-cudagraphs")

t_decayed_compiled = bench(compiled_decayed, values, decay_param, warmup=3, iters=10)
t_hgrn_compiled = bench(compiled_hgrn, values, decay_param, warmup=3, iters=10)
print(f"decayed_scan (compiled):        {t_decayed_compiled:.2f} ms/iter")
print(f"compiled_chunk_hgrn (compiled): {t_hgrn_compiled:.2f} ms/iter")
print(f"speedup: {t_decayed_compiled / t_hgrn_compiled:.2f}x")
