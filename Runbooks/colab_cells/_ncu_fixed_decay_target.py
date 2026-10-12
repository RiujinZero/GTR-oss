"""Minimal target for `ncu` profiling: just fixed_decay_chunk_hgrn at
production receiver-memory shape, a few forward+backward iterations.
No torch.compile (ncu profiles the raw kernel launches; compile adds
autotune/dynamo overhead that isn't the point here) and no CUDA
graphs, so ncu sees plain, individually launched kernels it can
attribute cleanly.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator_linear._hgrn_fixed_decay import fixed_decay_chunk_hgrn

ROWS, LENGTH, CHANNELS = 2626, 2032, 32
DEVICE = "cuda"
DTYPE = torch.bfloat16

torch.manual_seed(0)
values = torch.randn(ROWS, LENGTH, CHANNELS, device=DEVICE, dtype=DTYPE, requires_grad=True)
log_decay = (torch.rand(CHANNELS, device=DEVICE, dtype=DTYPE) * 0.9 + 0.05).log().requires_grad_(True)
initial_state = torch.zeros(ROWS, CHANNELS, device=DEVICE, dtype=DTYPE)

# One untimed warmup (autotune cache warm, allocator warm) BEFORE ncu attaches its
# own launch counting -- ncu is invoked around this whole process, so warmup here
# just means the profiled iterations below are the steady-state kernel, not autotune.
for _ in range(3):
    o, final = fixed_decay_chunk_hgrn(values, log_decay, initial_state=initial_state, output_final_state=True)
    (o.float().sum() + final.float().sum()).backward()
    values.grad = None
    log_decay.grad = None
torch.cuda.synchronize()

print("PROFILE_TARGET_START", flush=True)
for _ in range(3):
    o, final = fixed_decay_chunk_hgrn(values, log_decay, initial_state=initial_state, output_final_state=True)
    (o.float().sum() + final.float().sum()).backward()
    values.grad = None
    log_decay.grad = None
torch.cuda.synchronize()
print("PROFILE_TARGET_END", flush=True)
