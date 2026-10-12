"""Final throughput measurement using the REAL shipped
gator.mx_precision.te_fp8_cudagraph_node_linear (not a throwaway
standalone benchmark script) at the exact production node-compute shape:
cells=8, in/out_features=1312, batch=2, context=2064 (2048+16 response
tokens). Compares against the real cell_linear bf16 reference. Throwaway
benchmark, WSL-only.
"""
import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.mx_precision import te_fp8_cudagraph_node_linear
from gator.models.dense_kernels import cell_linear

device = "cuda"
torch.manual_seed(0)

# Real production config: n_cells=8, n_nodes=4, d_model=1312, batch=2,
# context=2064. Node compute weight shape [cells, out, in] where
# out = n_nodes * d_model (folded node transform).
cells, d_model, n_nodes = 8, 1312, 4
batch, context = 2, 2064
out_features = n_nodes * d_model

inputs = torch.randn(batch, cells, context, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(cells, out_features, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bf16_step():
    out = cell_linear(inputs, weight, None, backend="grouped_bmm", pad_to_multiple=32)
    out.sum().backward()
    inputs.grad = None
    weight.grad = None


def fp8_step():
    out = te_fp8_cudagraph_node_linear(inputs, weight, None)
    out.sum().backward()
    inputs.grad = None
    weight.grad = None


print("warming up FP8 graph (pays one-time capture cost)...", flush=True)
fp8_step()

t_bf16 = bench(bf16_step)
print(f"bf16 (real cell_linear):              {t_bf16*1e3:.4f} ms/iter", flush=True)

t_fp8 = bench(fp8_step)
print(f"fp8 (real te_fp8_cudagraph_node_linear): {t_fp8*1e3:.4f} ms/iter", flush=True)

print(f"speedup: {t_bf16/t_fp8:.3f}x", flush=True)

torch.cuda.reset_peak_memory_stats()
for _ in range(5):
    bf16_step()
torch.cuda.synchronize()
mem_bf16 = torch.cuda.max_memory_allocated() / 1e9

torch.cuda.reset_peak_memory_stats()
for _ in range(5):
    fp8_step()
torch.cuda.synchronize()
mem_fp8 = torch.cuda.max_memory_allocated() / 1e9

print(f"peak mem bf16: {mem_bf16:.3f} GB", flush=True)
print(f"peak mem fp8:  {mem_fp8:.3f} GB", flush=True)
print("done", flush=True)
