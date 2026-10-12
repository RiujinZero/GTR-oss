"""Real throughput measurement using the shipped
gator.mx_precision.te_fp8_cudagraph_readout_linear at the exact
production readout shape: vocab_size=128256, embedding_dim=256,
rows=batch*context=2*2064=4128. Compares against F.linear bf16.
Throwaway benchmark, WSL-only.
"""
import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch
import torch.nn.functional as F

from gator.mx_precision import te_fp8_cudagraph_readout_linear

device = "cuda"
torch.manual_seed(0)

vocab_size = 128256
embedding_dim = 256
batch, context = 2, 2064
rows = batch * context

x = torch.randn(rows, embedding_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.bfloat16, requires_grad=True)


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bf16_step():
    out = F.linear(x, weight, None)
    out.sum().backward()
    x.grad = None
    weight.grad = None


def fp8_step():
    out = te_fp8_cudagraph_readout_linear(x, weight, None)
    out.sum().backward()
    x.grad = None
    weight.grad = None


print("warming up FP8 graph (pays one-time capture cost)...", flush=True)
fp8_step()

t_bf16 = bench(bf16_step)
print(f"bf16 (F.linear):                          {t_bf16*1e3:.4f} ms/iter", flush=True)

t_fp8 = bench(fp8_step)
print(f"fp8 (real te_fp8_cudagraph_readout_linear): {t_fp8*1e3:.4f} ms/iter", flush=True)

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
