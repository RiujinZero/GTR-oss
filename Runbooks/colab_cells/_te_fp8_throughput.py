"""Standalone throughput probe: plain scalar FP8 (TE DelayedScaling) vs bf16
at production-representative node-compute shapes. Not part of the package,
throwaway benchmark, WSL-only. Uses the confirmed-working recipe from the
earlier isolated smoke test (DelayedScaling only -- MXFP8/NVFP4 block-scaled
recipes are separately gated on this GPU, see the ablation queue doc).
"""
import time

import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

device = "cuda"
torch.manual_seed(0)

# Production config: n_cells=8, n_nodes=4, d_model=1312 (gtr_arw_100m_smollm3
# naked-broadcast run). Node compute is a per-cell grouped affine transform;
# approximate its matmul shape as (n_cells * n_nodes, d_model) x (d_model, d_model)
# per "group", batched -- benchmark the underlying GEMM shape directly since
# that's what a real TE integration would dispatch to.
n_cells, n_nodes, d_model = 8, 4, 1312
batch = 2
context = 2064  # 2048 + 16 response tokens, matches confirmed production run

rows = batch * context  # flattened token dimension feeding node compute
in_features = d_model
out_features = d_model

x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
linear_bf16 = torch.nn.Linear(in_features, out_features, bias=False).to(device=device, dtype=torch.bfloat16)
linear_te = te.Linear(in_features, out_features, bias=False).to(device=device, dtype=torch.bfloat16)
with torch.no_grad():
    linear_te.weight.copy_(linear_bf16.weight)

recipe = te_recipe.DelayedScaling()


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def bf16_step():
    y = linear_bf16(x)
    y.sum().backward()
    linear_bf16.zero_grad(set_to_none=True)


def fp8_step():
    with te.autocast(enabled=True, recipe=recipe):
        y = linear_te(x)
    y.sum().backward()
    linear_te.zero_grad(set_to_none=True)


x.requires_grad_(False)
print(f"shape: rows={rows} in_features={in_features} out_features={out_features}", flush=True)

bf16_time = bench(bf16_step)
print(f"bf16:  {bf16_time*1e3:.4f} ms/iter", flush=True)

fp8_time = bench(fp8_step)
print(f"fp8:   {fp8_time*1e3:.4f} ms/iter", flush=True)

print(f"speedup: {bf16_time/fp8_time:.3f}x", flush=True)

peak_mem_bf16 = None
torch.cuda.reset_peak_memory_stats()
for _ in range(10):
    bf16_step()
torch.cuda.synchronize()
peak_mem_bf16 = torch.cuda.max_memory_allocated() / 1e9

torch.cuda.reset_peak_memory_stats()
for _ in range(10):
    fp8_step()
torch.cuda.synchronize()
peak_mem_fp8 = torch.cuda.max_memory_allocated() / 1e9

print(f"peak mem bf16: {peak_mem_bf16:.3f} GB", flush=True)
print(f"peak mem fp8:  {peak_mem_fp8:.3f} GB", flush=True)
print("done", flush=True)
