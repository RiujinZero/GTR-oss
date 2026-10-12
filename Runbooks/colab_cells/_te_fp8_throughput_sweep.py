"""FP8 vs bf16 throughput sweep: does a larger matmul shape or torch.compile
fusion change the overhead verdict from _te_fp8_throughput.py (which found
FP8 2.2x SLOWER than bf16 at the exact production node-compute shape,
rows=4128, d_model=1312)? Throwaway benchmark, WSL-only, not part of the
package.
"""
import time

import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

device = "cuda"
torch.manual_seed(0)
d_model = 1312
recipe = te_recipe.DelayedScaling()


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def make_pair(rows, in_f, out_f):
    linear_bf16 = torch.nn.Linear(in_f, out_f, bias=False).to(device=device, dtype=torch.bfloat16)
    linear_te = te.Linear(in_f, out_f, bias=False).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        linear_te.weight.copy_(linear_bf16.weight)
    x = torch.randn(rows, in_f, device=device, dtype=torch.bfloat16)
    return linear_bf16, linear_te, x


print("=== Shape sweep (eager, no compile) ===", flush=True)
for rows in [4128, 16512, 65536, 262144]:
    linear_bf16, linear_te, x = make_pair(rows, d_model, d_model)

    def bf16_step(linear_bf16=linear_bf16, x=x):
        y = linear_bf16(x)
        y.sum().backward()
        linear_bf16.zero_grad(set_to_none=True)

    def fp8_step(linear_te=linear_te, x=x):
        with te.autocast(enabled=True, recipe=recipe):
            y = linear_te(x)
        y.sum().backward()
        linear_te.zero_grad(set_to_none=True)

    t_bf16 = bench(bf16_step)
    t_fp8 = bench(fp8_step)
    print(f"rows={rows:>7} bf16={t_bf16*1e3:8.4f}ms fp8={t_fp8*1e3:8.4f}ms speedup={t_bf16/t_fp8:.3f}x", flush=True)

print("\n=== torch.compile fused, production shape (rows=4128) ===", flush=True)
linear_bf16, linear_te, x = make_pair(4128, d_model, d_model)


def bf16_step_eager(linear_bf16=linear_bf16, x=x):
    y = linear_bf16(x)
    y.sum().backward()
    linear_bf16.zero_grad(set_to_none=True)


def fp8_step_eager(linear_te=linear_te, x=x):
    with te.autocast(enabled=True, recipe=recipe):
        y = linear_te(x)
    y.sum().backward()
    linear_te.zero_grad(set_to_none=True)


compiled_bf16_step = torch.compile(bf16_step_eager, mode="max-autotune")
compiled_fp8_step = torch.compile(fp8_step_eager, mode="max-autotune")

try:
    t_bf16_c = bench(compiled_bf16_step, iters=50, warmup=10)
    print(f"bf16 compiled: {t_bf16_c*1e3:.4f} ms/iter", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"bf16 compiled FAILED: {type(exc).__name__}: {exc}", flush=True)
    t_bf16_c = None

try:
    t_fp8_c = bench(compiled_fp8_step, iters=50, warmup=10)
    print(f"fp8 compiled:  {t_fp8_c*1e3:.4f} ms/iter", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"fp8 compiled FAILED: {type(exc).__name__}: {exc}", flush=True)
    t_fp8_c = None

if t_bf16_c is not None and t_fp8_c is not None:
    print(f"compiled speedup: {t_bf16_c/t_fp8_c:.3f}x", flush=True)

print("done", flush=True)
