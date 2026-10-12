"""Chase FP8 overhead reduction at the exact production node-compute shape
(rows=4128, d_model=1312), where DelayedScaling FP8 measured ~2x SLOWER
than bf16 in eager mode. Tests:
  1. Float8CurrentScaling (no amax history/reduction bookkeeping) vs
     DelayedScaling.
  2. Manual torch.cuda.CUDAGraph capture around the FP8 step (bypasses
     torch.compile's inability to trace TE's custom amax-update op,
     while still eliminating per-call CPU/launch overhead directly).
Throwaway benchmark, WSL-only, not part of the package.
"""
import time

import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

device = "cuda"
torch.manual_seed(0)
d_model = 1312
rows = 4128


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def make_pair():
    linear_bf16 = torch.nn.Linear(d_model, d_model, bias=False).to(device=device, dtype=torch.bfloat16)
    linear_te = te.Linear(d_model, d_model, bias=False).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        linear_te.weight.copy_(linear_bf16.weight)
    x = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16)
    return linear_bf16, linear_te, x


print("=== Recipe comparison, eager, production shape (rows=4128) ===", flush=True)
linear_bf16, linear_te, x = make_pair()


def bf16_step():
    y = linear_bf16(x)
    y.sum().backward()
    linear_bf16.zero_grad(set_to_none=True)


t_bf16 = bench(bf16_step)
print(f"bf16:                 {t_bf16*1e3:.4f} ms/iter", flush=True)

for recipe_name in ["DelayedScaling", "Float8CurrentScaling", "Float8BlockScaling"]:
    recipe_cls = getattr(te_recipe, recipe_name, None)
    if recipe_cls is None:
        print(f"{recipe_name}: not available in this TE version", flush=True)
        continue
    recipe = recipe_cls()
    linear_bf16_r, linear_te_r, x_r = make_pair()

    def fp8_step(linear_te=linear_te_r, x=x_r, recipe=recipe):
        with te.autocast(enabled=True, recipe=recipe):
            y = linear_te(x)
        y.sum().backward()
        linear_te.zero_grad(set_to_none=True)

    try:
        t_fp8 = bench(fp8_step)
        print(f"{recipe_name:22} {t_fp8*1e3:.4f} ms/iter  speedup vs bf16: {t_bf16/t_fp8:.3f}x", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{recipe_name}: FAILED -> {type(exc).__name__}: {exc}", flush=True)

print("\n=== Manual CUDA graph capture, DelayedScaling, production shape ===", flush=True)
linear_bf16_g, linear_te_g, x_g = make_pair()
recipe = te_recipe.DelayedScaling()

static_x = x_g.clone()
static_x.requires_grad_(False)

# warmup on a side stream, required before graph capture
warmup_stream = torch.cuda.Stream()
warmup_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup_stream):
    for _ in range(5):
        with te.autocast(enabled=True, recipe=recipe):
            y = linear_te_g(static_x)
        y.sum().backward()
        linear_te_g.zero_grad(set_to_none=True)
torch.cuda.current_stream().wait_stream(warmup_stream)

try:
    graph = torch.cuda.CUDAGraph()
    linear_te_g.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        with te.autocast(enabled=True, recipe=recipe):
            static_y = linear_te_g(static_x)
        static_loss = static_y.sum()
        static_loss.backward()

    def fp8_graph_step():
        graph.replay()

    t_fp8_graph = bench(fp8_graph_step)
    print(f"fp8 (cuda graph):     {t_fp8_graph*1e3:.4f} ms/iter  speedup vs bf16: {t_bf16/t_fp8_graph:.3f}x", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"fp8 cuda graph capture FAILED: {type(exc).__name__}: {exc}", flush=True)

print("\n=== Manual CUDA graph capture, bf16 baseline (for fair comparison) ===", flush=True)
linear_bf16_g2, _, x_g2 = make_pair()
static_x2 = x_g2.clone()

warmup_stream2 = torch.cuda.Stream()
warmup_stream2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup_stream2):
    for _ in range(5):
        y = linear_bf16_g2(static_x2)
        y.sum().backward()
        linear_bf16_g2.zero_grad(set_to_none=True)
torch.cuda.current_stream().wait_stream(warmup_stream2)

try:
    graph2 = torch.cuda.CUDAGraph()
    linear_bf16_g2.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph2):
        static_y2 = linear_bf16_g2(static_x2)
        static_loss2 = static_y2.sum()
        static_loss2.backward()

    def bf16_graph_step():
        graph2.replay()

    t_bf16_graph = bench(bf16_graph_step)
    print(f"bf16 (cuda graph):    {t_bf16_graph*1e3:.4f} ms/iter", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"bf16 cuda graph capture FAILED: {type(exc).__name__}: {exc}", flush=True)

print("done", flush=True)
