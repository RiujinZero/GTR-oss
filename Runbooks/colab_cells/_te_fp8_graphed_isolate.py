"""Isolate why make_graphed_callables showed ~1.0x (no speedup) in the
realistic full-step test, versus 1.4-1.5x in the hand-rolled raw
graph.replay() benchmark. Tests three variants at the same production
shape (rows=4128, d_model=1312):
  A. Reused static input (no fresh tensor/copy each call) -- matches the
     earlier hand-rolled benchmark's condition exactly.
  B. Fresh random input each call (realistic training condition) --
     matches the full-step test's condition.
  C. Same as B but measuring make_graphed_callables' internal copy-in
     overhead in isolation (graphed module call with a fixed dummy
     downstream use, no separate upstream/downstream matmuls muddying
     the comparison).
Throwaway, WSL-only.
"""
import time

import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

device = "cuda"
torch.manual_seed(0)
d_model = 1312
rows = 4128
recipe = te_recipe.DelayedScaling()


def bench(fn, iters=150, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


class FP8Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = te.Linear(d_model, d_model, bias=False).to(device=device, dtype=torch.bfloat16)

    def forward(self, x):
        with te.autocast(enabled=True, recipe=recipe):
            return self.linear(x)


print("=== A: graphed module, REUSED static input (matches earlier clean benchmark) ===", flush=True)
module_a = FP8Block()
sample_input_a = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
graphed_a = torch.cuda.make_graphed_callables(module_a, (sample_input_a,), num_warmup_iters=5)

fixed_input = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)


def step_a():
    y = graphed_a(fixed_input)
    y.sum().backward()
    module_a.linear.weight.grad = None
    fixed_input.grad = None


t_a = bench(step_a)
print(f"A (reused input): {t_a*1e3:.4f} ms/iter", flush=True)

print("\n=== A-eager: same reused-input condition, no graph ===", flush=True)
module_a_eager = FP8Block()


def step_a_eager():
    y = module_a_eager(fixed_input)
    y.sum().backward()
    module_a_eager.linear.weight.grad = None
    fixed_input.grad = None


t_a_eager = bench(step_a_eager)
print(f"A-eager (reused input): {t_a_eager*1e3:.4f} ms/iter  speedup: {t_a_eager/t_a:.3f}x", flush=True)

print("\n=== B: graphed module, FRESH random input each call (realistic) ===", flush=True)
module_b = FP8Block()
sample_input_b = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
graphed_b = torch.cuda.make_graphed_callables(module_b, (sample_input_b,), num_warmup_iters=5)


def step_b():
    x = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = graphed_b(x)
    y.sum().backward()
    module_b.linear.weight.grad = None


t_b = bench(step_b)
print(f"B (fresh input): {t_b*1e3:.4f} ms/iter", flush=True)

print("\n=== B-eager: same fresh-input condition, no graph ===", flush=True)
module_b_eager = FP8Block()


def step_b_eager():
    x = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = module_b_eager(x)
    y.sum().backward()
    module_b_eager.linear.weight.grad = None


t_b_eager = bench(step_b_eager)
print(f"B-eager (fresh input): {t_b_eager*1e3:.4f} ms/iter  speedup: {t_b_eager/t_b:.3f}x", flush=True)

print("\n=== C: cost of just the torch.randn call itself, for reference ===", flush=True)


def just_randn():
    x = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    return x


t_randn = bench(just_randn)
print(f"torch.randn({rows},{d_model}) alone: {t_randn*1e3:.4f} ms/iter", flush=True)

print("done", flush=True)
