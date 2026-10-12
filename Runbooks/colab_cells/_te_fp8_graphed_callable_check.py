"""Verify torch.cuda.make_graphed_callables works with a TE FP8 Linear in
this environment, with a REAL (non-synthetic) upstream/downstream gradient
flow -- not just .sum() as the loss. Throwaway check, WSL-only.
"""
import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

device = "cuda"
torch.manual_seed(0)
d_model = 1312
rows = 4128
recipe = te_recipe.DelayedScaling()


class FP8Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = te.Linear(d_model, d_model, bias=False).to(device=device, dtype=torch.bfloat16)

    def forward(self, x):
        with te.autocast(enabled=True, recipe=recipe):
            return self.linear(x)


module = FP8Block()
sample_input = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)

graphed_module = torch.cuda.make_graphed_callables(module, (sample_input,), num_warmup_iters=5)
print("make_graphed_callables succeeded", flush=True)

# Simulate real usage: input from an upstream op, output feeds a downstream op.
upstream_weight = torch.randn(d_model, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
x0 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
x1 = x0 @ upstream_weight  # upstream op
y = graphed_module(x1)
downstream_weight = torch.randn(d_model, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
loss = (y @ downstream_weight).sum()
loss.backward()

print("x0.grad finite:", torch.isfinite(x0.grad).all().item(), flush=True)
print("upstream_weight.grad finite:", torch.isfinite(upstream_weight.grad).all().item(), flush=True)
print("module.linear.weight.grad finite:", torch.isfinite(module.linear.weight.grad).all().item(), flush=True)
print("downstream_weight.grad finite:", torch.isfinite(downstream_weight.grad).all().item(), flush=True)
print("loss finite:", torch.isfinite(loss).item(), flush=True)

import time

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def full_step():
    x0 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    x1 = x0 @ upstream_weight
    y = graphed_module(x1)
    loss = (y @ downstream_weight).sum()
    loss.backward()
    upstream_weight.grad = None
    downstream_weight.grad = None
    module.linear.weight.grad = None


t_graphed = bench(full_step)
print(f"graphed FP8 full step (with real upstream/downstream ops): {t_graphed*1e3:.4f} ms/iter", flush=True)


class FP8BlockEager(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = te.Linear(d_model, d_model, bias=False).to(device=device, dtype=torch.bfloat16)

    def forward(self, x):
        with te.autocast(enabled=True, recipe=recipe):
            return self.linear(x)


eager_module = FP8BlockEager()


def full_step_eager():
    x0 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    x1 = x0 @ upstream_weight
    y = eager_module(x1)
    loss = (y @ downstream_weight).sum()
    loss.backward()
    upstream_weight.grad = None
    downstream_weight.grad = None
    eager_module.linear.weight.grad = None


t_eager = bench(full_step_eager)
print(f"eager FP8 full step:                                        {t_eager*1e3:.4f} ms/iter", flush=True)
print(f"speedup: {t_eager/t_graphed:.3f}x", flush=True)

print("done", flush=True)
