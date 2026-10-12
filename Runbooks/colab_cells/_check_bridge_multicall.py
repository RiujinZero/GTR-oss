"""Precisely isolate: does wrapping the graphed call through a bridge
torch.autograd.Function (value-copy-in on forward, gradient-copy-out on
backward -- same pattern as _BridgeFP8CellWeights/_BridgeReceiverMemoryBias)
break multi-call-before-backward, even for the simplest possible case
(single te.Linear, no per-cell loop)? Throwaway, WSL-only.
"""
import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

device = "cuda"
torch.manual_seed(0)
d_model = 64
rows = 128
recipe = te_recipe.DelayedScaling()


class FP8Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = te.Linear(d_model, d_model, bias=False).to(device=device, dtype=torch.bfloat16)

    def forward(self, x):
        with te.autocast(enabled=True, recipe=recipe):
            return self.linear(x)


class BridgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, raw_module):
        with torch.no_grad():
            raw_module.linear.weight.copy_(weight)
            # NOTE: .grad reset removed for this test -- isolating whether
            # THAT specific line is the trigger.
        ctx.raw_module = raw_module
        return inputs

    @staticmethod
    def backward(ctx, grad_inputs):
        grad_weight = ctx.raw_module.linear.weight.grad.clone()
        return grad_inputs, grad_weight, None


module = FP8Block()
sample = torch.zeros(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
graphed = torch.cuda.make_graphed_callables(module, (sample,), num_warmup_iters=5)

x1 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
x2 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
live_weight = torch.randn(d_model, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)

print("=== Bridge + shared graphed instance, two calls before backward ===", flush=True)
try:
    bridged_x1 = BridgeWeight.apply(x1, live_weight, module)
    out1 = graphed(bridged_x1)
    bridged_x2 = BridgeWeight.apply(x2, live_weight, module)
    out2 = graphed(bridged_x2)
    loss = out1.sum() + out2.sum()
    loss.backward()
    print("backward succeeded", flush=True)
    print("live_weight.grad is None:", live_weight.grad is None, flush=True)
    if live_weight.grad is not None:
        print("live_weight.grad finite:", torch.isfinite(live_weight.grad).all().item(), flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"FAILED -> {type(exc).__name__}: {exc}", flush=True)

print("done", flush=True)
