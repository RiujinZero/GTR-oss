"""Test the hypothesis: does using a SEPARATE make_graphed_callables
instance per call-site (not one shared cached instance) fix the
multi-call-before-backward bug found in te_fp8_cudagraph_node_linear
and cuda_graph_linear_scan_receiver_context? Throwaway, WSL-only.
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


print("=== Scenario A: ONE shared graphed instance, called twice before backward ===", flush=True)
module_shared = FP8Block()
sample = torch.zeros(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
graphed_shared = torch.cuda.make_graphed_callables(module_shared, (sample,), num_warmup_iters=5)

x1 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
x2 = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
try:
    out1 = graphed_shared(x1)
    out2 = graphed_shared(x2)
    loss = out1.sum() + out2.sum()
    loss.backward()
    print("A: backward succeeded, weight.grad is None:", module_shared.linear.weight.grad is None, flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"A: FAILED -> {type(exc).__name__}: {exc}", flush=True)

print("\n=== Scenario B: TWO SEPARATE graphed instances, one per call-site ===", flush=True)
module_a = FP8Block()
sample_a = torch.zeros(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
graphed_a = torch.cuda.make_graphed_callables(module_a, (sample_a,), num_warmup_iters=5)

module_b = FP8Block()
sample_b = torch.zeros(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
graphed_b = torch.cuda.make_graphed_callables(module_b, (sample_b,), num_warmup_iters=5)

x1b = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
x2b = torch.randn(rows, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
try:
    out1b = graphed_a(x1b)
    out2b = graphed_b(x2b)
    loss_b = out1b.sum() + out2b.sum()
    loss_b.backward()
    print("B: backward succeeded", flush=True)
    print("B: module_a.weight.grad is None:", module_a.linear.weight.grad is None, flush=True)
    print("B: module_b.weight.grad is None:", module_b.linear.weight.grad is None, flush=True)
    if module_a.linear.weight.grad is not None:
        print("B: module_a.weight.grad finite:", torch.isfinite(module_a.linear.weight.grad).all().item(), flush=True)
    if module_b.linear.weight.grad is not None:
        print("B: module_b.weight.grad finite:", torch.isfinite(module_b.linear.weight.grad).all().item(), flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"B: FAILED -> {type(exc).__name__}: {exc}", flush=True)

print("\ndone", flush=True)
