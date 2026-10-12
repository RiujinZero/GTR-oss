"""Check whether te_fp8_cudagraph_node_linear (already shipped) has the
same multi-call-before-backward bug just found in the receiver-memory
integration test. Calls it TWICE (simulating relays=2) before a single
combined .backward(), then compares gradients against calling the
eager bf16 cell_linear reference the same way. Throwaway, WSL-only.
"""
import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.mx_precision import te_fp8_cudagraph_node_linear
from gator.models.dense_kernels import cell_linear

torch.manual_seed(0)
batch, cells, tokens, d_model, nodes = 2, 8, 16, 64, 4

inputs1 = torch.randn(batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
inputs2 = torch.randn(batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(cells, nodes * d_model, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)

print("=== FP8 CUDA-graph path, two calls sharing one weight, one combined backward ===", flush=True)
try:
    out1 = te_fp8_cudagraph_node_linear(inputs1, weight, None)
    out2 = te_fp8_cudagraph_node_linear(inputs2, weight, None)
    loss = out1.sum() + out2.sum()
    loss.backward()
    print("backward succeeded", flush=True)
    print("weight.grad is None:", weight.grad is None, flush=True)
    if weight.grad is not None:
        print("weight.grad finite:", torch.isfinite(weight.grad).all().item(), flush=True)
        fp8_weight_grad = weight.grad.clone()
    else:
        fp8_weight_grad = None
except Exception as exc:  # noqa: BLE001
    print(f"FAILED -> {type(exc).__name__}: {exc}", flush=True)
    fp8_weight_grad = None

print("\n=== Eager bf16 reference, same two-call-then-combined-backward pattern ===", flush=True)
weight.grad = None
ref_out1 = cell_linear(inputs1, weight, None, backend="grouped_bmm", pad_to_multiple=32)
ref_out2 = cell_linear(inputs2, weight, None, backend="grouped_bmm", pad_to_multiple=32)
ref_loss = ref_out1.sum() + ref_out2.sum()
ref_loss.backward()
print("ref weight.grad finite:", torch.isfinite(weight.grad).all().item(), flush=True)
ref_weight_grad = weight.grad.clone()

if fp8_weight_grad is not None:
    diff = (fp8_weight_grad.float() - ref_weight_grad.float()).abs()
    print(f"\nweight.grad diff: max={diff.max().item():.4f} mean={diff.mean().item():.4f} "
          f"ref_absmax={ref_weight_grad.float().abs().max().item():.4f}", flush=True)

print("done", flush=True)
