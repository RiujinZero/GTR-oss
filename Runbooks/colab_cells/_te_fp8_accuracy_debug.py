"""Debug why te_fp8_cudagraph_node_linear diverged from the bf16 reference
beyond a loose (rtol=0.2, atol=0.2) tolerance on the FIRST call. Checks
whether it's DelayedScaling's cold-start scale calibration (warms up over
several calls) rather than a real correctness bug. Throwaway, WSL-only.
"""
import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.mx_precision import te_fp8_cudagraph_node_linear
from gator.models.dense_kernels import cell_linear

torch.manual_seed(0)
batch, cells, tokens, d_model, nodes = 2, 8, 16, 64, 4
inputs = torch.randn(batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(cells, nodes * d_model, d_model, device="cuda", dtype=torch.bfloat16)

reference = cell_linear(inputs, weight, None, backend="grouped_bmm", pad_to_multiple=32)

for call_idx in range(6):
    output = te_fp8_cudagraph_node_linear(inputs, weight, None)
    diff = (output.float() - reference.float()).abs()
    rel = diff / (reference.float().abs() + 1e-6)
    print(
        f"call {call_idx}: max_abs_diff={diff.max().item():.4f} "
        f"mean_abs_diff={diff.mean().item():.4f} "
        f"max_rel={rel.max().item():.4f} mean_rel={rel.mean().item():.4f} "
        f"ref_absmax={reference.float().abs().max().item():.4f} "
        f"out_absmax={output.float().abs().max().item():.4f}",
        flush=True,
    )

print("done", flush=True)
