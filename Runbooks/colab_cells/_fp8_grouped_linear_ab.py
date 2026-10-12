"""Correctness + speed comparison: the current per-cell FP8 GEMM loop
(gator.mx_precision._FP8CellBlock, [self.linears[c](x[c]) for c in
range(cells)] -- 8 separate te.Linear kernel launches) vs
transformer_engine.pytorch.GroupedLinear (num_gemms=cells, one call
covering all cells' GEMMs with per-cell weights via m_splits).

Real production shape: cells=8, in_features=d_model=1312,
out_features=n_nodes*d_model=5248 (gator/models/language.py:496,
self.node_weight = nn.Parameter(torch.empty(c, n, d, d)), folded to
[cells, n*d, d] before reaching node_compute_linear). rows=batch*tokens
-- using 4128 to match te_fp8_cudagraph_node_linear's own docstring
benchmark shape.

No CUDA-graph wrapping on either side here (isolating the GEMM-count
question itself, not launch-overhead reduction -- that's already
measured separately, see te_fp8_cudagraph_node_linear's docstring).
"""

import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch
import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te

CELLS = 8
ROWS = 4128
IN_FEATURES = 1312
OUT_FEATURES = 5248
DEVICE = "cuda"

recipe = te_recipe.DelayedScaling()


class PerCellLoop(torch.nn.Module):
    def __init__(self, cells, in_features, out_features):
        super().__init__()
        self.cells = cells
        self.linears = torch.nn.ModuleList(
            [te.Linear(in_features, out_features, bias=False) for _ in range(cells)]
        )

    def forward(self, x):
        # x: [cells, rows, in_features]
        with te.autocast(enabled=True, recipe=recipe):
            outputs = [self.linears[c](x[c]) for c in range(self.cells)]
        return torch.stack(outputs, dim=0)


class Grouped(torch.nn.Module):
    def __init__(self, cells, in_features, out_features):
        super().__init__()
        self.cells = cells
        self.grouped = te.GroupedLinear(cells, in_features, out_features, bias=False)

    def forward(self, x):
        # x: [cells, rows, in_features] -> flatten to [cells*rows, in_features]
        rows = x.shape[1]
        flat = x.reshape(self.cells * rows, x.shape[-1])
        m_splits = torch.full((self.cells,), rows, dtype=torch.int64)
        with te.autocast(enabled=True, recipe=recipe):
            out = self.grouped(flat, m_splits)
        return out.reshape(self.cells, rows, -1)


def bench(module, x, warmup=8, iters=20):
    for _ in range(warmup):
        out = module(x)
        out.float().sum().backward()
        x.grad = None
        for p in module.parameters():
            p.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = module(x)
        out.float().sum().backward()
        x.grad = None
        for p in module.parameters():
            p.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


torch.manual_seed(0)
per_cell = PerCellLoop(CELLS, IN_FEATURES, OUT_FEATURES).to(DEVICE, dtype=torch.bfloat16)
grouped = Grouped(CELLS, IN_FEATURES, OUT_FEATURES).to(DEVICE, dtype=torch.bfloat16)

# Match weights so forward outputs can be compared directly.
with torch.no_grad():
    # GroupedLinear stores weights as a single [cells * out_features, in_features]
    # (or similar concatenated) parameter -- introspect its actual parameter shape
    # rather than assume, to get the copy right.
    grouped_params = dict(grouped.grouped.named_parameters())
    print("GroupedLinear parameter names/shapes:", {k: tuple(v.shape) for k, v in grouped_params.items()})

x = torch.randn(CELLS, ROWS, IN_FEATURES, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

print("\n=== forward-only sanity (no weight matching yet) ===")
out_per_cell = per_cell(x)
out_grouped = grouped(x)
print("per_cell output shape:", out_per_cell.shape)
print("grouped  output shape:", out_grouped.shape)
print("both finite:", torch.isfinite(out_per_cell).all().item(), torch.isfinite(out_grouped).all().item())

# --- Correctness: match weights exactly, compare forward + gradients ---
print("\n=== correctness: weight-matched forward + gradients ===")
with torch.no_grad():
    for c in range(CELLS):
        getattr(grouped.grouped, f"weight{c}").copy_(per_cell.linears[c].weight)

x_ref = x.detach().clone().requires_grad_(True)
x_my = x.detach().clone().requires_grad_(True)

# DelayedScaling's amax history starts empty -- run a few untimed
# forward+backward passes on BOTH first so their FP8 scales are warmed
# to the same steady state before comparing (per te_fp8_cudagraph_node_
# linear's own documented cold-start warning).
for _ in range(20):
    per_cell(x_ref).sum().backward()
    x_ref.grad = None
    for p in per_cell.parameters():
        p.grad = None
    grouped(x_my).sum().backward()
    x_my.grad = None
    for p in grouped.parameters():
        p.grad = None

out_ref = per_cell(x_ref)
out_my = grouped(x_my)
diff = (out_ref.float() - out_my.float()).abs()
print("forward matches:", torch.allclose(out_ref, out_my, rtol=5e-2, atol=5e-2))
print("forward max abs diff:", diff.max().item(), "mean abs diff:", diff.mean().item())
print("out_ref sample:", out_ref.flatten()[:6])
print("out_my  sample:", out_my.flatten()[:6])
print("out_ref abs().mean():", out_ref.float().abs().mean().item())

out_ref.sum().backward()
out_my.sum().backward()
dx_diff = (x_ref.grad.float() - x_my.grad.float()).abs()
print("dx matches:", torch.allclose(x_ref.grad, x_my.grad, rtol=5e-2, atol=5e-2))
print("dx max abs diff:", dx_diff.max().item(), "mean abs diff:", dx_diff.mean().item())
dw_ref = torch.stack([lin.weight.grad for lin in per_cell.linears])
dw_my = torch.stack([getattr(grouped.grouped, f"weight{c}").grad for c in range(CELLS)])
print("d_weight matches:", torch.allclose(dw_ref, dw_my, rtol=5e-2, atol=5e-2))
print("d_weight max abs diff:", (dw_ref.float() - dw_my.float()).abs().max().item())

print("\n=== EAGER speed (no CUDA graph on either side) ===")
t_per_cell = bench(per_cell, x)
t_grouped = bench(grouped, x)
print(f"per-cell loop (8x te.Linear): {t_per_cell:.3f} ms/iter")
print(f"GroupedLinear (1x call):      {t_grouped:.3f} ms/iter")
print(f"speedup: {t_per_cell/t_grouped:.2f}x")
