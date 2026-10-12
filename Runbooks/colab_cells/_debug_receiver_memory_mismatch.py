"""Debug why test_matches_eager_reference_across_two_sequential_calls
fails while the structurally similar single-call test passes. Reproduces
each scenario in an isolated process (avoiding any cross-test module
cache contamination) to narrow down the real cause. Throwaway, WSL-only.
"""
import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.cuda_graph_receiver_memory import cuda_graph_linear_scan_receiver_context
from gator.models.memory import LinearScanReceiverCache, linear_scan_receiver_context


def shapes():
    torch.manual_seed(0)
    batch, cells, tokens, hidden, nodes = 2, 8, 32, 48, 4
    block_size = 16
    states = torch.randn(batch, cells, tokens, hidden, device="cuda", dtype=torch.bfloat16)
    node_locations = torch.rand(batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16)
    writer_mass = torch.rand(batch, cells, tokens, device="cuda", dtype=torch.bfloat16)
    relative_node_bias = torch.randn(cells, nodes, 4, device="cuda", dtype=torch.bfloat16)
    rows = batch * (hidden + 1)
    channels = cells * nodes
    cache_forward_state = torch.zeros(rows, channels, device="cuda", dtype=torch.bfloat16)
    return states, node_locations, writer_mass, relative_node_bias, cache_forward_state, block_size


print("=== Scenario A: single isolated call (mirrors passing test) ===", flush=True)
states, locations, writer_mass, bias, cache0, block_size = shapes()
ref_ctx, ref_cache = linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size,
    cache=LinearScanReceiverCache(cache0), return_cache=True,
)
graphed_ctx, graphed_cache = cuda_graph_linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size, cache_forward_state=cache0,
)
diff = (graphed_ctx.float() - ref_ctx.float()).abs()
print(f"A: max_abs_diff={diff.max().item():.6f} mean_abs_diff={diff.mean().item():.6f} "
      f"ref_absmax={ref_ctx.float().abs().max().item():.6f}", flush=True)
print(f"A: cache0 mutated by reference call? {not torch.equal(cache0, torch.zeros_like(cache0))}", flush=True)

print("\n=== Scenario B: generate unrelated states2 BEFORE the comparison (mirrors failing test) ===", flush=True)
states, locations, writer_mass, bias, cache0, block_size = shapes()
torch.manual_seed(1)
states2 = torch.randn_like(states)
locations2 = torch.rand_like(locations)
writer_mass2 = torch.rand_like(writer_mass)

ref_ctx_b, ref_cache_b = linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size,
    cache=LinearScanReceiverCache(cache0.clone()), return_cache=True,
)
graphed_ctx_b, graphed_cache_b = cuda_graph_linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size, cache_forward_state=cache0.clone(),
)
diff_b = (graphed_ctx_b.float() - ref_ctx_b.float()).abs()
print(f"B: max_abs_diff={diff_b.max().item():.6f} mean_abs_diff={diff_b.mean().item():.6f} "
      f"ref_absmax={ref_ctx_b.float().abs().max().item():.6f}", flush=True)

print("\n=== Scenario C: same as A but call the graphed function TWICE in a row first (cache reuse) ===", flush=True)
states, locations, writer_mass, bias, cache0, block_size = shapes()
# Prime the cache/graph with one throwaway call first.
_ = cuda_graph_linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size, cache_forward_state=cache0.clone(),
)
ref_ctx_c, _ = linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size,
    cache=LinearScanReceiverCache(cache0.clone()), return_cache=True,
)
graphed_ctx_c, _ = cuda_graph_linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size, cache_forward_state=cache0.clone(),
)
diff_c = (graphed_ctx_c.float() - ref_ctx_c.float()).abs()
print(f"C: max_abs_diff={diff_c.max().item():.6f} mean_abs_diff={diff_c.mean().item():.6f} "
      f"ref_absmax={ref_ctx_c.float().abs().max().item():.6f}", flush=True)

print("\n=== Scenario D: call .backward() once first, THEN do a plain forward-only comparison (matches actual pytest failure order) ===", flush=True)
states, locations, writer_mass, bias, cache0, block_size = shapes()
states_g = states.clone().requires_grad_(True)
locations_g = locations.clone().requires_grad_(True)
writer_mass_g = writer_mass.clone().requires_grad_(True)
bias_g = bias.clone().requires_grad_(True)
ctx_priming, cache_priming = cuda_graph_linear_scan_receiver_context(
    states_g, locations_g, writer_mass_g, bias_g, block_size=block_size, cache_forward_state=cache0.clone(),
)
(ctx_priming.sum() + cache_priming.sum()).backward()
del ctx_priming, cache_priming, states_g, locations_g, writer_mass_g, bias_g

states, locations, writer_mass, bias, cache0, block_size = shapes()
ref_ctx_d, _ = linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size,
    cache=LinearScanReceiverCache(cache0.clone()), return_cache=True,
)
graphed_ctx_d, _ = cuda_graph_linear_scan_receiver_context(
    states, locations, writer_mass, bias, block_size=block_size, cache_forward_state=cache0.clone(),
)
diff_d = (graphed_ctx_d.float() - ref_ctx_d.float()).abs()
print(f"D: max_abs_diff={diff_d.max().item():.6f} mean_abs_diff={diff_d.mean().item():.6f} "
      f"ref_absmax={ref_ctx_d.float().abs().max().item():.6f}", flush=True)

print("done", flush=True)
