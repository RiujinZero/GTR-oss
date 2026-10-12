"""Real throughput measurement using the shipped
gator.cuda_graph_receiver_memory.cuda_graph_linear_scan_receiver_context
at the exact production shape: batch=2, cells=8, tokens=2064 (2048+16
response), hidden=d_model=1312, nodes=n_nodes=4, block_size=16
(block-causal). Compares against the eager linear_scan_receiver_context.
This benchmark includes the .clone() correctness fix from the start.
Throwaway, WSL-only.
"""
import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.cuda_graph_receiver_memory import cuda_graph_linear_scan_receiver_context
from gator.models.memory import LinearScanReceiverCache, linear_scan_receiver_context

device = "cuda"
torch.manual_seed(0)

batch, cells, tokens, hidden, nodes = 2, 8, 2064, 1312, 4
block_size = 16

states = torch.randn(batch, cells, tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
node_locations = torch.rand(batch, cells, tokens, nodes, device=device, dtype=torch.bfloat16, requires_grad=True)
writer_mass = torch.rand(batch, cells, tokens, device=device, dtype=torch.bfloat16, requires_grad=True)
bias = torch.randn(cells, nodes, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
rows = batch * (hidden + 1)
channels = cells * nodes
cache0 = torch.zeros(rows, channels, device=device, dtype=torch.bfloat16)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def eager_step():
    context, next_cache = linear_scan_receiver_context(
        states, node_locations, writer_mass, bias,
        block_size=block_size, cache=LinearScanReceiverCache(cache0), return_cache=True,
    )
    (context.sum() + next_cache.forward_state.sum()).backward()
    states.grad = None
    node_locations.grad = None
    writer_mass.grad = None
    bias.grad = None


def graphed_step():
    context, next_cache = cuda_graph_linear_scan_receiver_context(
        states, node_locations, writer_mass, bias,
        block_size=block_size, cache_forward_state=cache0,
    )
    (context.sum() + next_cache.sum()).backward()
    states.grad = None
    node_locations.grad = None
    writer_mass.grad = None
    bias.grad = None


print("warming up graphed path (pays one-time capture cost)...", flush=True)
graphed_step()

t_eager = bench(eager_step)
print(f"eager (linear_scan_receiver_context):        {t_eager*1e3:.4f} ms/iter", flush=True)

t_graphed = bench(graphed_step)
print(f"graphed (cuda_graph_linear_scan_receiver_context): {t_graphed*1e3:.4f} ms/iter", flush=True)

print(f"speedup: {t_eager/t_graphed:.3f}x", flush=True)

torch.cuda.reset_peak_memory_stats()
for _ in range(5):
    eager_step()
torch.cuda.synchronize()
mem_eager = torch.cuda.max_memory_allocated() / 1e9

torch.cuda.reset_peak_memory_stats()
for _ in range(5):
    graphed_step()
torch.cuda.synchronize()
mem_graphed = torch.cuda.max_memory_allocated() / 1e9

print(f"peak mem eager:   {mem_eager:.3f} GB", flush=True)
print(f"peak mem graphed: {mem_graphed:.3f} GB", flush=True)
print("done", flush=True)
