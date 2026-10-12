"""Isolated throughput check: eager vs CUDA-graphed stewardship traffic
bookkeeping at production shape (batch=2, cells=8, tokens=2064, nodes=4,
block_size=16). Forward+backward timed with cuda events after warmup,
same methodology as _broadcast_mix_throughput.py.
"""

import time

import torch

from gator.cuda_graph_stewardship_traffic import cuda_graph_stewardship_traffic

torch.manual_seed(0)
batch, cells, tokens, nodes, block_size = 2, 8, 2064, 4, 16


def eager_call(locations, writer_mass, write, prev_num, prev_write):
    blocks = tokens // block_size
    accumulator_dtype = torch.float32
    token_numerator = locations * writer_mass.unsqueeze(-1)
    block_numerator = token_numerator.reshape(
        batch, cells, blocks, block_size, nodes
    ).sum(dim=3, dtype=accumulator_dtype).permute(0, 2, 1, 3)
    block_write = write.reshape(batch, blocks, block_size).sum(
        dim=2, dtype=accumulator_dtype
    )
    cumulative_numerator = block_numerator.cumsum(dim=1) + prev_num.unsqueeze(1)
    cumulative_write = (block_write.cumsum(dim=1) + prev_write.unsqueeze(1)).clamp_min(1e-6)
    traffic_blocks = (cumulative_numerator / cumulative_write[:, :, None, None]).to(torch.bfloat16)
    traffic = traffic_blocks[:, :, None].expand(
        -1, -1, block_size, -1, -1
    ).reshape(batch, tokens, cells, nodes).permute(0, 2, 1, 3)
    return traffic


def graphed_call(locations, writer_mass, write, prev_num, prev_write, slot):
    traffic, _, _ = cuda_graph_stewardship_traffic(
        locations, writer_mass, write, prev_num, prev_write,
        block_size=block_size, states_dtype=torch.bfloat16, graph_slot=slot,
    )
    return traffic


def bench(fn, iters=30, warmup=10):
    locations = torch.rand(
        batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    writer_mass = torch.rand(
        batch, cells, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    write = torch.rand(batch, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    prev_num = torch.zeros(batch, cells, nodes, device="cuda", dtype=torch.float32)
    prev_write = torch.zeros(batch, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        out = fn(locations, writer_mass, write, prev_num, prev_write)
        out.sum().backward()
        locations.grad = None
        writer_mass.grad = None
        write.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(locations, writer_mass, write, prev_num, prev_write)
        out.sum().backward()
        locations.grad = None
        writer_mass.grad = None
        write.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    eager_ms = bench(eager_call)
    graphed_ms = bench(lambda *a: graphed_call(*a, 0))

print(f"eager:   {eager_ms:.3f} ms/iter")
print(f"graphed: {graphed_ms:.3f} ms/iter")
print(f"speedup: {eager_ms / graphed_ms:.3f}x")
