"""Kernel-level compute/memory profile of the exact production GTR-ARW
config (fp8_te node compute + readout, broadcast-mix + stewardship-
traffic CUDA graphs), to get real per-op data on where GPU time and
memory traffic actually go -- direct evidence for/against the memory-
bandwidth-bound hypothesis discussed throughout this investigation,
rather than the indirect reasoning (FP8 speedup below theoretical,
receiver-memory CUDA-graph giving zero speedup) used so far.

Deliberately EAGER (no torch.compile): inductor's kernel fusion merges
many small ops into fused kernels, which would obscure per-component
attribution (which part of the model is expensive) -- the exact
opposite of what this analysis is for. This still reflects the real
production model/config, just without the compiler's fusion boundary
changes. Our own custom CUDA graphs (node compute/readout/broadcast-
mix/stewardship-traffic) ARE active, matching production exactly.

Uses torch.profiler with record_shapes + profile_memory + with_flops:
- with_flops=True gives PyTorch's own FLOP estimate for supported ops
  (mm/bmm/addmm/linear/conv) -- the compute-side number.
- self_cuda_memory_usage (net allocated/freed CUDA memory per op) is
  used as a practical proxy for memory traffic -- not a true read+write
  byte count (that needs Nsight Compute), but a reasonable, readily
  available signal for which ops move the most memory, sufficient for
  directional "what to target next" guidance rather than a publication-
  grade roofline study.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

VOCAB_SIZE = 128256  # meta-llama/Llama-3.2-1B tokenizer, --hf-tokenizer default

config = GatorLanguageConfig(
    vocab_size=VOCAB_SIZE,
    d_model=1312,
    embedding_dim=2624,
    lexical_embedding_dim=256,
    n_cells=8,
    n_nodes=4,
    relays=2,
    neighbors=4,
    node_topology="asymmetric",
    cell_input_mode="shared_projection",
    cell_output_mode="shared_pool",
    cell_transport_mode="broadcast_mix",
    stewardship_execution="dense",
    receiver_memory_backend="linear_scan",
    block_causal=True,
    block_size=16,
    position_reference_length=4096,
    node_compute_precision="fp8_te",
    readout_precision="fp8_te",
    broadcast_mix_cuda_graph=True,
    stewardship_traffic_cuda_graph=True,
    dense_kernel_padding=8,
    dense_node_backend="folded_bmm",
    mask_token_id=3,
)

torch.manual_seed(0)
model = GatorLanguageModel(config).cuda().train()

batch_size = 2
sequence_length = 2048  # --smollm3-sequence-length: TOTAL length including response tokens
output_slots = 16
context_length = sequence_length - output_slots

context = torch.randint(4, VOCAB_SIZE, (batch_size, context_length), device="cuda")
mask = torch.ones_like(context, dtype=torch.bool)

print("Warming up (CUDA graph capture for node compute/readout/broadcast-mix/stewardship-traffic)...")
for step in range(8):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(context, mask, output_slots=output_slots)
    logits.float().sum().backward()
    torch.cuda.synchronize()
    print(f"  warmup step {step + 1}/8 done")

print("Profiling 3 real forward+backward steps...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    for step in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(context, mask, output_slots=output_slots)
        logits.float().sum().backward()
        torch.cuda.synchronize()

print("\n" + "=" * 100)
print("TOP 40 OPS BY SELF CUDA TIME")
print("=" * 100)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))

print("\n" + "=" * 100)
print("TOP 40 OPS BY SELF CUDA MEMORY USAGE (net alloc/free, bandwidth-traffic proxy)")
print("=" * 100)
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=40))

print("\n" + "=" * 100)
print("TOP 40 OPS BY FLOPS (compute-side, PyTorch's own estimate for mm/bmm/linear/conv)")
print("=" * 100)
print(prof.key_averages().table(sort_by="flops", row_limit=40))

trace_path = "/home/user/gtr_arw_100m_kernel_profile.json"
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace exported to {trace_path}")

# Aggregate empirical arithmetic-intensity read: total FLOPs vs total
# self CUDA memory traffic across the whole profiled window, as a
# single directional number to compare against the roofline point
# where this GPU's compute:bandwidth ratio makes a kernel compute-bound.
total_flops = sum(
    (evt.flops or 0) for evt in prof.key_averages()
)
total_cuda_mem = sum(
    # PyTorch 2.4+ generalized the profiler's per-device memory fields
    # (CUDA/XPU/MTIA); FunctionEventAvg exposes self_device_memory_usage
    # now, not the old self_cuda_memory_usage name used in the printed
    # table headers/older torch versions.
    abs(evt.self_device_memory_usage) for evt in prof.key_averages()
)
print("\n" + "=" * 100)
print("AGGREGATE (directional, not a true roofline measurement)")
print("=" * 100)
print(f"Total FLOPs (measurable ops only): {total_flops:,}")
print(f"Total |self CUDA memory usage| across ops: {total_cuda_mem:,} bytes")
if total_cuda_mem:
    print(f"FLOPs / byte (memory-traffic-proxy): {total_flops / total_cuda_mem:.2f}")
