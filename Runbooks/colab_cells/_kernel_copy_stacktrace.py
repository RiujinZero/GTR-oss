"""Find the real call sites behind aten::copy_/_to_copy's outsized share
of CUDA time in the production config (16%+ across ~4,700 calls per 3
profiled steps, per _kernel_bandwidth_profile.py) -- with_stack=True to
see WHERE these calls originate, since batching attempts targeted at
the CUDA-graph bridges' own destination-side copies didn't move the
needle (see Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md), which
means the bulk of these calls are coming from somewhere else -- the
leading hypothesis is the widespread `.to(torch.bfloat16)` dtype casts
used throughout every graphed function's call site, not just inside
the bridges.

Same config/setup as _kernel_bandwidth_profile.py; only difference is
with_stack=True and a narrower analysis focused on copy_/_to_copy
specifically, grouped by originating call site.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

VOCAB_SIZE = 128256

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
sequence_length = 2048
output_slots = 16
context_length = sequence_length - output_slots

context = torch.randint(4, VOCAB_SIZE, (batch_size, context_length), device="cuda")
mask = torch.ones_like(context, dtype=torch.bool)

print("Warming up...")
for step in range(8):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(context, mask, output_slots=output_slots)
    logits.float().sum().backward()
    torch.cuda.synchronize()
    print(f"  warmup step {step + 1}/8 done")

print("Profiling 1 real forward+backward step with with_stack=True (slower, stack unwinding)...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(context, mask, output_slots=output_slots)
    logits.float().sum().backward()
    torch.cuda.synchronize()

print("\n" + "=" * 100)
print("ALL OPS GROUPED BY CALL STACK (top 15 frames) -- table's Source Location")
print("column shows file:line directly, the well-tested rendering path")
print("=" * 100)
grouped = prof.key_averages(group_by_stack_n=15)
print(
    grouped.table(
        sort_by="self_device_time_total",
        row_limit=40,
        max_name_column_width=60,
        max_src_column_width=120,
    )
)
