"""Re-profile the REAL, full-scale, compiled production model (Phase 2
fixed-decay receiver-memory kernel active) to see where GPU time
actually goes now. The original component-attribution profiling
(Runbooks/colab_cells/_kernel_component_attribution.py) was EAGER --
receiver-memory showed up as ~96% of self time there, but that number
reflected exactly the kind of interpreter/loop overhead torch.compile
already erases for every OTHER component too. Production throughput
staying flat across three different receiver-memory kernel
implementations (stock chunk_hgrn, compiled general kernel, compiled
fixed-decay kernel -- 1.21x and then 1.6-1.8x faster in ISOLATION at
each step) despite zero graph breaks suggests receiver-memory's actual
share of compiled wall-clock time is much smaller than the eager
profile implied. This script builds the real production-shaped model,
compiles it with the exact same mode production uses, and profiles
several real (post-compile) forward+backward steps with
torch.profiler, then reports self CUDA time grouped by name pattern:
receiver-memory (gator::fixed_decay_hgrn_* / gator::chunk_hgrn_* /
decayed_scan-adjacent), fp8 node-compute (mx_precision /
te_fp8_cudagraph), broadcast-mix / stewardship-traffic CUDA graphs,
and everything else.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

config = GatorLanguageConfig(
    vocab_size=128256,
    d_model=1312,
    embedding_dim=2624,
    lexical_embedding_dim=256,
    n_cells=8,
    n_nodes=4,
    relays=2,
    neighbors=4,
    mask_token_id=3,
    lexical_mode="private_hash",
    cell_input_mode="shared_projection",
    cell_output_mode="shared_pool",
    cell_transport_mode="broadcast_mix",
    stewardship_execution="dense",
    receiver_memory_backend="linear_scan",
    receiver_memory_hgrn_scan=True,
    dense_node_backend="folded_bmm",
    dense_kernel_padding=8,
    node_topology="asymmetric",
    block_causal=True,
    block_size=16,
    position_reference_length=4096,
    node_compute_precision="fp8_te",
    readout_precision="fp8_te",
    broadcast_mix_cuda_graph=True,
    stewardship_traffic_cuda_graph=True,
)

torch.manual_seed(0)
model = GatorLanguageModel(config).cuda().train()

BATCH = 2
CONTEXT = 2032  # matches production: sequence_length=2048 - response_tokens=16
OUTPUT_SLOTS = 16

context = torch.randint(4, config.vocab_size, (BATCH, CONTEXT), device="cuda")
mask = torch.ones_like(context, dtype=torch.bool)

def fake_loss(logits):
    # A naive .float().sum() over the FULL logits tensor is NOT
    # representative of the real training loss (masked_cross_entropy),
    # which only touches the response-slot positions via a fused
    # cross_entropy kernel -- confirmed by a first profiling attempt
    # where that naive sum alone ate >55% of total profiled GPU time,
    # an artifact of the test harness, not the real model. Slice down
    # to the last OUTPUT_SLOTS positions and use real cross_entropy
    # against random targets to stay in the right ballpark.
    slots = logits[:, -OUTPUT_SLOTS:, :] if logits.shape[1] > OUTPUT_SLOTS else logits
    targets = torch.randint(0, slots.shape[-1], slots.shape[:-1], device=slots.device)
    return F.cross_entropy(slots.float().reshape(-1, slots.shape[-1]), targets.reshape(-1))


print("Eager warmup (forces lazy _FP8CellBlock construction OUTSIDE any Dynamo trace --")
print("without this, the first compiled call hits a Dynamo internal bug tracing")
print("TransformerEngine's lazy nn.Module construction; production never hits this")
print("because something earlier in its pipeline already materializes these eagerly)...")
logits = model(context, mask, output_slots=OUTPUT_SLOTS)
print("  logits.shape:", tuple(logits.shape))
fake_loss(logits).backward()
for p in model.parameters():
    p.grad = None
torch.cuda.synchronize()
print("  eager warmup done")

compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs")

print("Compiling + warming up (this triggers the real max-autotune-no-cudagraphs compile)...")
for step in range(3):
    logits = compiled_model(context, mask, output_slots=OUTPUT_SLOTS)
    fake_loss(logits).backward()
    for p in model.parameters():
        p.grad = None
    torch.cuda.synchronize()
    print(f"  warmup step {step} done")

print("Profiling 3 post-compile steps...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for step in range(3):
        logits = compiled_model(context, mask, output_slots=OUTPUT_SLOTS)
        fake_loss(logits).backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()

events = prof.key_averages()

def self_us(evt):
    return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)

# NOTE: key_averages() can list the SAME underlying kernel launch under
# two different entries -- e.g. a torch.library.triton_op's dispatch
# wrapper name (often with a numeric "_1" disambiguation suffix) AND
# the raw kernel name, both reporting the identical self CUDA time.
# Confirmed by a first pass here where every custom fixed_decay kernel
# appeared as an exact-value pair. Summing ALL entries naively double-
# counts those and inflates totals ~2x vs the table's own official
# "Self CUDA time total" footer. So: use EXACT kernel names only (no
# substring/suffix matching) for the receiver-memory figure, and read
# the trustworthy overall total from the table footer below by eye.
RECEIVER_MEMORY_KERNEL_NAMES = {
    "_fixed_decay_fwd_kernel_h",
    "_fixed_decay_fwd_kernel_o",
    "_fixed_decay_bwd_kernel_h",
    "_fixed_decay_bwd_kernel_o",
}
receiver_memory_us = sum(
    self_us(e) for e in events if e.key in RECEIVER_MEMORY_KERNEL_NAMES
)
print(f"\nReceiver-memory (fixed_decay kernels, exact-name match, no double-count): {receiver_memory_us/1000:.2f} ms (3 steps)")
print("Compare against 'Self CUDA time total' in the table footer below for the percentage.")

print("\n=== Top 40 by self CUDA time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
