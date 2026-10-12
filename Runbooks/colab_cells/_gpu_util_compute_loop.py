"""Runs the real, full-scale production model (Phase 2 fixed-decay
receiver-memory kernel active) compiled under the exact production
mode, then loops forward+backward continuously for a fixed duration
with NO synchronization/profiling overhead inside the loop -- so a
concurrent `nvidia-smi --query-gpu=utilization.gpu -l 1` sampler can
see genuine steady-state GPU business (or idle gaps from CPU-side
dispatch/data-loading stalls) without the loop itself introducing
artificial gaps.

Prints START_COMPUTE_LOOP / END_COMPUTE_LOOP markers so the sampler's
output can be trimmed to just the steady-state compute window,
excluding compile time.
"""

import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch
import torch.nn.functional as F

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
CONTEXT = 2032
OUTPUT_SLOTS = 16

context = torch.randint(4, config.vocab_size, (BATCH, CONTEXT), device="cuda")
mask = torch.ones_like(context, dtype=torch.bool)


def fake_loss(logits):
    slots = logits[:, -OUTPUT_SLOTS:, :] if logits.shape[1] > OUTPUT_SLOTS else logits
    targets = torch.randint(0, slots.shape[-1], slots.shape[:-1], device=slots.device)
    return F.cross_entropy(slots.float().reshape(-1, slots.shape[-1]), targets.reshape(-1))


print("Eager warmup (lazy FP8 module construction outside any trace)...", flush=True)
logits = model(context, mask, output_slots=OUTPUT_SLOTS)
fake_loss(logits).backward()
for p in model.parameters():
    p.grad = None
torch.cuda.synchronize()

compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs")

print("Compiling...", flush=True)
for step in range(3):
    logits = compiled_model(context, mask, output_slots=OUTPUT_SLOTS)
    fake_loss(logits).backward()
    for p in model.parameters():
        p.grad = None
    torch.cuda.synchronize()
print("Compile done.", flush=True)

DURATION_S = 20
print("START_COMPUTE_LOOP", flush=True)
start = time.perf_counter()
steps = 0
while time.perf_counter() - start < DURATION_S:
    logits = compiled_model(context, mask, output_slots=OUTPUT_SLOTS)
    fake_loss(logits).backward()
    for p in model.parameters():
        p.grad = None
    steps += 1
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print("END_COMPUTE_LOOP", flush=True)
print(f"steps={steps} elapsed={elapsed:.2f}s steps_per_second={steps/elapsed:.3f}", flush=True)
