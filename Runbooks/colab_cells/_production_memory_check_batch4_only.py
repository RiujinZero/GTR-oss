"""batch=4 ALONE, fresh process -- isolates whether batch=4 genuinely
fits once CUDA graphs are captured for that shape from a clean process,
without the confound of a prior batch=2 run's graph pools (which
CUDA graphs don't release via empty_cache()) still resident.
WSL-only, throwaway.
"""
import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

device = "cuda"

BASE_CONFIG = dict(
    vocab_size=128256,
    d_model=1312,
    embedding_dim=2624,
    lexical_embedding_dim=256,
    n_cells=8,
    n_nodes=4,
    relays=2,
    neighbors=4,
    mask_token_id=3,
    node_topology="asymmetric",
    cell_input_mode="feature_gated",
    cell_output_mode="grouped_concat",
    cell_transport_mode="stewardship",
    block_causal=True,
    block_size=16,
    position_reference_length=4096,
    receiver_memory_backend="linear_scan",
    stewardship_execution="dense",
    dense_kernel_padding=8,
    dense_node_backend="folded_bmm",
    node_compute_precision="fp8_te",
    readout_precision="fp8_te",
    receiver_memory_cuda_graph=True,
)

CONTEXT = 2048
OUTPUT_SLOTS = 16

torch.manual_seed(0)
config = GatorLanguageConfig(**BASE_CONFIG)
model = GatorLanguageModel(config).to(device=device).train()
context = torch.randint(4, config.vocab_size, (4, CONTEXT), device=device)
mask = torch.ones_like(context, dtype=torch.bool)

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
try:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = model(context, mask, output_slots=OUTPUT_SLOTS)
    logits.float().sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch=4 fp8_te+graphs: peak mem {peak_gb:.3f} GB, step time {elapsed:.3f}s", flush=True)
except torch.cuda.OutOfMemoryError as exc:
    print(f"batch=4 fp8_te+graphs: OOM -> {exc}", flush=True)

print("done", flush=True)
