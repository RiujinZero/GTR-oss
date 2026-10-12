"""Real, full-model, production-config peak-memory measurement:
node_compute_precision=fp8_te + readout_precision=fp8_te +
receiver_memory_cuda_graph=True vs. the bf16 baseline, at the actual
production shapes (not the tiny toy configs the unit/integration tests
use). Confirms whether today's CUDA-graph work actually frees enough
VRAM to justify a larger batch size, and whether batch=4 fits.
WSL-only, throwaway.
"""
import sys
import time

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

device = "cuda"

# Matches the confirmed production launch config (see
# Runbooks/GTR_RUN_REGISTRY.md's gtr_arw_100m_smollm3_naked_
# local_sanity_check row) except for the precision/graphing flags under
# test here.
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
)

CONTEXT = 2048
OUTPUT_SLOTS = 16  # response_tokens


def run_step(config: GatorLanguageConfig, batch: int):
    # Matches the real trainer's mixed-precision handling
    # (gator_language_train.py): torch.autocast, not a blanket .to(bf16)
    # cast of the whole model -- some internal computations (e.g.
    # position codes) are meant to stay fp32 for numerical stability.
    torch.manual_seed(0)
    model = GatorLanguageModel(config).to(device=device).train()
    context = torch.randint(4, config.vocab_size, (batch, CONTEXT), device=device)
    mask = torch.ones_like(context, dtype=torch.bool)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = model(context, mask, output_slots=OUTPUT_SLOTS)
    logits.float().sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del model, logits, context, mask
    torch.cuda.empty_cache()
    return peak_gb, elapsed


print("=== batch=2, bf16 baseline (no fp8_te, no receiver-memory graph) ===", flush=True)
config_baseline_b2 = GatorLanguageConfig(**BASE_CONFIG)
peak_baseline_b2, t_baseline_b2 = run_step(config_baseline_b2, batch=2)
print(f"peak mem: {peak_baseline_b2:.3f} GB, step time: {t_baseline_b2:.3f}s", flush=True)

print("\n=== batch=2, fp8_te node+readout + receiver-memory CUDA graph ===", flush=True)
config_fp8_b2 = GatorLanguageConfig(
    **BASE_CONFIG,
    node_compute_precision="fp8_te",
    readout_precision="fp8_te",
    receiver_memory_cuda_graph=True,
)
peak_fp8_b2, t_fp8_b2 = run_step(config_fp8_b2, batch=2)
print(f"peak mem: {peak_fp8_b2:.3f} GB, step time: {t_fp8_b2:.3f}s", flush=True)
print(f"memory delta at batch=2: {peak_baseline_b2 - peak_fp8_b2:.3f} GB "
      f"({(1 - peak_fp8_b2/peak_baseline_b2)*100:.1f}% less)", flush=True)

print("\n=== batch=4, fp8_te node+readout + receiver-memory CUDA graph (does it fit?) ===", flush=True)
try:
    config_fp8_b4 = GatorLanguageConfig(
        **BASE_CONFIG,
        node_compute_precision="fp8_te",
        readout_precision="fp8_te",
        receiver_memory_cuda_graph=True,
    )
    peak_fp8_b4, t_fp8_b4 = run_step(config_fp8_b4, batch=4)
    print(f"peak mem: {peak_fp8_b4:.3f} GB, step time: {t_fp8_b4:.3f}s", flush=True)
except torch.cuda.OutOfMemoryError as exc:
    print(f"OOM at batch=4: {exc}", flush=True)

print("\ndone", flush=True)
