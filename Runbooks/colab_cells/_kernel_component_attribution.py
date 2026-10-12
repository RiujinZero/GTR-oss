"""Attribute CUDA time/kernel-launch count to GTR-ARW's major forward-
pass components (broadcast-mix coupling, stewardship-traffic
bookkeeping, node compute, readout, receiver-memory scan) via labeled
torch.profiler.record_function blocks, rather than relying on
torch.profiler's stack-grouping APIs -- those turned out to collapse
almost everything into one bucket with no usable per-call-site info in
this torch version (see Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_
QUEUE.md for that dead end).

Monkey-patches the relevant function NAMES as they exist in
gator.models.language's own module namespace (that's where the actual
calls execute from at runtime, since language.py did `from ..x import
y` -- patching the SOURCE module's attribute wouldn't affect
language.py's already-bound name).
"""

import functools

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import gator.models.language as lang_mod
from gator.models.language import GatorLanguageConfig, GatorLanguageModel

VOCAB_SIZE = 128256


def _wrap(fn, label):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with record_function(label):
            return fn(*args, **kwargs)
    return wrapped


_ORIGINALS = {
    name: getattr(lang_mod, name)
    for name in (
        "cuda_graph_broadcast_mix",
        "cuda_graph_stewardship_traffic",
        "node_compute_linear",
        "mx_readout_linear",
        "relative_receiver_context",
    )
}
for name, fn in _ORIGINALS.items():
    setattr(lang_mod, name, _wrap(fn, f"COMPONENT:{name}"))

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

print("Profiling 3 real forward+backward steps with component labels...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for step in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(context, mask, output_slots=output_slots)
        logits.float().sum().backward()
        torch.cuda.synchronize()

print("\n" + "=" * 100)
print("COMPONENT LABELS (self+child CUDA time includes everything nested inside)")
print("=" * 100)
averages = prof.key_averages()
component_rows = [evt for evt in averages if evt.key.startswith("COMPONENT:")]
component_rows.sort(key=lambda evt: evt.self_device_time_total, reverse=True)
for evt in component_rows:
    print(
        f"{evt.key:45s}  count={evt.count:5d}  "
        f"self_device_time={evt.self_device_time_total / 1000:9.3f}ms  "
        f"device_time_total(incl. children)={evt.device_time_total / 1000:9.3f}ms"
    )

print("\n" + "=" * 100)
print("FULL TABLE sorted by self CUDA time (component rows will appear inline)")
print("=" * 100)
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=50))
