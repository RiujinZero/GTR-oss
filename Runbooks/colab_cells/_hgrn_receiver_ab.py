"""A/B: FLA's fused chunk_hgrn Triton kernel vs the existing chunked
decayed_scan (gator_linear/core.py), at GTR-ARW's real production
receiver-memory shape.

Mapping (user-specified, matches gator_linear/core.py's decayed_scan
recurrence exactly): our recurrence h_t = decay*h_{t-1} + v_t is HGRN's
h_t = exp(g_t)*h_{t-1} + x_t with g_t = log(decay), broadcast constant
across t (decay is learned but time-independent here) -- Phase 1 per
the user's spec: use stock chunk_hgrn unchanged, broadcasting the
constant gate to [B,T,D] rather than forking a specialized kernel yet.

Only the FORWARD (full-sequence, cache-carrying) direction is targeted,
matching the FFT A/B's scope -- the BACKWARD direction in block_causal
mode already resets every block_size=16 positions, too short for a
fused chunk kernel's overhead to pay off differently than the existing
tiny-chunk implementation.
"""

import torch

from fla.ops.hgrn import chunk_hgrn
from gator_linear.core import decayed_scan


def hgrn_forward(values: torch.Tensor, decay: torch.Tensor, initial_state=None):
    """values: [rows, L, C] (decayed_scan's convention).
    decay: [C]. Returns (forward[rows,L,C], final_state[rows,C])."""
    rows, length, channels = values.shape
    g = decay.log()[None, None, :].expand(rows, length, channels).contiguous()
    y, final_state = chunk_hgrn(values, g, initial_state=initial_state, output_final_state=True)
    return y, final_state


def reference_forward(values, decay, chunk_size=64, initial_state=None):
    if initial_state is None:
        return decayed_scan(values, decay, chunk_size=chunk_size)
    return decayed_scan(
        values, decay, chunk_size=chunk_size,
        initial_state=initial_state, return_final_state=True,
    )


# --- Correctness check at a small shape ---
torch.manual_seed(0)
rows_s, length_s, channels_s = 4, 130, 6
values_s = torch.randn(rows_s, length_s, channels_s, device="cuda", dtype=torch.float32, requires_grad=True)
decay_s = torch.rand(channels_s, device="cuda", dtype=torch.float32) * 0.9 + 0.05

ref_s = reference_forward(values_s, decay_s)
hgrn_s, hgrn_final_s = hgrn_forward(values_s, decay_s)
max_abs_diff = (ref_s - hgrn_s).abs().max().item()
max_rel_diff = ((ref_s - hgrn_s).abs() / ref_s.abs().clamp_min(1e-6)).max().item()
print(f"Correctness check (small shape): max abs diff={max_abs_diff:.3e}, max rel diff={max_rel_diff:.3e}")

final_state_ref = ref_s[:, -1, :]
final_state_diff = (final_state_ref - hgrn_final_s).abs().max().item()
print(f"Final-state check: max abs diff={final_state_diff:.3e}")

# Gradient check (w.r.t. values)
ref_s.sum().backward()
grad_ref = values_s.grad.clone()
values_s.grad = None
hgrn_s2, _ = hgrn_forward(values_s, decay_s)
hgrn_s2.sum().backward()
grad_hgrn = values_s.grad.clone()
grad_diff = (grad_ref - grad_hgrn).abs().max().item()
print(f"Gradient check (d/d values): max abs diff={grad_diff:.3e}")

# Gradient check w.r.t. decay (via autograd through log/exp; decay_s
# itself doesn't require grad above -- redo with requires_grad=True)
decay_s2 = decay_s.detach().clone().requires_grad_(True)
values_s3 = values_s.detach().clone().requires_grad_(True)
ref_s2 = reference_forward(values_s3, decay_s2)
ref_s2.sum().backward()
grad_decay_ref = decay_s2.grad.clone()

decay_s3 = decay_s.detach().clone().requires_grad_(True)
values_s4 = values_s.detach().clone().requires_grad_(True)
hgrn_s3, _ = hgrn_forward(values_s4, decay_s3)
hgrn_s3.sum().backward()
grad_decay_hgrn = decay_s3.grad.clone()
grad_decay_diff = (grad_decay_ref - grad_decay_hgrn).abs().max().item()
print(f"Gradient check (d/d decay): max abs diff={grad_decay_diff:.3e}")

# --- bf16 stability check ---
values_bf16 = values_s.detach().clone().to(torch.bfloat16).requires_grad_(True)
decay_bf16 = decay_s.detach().clone().to(torch.bfloat16)
ref_bf16 = reference_forward(values_bf16, decay_bf16)
hgrn_bf16, _ = hgrn_forward(values_bf16, decay_bf16)
bf16_diff = (ref_bf16.float() - hgrn_bf16.float()).abs().max().item()
print(f"bf16 forward diff: max abs diff={bf16_diff:.3e}")
print(f"bf16 finite: ref={torch.isfinite(ref_bf16).all().item()} hgrn={torch.isfinite(hgrn_bf16).all().item()}")

# --- Benchmark at GTR-ARW production receiver-memory shape ---
batch, hidden, length, cells, nodes = 2, 1312, 2032, 8, 4
rows = batch * (hidden + 1)
channels = cells * nodes

print(f"\nProduction shape: rows={rows}, length={length}, channels={channels}")

torch.manual_seed(1)
values = torch.randn(rows, length, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
decay = (torch.rand(channels, device="cuda", dtype=torch.bfloat16) * 0.9 + 0.05)


def bench(fn, iters=10, warmup=5):
    for _ in range(warmup):
        out = fn(values, decay)
        out.sum().backward()
        values.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(values, decay)
        out.sum().backward()
        values.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


chunked_ms = bench(lambda v, d: reference_forward(v, d, chunk_size=64))
hgrn_ms = bench(lambda v, d: hgrn_forward(v, d)[0])

print(f"\nchunked decayed_scan (chunk_size=64): {chunked_ms:.3f} ms/iter (fwd+bwd)")
print(f"FLA chunk_hgrn:                        {hgrn_ms:.3f} ms/iter (fwd+bwd)")
print(f"speedup: {chunked_ms / hgrn_ms:.3f}x")
