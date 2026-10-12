"""A/B: FFT-based causal decay scan vs the existing chunked decayed_scan
(gator_linear/core.py), at GTR-ARW's real production receiver-memory
shape. Only the FORWARD (full-sequence, cache-carrying) direction is
targeted -- the BACKWARD direction in block_causal mode already resets
every block_size=16 positions, far too short for FFT overhead to pay
off, so it stays on the existing chunked implementation regardless.

forward[l] = sum_{l' <= l} decay^(l-l') * values[l']

This is a plain causal convolution of `values` with a per-channel
geometric kernel `decay^j` (j=0..L-1) -- computed via FFT as a linear
(not circular) convolution: zero-pad to >= 2L-1, multiply spectra,
inverse-FFT, keep the first L samples. Kernel/FFT math done in fp32
regardless of input dtype (bf16 decay^j underflows to exact zero much
earlier than fp32, which would silently truncate long-range history).

Prior evidence in this codebase (Runbooks/GTR_ARW_100M_PRODUCTION_
ABLATION_QUEUE.md, referencing PERFORMANCE_TODO.md /
BLACKWELL_FFT_TRITON_GATE.md): an FFT receiver-memory backend measured
~3.6x SLOWER than dense matrix for the Convergent (packed/sparse)
architecture at context length 640 -- but that entangled packed-routing
overhead with the FFT choice, and 640 is well under GTR-ARW's real
~2032-token context, where the chunked scan's O(L/chunk_size) sequential
Python-loop iterations (~32 of them here) cost more. This script
isolates the comparison: dense-only, GTR-ARW's actual shape, forward-
direction only.
"""

import time

import torch

from gator_linear.core import decayed_scan


def fft_causal_decay_scan(values: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """forward[l] = sum_{l'<=l} decay^(l-l') * values[l'], via FFT.

    values: [rows, L, C]. decay: [C] in (0,1). Returns [rows, L, C].
    """
    rows, length, channels = values.shape
    compute_dtype = torch.float32
    values32 = values.to(compute_dtype)
    decay32 = decay.to(compute_dtype)

    exponents = torch.arange(length, device=values.device, dtype=compute_dtype)
    kernel = decay32[None, :] ** exponents[:, None]  # [L, C], kernel[j,c] = decay_c^j

    fft_len = 2 * length  # zero-pad to avoid circular wraparound
    V = torch.fft.rfft(values32, n=fft_len, dim=1)     # [rows, fft_len//2+1, C]
    K = torch.fft.rfft(kernel, n=fft_len, dim=0)       # [fft_len//2+1, C]
    Y = V * K[None, :, :]
    y = torch.fft.irfft(Y, n=fft_len, dim=1)           # [rows, fft_len, C]
    return y[:, :length, :].to(values.dtype)


def reference_forward(values, decay, chunk_size=64):
    return decayed_scan(values, decay, chunk_size=chunk_size)


# --- Correctness check at a small shape ---
torch.manual_seed(0)
rows_s, length_s, channels_s = 4, 130, 6
values_s = torch.randn(rows_s, length_s, channels_s, device="cuda", dtype=torch.float32, requires_grad=True)
decay_s = torch.rand(channels_s, device="cuda", dtype=torch.float32) * 0.9 + 0.05

ref_s = reference_forward(values_s, decay_s)
fft_s = fft_causal_decay_scan(values_s, decay_s)
max_abs_diff = (ref_s - fft_s).abs().max().item()
max_rel_diff = ((ref_s - fft_s).abs() / ref_s.abs().clamp_min(1e-6)).max().item()
print(f"Correctness check (small shape): max abs diff={max_abs_diff:.3e}, max rel diff={max_rel_diff:.3e}")

# Gradient check
ref_s.sum().backward()
grad_ref = values_s.grad.clone()
values_s.grad = None
fft_s2 = fft_causal_decay_scan(values_s, decay_s)
fft_s2.sum().backward()
grad_fft = values_s.grad.clone()
grad_diff = (grad_ref - grad_fft).abs().max().item()
print(f"Gradient check (small shape): max abs diff={grad_diff:.3e}")

# --- Benchmark at GTR-ARW production receiver-memory shape ---
# batch=2, hidden=d_model=1312 (+1 for the denominator channel),
# tokens=2032 (sequence_length=2048 - output_slots=16), cells=8, nodes=4
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
fft_ms = bench(fft_causal_decay_scan)

print(f"\nchunked decayed_scan (chunk_size=64): {chunked_ms:.3f} ms/iter (fwd+bwd)")
print(f"FFT causal decay scan:                 {fft_ms:.3f} ms/iter (fwd+bwd)")
print(f"speedup: {chunked_ms / fft_ms:.3f}x")
