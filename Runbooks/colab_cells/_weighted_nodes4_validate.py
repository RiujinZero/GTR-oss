"""Validate gator.weighted_nodes4.weighted_nodes4 against the
reference formula it replaces: (q.unsqueeze(-1) * destination).sum(-2)
[+ h]. Forward, gradients w.r.t. q/destination/h, both with and
without the fused residual, bf16 finiteness, and a fullgraph=True
compile check (forward captured with zero graph breaks -- same bar as
the HGRN work, avoiding re-creating that graph-break problem here).
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.weighted_nodes4 import weighted_nodes4

torch.manual_seed(0)
B, C, T, N, D = 2, 4, 16, 4, 32
q = torch.randn(B, C, T, N, device="cuda", dtype=torch.bfloat16)
destination = torch.randn(B, C, T, N, D, device="cuda", dtype=torch.bfloat16)
h = torch.randn(B, C, T, D, device="cuda", dtype=torch.bfloat16)


def reference(q, destination, h=None):
    mixed = (q.unsqueeze(-1) * destination).sum(-2)
    return mixed + h if h is not None else mixed


for label, use_h in [("no residual", False), ("with residual", True)]:
    print(f"\n=== {label} ===")
    h_arg = h if use_h else None

    ref = reference(q, destination, h_arg)
    my = weighted_nodes4(q, destination, h_arg)
    # bf16 tolerance: a 4-term FMA accumulation carries more rounding
    # noise than a single op -- matches the rtol=5e-2 bar used
    # elsewhere in this session for bf16 comparisons (e.g. HGRN).
    print("forward matches:", torch.allclose(ref, my, atol=5e-2, rtol=5e-2))

    q_ref = q.detach().clone().requires_grad_(True)
    dest_ref = destination.detach().clone().requires_grad_(True)
    h_ref = h.detach().clone().requires_grad_(True) if use_h else None
    ref2 = reference(q_ref, dest_ref, h_ref)
    ref2.sum().backward()

    q_my = q.detach().clone().requires_grad_(True)
    dest_my = destination.detach().clone().requires_grad_(True)
    h_my = h.detach().clone().requires_grad_(True) if use_h else None
    my2 = weighted_nodes4(q_my, dest_my, h_my)
    my2.sum().backward()

    print("dq matches:", torch.allclose(q_ref.grad, q_my.grad, atol=1e-1, rtol=5e-2))
    print("d_destination matches:", torch.allclose(dest_ref.grad, dest_my.grad, atol=1e-2, rtol=1e-2))
    if use_h:
        print("dh matches:", torch.allclose(h_ref.grad, h_my.grad, atol=1e-2, rtol=1e-2))

print("\n=== bf16 forward finiteness ===")
print("output finite:", torch.isfinite(my).all().item())

print("\n=== fullgraph=True compile (forward only, with residual) ===")


def wrapper(q, destination, h):
    return weighted_nodes4(q, destination, h).float().sum()


compiled = torch.compile(wrapper, fullgraph=True)
try:
    result = compiled(q, destination, h)
    print("SUCCEEDED, result:", result.item())
except Exception as error:
    print("FAILED:")
    print(repr(error))

print("\n=== fullgraph=True compile, real training-shaped autograd round trip ===")
q_c = q.detach().clone().requires_grad_(True)
dest_c = destination.detach().clone().requires_grad_(True)
h_c = h.detach().clone().requires_grad_(True)
compiled_fwd = torch.compile(weighted_nodes4, fullgraph=True)
try:
    out_c = compiled_fwd(q_c, dest_c, h_c)
    loss = out_c.float().sum()
    loss.backward()
    print("SUCCEEDED, loss:", loss.item())
    print("q_c.grad finite:", torch.isfinite(q_c.grad).all().item())
    print("dest_c.grad finite:", torch.isfinite(dest_c.grad).all().item())
    print("h_c.grad finite:", torch.isfinite(h_c.grad).all().item())
except Exception as error:
    print("FAILED:")
    print(repr(error))
