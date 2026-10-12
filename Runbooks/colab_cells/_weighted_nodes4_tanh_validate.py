"""Validate gator.weighted_nodes4.weighted_nodes4_tanh (tanh fused
into the kernel, operating on node_affine directly) against the
reference it replaces: torch.tanh(node_affine) then broadcast-sum
[+ h]. Forward, gradients w.r.t. q/node_affine/h, both with and
without residual, bf16 finiteness, fullgraph=True compile.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.weighted_nodes4 import weighted_nodes4_tanh

torch.manual_seed(0)
B, C, T, N, D = 2, 4, 16, 4, 32
q = torch.randn(B, C, T, N, device="cuda", dtype=torch.bfloat16)
node_affine = torch.randn(B, C, T, N, D, device="cuda", dtype=torch.bfloat16) * 2.0
h = torch.randn(B, C, T, D, device="cuda", dtype=torch.bfloat16)


def reference(q, node_affine, h=None):
    destination = torch.tanh(node_affine)
    mixed = (q.unsqueeze(-1) * destination).sum(-2)
    return mixed + h if h is not None else mixed


for label, use_h in [("no residual", False), ("with residual", True)]:
    print(f"\n=== {label} ===")
    h_arg = h if use_h else None

    ref = reference(q, node_affine, h_arg)
    my = weighted_nodes4_tanh(q, node_affine, h_arg)
    print("forward matches:", torch.allclose(ref, my, atol=5e-2, rtol=5e-2))

    q_ref = q.detach().clone().requires_grad_(True)
    affine_ref = node_affine.detach().clone().requires_grad_(True)
    h_ref = h.detach().clone().requires_grad_(True) if use_h else None
    ref2 = reference(q_ref, affine_ref, h_ref)
    ref2.sum().backward()

    q_my = q.detach().clone().requires_grad_(True)
    affine_my = node_affine.detach().clone().requires_grad_(True)
    h_my = h.detach().clone().requires_grad_(True) if use_h else None
    my2 = weighted_nodes4_tanh(q_my, affine_my, h_my)
    my2.sum().backward()

    print("dq matches:", torch.allclose(q_ref.grad, q_my.grad, atol=1e-1, rtol=5e-2))
    print("d_node_affine matches:", torch.allclose(affine_ref.grad, affine_my.grad, atol=5e-2, rtol=5e-2))
    if use_h:
        print("dh matches:", torch.allclose(h_ref.grad, h_my.grad, atol=1e-2, rtol=1e-2))

print("\n=== bf16 forward finiteness ===")
print("output finite:", torch.isfinite(my).all().item())

print("\n=== fullgraph=True compile (forward only, with residual) ===")


def wrapper(q, node_affine, h):
    return weighted_nodes4_tanh(q, node_affine, h).float().sum()


compiled = torch.compile(wrapper, fullgraph=True)
try:
    result = compiled(q, node_affine, h)
    print("SUCCEEDED, result:", result.item())
except Exception as error:
    print("FAILED:")
    print(repr(error))

print("\n=== fullgraph=True compile, real training-shaped autograd round trip ===")
q_c = q.detach().clone().requires_grad_(True)
affine_c = node_affine.detach().clone().requires_grad_(True)
h_c = h.detach().clone().requires_grad_(True)
compiled_fwd = torch.compile(weighted_nodes4_tanh, fullgraph=True)
try:
    out_c = compiled_fwd(q_c, affine_c, h_c)
    loss = out_c.float().sum()
    loss.backward()
    print("SUCCEEDED, loss:", loss.item())
    print("q_c.grad finite:", torch.isfinite(q_c.grad).all().item())
    print("affine_c.grad finite:", torch.isfinite(affine_c.grad).all().item())
    print("h_c.grad finite:", torch.isfinite(h_c.grad).all().item())
except Exception as error:
    print("FAILED:")
    print(repr(error))
