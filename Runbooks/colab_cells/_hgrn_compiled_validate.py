"""Validate gator_linear._hgrn_compiled.compiled_chunk_hgrn against
stock fla.ops.hgrn.chunk_hgrn: forward, final_state, gradients (both
initial_state=None and a real carry tensor), bf16 finiteness, and a
fullgraph=True compile check (forward captured with zero graph breaks).
Run before swapping gator_linear/core.py's chunk_hgrn_decayed_scan over
to the compiled version.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from fla.ops.hgrn import chunk_hgrn as fla_chunk_hgrn
from gator_linear._hgrn_compiled import compiled_chunk_hgrn

torch.manual_seed(0)
B, T, D = 4, 130, 32
x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
decay = torch.rand(D, device="cuda", dtype=torch.bfloat16) * 0.9 + 0.05
g = decay.log()[None, None, :].expand(B, T, D).contiguous()
carry = torch.randn(B, D, device="cuda", dtype=torch.bfloat16) * 0.1

print("=== initial_state=None ===")
ref_o, ref_final = fla_chunk_hgrn(x, g, initial_state=None, output_final_state=True)
my_o, my_final = compiled_chunk_hgrn(x, g, initial_state=None, output_final_state=True)
print("forward matches:", torch.allclose(ref_o, my_o, atol=1e-3, rtol=1e-3))
print("final_state matches:", torch.allclose(ref_final, my_final, atol=1e-3, rtol=1e-3))

x_ref = x.detach().clone().requires_grad_(True)
g_ref = g.detach().clone().requires_grad_(True)
ref_o2, _ = fla_chunk_hgrn(x_ref, g_ref, initial_state=None, output_final_state=True)
ref_o2.sum().backward()

x_my = x.detach().clone().requires_grad_(True)
g_my = g.detach().clone().requires_grad_(True)
my_o2, my_final2 = compiled_chunk_hgrn(x_my, g_my, initial_state=None, output_final_state=True)
my_o2.sum().backward()
print("dx matches:", torch.allclose(x_ref.grad, x_my.grad, atol=1e-2, rtol=1e-2))
print("dg matches:", torch.allclose(g_ref.grad, g_my.grad, atol=1e-2, rtol=1e-2))

print()
print("=== initial_state=<real carry tensor> ===")
ref_o3, ref_final3 = fla_chunk_hgrn(x, g, initial_state=carry, output_final_state=True)
my_o3, my_final3 = compiled_chunk_hgrn(x, g, initial_state=carry, output_final_state=True)
print("forward matches:", torch.allclose(ref_o3, my_o3, atol=1e-3, rtol=1e-3))
print("final_state matches:", torch.allclose(ref_final3, my_final3, atol=1e-3, rtol=1e-3))

x_ref2 = x.detach().clone().requires_grad_(True)
g_ref2 = g.detach().clone().requires_grad_(True)
ref_o4, _ = fla_chunk_hgrn(x_ref2, g_ref2, initial_state=carry, output_final_state=True)
ref_o4.sum().backward()

x_my2 = x.detach().clone().requires_grad_(True)
g_my2 = g.detach().clone().requires_grad_(True)
my_o4, _ = compiled_chunk_hgrn(x_my2, g_my2, initial_state=carry, output_final_state=True)
my_o4.sum().backward()
print("dx matches (with carry):", torch.allclose(x_ref2.grad, x_my2.grad, atol=1e-2, rtol=1e-2))
print("dg matches (with carry):", torch.allclose(g_ref2.grad, g_my2.grad, atol=1e-2, rtol=1e-2))

print()
print("=== bf16 forward finiteness ===")
print("output finite:", torch.isfinite(my_o3).all().item())

print()
print("=== fullgraph=True compile (forward only) ===")


def wrapper(x, g, carry):
    o, final = compiled_chunk_hgrn(x, g, initial_state=carry, output_final_state=True)
    return o.float().sum() + final.float().sum()


compiled = torch.compile(wrapper, fullgraph=True)
try:
    result = compiled(x, g, carry)
    print("SUCCEEDED, result:", result.item())
except Exception as error:
    print("FAILED:")
    print(repr(error))

print()
print("=== fullgraph=True compile, real training-shaped autograd round trip ===")
x_c = x.detach().clone().requires_grad_(True)
g_c = g.detach().clone().requires_grad_(True)
compiled_fwd = torch.compile(
    lambda x, g, carry: compiled_chunk_hgrn(x, g, initial_state=carry, output_final_state=True),
    fullgraph=True,
)
try:
    o_c, final_c = compiled_fwd(x_c, g_c, carry)
    loss = o_c.float().sum() + final_c.float().sum()
    loss.backward()
    print("SUCCEEDED, loss:", loss.item())
    print("x_c.grad finite:", torch.isfinite(x_c.grad).all().item())
    print("g_c.grad finite:", torch.isfinite(g_c.grad).all().item())
except Exception as error:
    print("FAILED:")
    print(repr(error))
