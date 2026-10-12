"""Validate gator_linear._hgrn_fixed_decay.fixed_decay_chunk_hgrn
against stock fla.ops.hgrn.chunk_hgrn called with a broadcast-expanded
gate (the mathematically equivalent general-kernel path): forward,
final_state, gradient w.r.t. x, and the REDUCED gradient w.r.t.
log_decay (comparing against the general kernel's dg summed over batch
and time, which is what PyTorch's own .expand().backward() would
compute for the Phase 1.5 chunk_hgrn_decayed_scan path). Both
initial_state=None and a real carry tensor. Then a fullgraph=True
compile check.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from fla.ops.hgrn import chunk_hgrn as fla_chunk_hgrn
from gator_linear._hgrn_fixed_decay import fixed_decay_chunk_hgrn

torch.manual_seed(0)
B, T, D = 4, 130, 32
x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
log_decay = (torch.rand(D, device="cuda", dtype=torch.bfloat16) * 0.9 + 0.05).log()
carry = torch.randn(B, D, device="cuda", dtype=torch.bfloat16) * 0.1


def general_kernel_ref(x, log_decay, initial_state):
    g = log_decay[None, None, :].expand(B, T, D).contiguous().requires_grad_(True)
    x_ = x.detach().clone().requires_grad_(True)
    o, final = fla_chunk_hgrn(x_, g, initial_state=initial_state, output_final_state=True)
    return x_, g, o, final


for label, init_state in [("initial_state=None", None), ("initial_state=<carry>", carry)]:
    print(f"\n=== {label} ===")

    ref_o, ref_final = fla_chunk_hgrn(
        x, log_decay[None, None, :].expand(B, T, D).contiguous(),
        initial_state=init_state, output_final_state=True,
    )
    my_o, my_final = fixed_decay_chunk_hgrn(x, log_decay, initial_state=init_state, output_final_state=True)
    print("forward matches:", torch.allclose(ref_o, my_o, atol=1e-3, rtol=1e-3))
    print("final_state matches:", torch.allclose(ref_final, my_final, atol=1e-3, rtol=1e-3))

    x_ref, g_ref, ref_o2, _ = general_kernel_ref(x, log_decay, init_state)
    ref_o2.sum().backward()
    ref_dx = x_ref.grad
    ref_dlog_decay = g_ref.grad.float().sum(dim=(0, 1))  # what .expand()'s own backward would produce

    x_my = x.detach().clone().requires_grad_(True)
    log_decay_my = log_decay.detach().clone().requires_grad_(True)
    my_o2, _ = fixed_decay_chunk_hgrn(x_my, log_decay_my, initial_state=init_state, output_final_state=True)
    my_o2.sum().backward()

    print("dx matches:", torch.allclose(ref_dx, x_my.grad, atol=1e-2, rtol=1e-2))
    print(
        "d(log_decay) matches:",
        torch.allclose(ref_dlog_decay, log_decay_my.grad.float(), atol=2e-1, rtol=2e-2),
        "  max abs diff:", (ref_dlog_decay - log_decay_my.grad.float()).abs().max().item(),
    )

print("\n=== bf16 forward finiteness ===")
print("output finite:", torch.isfinite(my_o).all().item())

print("\n=== fullgraph=True compile (forward only) ===")


def wrapper(x, log_decay, carry):
    o, final = fixed_decay_chunk_hgrn(x, log_decay, initial_state=carry, output_final_state=True)
    return o.float().sum() + final.float().sum()


compiled = torch.compile(wrapper, fullgraph=True)
try:
    result = compiled(x, log_decay, carry)
    print("SUCCEEDED, result:", result.item())
except Exception as error:
    print("FAILED:")
    print(repr(error))

print("\n=== fullgraph=True compile, real training-shaped autograd round trip ===")
x_c = x.detach().clone().requires_grad_(True)
log_decay_c = log_decay.detach().clone().requires_grad_(True)
compiled_fwd = torch.compile(
    lambda x, log_decay, carry: fixed_decay_chunk_hgrn(x, log_decay, initial_state=carry, output_final_state=True),
    fullgraph=True,
)
try:
    o_c, final_c = compiled_fwd(x_c, log_decay_c, carry)
    loss = o_c.float().sum() + final_c.float().sum()
    loss.backward()
    print("SUCCEEDED, loss:", loss.item())
    print("x_c.grad finite:", torch.isfinite(x_c.grad).all().item())
    print("log_decay_c.grad finite:", torch.isfinite(log_decay_c.grad).all().item())
except Exception as error:
    print("FAILED:")
    print(repr(error))
