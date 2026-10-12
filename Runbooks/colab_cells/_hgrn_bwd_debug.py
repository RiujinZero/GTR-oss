"""Debug: isolate whether the gradient mismatch in the triton_op-wrapped
chunk_hgrn backward is a kernel-call bug or an autograd-wiring bug, by
calling the SAME (g, o, initial_state, do) inputs through both FLA's
own ChunkHGRNFunction.backward and my chunk_hgrn_bwd_op directly,
bypassing autograd's register_autograd/setup_context machinery
entirely.
"""

import torch
import triton

from fla.ops.hgrn.chunk import (
    ChunkHGRNFunction,
    chunk_hgrn_bwd_kernel_h,
    chunk_hgrn_bwd_kernel_o,
)

torch.manual_seed(0)
B, T, D = 2, 130, 32
x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
decay = torch.rand(D, device="cuda", dtype=torch.bfloat16) * 0.9 + 0.05
g = decay.log()[None, None, :].expand(B, T, D).contiguous()

# Run FLA's own forward to get the exact (g, o, initial_state) triple
# its backward expects, plus a matching `do`.
o, final_state = ChunkHGRNFunction.apply(x, g, None, True)
do = torch.randn_like(o)

# Reference: reimplement ChunkHGRNFunction.backward's body inline
# (the @input_guard decorator wraps .backward, easier to bypass than
# fight ctx-object construction for a direct call).
def reference_backward(g, o, do, initial_state):
    B, T, D = do.shape
    BT, BD = 128, min(64, triton.next_power_of_2(D))
    num_warps = 8 if BD == 64 else 4
    gc = torch.empty_like(g, dtype=torch.float)
    dx = torch.empty_like(o, dtype=torch.float)
    def grid(meta): return (triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B)
    chunk_hgrn_bwd_kernel_h[grid](g, gc, dx, do, T=T, D=D, BT=BT)
    dg = torch.empty_like(g, dtype=torch.float)
    def grid2(meta): return (triton.cdiv(D, meta['BD']), B)
    chunk_hgrn_bwd_kernel_o[grid2](
        g, gc, o, dx, dg,
        o.stride(-3), o.stride(-2), o.stride(-1),
        T=T, D=D, BT=BT, BD=BD, num_warps=num_warps,
    )
    if initial_state is not None:
        dg[:, 0] = (initial_state * dx[:, 0] * g[:, 0].float().exp()).to(dg.dtype)
    return dx.to(o.dtype), dg


ref_dx, ref_dg = reference_backward(g, o, do, None)


@torch.library.triton_op("debug::chunk_hgrn_bwd", mutates_args={})
def chunk_hgrn_bwd_op(
    g: torch.Tensor, o: torch.Tensor, do: torch.Tensor,
    initial_state: torch.Tensor, use_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = do.shape
    BT, BD = 128, min(64, triton.next_power_of_2(D))
    num_warps = 8 if BD == 64 else 4
    gc = torch.empty_like(g, dtype=torch.float)
    dx = torch.empty_like(o, dtype=torch.float)
    def grid_h(meta): return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)
    torch.library.wrap_triton(chunk_hgrn_bwd_kernel_h)[grid_h](g, gc, dx, do, T=T, D=D, BT=BT)
    dg = torch.empty_like(g, dtype=torch.float)
    def grid_o(meta): return (triton.cdiv(D, meta["BD"]), B)
    torch.library.wrap_triton(chunk_hgrn_bwd_kernel_o)[grid_o](
        g, gc, o, dx, dg,
        o.stride(-3), o.stride(-2), o.stride(-1),
        T=T, D=D, BT=BT, BD=BD, num_warps=num_warps,
    )
    if use_initial_state:
        dg = dg.clone()
        dg[:, 0] = (initial_state * dx[:, 0] * g[:, 0].float().exp()).to(dg.dtype)
    return dx.to(o.dtype), dg


zero_state = torch.zeros(B, D, device="cuda", dtype=torch.bfloat16)
my_dx, my_dg = chunk_hgrn_bwd_op(g, o, do, zero_state, False)

print("dx matches:", torch.allclose(ref_dx, my_dx, atol=1e-3, rtol=1e-3))
print("dg matches:", torch.allclose(ref_dg, my_dg, atol=1e-3, rtol=1e-3))
print("ref_dx[0,0,:4]", ref_dx[0, 0, :4])
print("my_dx[0,0,:4] ", my_dx[0, 0, :4])
print("ref_dg[:4]", ref_dg.flatten()[:4])
print("my_dg[:4] ", my_dg.flatten()[:4])
