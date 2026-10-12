"""Phase 1.5: make chunk_hgrn compile-friendly.

fla.ops.hgrn.chunk.chunk_hgrn is decorated @torch.compiler.disable,
forcing a graph break every time it's called from inside a
torch.compile'd region -- confirmed as the cause of the production
300-step run showing no net win despite chunk_hgrn's isolated 5.48x
kernel speedup (Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md).

ChunkHGRNFunction's forward/backward are each just two Triton kernel
launches plus some torch.empty_like/slice/clone/cast glue (verified by
reading fla.ops.hgrn.chunk.ChunkHGRNFunction's source directly) -- no
complex Python control flow, so wrapping the SAME kernel calls as a
torch.library.triton_op (with an explicit register_autograd backward)
should let torch.compile treat this as one opaque node inside the
compiled graph, instead of a Dynamo graph-break-and-reenter boundary.

This step: forward-only correctness against fla's own chunk_hgrn, then
a fullgraph=True trace check (turns any remaining graph break into a
hard error instead of silently fragmenting).
"""

import torch
import triton

from fla.ops.hgrn.chunk import (
    chunk_hgrn_bwd_kernel_h,
    chunk_hgrn_bwd_kernel_o,
    chunk_hgrn_fwd_kernel_h,
    chunk_hgrn_fwd_kernel_o,
)
from fla.ops.hgrn import chunk_hgrn as fla_chunk_hgrn


@torch.library.triton_op("gator::chunk_hgrn_fwd", mutates_args={})
def chunk_hgrn_fwd_op(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    use_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    BT, BD = 128, min(64, triton.next_power_of_2(D))
    num_warps = 8 if BD == 64 else 4

    gc = torch.empty_like(g, dtype=torch.float)
    o = torch.empty_like(x, dtype=torch.float)

    def grid_h(meta):
        return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

    torch.library.wrap_triton(chunk_hgrn_fwd_kernel_h)[grid_h](
        x, g, gc, o, initial_state,
        T=T, D=D, BT=BT,
        USE_INITIAL_STATE=use_initial_state,
    )

    def grid_o(meta):
        return (triton.cdiv(D, meta["BD"]), B)

    torch.library.wrap_triton(chunk_hgrn_fwd_kernel_o)[grid_o](
        gc, o,
        o.stride(-3), o.stride(-2), o.stride(-1),
        T=T, D=D, BT=BT, BD=BD,
        num_warps=num_warps,
    )
    final_state = o[:, -1].clone()
    o = o.to(x.dtype)
    return o, final_state


@chunk_hgrn_fwd_op.register_fake
def _(x, g, initial_state, use_initial_state):
    B, T, D = x.shape
    return torch.empty_like(x), torch.empty(B, D, device=x.device, dtype=x.dtype)


@torch.library.triton_op("gator::chunk_hgrn_bwd", mutates_args={})
def chunk_hgrn_bwd_op(
    g: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    initial_state: torch.Tensor,
    use_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = do.shape
    BT, BD = 128, min(64, triton.next_power_of_2(D))
    num_warps = 8 if BD == 64 else 4

    gc = torch.empty_like(g, dtype=torch.float)
    dx = torch.empty_like(o, dtype=torch.float)

    def grid_h(meta):
        return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

    torch.library.wrap_triton(chunk_hgrn_bwd_kernel_h)[grid_h](
        g, gc, dx, do, T=T, D=D, BT=BT,
    )

    dg = torch.empty_like(g, dtype=torch.float)

    def grid_o(meta):
        return (triton.cdiv(D, meta["BD"]), B)

    torch.library.wrap_triton(chunk_hgrn_bwd_kernel_o)[grid_o](
        g, gc, o, dx, dg,
        o.stride(-3), o.stride(-2), o.stride(-1),
        T=T, D=D, BT=BT, BD=BD,
        num_warps=num_warps,
    )
    if use_initial_state:
        dg = dg.clone()
        dg[:, 0] = (initial_state * dx[:, 0] * g[:, 0].float().exp()).to(dg.dtype)
    return dx.to(o.dtype), dg


@chunk_hgrn_bwd_op.register_fake
def _(g, o, do, initial_state, use_initial_state):
    return torch.empty_like(o), torch.empty_like(g)


def _setup_context(ctx, inputs, output):
    x, g, initial_state, use_initial_state = inputs
    o, final_state = output
    ctx.save_for_backward(g, o, initial_state)
    ctx.use_initial_state = use_initial_state


def _backward(ctx, do, dfinal_state):
    # Root cause (confirmed): autograd may hand backward an EXPANDED
    # stride-0 gradient (e.g. from `y.sum().backward()`, conceptually
    # "every element is 1" without materializing memory). The HGRN
    # backward kernel does flat pointer arithmetic assuming real
    # per-element strides, so a stride-0 `do` produces garbage/NaN --
    # exactly what FLA's own @input_guard decorator silently prevents
    # by calling .contiguous() on every arg before ChunkHGRNFunction
    # ever runs. register_autograd has no equivalent guard, so it must
    # be done explicitly here.
    g, o, initial_state = ctx.saved_tensors
    do = do.contiguous()
    dx, dg = chunk_hgrn_bwd_op(g, o, do, initial_state, ctx.use_initial_state)
    return dx, dg, None, None


torch.library.register_autograd(
    "gator::chunk_hgrn_fwd", _backward, setup_context=_setup_context,
)


# --- Correctness: matches fla's own chunk_hgrn forward ---
torch.manual_seed(0)
B, T, D = 2, 130, 32
x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
decay = torch.rand(D, device="cuda", dtype=torch.bfloat16) * 0.9 + 0.05
g = decay.log()[None, None, :].expand(B, T, D).contiguous()
zero_state = torch.zeros(B, D, device="cuda", dtype=torch.bfloat16)

ref_o, ref_final = fla_chunk_hgrn(x, g, initial_state=None, output_final_state=True)
my_o, my_final = chunk_hgrn_fwd_op(x, g, zero_state, False)

print("forward matches fla chunk_hgrn:", torch.allclose(ref_o, my_o, atol=1e-3, rtol=1e-3))
print("final_state matches:", torch.allclose(ref_final, my_final, atol=1e-3, rtol=1e-3))

# --- fullgraph=True trace check ---
def wrapper(x, g, state):
    o, final = chunk_hgrn_fwd_op(x, g, state, False)
    return o.sum() + final.sum()


compiled = torch.compile(wrapper, fullgraph=True)
try:
    result = compiled(x, g, zero_state)
    print("fullgraph=True compile SUCCEEDED (forward only), result:", result.item())
except Exception as error:
    print("fullgraph=True compile FAILED (forward only):")
    print(repr(error))

# --- Gradient correctness vs fla's own ChunkHGRNFunction ---
x_ref = x.detach().clone().requires_grad_(True)
g_ref = g.detach().clone().requires_grad_(True)
ref_o2, _ = fla_chunk_hgrn(x_ref, g_ref, initial_state=None, output_final_state=True)
ref_o2.sum().backward()

x_my = x.detach().clone().requires_grad_(True)
g_my = g.detach().clone().requires_grad_(True)
my_o2, my_final2 = chunk_hgrn_fwd_op(x_my, g_my, zero_state, False)
my_o2.sum().backward()

print("dx matches:", torch.allclose(x_ref.grad, x_my.grad, atol=1e-2, rtol=1e-2))
print("dg matches:", torch.allclose(g_ref.grad, g_my.grad, atol=1e-2, rtol=1e-2))

# --- fullgraph=True compile of forward; .backward() called OUTSIDE
# the compiled region (Dynamo doesn't support tracing .backward()
# itself regardless of the op -- this is the correct usage pattern,
# not a workaround for the custom op specifically).
def compiled_forward(x, g, state):
    o, final = chunk_hgrn_fwd_op(x, g, state, False)
    return o.float().sum() + final.float().sum()


x_c = x.detach().clone().requires_grad_(True)
g_c = g.detach().clone().requires_grad_(True)
compiled_forward_fn = torch.compile(compiled_forward, fullgraph=True)
try:
    loss = compiled_forward_fn(x_c, g_c, zero_state)
    loss.backward()
    print("fullgraph=True compile SUCCEEDED (forward compiled, backward outside), loss:", loss.item())
    print("x_c.grad is not None and finite:", x_c.grad is not None and torch.isfinite(x_c.grad).all().item())
    print("g_c.grad is not None and finite:", g_c.grad is not None and torch.isfinite(g_c.grad).all().item())
except Exception as error:
    print("fullgraph=True compile FAILED (forward compiled, backward outside):")
    print(repr(error))
