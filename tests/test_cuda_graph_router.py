"""Correctness/gradient checks for gator/cuda_graph_router.py.

Requires CUDA; skipped everywhere else, matching
tests/test_cuda_graph_broadcast_mix.py's pattern for GPU-only
regression tests. This is plain bf16 (not a lower-precision format), so
the correctness bar is tighter than the FP8 tests -- output should
closely match the eager reference, not just be "reasonably close."

**This module has a confirmed, unresolved bug** -- see
gator/cuda_graph_router.py's module docstring for the full writeup.
``test_multi_relay_shared_weights_gradients_not_corrupted`` and
``test_reusing_one_graph_slot_across_two_training_steps_corrupts_
gradient`` are both marked ``@unittest.expectedFailure`` for this
reason: they demonstrate the real bug rather than assert it's fixed.
Do not silently mark them passing/skip without actually fixing the
underlying issue.
"""

import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from gator.cuda_graph_router import cuda_graph_router_edges
from gator.models.dense_kernels import cell_linear


def _eager_router(h, context, occupancy, q, dense_address_edges, temperature, router_weight, router_bias, adjacency, n, backend, pad_to_multiple):
    batch, c, total_length = h.shape[0], h.shape[1], h.shape[2]
    router_input = torch.cat([h, context, occupancy], -1)
    edge_logits = cell_linear(
        router_input, router_weight, router_bias, backend=backend, pad_to_multiple=pad_to_multiple,
    )
    edge_logits = (edge_logits + dense_address_edges[None]).view(batch, c, total_length, n, n)
    adjacency_broadcast = (
        adjacency[None, None, None] if adjacency.ndim == 2 else adjacency[None, :, None]
    )
    transition = F.softmax(edge_logits.masked_fill(~adjacency_broadcast, -1e9) / temperature, -1)
    return torch.einsum("bcts,bctsd->bctd", q, transition)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaGraphRouterTests(unittest.TestCase):
    @staticmethod
    def _setup(cells=4, tokens=16, n=4, h_dim=8, ctx_dim=8, occ_dim=4, seed=0):
        torch.manual_seed(seed)
        batch = 2
        edges = n * n
        router_inputs = h_dim + ctx_dim + occ_dim
        router_weight = nn.Parameter(
            torch.randn(cells, edges, router_inputs, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        router_bias = nn.Parameter(torch.zeros(cells, edges, device="cuda", dtype=torch.bfloat16))
        adjacency = torch.ones(n, n, device="cuda", dtype=torch.bool)
        h = torch.randn(batch, cells, tokens, h_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        context = torch.randn(batch, cells, tokens, ctx_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        occupancy = torch.randn(batch, cells, tokens, occ_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q = torch.rand(batch, cells, tokens, n, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dense_address_edges = torch.randn(
            cells, tokens, edges, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        return h, context, occupancy, q, dense_address_edges, router_weight, router_bias, adjacency, n

    def test_forward_and_backward_are_finite(self):
        h, context, occupancy, q, edges, weight, bias, adjacency, n = self._setup()
        q_next = cuda_graph_router_edges(
            h, context, occupancy, q, edges, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8,
        )
        self.assertTrue(torch.isfinite(q_next).all())
        q_next.sum().backward()
        for name, tensor in [
            ("h", h), ("context", context), ("occupancy", occupancy), ("q", q),
            ("edges", edges), ("weight", weight), ("bias", bias),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")

    def test_matches_eager_reference_first_call(self):
        h, context, occupancy, q, edges, weight, bias, adjacency, n = self._setup()
        ref = _eager_router(
            h, context, occupancy, q, edges, 0.35, weight, bias, adjacency, n,
            "grouped_bmm", 8,
        )
        graphed = cuda_graph_router_edges(
            h, context, occupancy, q, edges, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8,
        )
        self.assertTrue(
            torch.allclose(graphed, ref, rtol=1e-2, atol=1e-2),
            "graphed output diverged from the eager reference on the first call",
        )

    def test_matches_eager_reference_with_different_temperature(self):
        # temperature is a scheduled value, threaded through as a tensor
        # input rather than baked in at capture time -- regression test
        # for exactly that: call once at temperature=0.35 (builds the
        # graph), then again at a different temperature, and confirm
        # the SECOND call's output reflects the NEW value, not a stale
        # baked-in constant from the first call's capture. Uses a
        # moderate second temperature (0.5), not something extreme like
        # 0.05 -- a very low temperature makes softmax near-one-hot,
        # which amplifies ordinary bf16 rounding differences between
        # graphed and eager compute well past this test's tolerance,
        # a precision artifact unrelated to the staleness bug this test
        # targets (confirmed separately: at temperature=0.05 the two
        # still clearly track each other, just with a few elements
        # exceeding a tight atol/rtol in the sharpest bins).
        h, context, occupancy, q, edges, weight, bias, adjacency, n = self._setup()
        _ = cuda_graph_router_edges(
            h, context, occupancy, q, edges, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8,
        )
        graphed_new_temp = cuda_graph_router_edges(
            h, context, occupancy, q, edges, 0.5,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8,
        )
        ref_new_temp = _eager_router(
            h, context, occupancy, q, edges, 0.5, weight, bias, adjacency, n,
            "grouped_bmm", 8,
        )
        self.assertTrue(
            torch.allclose(graphed_new_temp, ref_new_temp, rtol=1e-2, atol=1e-2),
            "graphed router output did not pick up a changed temperature on "
            "a later call -- possible stale-baked-in-constant bug",
        )

    @unittest.expectedFailure
    def test_multi_relay_shared_weights_gradients_not_corrupted(self):
        # router_weight/router_bias are genuinely shared across relays --
        # direct regression test for the multi-call-before-backward
        # gradient-corruption bug class found for node compute/receiver
        # memory/broadcast-mix. Compares against an eager reference
        # computed on deep-cloned weights so neither computation's .grad
        # accumulation can interfere with the other.
        h1, context1, occupancy1, q1, edges1, weight, bias, adjacency, n = self._setup(seed=5)
        h2, context2, occupancy2, q2, edges2, _, _, _, _ = self._setup(seed=6)

        ref_weight = nn.Parameter(weight.detach().clone())
        ref_bias = nn.Parameter(bias.detach().clone())
        ref_h1 = h1.detach().clone().requires_grad_(True)
        ref_h2 = h2.detach().clone().requires_grad_(True)

        ref_out1 = _eager_router(
            ref_h1, context1.detach(), occupancy1.detach(), q1.detach(), edges1.detach(),
            0.35, ref_weight, ref_bias, adjacency, n, "grouped_bmm", 8,
        )
        ref_out2 = _eager_router(
            ref_h2, context2.detach(), occupancy2.detach(), q2.detach(), edges2.detach(),
            0.35, ref_weight, ref_bias, adjacency, n, "grouped_bmm", 8,
        )
        (ref_out1.sum() + ref_out2.sum()).backward()

        graphed_out1 = cuda_graph_router_edges(
            h1, context1, occupancy1, q1, edges1, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8, graph_slot=0,
        )
        graphed_out2 = cuda_graph_router_edges(
            h2, context2, occupancy2, q2, edges2, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8, graph_slot=1,
        )
        (graphed_out1.sum() + graphed_out2.sum()).backward()

        self.assertTrue(torch.allclose(graphed_out1, ref_out1, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(graphed_out2, ref_out2, rtol=1e-2, atol=1e-2))
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        # NOT a bf16-precision issue -- this is the real, confirmed bug
        # documented in gator/cuda_graph_router.py's module docstring.
        # Using a relative-norm check rather than per-element allclose
        # here doesn't paper over it (the relative norm error is large,
        # not a marginal few-percent-of-elements case); it's just a
        # clearer failure signal for @unittest.expectedFailure than an
        # element-wise assertion would give.
        for name, graphed_grad, ref_grad in [
            ("weight", weight.grad, ref_weight.grad),
            ("bias", bias.grad, ref_bias.grad),
        ]:
            relative_error = (
                (graphed_grad.float() - ref_grad.float()).norm()
                / ref_grad.float().norm().clamp_min(1e-6)
            ).item()
            self.assertLess(
                relative_error, 0.05,
                f"{name}.grad relative norm error {relative_error:.4f} too "
                "large -- possible cross-relay gradient corruption",
            )

    @unittest.expectedFailure
    def test_reusing_one_graph_slot_across_two_training_steps_corrupts_gradient(self):
        # Minimal, precise repro of the real bug (see the module
        # docstring in gator/cuda_graph_router.py): a SINGLE graph_slot
        # used for one full forward+backward "training step", then used
        # AGAIN for a second, independent forward+backward "training
        # step" with fresh data -- exactly production's real usage
        # pattern (the same cached graph_slot is reused every training
        # step for the life of a run). Unlike
        # test_multi_relay_shared_weights_gradients_not_corrupted (two
        # DIFFERENT graph_slots sharing one combined backward, which
        # this module handles correctly), this is ONE graph_slot used
        # across TWO SEPARATE backward() calls -- confirmed broken
        # regardless of whether the intervening .grad is reset via
        # `= None` or `.zero_()`, and regardless of whether the two
        # calls share the same weight/bias Python objects or use
        # different ones. Forward output stays correct on both calls;
        # only the second call's gradient is wrong.
        h, context, occupancy, q, edges, weight, bias, adjacency, n = self._setup(seed=0)
        first_step_out = cuda_graph_router_edges(
            h, context, occupancy, q, edges, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8, graph_slot=0,
        )
        first_step_out.sum().backward()
        weight.grad = None
        bias.grad = None

        h2, context2, occupancy2, q2, edges2, _, _, _, _ = self._setup(seed=5)
        ref_weight = nn.Parameter(weight.detach().clone())
        ref_bias = nn.Parameter(bias.detach().clone())
        ref_h2 = h2.detach().clone().requires_grad_(True)
        ref_out = _eager_router(
            ref_h2, context2.detach(), occupancy2.detach(), q2.detach(), edges2.detach(),
            0.35, ref_weight, ref_bias, adjacency, n, "grouped_bmm", 8,
        )
        ref_out.sum().backward()

        second_step_out = cuda_graph_router_edges(
            h2, context2, occupancy2, q2, edges2, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8, graph_slot=0,
        )
        second_step_out.sum().backward()

        self.assertTrue(torch.allclose(second_step_out, ref_out, rtol=1e-2, atol=1e-2))
        relative_error = (
            (weight.grad.float() - ref_weight.grad.float()).norm()
            / ref_weight.grad.float().norm().clamp_min(1e-6)
        ).item()
        self.assertLess(
            relative_error, 0.05,
            f"weight.grad relative norm error {relative_error:.4f} on the "
            "second use of a reused graph_slot -- this is the real bug",
        )

    def test_gradients_flow_through_real_upstream_and_downstream_ops(self):
        h, context, occupancy, q, edges, weight, bias, adjacency, n = self._setup()
        upstream_weight = torch.randn(
            h.shape[-1], h.shape[-1], device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        raw_h = torch.randn_like(h, requires_grad=True)
        downstream_weight = torch.randn(
            n, n, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        h_derived = raw_h @ upstream_weight

        q_next = cuda_graph_router_edges(
            h_derived, context, occupancy, q, edges, 0.35,
            router_weight=weight, router_bias=bias, adjacency=adjacency, n=n,
            backend="grouped_bmm", pad_to_multiple=8,
        )
        loss = (q_next @ downstream_weight).sum()
        loss.backward()

        for name, tensor in [
            ("raw_h", raw_h), ("upstream_weight", upstream_weight),
            ("downstream_weight", downstream_weight), ("weight", weight),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")


if __name__ == "__main__":
    unittest.main()
