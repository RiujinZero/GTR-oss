"""Correctness/gradient checks for gator/cuda_graph_receiver_memory.py.

Requires CUDA; skipped everywhere else, matching
tests/test_mx_precision.py's pattern for GPU-only regression tests. This
is plain bf16 (not a lower-precision format), so the correctness bar is
tighter than the FP8 tests -- output should closely match the eager
linear_scan_receiver_context reference, not just be "reasonably close."
"""

import unittest

import torch

from gator.cuda_graph_receiver_memory import cuda_graph_linear_scan_receiver_context
from gator.models.memory import LinearScanReceiverCache, linear_scan_receiver_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaGraphReceiverMemoryTests(unittest.TestCase):
    @staticmethod
    def _shapes():
        torch.manual_seed(0)
        batch, cells, tokens, hidden, nodes = 2, 8, 32, 48, 4
        block_size = 16
        states = torch.randn(
            batch, cells, tokens, hidden, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        node_locations = torch.rand(
            batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        writer_mass = torch.rand(
            batch, cells, tokens, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        relative_node_bias = torch.randn(
            cells, nodes, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        rows = batch * (hidden + 1)
        channels = cells * nodes
        cache_forward_state = torch.zeros(
            rows, channels, device="cuda", dtype=torch.bfloat16
        )
        return (
            states, node_locations, writer_mass, relative_node_bias,
            cache_forward_state, block_size,
        )

    def test_forward_and_backward_are_finite(self):
        states, locations, writer_mass, bias, cache, block_size = self._shapes()
        context, next_cache = cuda_graph_linear_scan_receiver_context(
            states, locations, writer_mass, bias,
            block_size=block_size, cache_forward_state=cache,
        )
        self.assertTrue(torch.isfinite(context).all())
        self.assertTrue(torch.isfinite(next_cache).all())
        (context.sum() + next_cache.sum()).backward()
        for name, tensor in [
            ("states", states), ("locations", locations),
            ("writer_mass", writer_mass), ("bias", bias),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")

    def test_matches_eager_reference_first_call(self):
        states, locations, writer_mass, bias, cache, block_size = self._shapes()
        ref_context, ref_next_cache = linear_scan_receiver_context(
            states, locations, writer_mass, bias,
            block_size=block_size,
            cache=LinearScanReceiverCache(cache),
            return_cache=True,
        )
        graphed_context, graphed_next_cache = cuda_graph_linear_scan_receiver_context(
            states, locations, writer_mass, bias,
            block_size=block_size, cache_forward_state=cache,
        )
        self.assertTrue(
            torch.allclose(graphed_context, ref_context, rtol=1e-2, atol=1e-2),
            "graphed output diverged from the eager reference on the first call",
        )
        self.assertTrue(
            torch.allclose(
                graphed_next_cache, ref_next_cache.forward_state, rtol=1e-2, atol=1e-2
            ),
            "graphed next-cache diverged from the eager reference on the first call",
        )

    def test_matches_eager_reference_across_two_sequential_calls(self):
        # Simulates two relays: cache from call 1 feeds call 2, matching
        # the real block-causal usage pattern in _dense_stewardship_context.
        #
        # This is also the regression test for a real bug this
        # investigation found: make_graphed_callables' output is a live
        # view into the CUDA graph's static output buffer, silently
        # overwritten by the NEXT call to the same graphed callable --
        # this test failed before cuda_graph_linear_scan_receiver_context
        # added an explicit .clone() on its outputs, because computing
        # call 2 (using graphed_cache1) before asserting on call 1's
        # result silently corrupted graphed_ctx1's memory in place.
        # Without the fix, real multi-relay block-causal training would
        # silently produce wrong (not crashing) receiver memory.
        states1, locations1, writer_mass1, bias, cache0, block_size = self._shapes()
        torch.manual_seed(1)
        states2 = torch.randn_like(states1)
        locations2 = torch.rand_like(locations1)
        writer_mass2 = torch.rand_like(writer_mass1)

        ref_ctx1, ref_cache1 = linear_scan_receiver_context(
            states1, locations1, writer_mass1, bias,
            block_size=block_size, cache=LinearScanReceiverCache(cache0.clone()),
            return_cache=True,
        )
        ref_ctx2, _ = linear_scan_receiver_context(
            states2, locations2, writer_mass2, bias,
            block_size=block_size, cache=ref_cache1, return_cache=True,
        )

        graphed_ctx1, graphed_cache1 = cuda_graph_linear_scan_receiver_context(
            states1, locations1, writer_mass1, bias,
            block_size=block_size, cache_forward_state=cache0.clone(),
        )
        graphed_ctx2, _ = cuda_graph_linear_scan_receiver_context(
            states2, locations2, writer_mass2, bias,
            block_size=block_size, cache_forward_state=graphed_cache1,
        )

        self.assertTrue(torch.allclose(graphed_ctx1, ref_ctx1, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(graphed_ctx2, ref_ctx2, rtol=1e-2, atol=1e-2))

    def test_gradients_flow_through_real_upstream_and_downstream_ops(self):
        torch.manual_seed(0)
        cells, nodes, hidden = 8, 4, 48
        batch, tokens, block_size = 2, 32, 16
        upstream_weight = torch.randn(
            hidden, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        raw_states = torch.randn(
            batch, cells, tokens, hidden, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        locations = torch.rand(
            batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        writer_mass = torch.rand(
            batch, cells, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        bias = torch.randn(
            cells, nodes, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        rows = batch * (hidden + 1)
        channels = cells * nodes
        cache0 = torch.zeros(rows, channels, device="cuda", dtype=torch.bfloat16)
        downstream_weight = torch.randn(
            hidden, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        states = raw_states @ upstream_weight
        context, next_cache = cuda_graph_linear_scan_receiver_context(
            states, locations, writer_mass, bias,
            block_size=block_size, cache_forward_state=cache0,
        )
        loss = (context @ downstream_weight).sum() + next_cache.sum()
        loss.backward()

        for name, tensor in [
            ("raw_states", raw_states), ("upstream_weight", upstream_weight),
            ("locations", locations), ("writer_mass", writer_mass),
            ("bias", bias), ("downstream_weight", downstream_weight),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")


if __name__ == "__main__":
    unittest.main()
