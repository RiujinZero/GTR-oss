"""Correctness/gradient checks for gator/cuda_graph_stewardship_traffic.py.

Requires CUDA; skipped everywhere else, matching
tests/test_cuda_graph_receiver_memory.py's pattern for GPU-only
regression tests. This is plain arithmetic (fp32 accumulator over
bf16 inputs, not a lower-precision format), so the correctness bar is
tighter than the FP8 tests -- output should closely match the eager
reference, not just be "reasonably close."
"""

import unittest

import torch

from gator.cuda_graph_stewardship_traffic import cuda_graph_stewardship_traffic


def _eager_traffic(node_locations, writer_mass, write, previous_numerator, previous_write, block_size):
    batch, cells, tokens, nodes = node_locations.shape
    blocks = tokens // block_size
    accumulator_dtype = previous_numerator.dtype
    token_numerator = node_locations * writer_mass.unsqueeze(-1)
    block_numerator = token_numerator.reshape(
        batch, cells, blocks, block_size, nodes
    ).sum(dim=3, dtype=accumulator_dtype).permute(0, 2, 1, 3)
    block_write = write.reshape(batch, blocks, block_size).sum(
        dim=2, dtype=accumulator_dtype
    )
    cumulative_numerator = (
        block_numerator.cumsum(dim=1) + previous_numerator.unsqueeze(1)
    )
    cumulative_write = (
        block_write.cumsum(dim=1) + previous_write.unsqueeze(1)
    ).clamp_min(1e-6)
    traffic_blocks = (cumulative_numerator / cumulative_write[:, :, None, None]).to(
        node_locations.dtype
    )
    traffic = traffic_blocks[:, :, None].expand(
        -1, -1, block_size, -1, -1
    ).reshape(batch, tokens, cells, nodes).permute(0, 2, 1, 3)
    next_numerator = previous_numerator + block_numerator.sum(dim=1)
    next_write = previous_write + block_write.sum(dim=1)
    return traffic, next_numerator, next_write


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaGraphStewardshipTrafficTests(unittest.TestCase):
    @staticmethod
    def _shapes(seed=0):
        torch.manual_seed(seed)
        batch, cells, tokens, nodes, block_size = 2, 8, 32, 4, 16
        node_locations = torch.rand(
            batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        writer_mass = torch.rand(
            batch, cells, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        write = torch.rand(
            batch, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        previous_numerator = torch.zeros(batch, cells, nodes, device="cuda", dtype=torch.float32)
        previous_write = torch.zeros(batch, device="cuda", dtype=torch.float32)
        return node_locations, writer_mass, write, previous_numerator, previous_write, block_size

    def test_forward_and_backward_are_finite(self):
        locations, writer_mass, write, prev_num, prev_write, block_size = self._shapes()
        traffic, next_num, next_write = cuda_graph_stewardship_traffic(
            locations, writer_mass, write, prev_num, prev_write,
            block_size=block_size, states_dtype=torch.bfloat16,
        )
        self.assertTrue(torch.isfinite(traffic).all())
        self.assertTrue(torch.isfinite(next_num).all())
        self.assertTrue(torch.isfinite(next_write).all())
        (traffic.sum() + next_num.sum() + next_write.sum()).backward()
        for name, tensor in [
            ("locations", locations), ("writer_mass", writer_mass), ("write", write),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")

    def test_matches_eager_reference_first_call(self):
        locations, writer_mass, write, prev_num, prev_write, block_size = self._shapes()
        ref_traffic, ref_num, ref_write = _eager_traffic(
            locations, writer_mass, write, prev_num, prev_write, block_size
        )
        graphed_traffic, graphed_num, graphed_write = cuda_graph_stewardship_traffic(
            locations, writer_mass, write, prev_num, prev_write,
            block_size=block_size, states_dtype=torch.bfloat16,
        )
        self.assertTrue(torch.allclose(graphed_traffic, ref_traffic, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(graphed_num, ref_num, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(graphed_write, ref_write, rtol=1e-2, atol=1e-2))

    def test_matches_eager_reference_across_two_sequential_relays(self):
        # Simulates two relays sharing one cache chain, the real usage
        # pattern in _dense_stewardship_context. Also the regression test
        # for the output-buffer-aliasing bug class already fixed in every
        # other CUDA-graph module in this codebase: without .clone() on
        # next_numerator/next_write, computing relay 2 (using relay 1's
        # cache output) would silently corrupt relay 1's own result in
        # place before it's consumed.
        locations1, writer_mass1, write1, cache_num0, cache_write0, block_size = self._shapes(seed=0)
        torch.manual_seed(1)
        batch, cells, tokens, nodes = locations1.shape
        locations2 = torch.rand_like(locations1)
        writer_mass2 = torch.rand_like(writer_mass1)
        write2 = torch.rand_like(write1)

        ref_ctx1, ref_num1, ref_write1 = _eager_traffic(
            locations1, writer_mass1, write1, cache_num0.clone(), cache_write0.clone(), block_size
        )
        ref_ctx2, _, _ = _eager_traffic(
            locations2, writer_mass2, write2, ref_num1, ref_write1, block_size
        )

        graphed_ctx1, graphed_num1, graphed_write1 = cuda_graph_stewardship_traffic(
            locations1, writer_mass1, write1, cache_num0.clone(), cache_write0.clone(),
            block_size=block_size, states_dtype=torch.bfloat16, graph_slot=0,
        )
        graphed_ctx2, _, _ = cuda_graph_stewardship_traffic(
            locations2, writer_mass2, write2, graphed_num1, graphed_write1,
            block_size=block_size, states_dtype=torch.bfloat16, graph_slot=1,
        )

        self.assertTrue(torch.allclose(graphed_ctx1, ref_ctx1, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(graphed_ctx2, ref_ctx2, rtol=1e-2, atol=1e-2))

    def test_gradients_flow_through_real_upstream_and_downstream_ops(self):
        torch.manual_seed(0)
        batch, cells, tokens, nodes, block_size = 2, 8, 32, 4, 16
        upstream_weight = torch.randn(
            nodes, nodes, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        raw_locations = torch.randn(
            batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        writer_mass = torch.rand(
            batch, cells, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        write = torch.rand(batch, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        prev_num = torch.zeros(batch, cells, nodes, device="cuda", dtype=torch.float32)
        prev_write = torch.zeros(batch, device="cuda", dtype=torch.float32)
        downstream_weight = torch.randn(
            nodes, nodes, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        locations = raw_locations @ upstream_weight
        traffic, next_num, next_write = cuda_graph_stewardship_traffic(
            locations, writer_mass, write, prev_num, prev_write,
            block_size=block_size, states_dtype=torch.bfloat16,
        )
        loss = (traffic @ downstream_weight).sum() + next_num.sum() + next_write.sum()
        loss.backward()

        for name, tensor in [
            ("raw_locations", raw_locations), ("upstream_weight", upstream_weight),
            ("writer_mass", writer_mass), ("write", write),
            ("downstream_weight", downstream_weight),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")


if __name__ == "__main__":
    unittest.main()
