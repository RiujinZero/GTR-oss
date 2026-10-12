"""Correctness/gradient checks for gator/cuda_graph_broadcast_mix.py.

Requires CUDA; skipped everywhere else, matching
tests/test_cuda_graph_receiver_memory.py's pattern for GPU-only
regression tests. This is plain bf16 (not a lower-precision format), so
the correctness bar is tighter than the FP8 tests -- output should
closely match the eager reference, not just be "reasonably close."
"""

import copy
import unittest

import torch
import torch.nn as nn

from gator.cuda_graph_broadcast_mix import cuda_graph_broadcast_mix


def _eager_broadcast_mix(local, left_projection, right_projection, coupling_norm, neighbor_gates, self_gates, radius):
    left = torch.stack(
        [torch.roll(local, distance, 1) for distance in range(1, radius + 1)]
    ).mean(0)
    right = torch.stack(
        [torch.roll(local, -distance, 1) for distance in range(1, radius + 1)]
    ).mean(0)
    gates = torch.sigmoid(neighbor_gates)
    mixed = self_gates[None, :, None, None] * local
    mixed = mixed + gates[None, :, 0, None, None] * left_projection(left)
    mixed = mixed + gates[None, :, 1, None, None] * right_projection(right)
    return coupling_norm(mixed)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaGraphBroadcastMixTests(unittest.TestCase):
    @staticmethod
    def _modules(cells, d_model):
        torch.manual_seed(0)
        left_projection = nn.Linear(d_model, d_model, device="cuda", dtype=torch.bfloat16)
        right_projection = nn.Linear(d_model, d_model, device="cuda", dtype=torch.bfloat16)
        coupling_norm = nn.LayerNorm(d_model, device="cuda", dtype=torch.bfloat16)
        neighbor_gates = nn.Parameter(
            torch.randn(cells, 2, device="cuda", dtype=torch.bfloat16)
        )
        self_gates = nn.Parameter(
            torch.randn(cells, device="cuda", dtype=torch.bfloat16)
        )
        return left_projection, right_projection, coupling_norm, neighbor_gates, self_gates

    @staticmethod
    def _local(batch, cells, tokens, d_model):
        return torch.randn(
            batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )

    def test_forward_and_backward_are_finite(self):
        batch, cells, tokens, d_model, radius = 2, 8, 16, 48, 2
        left, right, norm, neighbor_gates, self_gates = self._modules(cells, d_model)
        local = self._local(batch, cells, tokens, d_model)
        output = cuda_graph_broadcast_mix(
            local,
            left_projection=left, right_projection=right, coupling_norm=norm,
            neighbor_gates=neighbor_gates, self_gates=self_gates, radius=radius,
        )
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        for name, tensor in [
            ("local", local), ("left.weight", left.weight), ("left.bias", left.bias),
            ("right.weight", right.weight), ("right.bias", right.bias),
            ("norm.weight", norm.weight), ("norm.bias", norm.bias),
            ("neighbor_gates", neighbor_gates), ("self_gates", self_gates),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")

    def test_matches_eager_reference_first_call(self):
        batch, cells, tokens, d_model, radius = 2, 8, 16, 48, 2
        left, right, norm, neighbor_gates, self_gates = self._modules(cells, d_model)
        local = self._local(batch, cells, tokens, d_model)

        ref = _eager_broadcast_mix(local, left, right, norm, neighbor_gates, self_gates, radius)
        graphed = cuda_graph_broadcast_mix(
            local,
            left_projection=left, right_projection=right, coupling_norm=norm,
            neighbor_gates=neighbor_gates, self_gates=self_gates, radius=radius,
        )
        self.assertTrue(
            torch.allclose(graphed, ref, rtol=1e-2, atol=1e-2),
            "graphed output diverged from the eager reference on the first call",
        )

    def test_multi_relay_shared_params_gradients_not_corrupted(self):
        # Direct regression test for the class of bug found and fixed for
        # node compute/receiver memory: a single graphed instance shared
        # across more than one call before a shared backward() silently
        # corrupts gradients. Here left_projection/right_projection/
        # coupling_norm/neighbor_gates/self_gates are genuinely SHARED
        # across relays (same nn.Module instances, unlike node compute's
        # per-relay-refolded weight) -- graph_slot=relay is what keeps
        # each relay's graphed instance separate. Compares against an
        # eager reference computed on deep-cloned modules so neither
        # computation's .grad accumulation can interfere with the other.
        batch, cells, tokens, d_model, radius = 2, 8, 16, 48, 2
        left, right, norm, neighbor_gates, self_gates = self._modules(cells, d_model)
        ref_left = copy.deepcopy(left)
        ref_right = copy.deepcopy(right)
        ref_norm = copy.deepcopy(norm)
        ref_neighbor_gates = nn.Parameter(neighbor_gates.detach().clone())
        ref_self_gates = nn.Parameter(self_gates.detach().clone())

        torch.manual_seed(5)
        local1 = self._local(batch, cells, tokens, d_model)
        torch.manual_seed(6)
        local2 = self._local(batch, cells, tokens, d_model)
        ref_local1 = local1.detach().clone().requires_grad_(True)
        ref_local2 = local2.detach().clone().requires_grad_(True)

        ref_out1 = _eager_broadcast_mix(
            ref_local1, ref_left, ref_right, ref_norm, ref_neighbor_gates, ref_self_gates, radius
        )
        ref_out2 = _eager_broadcast_mix(
            ref_local2, ref_left, ref_right, ref_norm, ref_neighbor_gates, ref_self_gates, radius
        )
        (ref_out1.sum() + ref_out2.sum()).backward()

        graphed_out1 = cuda_graph_broadcast_mix(
            local1,
            left_projection=left, right_projection=right, coupling_norm=norm,
            neighbor_gates=neighbor_gates, self_gates=self_gates, radius=radius,
            graph_slot=0,
        )
        graphed_out2 = cuda_graph_broadcast_mix(
            local2,
            left_projection=left, right_projection=right, coupling_norm=norm,
            neighbor_gates=neighbor_gates, self_gates=self_gates, radius=radius,
            graph_slot=1,
        )
        (graphed_out1.sum() + graphed_out2.sum()).backward()

        self.assertTrue(torch.allclose(graphed_out1, ref_out1, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(graphed_out2, ref_out2, rtol=1e-2, atol=1e-2))
        for name, graphed_param, ref_param in [
            ("left.weight", left.weight, ref_left.weight),
            ("left.bias", left.bias, ref_left.bias),
            ("right.weight", right.weight, ref_right.weight),
            ("right.bias", right.bias, ref_right.bias),
            ("norm.weight", norm.weight, ref_norm.weight),
            ("norm.bias", norm.bias, ref_norm.bias),
            ("neighbor_gates", neighbor_gates, ref_neighbor_gates),
            ("self_gates", self_gates, ref_self_gates),
        ]:
            self.assertIsNotNone(graphed_param.grad, f"{name}.grad is None")
            self.assertTrue(
                torch.allclose(graphed_param.grad, ref_param.grad, rtol=1e-2, atol=1e-2),
                f"gradient for {name} diverged between eager and graphed "
                "broadcast-mix across two relays sharing parameters -- "
                "possible cross-relay gradient corruption",
            )

    def test_gradients_flow_through_real_upstream_and_downstream_ops(self):
        batch, cells, tokens, d_model, radius = 2, 8, 16, 48, 2
        left, right, norm, neighbor_gates, self_gates = self._modules(cells, d_model)
        upstream_weight = torch.randn(
            d_model, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        downstream_weight = torch.randn(
            d_model, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        raw_local = torch.randn(
            batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        local = raw_local @ upstream_weight

        output = cuda_graph_broadcast_mix(
            local,
            left_projection=left, right_projection=right, coupling_norm=norm,
            neighbor_gates=neighbor_gates, self_gates=self_gates, radius=radius,
        )
        loss = (output @ downstream_weight).sum()
        loss.backward()

        for name, tensor in [
            ("raw_local", raw_local), ("upstream_weight", upstream_weight),
            ("downstream_weight", downstream_weight),
            ("left.weight", left.weight), ("neighbor_gates", neighbor_gates),
            ("self_gates", self_gates),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad not finite")


if __name__ == "__main__":
    unittest.main()
