"""Correctness/gradient checks for gator_linear.core.chunk_hgrn_decayed_scan
and its wiring into gator.models.memory.linear_scan_receiver_context via
hgrn_scan=True.

Requires CUDA and flash-linear-attention; skipped otherwise. This is
plain bf16 (not a lower-precision format), so the correctness bar is
tighter than the FP8 tests -- output should closely match the eager
decayed_scan reference, not just be "reasonably close."
"""

import unittest

import torch

from gator_linear.core import chunk_hgrn_decayed_scan, decayed_scan

try:
    import fla  # noqa: F401

    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention is not installed")
class ChunkHgrnDecayedScanTests(unittest.TestCase):
    @staticmethod
    def _shapes(seed=0, rows=4, length=130, channels=6, dtype=torch.float32):
        torch.manual_seed(seed)
        values = torch.randn(
            rows, length, channels, device="cuda", dtype=dtype, requires_grad=True
        )
        decay = torch.rand(channels, device="cuda", dtype=dtype) * 0.9 + 0.05
        return values, decay

    def test_matches_decayed_scan_forward(self):
        values, decay = self._shapes()
        ref = decayed_scan(values, decay, chunk_size=64)
        hgrn = chunk_hgrn_decayed_scan(values, decay)
        self.assertTrue(torch.allclose(ref, hgrn, rtol=1e-3, atol=1e-4))

    def test_matches_decayed_scan_final_state(self):
        values, decay = self._shapes()
        ref = decayed_scan(values, decay, chunk_size=64)
        _, hgrn_final = chunk_hgrn_decayed_scan(values, decay, return_final_state=True)
        # decayed_scan's own carry IS the last output position -- see its
        # docstring/implementation (`carry = block[:, -1]`).
        self.assertTrue(torch.allclose(ref[:, -1, :], hgrn_final, rtol=1e-3, atol=1e-4))

    def test_gradients_match_decayed_scan(self):
        values, decay = self._shapes()
        ref = decayed_scan(values, decay, chunk_size=64)
        ref.sum().backward()
        grad_ref = values.grad.clone()
        values.grad = None

        hgrn = chunk_hgrn_decayed_scan(values, decay)
        hgrn.sum().backward()
        grad_hgrn = values.grad.clone()
        self.assertTrue(torch.allclose(grad_ref, grad_hgrn, rtol=1e-3, atol=1e-4))

    def test_decay_gradients_match_decayed_scan(self):
        values1, decay1 = self._shapes()
        values1 = values1.detach().requires_grad_(True)
        decay1 = decay1.detach().requires_grad_(True)
        ref = decayed_scan(values1, decay1, chunk_size=64)
        ref.sum().backward()
        grad_decay_ref = decay1.grad.clone()

        values2, decay2 = self._shapes()
        values2 = values2.detach().requires_grad_(True)
        decay2 = decay2.detach().requires_grad_(True)
        hgrn = chunk_hgrn_decayed_scan(values2, decay2)
        hgrn.sum().backward()
        grad_decay_hgrn = decay2.grad.clone()
        self.assertTrue(
            torch.allclose(grad_decay_ref, grad_decay_hgrn, rtol=1e-2, atol=1e-3)
        )

    def test_bf16_forward_finite_and_reasonably_close(self):
        values, decay = self._shapes(dtype=torch.bfloat16)
        ref = decayed_scan(values, decay, chunk_size=64)
        hgrn = chunk_hgrn_decayed_scan(values, decay)
        self.assertTrue(torch.isfinite(ref).all())
        self.assertTrue(torch.isfinite(hgrn).all())
        self.assertTrue(torch.allclose(ref, hgrn, rtol=5e-2, atol=5e-2))

    def test_initial_state_carries_correctly(self):
        # Two sequential relays sharing a cache chain -- the real
        # block-causal usage pattern.
        values1, decay = self._shapes(seed=0)
        values2, _ = self._shapes(seed=1)
        rows, _, channels = values1.shape
        zero_state = torch.zeros(rows, channels, device="cuda", dtype=values1.dtype)

        ref1 = decayed_scan(values1, decay, chunk_size=64)
        ref2, ref2_final = decayed_scan(
            values2, decay, chunk_size=64,
            initial_state=ref1[:, -1, :], return_final_state=True,
        )

        hgrn1, hgrn1_final = chunk_hgrn_decayed_scan(
            values1, decay, initial_state=zero_state, return_final_state=True
        )
        hgrn2, hgrn2_final = chunk_hgrn_decayed_scan(
            values2, decay, initial_state=hgrn1_final, return_final_state=True
        )
        self.assertTrue(torch.allclose(ref1, hgrn1, rtol=1e-3, atol=1e-4))
        self.assertTrue(torch.allclose(ref2, hgrn2, rtol=1e-3, atol=1e-4))
        self.assertTrue(torch.allclose(ref2_final, hgrn2_final, rtol=1e-3, atol=1e-4))

    def test_rejects_wrong_decay_shape(self):
        values, _ = self._shapes()
        bad_decay = torch.rand(3, device="cuda")
        with self.assertRaises(ValueError):
            chunk_hgrn_decayed_scan(values, bad_decay)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention is not installed")
class LinearScanReceiverContextHgrnTests(unittest.TestCase):
    """Tests linear_scan_receiver_context's hgrn_scan=True dispatch,
    one level up from chunk_hgrn_decayed_scan itself."""

    @staticmethod
    def _inputs(seed=0, batch=2, cells=4, tokens=32, hidden=8, nodes=4, block_size=16):
        torch.manual_seed(seed)
        states = torch.randn(
            batch, cells, tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        node_locations = torch.rand(
            batch, cells, tokens, nodes, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        writer_mass = torch.rand(
            batch, cells, tokens, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        relative_node_bias = torch.randn(
            cells, nodes, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        return states, node_locations, writer_mass, relative_node_bias, block_size

    def test_matches_default_scan_first_call(self):
        from gator.models.memory import linear_scan_receiver_context

        states, node_locations, writer_mass, relative_node_bias, block_size = self._inputs()
        ref = linear_scan_receiver_context(
            states, node_locations, writer_mass, relative_node_bias, block_size=block_size,
        )
        hgrn = linear_scan_receiver_context(
            states, node_locations, writer_mass, relative_node_bias,
            block_size=block_size, hgrn_scan=True,
        )
        self.assertTrue(torch.allclose(ref, hgrn, rtol=1e-2, atol=1e-2))

    def test_gradients_match_default_scan(self):
        from gator.models.memory import linear_scan_receiver_context

        states1, node_locations1, writer_mass1, bias1, block_size = self._inputs()
        ref = linear_scan_receiver_context(
            states1, node_locations1, writer_mass1, bias1, block_size=block_size,
        )
        ref.sum().backward()
        grad_ref = states1.grad.clone()

        states2, node_locations2, writer_mass2, bias2, _ = self._inputs()
        hgrn = linear_scan_receiver_context(
            states2, node_locations2, writer_mass2, bias2,
            block_size=block_size, hgrn_scan=True,
        )
        hgrn.sum().backward()
        grad_hgrn = states2.grad.clone()
        self.assertTrue(torch.allclose(grad_ref, grad_hgrn, rtol=5e-2, atol=5e-2))


if __name__ == "__main__":
    unittest.main()
