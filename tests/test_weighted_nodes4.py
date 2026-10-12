"""Correctness checks for gator.weighted_nodes4.weighted_nodes4
against the reference formula it was written to replace:
``(q.unsqueeze(-1) * destination).sum(-2) [+ h]``.

Confirmed NOT a production win under torch.compile (see
Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md) -- kept as a
documented negative result, correctness still matters for that
conclusion to mean anything.

Requires CUDA and triton; skipped otherwise.
"""

import unittest

import torch

try:
    import triton  # noqa: F401

    from gator.weighted_nodes4 import weighted_nodes4

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TRITON_AVAILABLE, "triton is not installed")
class WeightedNodes4Tests(unittest.TestCase):
    @staticmethod
    def _shapes(seed=0, B=2, C=4, T=16, N=4, D=32, dtype=torch.float32):
        torch.manual_seed(seed)
        q = torch.randn(B, C, T, N, device="cuda", dtype=dtype, requires_grad=True)
        destination = torch.randn(B, C, T, N, D, device="cuda", dtype=dtype, requires_grad=True)
        h = torch.randn(B, C, T, D, device="cuda", dtype=dtype, requires_grad=True)
        return q, destination, h

    @staticmethod
    def _reference(q, destination, h=None):
        mixed = (q.unsqueeze(-1) * destination).sum(-2)
        return mixed + h if h is not None else mixed

    def test_forward_matches_reference_no_residual(self):
        q, destination, _ = self._shapes()
        ref = self._reference(q, destination)
        my = weighted_nodes4(q, destination)
        self.assertTrue(torch.allclose(ref, my, rtol=1e-3, atol=1e-4))

    def test_forward_matches_reference_with_residual(self):
        q, destination, h = self._shapes()
        ref = self._reference(q, destination, h)
        my = weighted_nodes4(q, destination, h)
        self.assertTrue(torch.allclose(ref, my, rtol=1e-3, atol=1e-4))

    def test_gradients_match_reference_no_residual(self):
        q, destination, _ = self._shapes()
        ref = self._reference(q, destination)
        ref.sum().backward()
        dq_ref, ddest_ref = q.grad.clone(), destination.grad.clone()

        q2, destination2, _ = self._shapes()
        my = weighted_nodes4(q2, destination2)
        my.sum().backward()

        self.assertTrue(torch.allclose(dq_ref, q2.grad, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(ddest_ref, destination2.grad, rtol=1e-3, atol=1e-4))

    def test_gradients_match_reference_with_residual(self):
        q, destination, h = self._shapes()
        ref = self._reference(q, destination, h)
        ref.sum().backward()
        dq_ref, ddest_ref, dh_ref = q.grad.clone(), destination.grad.clone(), h.grad.clone()

        q2, destination2, h2 = self._shapes()
        my = weighted_nodes4(q2, destination2, h2)
        my.sum().backward()

        self.assertTrue(torch.allclose(dq_ref, q2.grad, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(ddest_ref, destination2.grad, rtol=1e-3, atol=1e-4))
        self.assertTrue(torch.allclose(dh_ref, h2.grad, rtol=1e-3, atol=1e-4))

    def test_bf16_forward_finite(self):
        q, destination, h = self._shapes(dtype=torch.bfloat16)
        out = weighted_nodes4(q, destination, h)
        self.assertTrue(torch.isfinite(out).all())

    def test_rejects_wrong_last_dim(self):
        q = torch.rand(2, 4, 16, 3, device="cuda")  # not 4
        destination = torch.rand(2, 4, 16, 3, 32, device="cuda")
        with self.assertRaises(ValueError):
            weighted_nodes4(q, destination)

    def test_fullgraph_compile_forward(self):
        q, destination, h = self._shapes(dtype=torch.bfloat16)

        def wrapper(q, destination, h):
            return weighted_nodes4(q, destination, h).float().sum()

        compiled = torch.compile(wrapper, fullgraph=True)
        result = compiled(q, destination, h)
        self.assertTrue(torch.isfinite(result))


if __name__ == "__main__":
    unittest.main()
