"""Correctness checks for gator_linear._hgrn_fixed_decay.fixed_decay_chunk_hgrn
(Phase 2) directly against the general kernel
(gator_linear._hgrn_compiled.compiled_chunk_hgrn / stock FLA
chunk_hgrn called with a broadcast-expanded gate) -- the specialized
kernel's whole point is to be bit-for-bit equivalent to the general
one while skipping the [rows, length, channels] gate/gc/dg memory
traffic. gator_linear/core.py's chunk_hgrn_decayed_scan (covered by
test_hgrn_receiver_scan.py) is the drop-in-replacement-level test
against decayed_scan; this file tests the kernel itself, one level
lower, including the reduced d(log_decay) gradient specifically since
that's the one behavior only the specialized kernel has (the general
kernel returns a full [rows, length, channels] gate gradient that the
caller's .expand() backward reduces instead).

Requires CUDA and flash-linear-attention; skipped otherwise.
"""

import unittest

import torch

try:
    import fla  # noqa: F401
    from fla.ops.hgrn import chunk_hgrn as fla_chunk_hgrn

    from gator_linear._hgrn_fixed_decay import fixed_decay_chunk_hgrn

    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention is not installed")
class FixedDecayChunkHgrnTests(unittest.TestCase):
    @staticmethod
    def _shapes(seed=0, batch=4, length=130, channels=6, dtype=torch.float32):
        torch.manual_seed(seed)
        x = torch.randn(batch, length, channels, device="cuda", dtype=dtype, requires_grad=True)
        log_decay = (torch.rand(channels, device="cuda", dtype=dtype) * 0.9 + 0.05).log()
        return x, log_decay

    @staticmethod
    def _general_kernel_forward(x, log_decay, initial_state):
        batch, length, channels = x.shape
        gate = log_decay[None, None, :].expand(batch, length, channels).contiguous()
        return fla_chunk_hgrn(x, gate, initial_state=initial_state, output_final_state=True)

    def test_matches_general_kernel_forward(self):
        x, log_decay = self._shapes()
        ref_o, ref_final = self._general_kernel_forward(x, log_decay, None)
        my_o, my_final = fixed_decay_chunk_hgrn(x, log_decay, output_final_state=True)
        self.assertTrue(torch.allclose(ref_o, my_o, rtol=1e-3, atol=1e-4))
        self.assertTrue(torch.allclose(ref_final, my_final, rtol=1e-3, atol=1e-4))

    def test_matches_general_kernel_with_initial_state(self):
        x, log_decay = self._shapes()
        batch, _, channels = x.shape
        carry = torch.randn(batch, channels, device="cuda") * 0.1
        ref_o, ref_final = self._general_kernel_forward(x, log_decay, carry)
        my_o, my_final = fixed_decay_chunk_hgrn(x, log_decay, initial_state=carry, output_final_state=True)
        self.assertTrue(torch.allclose(ref_o, my_o, rtol=1e-3, atol=1e-4))
        self.assertTrue(torch.allclose(ref_final, my_final, rtol=1e-3, atol=1e-4))

    def test_dx_matches_general_kernel(self):
        x, log_decay = self._shapes()
        x_ref = x.detach().clone().requires_grad_(True)
        gate_ref = log_decay[None, None, :].expand_as(x).contiguous().requires_grad_(True)
        ref_o, _ = fla_chunk_hgrn(x_ref, gate_ref, initial_state=None, output_final_state=True)
        ref_o.sum().backward()

        x_my = x.detach().clone().requires_grad_(True)
        log_decay_my = log_decay.detach().clone().requires_grad_(True)
        my_o, _ = fixed_decay_chunk_hgrn(x_my, log_decay_my, output_final_state=True)
        my_o.sum().backward()

        self.assertTrue(torch.allclose(x_ref.grad, x_my.grad, rtol=1e-2, atol=1e-2))

    def test_d_log_decay_matches_general_kernel_reduced_gradient(self):
        x, log_decay = self._shapes()
        x_ref = x.detach().clone().requires_grad_(True)
        gate_ref = log_decay[None, None, :].expand_as(x).contiguous().requires_grad_(True)
        ref_o, _ = fla_chunk_hgrn(x_ref, gate_ref, initial_state=None, output_final_state=True)
        ref_o.sum().backward()
        ref_d_log_decay = gate_ref.grad.sum(dim=(0, 1))

        x_my = x.detach().clone().requires_grad_(True)
        log_decay_my = log_decay.detach().clone().requires_grad_(True)
        my_o, _ = fixed_decay_chunk_hgrn(x_my, log_decay_my, output_final_state=True)
        my_o.sum().backward()

        self.assertTrue(torch.allclose(ref_d_log_decay, log_decay_my.grad, rtol=1e-2, atol=1e-2))

    def test_d_log_decay_matches_with_initial_state(self):
        # The position-0 correction (initial_state's own contribution
        # to d_log_decay) only applies when a real carry is used --
        # test that path specifically, since it's a separate code path
        # (post-hoc correction after the reduction kernel) from the
        # initial_state=None case above.
        x, log_decay = self._shapes()
        batch, _, channels = x.shape
        carry = torch.randn(batch, channels, device="cuda") * 0.1

        x_ref = x.detach().clone().requires_grad_(True)
        gate_ref = log_decay[None, None, :].expand_as(x).contiguous().requires_grad_(True)
        ref_o, _ = fla_chunk_hgrn(x_ref, gate_ref, initial_state=carry, output_final_state=True)
        ref_o.sum().backward()
        ref_d_log_decay = gate_ref.grad.sum(dim=(0, 1))

        x_my = x.detach().clone().requires_grad_(True)
        log_decay_my = log_decay.detach().clone().requires_grad_(True)
        my_o, _ = fixed_decay_chunk_hgrn(x_my, log_decay_my, initial_state=carry, output_final_state=True)
        my_o.sum().backward()

        self.assertTrue(torch.allclose(ref_d_log_decay, log_decay_my.grad, rtol=1e-2, atol=1e-2))

    def test_bf16_forward_finite(self):
        x, log_decay = self._shapes(dtype=torch.bfloat16)
        o, final = fixed_decay_chunk_hgrn(x, log_decay, output_final_state=True)
        self.assertTrue(torch.isfinite(o).all())
        self.assertTrue(torch.isfinite(final).all())

    def test_rejects_wrong_log_decay_shape(self):
        x, _ = self._shapes()
        bad_log_decay = torch.rand(3, device="cuda")
        with self.assertRaises(ValueError):
            fixed_decay_chunk_hgrn(x, bad_log_decay)

    def test_fullgraph_compile_forward(self):
        x, log_decay = self._shapes(dtype=torch.bfloat16)

        def wrapper(x, log_decay):
            o, final = fixed_decay_chunk_hgrn(x, log_decay, output_final_state=True)
            return o.float().sum() + final.float().sum()

        compiled = torch.compile(wrapper, fullgraph=True)
        result = compiled(x, log_decay)
        self.assertTrue(torch.isfinite(result))


if __name__ == "__main__":
    unittest.main()
