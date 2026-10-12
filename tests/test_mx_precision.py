"""Correctness/gradient checks for gator/mx_precision.py.

Requires torchao and CUDA (SM100+/Blackwell for the real kernels
underneath the fake-quant round trip); skipped everywhere else,
matching tests/test_gator_arw_cudagraph.py's pattern for GPU-only
regression tests. See gator/mx_precision.py's module docstring and
Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for the known-working
environment (Python 3.10, torch 2.13.0+cu130, torchao 0.18.0) this was
verified against -- the repo's main venv does not necessarily have
torchao installed, which is why this whole module is optional/lazy.
"""

import unittest

import torch

try:
    import torchao  # noqa: F401

    _TORCHAO_AVAILABLE = True
except ImportError:
    _TORCHAO_AVAILABLE = False

try:
    import transformer_engine  # noqa: F401

    _TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    _TRANSFORMER_ENGINE_AVAILABLE = False

from gator.mx_precision import (
    mx_readout_linear,
    mxfp8_fake_quant_node_linear,
    node_compute_linear,
    nvfp4_grouped_node_linear,
    te_fp8_cudagraph_node_linear,
    te_fp8_cudagraph_readout_linear,
)
from gator.models.dense_kernels import cell_linear


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TORCHAO_AVAILABLE, "torchao is not installed")
class MXPrecisionNodeComputeTests(unittest.TestCase):
    @staticmethod
    def _shapes():
        torch.manual_seed(0)
        batch, cells, tokens, d_model, nodes = 2, 8, 16, 64, 4
        inputs = torch.randn(
            batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            cells, nodes * d_model, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        bias = torch.randn(
            cells, nodes * d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        return inputs, weight, bias

    def test_bf16_precision_is_byte_identical_to_cell_linear(self):
        inputs, weight, bias = self._shapes()
        reference = cell_linear(
            inputs, weight, bias, backend="grouped_bmm", pad_to_multiple=32
        )
        dispatched = node_compute_linear(inputs, weight, bias, precision="bf16")
        self.assertTrue(torch.equal(reference, dispatched))

    def test_mxfp8_forward_and_backward_are_finite(self):
        inputs, weight, bias = self._shapes()
        output = mxfp8_fake_quant_node_linear(inputs, weight, bias)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_nvfp4_approx_forward_and_backward_are_finite(self):
        inputs, weight, bias = self._shapes()
        output = nvfp4_grouped_node_linear(inputs, weight, bias)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_gradient_actually_flows_through_quantization(self):
        # Regression test for a real bug found during implementation:
        # torchao's FP4 element dtype (float4_e2m1fn_x2) silently left
        # .grad as None without an explicit straight-through estimator
        # (MXTensor's ops are not autograd-transparent for FP4, unlike
        # FP8, which happened to work without one). This asserts
        # gradients are non-None, not just finite, so a regression back
        # to the un-STE'd version would be caught.
        inputs, weight, bias = self._shapes()
        output = nvfp4_grouped_node_linear(inputs, weight, bias)
        output.sum().backward()
        self.assertIsNotNone(
            inputs.grad, "gradient did not flow through FP4 fake-quant"
        )

    def test_outlier_input_value_does_not_produce_nan_or_inf(self):
        # User's specific focus: outlier gradient/activation handling.
        # A single very large value in an otherwise ordinary block should
        # degrade precision for that block, not blow up to NaN/Inf.
        inputs, weight, bias = self._shapes()
        with torch.no_grad():
            inputs[0, 0, 0, 0] = 1e4
        for fn in (mxfp8_fake_quant_node_linear, nvfp4_grouped_node_linear):
            output = fn(inputs, weight, bias)
            self.assertTrue(
                torch.isfinite(output).all(),
                f"{fn.__name__} produced non-finite output with an outlier input",
            )
            output.sum().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
            inputs.grad = None
            weight.grad = None
            if bias.grad is not None:
                bias.grad = None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TRANSFORMER_ENGINE_AVAILABLE, "transformer_engine is not installed")
class TEFP8CudaGraphNodeLinearTests(unittest.TestCase):
    """te_fp8_cudagraph_node_linear: real hardware FP8 + CUDA graph capture.

    Unlike the fake-quant precisions above, this path is genuinely
    hardware-accelerated (real TE FP8 tensor-core matmul, not
    quantize-dequantize-then-bf16-matmul) -- so its correctness bar is
    different: gradients must flow correctly through the CUDA-graph
    bridge (not just be non-None/finite in isolation, but survive a real
    upstream-op -> this-layer -> downstream-op chain, since that's what
    make_graphed_callables' documented failure modes tend to break) and
    the output must be numerically close to the bf16 reference (loose
    tolerance -- real FP8 quantization changes values, this isn't meant
    to be exact).
    """

    @staticmethod
    def _shapes():
        torch.manual_seed(0)
        batch, cells, tokens, d_model, nodes = 2, 8, 16, 64, 4
        inputs = torch.randn(
            batch, cells, tokens, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            cells, nodes * d_model, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        return inputs, weight

    def test_forward_and_backward_are_finite(self):
        inputs, weight = self._shapes()
        output = te_fp8_cudagraph_node_linear(inputs, weight, None)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_output_reasonably_close_to_bf16_reference_at_steady_state(self):
        # DelayedScaling's amax-history scale calibration is cold on the
        # very first call for a given shape (measured directly: ~10x
        # magnitude error, out_absmax=3.3 vs ref_absmax=40.3) -- inherent
        # to the recipe (an empty history has no real scale to compute
        # from yet), not specific to the CUDA-graph wrapper, and self-
        # corrects after a couple of real calls. One warmup call here
        # mirrors realistic usage (this layer runs every training step,
        # so steady-state accuracy is what matters). Elementwise
        # rtol/atol allclose is dominated by near-zero reference entries
        # (a small absolute error there is a huge relative error), so
        # this checks mean absolute error against the reference's own
        # scale instead -- measured directly at ~0.6% of ref_absmax at
        # steady state, well inside real FP8 (E4M3) precision loss.
        inputs, weight = self._shapes()
        reference = cell_linear(
            inputs, weight, None, backend="grouped_bmm", pad_to_multiple=32
        )
        te_fp8_cudagraph_node_linear(inputs, weight, None)  # warm up scale history
        output = te_fp8_cudagraph_node_linear(inputs, weight, None)
        mean_abs_diff = (output.float() - reference.float()).abs().mean()
        ref_scale = reference.float().abs().max()
        self.assertLess(
            mean_abs_diff / ref_scale,
            0.05,
            "FP8 output diverged too far from the bf16 reference at "
            "steady state to be the same computation at lower precision",
        )

    def test_gradients_flow_through_real_upstream_and_downstream_ops(self):
        # Regression test for the exact failure mode make_graphed_callables
        # would break under: gradients must correctly reach both this
        # layer's own inputs/weight AND tensors upstream/downstream of it,
        # not just be non-None when this layer is the entire graph (see
        # Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for the real
        # end-to-end check this mirrors).
        torch.manual_seed(0)
        cells, d_model, nodes = 8, 64, 4
        upstream_weight = torch.randn(
            d_model, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        raw_inputs = torch.randn(
            2, cells, 16, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        node_weight = torch.randn(
            cells, nodes * d_model, d_model, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        downstream_weight = torch.randn(
            nodes * d_model, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        upstream_out = raw_inputs @ upstream_weight
        node_out = te_fp8_cudagraph_node_linear(upstream_out, node_weight, None)
        loss = (node_out @ downstream_weight).sum()
        loss.backward()

        for name, tensor in [
            ("raw_inputs", raw_inputs),
            ("upstream_weight", upstream_weight),
            ("node_weight", node_weight),
            ("downstream_weight", downstream_weight),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(
                torch.isfinite(tensor.grad).all(), f"{name}.grad has non-finite values"
            )

    def test_outlier_input_value_does_not_produce_nan_or_inf(self):
        inputs, weight = self._shapes()
        with torch.no_grad():
            inputs[0, 0, 0, 0] = 1e4
        output = te_fp8_cudagraph_node_linear(inputs, weight, None)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_dispatched_through_node_compute_linear(self):
        inputs, weight = self._shapes()
        output = node_compute_linear(inputs, weight, None, precision="fp8_te")
        self.assertTrue(torch.isfinite(output).all())

    def test_first_call_output_survives_a_second_call(self):
        # Regression test for a real bug found during implementation:
        # make_graphed_callables' output is a live view into the CUDA
        # graph's static output buffer, silently overwritten by the NEXT
        # call to the same graphed callable -- fixed with an explicit
        # .clone() in te_fp8_cudagraph_node_linear. Without it, this is
        # exactly what happens in real multi-relay training: relay 1's
        # node-compute output would be silently corrupted by relay 2's
        # call to the same function before backward() ever uses relay
        # 1's activation -- wrong gradients, not a crash, which is worse.
        #
        # Snapshots output1 immediately after it's produced, then checks
        # it hasn't silently changed after a second call runs -- doesn't
        # rely on DelayedScaling's amax history being identical across
        # two separate calls with the same inputs (it isn't: the second
        # call's differing input updates the scale history), only on
        # output1's own in-memory values not mutating out from under it.
        inputs1, weight = self._shapes()
        torch.manual_seed(1)
        inputs2 = torch.randn_like(inputs1)

        output1 = te_fp8_cudagraph_node_linear(inputs1, weight, None)
        output1_snapshot = output1.clone()
        te_fp8_cudagraph_node_linear(inputs2, weight, None)  # a second call

        self.assertTrue(
            torch.equal(output1, output1_snapshot),
            "first call's output changed after a second, unrelated call ran "
            "-- the graph's static output buffer was not cloned before return",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TORCHAO_AVAILABLE, "torchao is not installed")
class MXPrecisionReadoutTests(unittest.TestCase):
    @staticmethod
    def _shapes():
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(
            256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        bias = torch.randn(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        return x, weight, bias

    def test_bf16_precision_matches_functional_linear(self):
        x, weight, bias = self._shapes()
        reference = torch.nn.functional.linear(x, weight, bias)
        dispatched = mx_readout_linear(x, weight, bias, precision="bf16")
        self.assertTrue(torch.equal(reference, dispatched))

    def test_mxfp8_and_mxfp4_are_finite_and_differentiable(self):
        for precision in ("mxfp8", "mxfp4"):
            x, weight, bias = self._shapes()
            logits = mx_readout_linear(x, weight, bias, precision=precision)
            self.assertTrue(torch.isfinite(logits).all())
            logits.sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(weight.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TRANSFORMER_ENGINE_AVAILABLE, "transformer_engine is not installed")
class TEFP8CudaGraphReadoutLinearTests(unittest.TestCase):
    """te_fp8_cudagraph_readout_linear: readout counterpart to
    TEFP8CudaGraphNodeLinearTests -- same correctness bar and same
    reasoning for why it's needed (see that class's docstring).
    """

    @staticmethod
    def _shapes():
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(
            256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        return x, weight

    def test_forward_and_backward_are_finite(self):
        x, weight = self._shapes()
        output = te_fp8_cudagraph_readout_linear(x, weight, None)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_output_reasonably_close_to_bf16_reference_at_steady_state(self):
        # Same DelayedScaling cold-start caveat as node compute -- one
        # warmup call before comparing, see
        # TEFP8CudaGraphNodeLinearTests.test_output_reasonably_close_to_bf16_reference_at_steady_state
        # for the full reasoning.
        x, weight = self._shapes()
        reference = torch.nn.functional.linear(x, weight, None)
        te_fp8_cudagraph_readout_linear(x, weight, None)  # warm up scale history
        output = te_fp8_cudagraph_readout_linear(x, weight, None)
        mean_abs_diff = (output.float() - reference.float()).abs().mean()
        ref_scale = reference.float().abs().max()
        self.assertLess(
            mean_abs_diff / ref_scale,
            0.05,
            "FP8 readout output diverged too far from the bf16 reference "
            "at steady state to be the same computation at lower precision",
        )

    def test_gradients_flow_through_real_upstream_and_downstream_ops(self):
        torch.manual_seed(0)
        d_model = 64
        upstream_weight = torch.randn(
            d_model, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        raw_inputs = torch.randn(
            2, 8, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        readout_weight = torch.randn(
            256, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        upstream_out = raw_inputs @ upstream_weight
        logits = te_fp8_cudagraph_readout_linear(upstream_out, readout_weight, None)
        loss = logits.sum()
        loss.backward()

        for name, tensor in [
            ("raw_inputs", raw_inputs),
            ("upstream_weight", upstream_weight),
            ("readout_weight", readout_weight),
        ]:
            self.assertIsNotNone(tensor.grad, f"{name}.grad is None")
            self.assertTrue(
                torch.isfinite(tensor.grad).all(), f"{name}.grad has non-finite values"
            )

    def test_outlier_input_value_does_not_produce_nan_or_inf(self):
        x, weight = self._shapes()
        with torch.no_grad():
            x[0, 0, 0] = 1e4
        output = te_fp8_cudagraph_readout_linear(x, weight, None)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_dispatched_through_mx_readout_linear(self):
        x, weight = self._shapes()
        output = mx_readout_linear(x, weight, None, precision="fp8_te")
        self.assertTrue(torch.isfinite(output).all())

    def test_first_call_output_survives_a_second_call(self):
        # Readout counterpart to
        # TEFP8CudaGraphNodeLinearTests.test_first_call_output_survives_a_second_call
        # -- see that test's comment for the full reasoning.
        x1, weight = self._shapes()
        torch.manual_seed(1)
        x2 = torch.randn_like(x1)

        output1 = te_fp8_cudagraph_readout_linear(x1, weight, None)
        output1_snapshot = output1.clone()
        te_fp8_cudagraph_readout_linear(x2, weight, None)  # a second call

        self.assertTrue(
            torch.equal(output1, output1_snapshot),
            "first call's output changed after a second, unrelated call ran "
            "-- the graph's static output buffer was not cloned before return",
        )


if __name__ == "__main__":
    unittest.main()
