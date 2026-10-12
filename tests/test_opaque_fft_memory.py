from __future__ import annotations

import unittest
from unittest import mock

import torch

import gator.models.opaque_fft_memory as opaque_fft_module
import gator.models.packed_convolution_memory as packed_memory_module
from gator.models.opaque_fft_memory import (
    opaque_packed_fft_node_context,
)
from gator.models.packed_convolution_memory import (
    packed_convolution_node_context,
)
from gator.models.triton_packed_memory import (
    triton_packed_reader_available,
)
from gator.models.triton_packed_writer import triton_packed_writer_available


class OpaquePackedFFTMemoryTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[torch.Tensor, ...]:
        torch.manual_seed(911)
        batch, tokens, lanes, width, route_width, nodes = 2, 5, 2, 6, 3, 7
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            dtype=torch.float64,
            requires_grad=True,
        )
        route_nodes = torch.randint(
            0,
            nodes,
            (batch, tokens, lanes, route_width),
        )
        # Exercise duplicate merging and one masked route.
        route_nodes[..., 1] = route_nodes[..., 0]
        valid = torch.ones_like(route_nodes, dtype=torch.bool)
        valid[0, -1, -1, -1] = False
        mass = torch.rand(
            batch,
            tokens,
            lanes,
            route_width,
            dtype=torch.float64,
            requires_grad=True,
        )
        write = torch.rand(
            batch,
            tokens,
            dtype=torch.float64,
            requires_grad=True,
        )
        with torch.no_grad():
            write[0, 1] = 0.0
        relative_node_bias = torch.randn(
            nodes,
            4,
            dtype=torch.float64,
            requires_grad=True,
        )
        return hidden, route_nodes, mass, valid, write, relative_node_bias

    @staticmethod
    def _clone_differentiable_inputs(
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden, nodes, mass, valid, write, bias = inputs
        return (
            hidden.detach().clone().requires_grad_(True),
            nodes.clone(),
            mass.detach().clone().requires_grad_(True),
            valid.clone(),
            write.detach().clone().requires_grad_(True),
            bias.detach().clone().requires_grad_(True),
        )

    def test_forward_and_gradients_match_eager_fft(self) -> None:
        cases = (
            # causal, compact active nodes, unique-route fast path, chunk
            (False, False, False, 1),
            (True, False, False, 4),
            (False, True, False, 2),
            (True, True, True, 6),
        )
        for causal, compact, unique, feature_chunk_size in cases:
            with self.subTest(
                causal=causal,
                compact=compact,
                unique=unique,
                feature_chunk_size=feature_chunk_size,
            ):
                source = list(self._inputs())
                if compact:
                    # Leave three bias rows inactive so this exercises actual
                    # route compaction rather than merely enabling its flag.
                    source[1] = source[1].remainder(4)
                if unique:
                    # The fast path requires unique node IDs within each
                    # lane. Build that contract explicitly, including for the
                    # compact-active-node case.
                    route_width = source[1].shape[-1]
                    node_count = 4 if compact else source[5].shape[0]
                    source[1] = (
                        source[1][..., :1]
                        + torch.arange(route_width).view(1, 1, 1, -1)
                    ).remainder(node_count)
                source_tuple = tuple(source)
                eager_inputs = self._clone_differentiable_inputs(source_tuple)
                opaque_inputs = self._clone_differentiable_inputs(source_tuple)

                eager = packed_convolution_node_context(
                    *eager_inputs,
                    causal=causal,
                    feature_chunk_size=feature_chunk_size,
                    compact_active_nodes=compact,
                    convolution_backend="fft",
                    assume_unique_routes=unique,
                )
                opaque = opaque_packed_fft_node_context(
                    *opaque_inputs,
                    causal=causal,
                    feature_chunk_size=feature_chunk_size,
                    compact_active_nodes=compact,
                    assume_unique_routes=unique,
                )

                self.assertFalse(torch.is_complex(opaque))
                torch.testing.assert_close(opaque, eager, atol=1e-12, rtol=1e-12)

                probe = torch.randn_like(eager)
                differentiable_indices = (0, 2, 4, 5)
                eager_gradients = torch.autograd.grad(
                    (eager * probe).sum(),
                    tuple(
                        eager_inputs[index]
                        for index in differentiable_indices
                    ),
                )
                opaque_gradients = torch.autograd.grad(
                    (opaque * probe).sum(),
                    tuple(
                        opaque_inputs[index]
                        for index in differentiable_indices
                    ),
                )
                for actual, expected in zip(
                    opaque_gradients,
                    eager_gradients,
                ):
                    self.assertFalse(torch.is_complex(actual))
                    # The oracle case intentionally retains epsilon=1e-9.
                    # Its large clamped-denominator gradients make the final
                    # few ULPs sensitive to explicit accumulation order.
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=2e-6,
                        rtol=2e-7,
                    )

    def test_fullgraph_compile_backend_eager_preserves_forward_and_backward(
        self,
    ) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        source = self._inputs()
        eager_inputs = self._clone_differentiable_inputs(source)
        compiled_inputs = self._clone_differentiable_inputs(source)

        def receiver_loss(
            hidden: torch.Tensor,
            nodes: torch.Tensor,
            mass: torch.Tensor,
            valid: torch.Tensor,
            write: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            context = opaque_packed_fft_node_context(
                hidden,
                nodes,
                mass,
                valid,
                write,
                bias,
                causal=False,
                feature_chunk_size=3,
                compact_active_nodes=False,
            )
            return context.square().mean()

        expected = receiver_loss(*eager_inputs)
        expected_gradients = torch.autograd.grad(
            expected,
            (eager_inputs[0], eager_inputs[2], eager_inputs[4], eager_inputs[5]),
        )

        compiled = torch.compile(
            receiver_loss,
            backend="eager",
            fullgraph=True,
        )
        actual = compiled(*compiled_inputs)
        actual_gradients = torch.autograd.grad(
            actual,
            (
                compiled_inputs[0],
                compiled_inputs[2],
                compiled_inputs[4],
                compiled_inputs[5],
            ),
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=1e-10,
                rtol=1e-10,
            )

    def test_custom_op_schema_fake_autograd_and_aot_registration(self) -> None:
        if not hasattr(torch.library, "opcheck"):
            self.skipTest("torch.library.opcheck is unavailable")
        inputs = self._clone_differentiable_inputs(self._inputs())
        result = torch.library.opcheck(
            torch.ops.gator.opaque_packed_fft_receiver.default,
            (*inputs, False, 1e-9, 3, False, False, False, False),
        )
        self.assertEqual(
            set(result),
            {
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            },
        )
        self.assertTrue(all(value == "SUCCESS" for value in result.values()))

    def test_explicit_triton_reader_rejects_cpu_before_custom_op(self) -> None:
        inputs = self._clone_differentiable_inputs(self._inputs())
        with mock.patch(
            "gator.models.opaque_fft_memory._opaque_packed_fft_receiver",
            side_effect=AssertionError("opaque op should not launch"),
        ):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                opaque_packed_fft_node_context(
                    *inputs,
                    reader_backend="triton",
                )

    def test_explicit_triton_writer_rejects_cpu_before_custom_op(self) -> None:
        inputs = self._clone_differentiable_inputs(self._inputs())
        with mock.patch(
            "gator.models.opaque_fft_memory._opaque_packed_fft_receiver",
            side_effect=AssertionError("opaque op should not launch"),
        ):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                opaque_packed_fft_node_context(
                    *inputs,
                    writer_backend="triton",
                )

    def test_fake_and_real_outputs_share_contiguous_stride_contract(self) -> None:
        hidden, nodes, mass, valid, write, bias = self._inputs()
        noncontiguous_hidden = torch.randn(
            hidden.shape[0],
            hidden.shape[1],
            hidden.shape[3],
            hidden.shape[2],
            dtype=hidden.dtype,
        ).permute(0, 1, 3, 2)
        self.assertFalse(noncontiguous_hidden.is_contiguous())
        actual = opaque_packed_fft_node_context(
            noncontiguous_hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            feature_chunk_size=2,
        )
        fake = opaque_fft_module._opaque_packed_fft_receiver_fake(
            noncontiguous_hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            False,
            1e-9,
            2,
            False,
            False,
            False,
            False,
        )
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(fake.is_contiguous())
        self.assertEqual(fake.stride(), actual.stride())

    def test_backward_fake_matches_real_contiguous_stride_contract(self) -> None:
        hidden, nodes, _, valid, _, bias = self._inputs()
        batch, tokens, lanes, width = hidden.shape
        routes = nodes.shape[-1]
        n_nodes = bias.shape[0]
        noncontiguous_hidden = torch.randn(
            batch,
            tokens,
            width,
            lanes,
            dtype=hidden.dtype,
        ).permute(0, 1, 3, 2)
        noncontiguous_mass = torch.rand(
            batch,
            tokens,
            routes,
            lanes,
            dtype=hidden.dtype,
        ).permute(0, 1, 3, 2)
        noncontiguous_write = torch.rand(
            tokens,
            batch,
            dtype=hidden.dtype,
        ).T
        noncontiguous_bias = torch.randn(
            4,
            n_nodes,
            dtype=hidden.dtype,
        ).T
        noncontiguous_grad_output = torch.randn(
            batch,
            tokens,
            width,
            lanes,
            dtype=hidden.dtype,
        ).permute(0, 1, 3, 2)
        inputs = (
            noncontiguous_hidden,
            nodes,
            noncontiguous_mass,
            valid,
            noncontiguous_write,
            noncontiguous_bias,
            noncontiguous_grad_output,
            False,
            1e-9,
            2,
            False,
            False,
            False,
            False,
        )
        actual = opaque_fft_module._opaque_packed_fft_receiver_backward(*inputs)
        fake = opaque_fft_module._opaque_packed_fft_receiver_backward_fake(*inputs)
        for actual_gradient, fake_gradient in zip(actual, fake):
            self.assertTrue(actual_gradient.is_contiguous())
            self.assertTrue(fake_gradient.is_contiguous())
            self.assertEqual(fake_gradient.stride(), actual_gradient.stride())

    def test_native_backward_never_replays_complete_receiver_vjp(self) -> None:
        hidden, nodes, mass, valid, write, bias = (
            self._clone_differentiable_inputs(self._inputs())
        )
        output = opaque_packed_fft_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            causal=True,
            feature_chunk_size=2,
            compact_active_nodes=True,
        )
        probe = torch.randn_like(output)
        native_reader_adjoint = (
            opaque_fft_module._packed_reader_backward_unchecked
        )
        native_fft = opaque_fft_module._fft_grouped_convolution
        native_fft_with_spectrum = (
            opaque_fft_module._fft_grouped_convolution_with_source_spectrum
        )
        native_fft_adjoint = (
            opaque_fft_module._fft_grouped_convolution_adjoint
        )
        with (
            mock.patch.object(
                torch.func,
                "vjp",
                side_effect=AssertionError("whole-receiver VJP replay"),
            ) as vjp,
            mock.patch.object(
                opaque_fft_module,
                "_eager_fft_receiver",
                side_effect=AssertionError("whole receiver recomputed"),
            ) as eager_receiver,
            mock.patch.object(
                opaque_fft_module,
                "_packed_reader_backward_unchecked",
                wraps=native_reader_adjoint,
            ) as reader_adjoint,
            mock.patch.object(
                opaque_fft_module,
                "_fft_grouped_convolution",
                wraps=native_fft,
            ) as fft_recompute,
            mock.patch.object(
                opaque_fft_module,
                "_fft_grouped_convolution_with_source_spectrum",
                wraps=native_fft_with_spectrum,
            ) as fft_recompute_with_spectrum,
            mock.patch.object(
                opaque_fft_module,
                "_fft_grouped_convolution_adjoint",
                wraps=native_fft_adjoint,
            ) as fft_adjoint,
        ):
            gradients = torch.autograd.grad(
                (output * probe).sum(),
                (hidden, mass, write, bias),
            )
        vjp.assert_not_called()
        eager_receiver.assert_not_called()
        self.assertEqual(reader_adjoint.call_count, 3)
        # Denominator and the first two-feature state chunk share one
        # three-feature FFT. Each later chunk returns the source spectrum from
        # its required numerator recomputation so its immediate adjoint does
        # not run the same RFFT a second time.
        self.assertEqual(fft_recompute.call_count, 1)
        self.assertEqual(fft_recompute_with_spectrum.call_count, 2)
        self.assertEqual(fft_adjoint.call_count, 3)
        self.assertEqual(
            [call.args[0].shape[-1] for call in fft_recompute.call_args_list],
            [3],
        )
        self.assertEqual(
            [
                call.args[0].shape[-1]
                for call in fft_recompute_with_spectrum.call_args_list
            ],
            [2, 2],
        )
        self.assertEqual(
            sorted(call.args[0].shape[-1] for call in fft_adjoint.call_args_list),
            [2, 2, 3],
        )
        for invocation in fft_adjoint.call_args_list:
            if invocation.args[0].shape[-1] == 2:
                self.assertIsNotNone(
                    invocation.kwargs.get("source_spectrum")
                )
            else:
                self.assertIsNone(
                    invocation.kwargs.get("source_spectrum")
                )
        for invocation in reader_adjoint.call_args_list:
            self.assertEqual(invocation.kwargs["backend"], "torch")
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_fft_adjoint_crop_is_contiguous_and_reuses_source_spectrum(
        self,
    ) -> None:
        torch.manual_seed(921)
        batch, active_nodes, tokens, features = 2, 3, 5, 4
        values = torch.randn(
            batch,
            active_nodes,
            tokens,
            features,
            dtype=torch.float64,
        )
        kernel = torch.randn(
            active_nodes,
            2 * tokens - 1,
            dtype=torch.float64,
        )
        transform_length, kernel_spectrum = (
            opaque_fft_module._fft_kernel_spectrum(
                kernel,
                tokens=tokens,
                feature_dtype=values.dtype,
            )
        )
        output, source_spectrum = (
            opaque_fft_module._fft_grouped_convolution_with_source_spectrum(
                values,
                kernel_spectrum,
                transform_length=transform_length,
            )
        )
        grad_output = torch.randn_like(output)
        original_rfft = torch.fft.rfft
        call_count = 0

        def counted_rfft(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return original_rfft(input, *args, **kwargs)

        with mock.patch.object(
            torch.fft,
            "rfft",
            side_effect=counted_rfft,
        ):
            grad_values, _ = (
                opaque_fft_module._fft_grouped_convolution_adjoint(
                    values,
                    kernel_spectrum,
                    grad_output,
                    transform_length=transform_length,
                    source_spectrum=source_spectrum,
                )
            )

        # Only the output-cotangent transform remains; the source transform
        # came directly from the numerator recomputation.
        self.assertEqual(call_count, 1)
        self.assertTrue(grad_values.is_contiguous())
        self.assertIsNone(grad_values._base)
        self.assertEqual(
            grad_values.untyped_storage().nbytes(),
            grad_values.numel() * grad_values.element_size(),
        )

    def test_fft_native_writer_layout_matches_portable_layout(self) -> None:
        torch.manual_seed(922)
        batch, active_nodes, tokens, features = 2, 4, 6, 5
        values = torch.randn(
            batch,
            active_nodes,
            tokens,
            features,
            dtype=torch.float64,
        )
        native_values = values.permute(0, 3, 1, 2).contiguous()
        kernel = torch.randn(
            active_nodes,
            2 * tokens - 1,
            dtype=torch.float64,
        )
        transform_length, kernel_spectrum = (
            opaque_fft_module._fft_kernel_spectrum(
                kernel,
                tokens=tokens,
                feature_dtype=values.dtype,
            )
        )
        expected, expected_spectrum = (
            opaque_fft_module._fft_grouped_convolution_with_source_spectrum(
                values,
                kernel_spectrum,
                transform_length=transform_length,
            )
        )
        actual, actual_spectrum = (
            opaque_fft_module._fft_native_grouped_convolution_with_source_spectrum(
                native_values,
                kernel_spectrum,
                transform_length=transform_length,
            )
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_spectrum, expected_spectrum)

        grad_output = torch.randn_like(expected)
        expected_gradient, expected_kernel_gradient = (
            opaque_fft_module._fft_grouped_convolution_adjoint(
                values,
                kernel_spectrum,
                grad_output,
                transform_length=transform_length,
                source_spectrum=expected_spectrum,
            )
        )
        actual_gradient, actual_kernel_gradient = (
            opaque_fft_module._fft_native_grouped_convolution_adjoint(
                native_values,
                kernel_spectrum,
                grad_output,
                transform_length=transform_length,
                source_spectrum=actual_spectrum,
            )
        )
        torch.testing.assert_close(
            actual_gradient.permute(0, 2, 3, 1),
            expected_gradient,
        )
        torch.testing.assert_close(
            actual_kernel_gradient,
            expected_kernel_gradient,
        )

    def test_empty_compact_frontier_has_exact_zero_gradients(self) -> None:
        hidden, nodes, mass, valid, write, bias = (
            self._clone_differentiable_inputs(self._inputs())
        )
        valid.zero_()
        output = opaque_packed_fft_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            compact_active_nodes=True,
            feature_chunk_size=2,
        )
        gradients = torch.autograd.grad(
            output.sum(),
            (hidden, mass, write, bias),
        )
        torch.testing.assert_close(output, torch.zeros_like(output))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@unittest.skipUnless(
    torch.cuda.is_available()
    and triton_packed_reader_available()
    and triton_packed_writer_available(),
    "CUDA Triton is unavailable",
)
class OpaquePackedFFTMemoryCudaTests(unittest.TestCase):
    def test_high_node_1337_receiver_smoke_is_finite(self) -> None:
        torch.manual_seed(924)
        batch, tokens, lanes, width, route_width, n_nodes = (
            1,
            17,
            2,
            16,
            4,
            1337,
        )
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            device="cuda",
            requires_grad=True,
        )
        nodes = torch.randint(
            0,
            n_nodes,
            (batch, tokens, lanes, route_width),
            device="cuda",
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        mass = torch.softmax(
            torch.randn_like(nodes, dtype=torch.float32),
            dim=-1,
        ).requires_grad_(True)
        write = torch.rand(
            batch,
            tokens,
            device="cuda",
            requires_grad=True,
        )
        bias = torch.randn(
            n_nodes,
            4,
            device="cuda",
            requires_grad=True,
        )
        context = opaque_packed_fft_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            feature_chunk_size=16,
            reader_backend="triton",
            writer_backend="triton",
        )
        loss = context.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(context).all())
        self.assertTrue(torch.isfinite(loss))
        for gradient in (hidden.grad, mass.grad, write.grad, bias.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_triton_writer_reader_match_torch_receiver_and_gradients(
        self,
    ) -> None:
        torch.manual_seed(923)
        batch, tokens, lanes, width, route_width, n_nodes = (
            2,
            9,
            3,
            13,
            4,
            17,
        )
        hidden = torch.randn(batch, tokens, lanes, width, device="cuda")
        nodes = torch.randint(
            0,
            n_nodes,
            (batch, tokens, lanes, route_width),
            device="cuda",
        )
        nodes[..., 1] = nodes[..., 0]
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, -1, -1, -1] = False
        mass = torch.rand_like(nodes, dtype=torch.float32).add_(0.1)
        write = torch.rand(batch, tokens, device="cuda").add_(0.1)
        bias = torch.randn(n_nodes, 4, device="cuda")
        source = (hidden, nodes, mass, valid, write, bias)
        torch_inputs = self._clone_cuda_differentiable_inputs(source)
        triton_inputs = self._clone_cuda_differentiable_inputs(source)

        expected = packed_convolution_node_context(
            *torch_inputs,
            causal=False,
            epsilon=1e-6,
            feature_chunk_size=7,
            convolution_backend="fft",
            reader_backend="torch",
            writer_backend="torch",
        )
        # The opaque forward delegates to packed_convolution_node_context,
        # whose module owns the writer-forward binding.  Backward is native
        # to opaque_fft_memory and is instrumented separately below.
        native_writer = packed_memory_module._packed_writer_features_unchecked
        with mock.patch.object(
            packed_memory_module,
            "_packed_writer_features_unchecked",
            wraps=native_writer,
        ) as writer_forward:
            actual = opaque_packed_fft_node_context(
                *triton_inputs,
                causal=False,
                epsilon=1e-6,
                feature_chunk_size=7,
                reader_backend="triton",
                writer_backend="triton",
            )
        self.assertEqual(writer_forward.call_count, 2)
        for invocation in writer_forward.call_args_list:
            self.assertEqual(invocation.kwargs["backend"], "triton")
        torch.testing.assert_close(actual, expected, atol=8e-4, rtol=8e-4)

        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (torch_inputs[0], torch_inputs[2], torch_inputs[4], torch_inputs[5]),
        )
        native_adjoint = opaque_fft_module._packed_writer_adjoint_unchecked
        with mock.patch.object(
            opaque_fft_module,
            "_packed_writer_adjoint_unchecked",
            wraps=native_adjoint,
        ) as writer_adjoint:
            actual_gradients = torch.autograd.grad(
                (actual * probe).sum(),
                (
                    triton_inputs[0],
                    triton_inputs[2],
                    triton_inputs[4],
                    triton_inputs[5],
                ),
            )
        self.assertEqual(writer_adjoint.call_count, 2)
        for invocation in writer_adjoint.call_args_list:
            self.assertEqual(invocation.kwargs["backend"], "triton")
        for candidate, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(
                candidate,
                reference,
                atol=2e-3,
                rtol=2e-3,
            )

    def test_triton_native_backward_matches_torch_oracle_and_is_invoked(
        self,
    ) -> None:
        torch.manual_seed(917)
        batch, tokens, lanes, width, route_width, n_nodes = 2, 6, 2, 7, 3, 9
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            device="cuda",
        )
        nodes = torch.randint(
            0,
            6,
            (batch, tokens, lanes, route_width),
            device="cuda",
        )
        nodes[..., 1] = nodes[..., 0]
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, -1, -1, -1] = False
        mass = torch.rand_like(nodes, dtype=torch.float32).add_(0.2)
        write = torch.rand(batch, tokens, device="cuda").add_(0.2)
        bias = torch.randn(n_nodes, 4, device="cuda")
        source = (hidden, nodes, mass, valid, write, bias)
        torch_inputs = self._clone_cuda_differentiable_inputs(source)
        triton_inputs = self._clone_cuda_differentiable_inputs(source)

        expected = packed_convolution_node_context(
            *torch_inputs,
            causal=True,
            epsilon=1e-6,
            feature_chunk_size=3,
            compact_active_nodes=True,
            convolution_backend="fft",
            reader_backend="torch",
            assume_unique_routes=False,
        )
        actual = opaque_packed_fft_node_context(
            *triton_inputs,
            causal=True,
            epsilon=1e-6,
            feature_chunk_size=3,
            compact_active_nodes=True,
            reader_backend="triton",
            assume_unique_routes=False,
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (
                torch_inputs[0],
                torch_inputs[2],
                torch_inputs[4],
                torch_inputs[5],
            ),
        )
        native_reader_adjoint = (
            opaque_fft_module._packed_reader_backward_unchecked
        )
        with mock.patch.object(
            opaque_fft_module,
            "_packed_reader_backward_unchecked",
            wraps=native_reader_adjoint,
        ) as reader_adjoint:
            actual_gradients = torch.autograd.grad(
                (actual * probe).sum(),
                (
                    triton_inputs[0],
                    triton_inputs[2],
                    triton_inputs[4],
                    triton_inputs[5],
                ),
            )

        self.assertEqual(reader_adjoint.call_count, 3)
        for invocation in reader_adjoint.call_args_list:
            self.assertEqual(invocation.kwargs["backend"], "triton")
        for candidate, reference in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                candidate,
                reference,
                atol=8e-4,
                rtol=8e-4,
            )

    @staticmethod
    def _clone_cuda_differentiable_inputs(
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden, nodes, mass, valid, write, bias = inputs
        return (
            hidden.detach().clone().requires_grad_(True),
            nodes.clone(),
            mass.detach().clone().requires_grad_(True),
            valid.clone(),
            write.detach().clone().requires_grad_(True),
            bias.detach().clone().requires_grad_(True),
        )

    def test_compiled_opaque_fft_and_triton_reader_train_together(self) -> None:
        torch.manual_seed(919)
        batch, tokens, lanes, width, route_width, n_nodes = 1, 7, 2, 5, 3, 9
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            device="cuda",
            requires_grad=True,
        )
        nodes = torch.randint(
            0,
            n_nodes,
            (batch, tokens, lanes, route_width),
            device="cuda",
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[:, -1, :, -1] = False
        mass = torch.rand_like(nodes, dtype=torch.float32).requires_grad_(True)
        write = torch.rand(batch, tokens, device="cuda", requires_grad=True)
        bias = torch.randn(n_nodes, 4, device="cuda", requires_grad=True)

        @torch.compile(fullgraph=True)
        def loss(
            hidden: torch.Tensor,
            nodes: torch.Tensor,
            mass: torch.Tensor,
            valid: torch.Tensor,
            write: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            context = opaque_packed_fft_node_context(
                hidden,
                nodes,
                mass,
                valid,
                write,
                bias,
                causal=True,
                feature_chunk_size=3,
                reader_backend="triton",
            )
            return context.square().mean()

        value = loss(hidden, nodes, mass, valid, write, bias)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        for gradient in (hidden.grad, mass.grad, write.grad, bias.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
