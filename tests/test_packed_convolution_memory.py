from __future__ import annotations

import ast
import inspect
import textwrap
import unittest
from unittest import mock

import torch

import gator.models.packed_convolution_memory as packed_memory_module
from gator.models.packed_convolution_memory import (
    PackedConvolutionMemoryDiagnostics,
    packed_convolution_node_context,
)
from gator_convergent import ConvergentGator


def _dense_locations(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    safe_nodes = torch.where(valid, nodes, torch.zeros_like(nodes))
    return mass.new_zeros(*mass.shape[:-1], n_nodes).scatter_add(
        -1,
        safe_nodes,
        mass * valid.to(mass.dtype),
    )


class PackedConvolutionMemoryTests(unittest.TestCase):
    def _compare(
        self,
        *,
        causal: bool,
        compact_active_nodes: bool,
        convolution_backend: str,
    ) -> None:
        torch.manual_seed(401 + int(causal) + 2 * int(compact_active_nodes))
        batch, tokens, lanes, width, route_width, n_nodes = 2, 6, 3, 7, 3, 9
        model = ConvergentGator(
            n_nodes=n_nodes,
            d_model=width,
            degree=3,
            relays=2,
            lanes=lanes,
        ).double()
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            dtype=torch.float64,
            requires_grad=True,
        )
        # Restrict IDs to a subset so the compact path has observable value.
        nodes = torch.randint(
            0,
            4,
            (batch, tokens, lanes, route_width),
        )
        nodes[..., 1] = nodes[..., 0]
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, -1, 1, -1] = False
        valid[1, 2, 2, -1] = False
        raw_mass = torch.rand_like(nodes, dtype=torch.float64)
        raw_mass[..., 0] *= 5e-10
        raw_mass[..., 1] *= 5e-10
        mass = raw_mass.requires_grad_(True)
        write_values = torch.rand(batch, tokens, dtype=torch.float64)
        write_values[0, 1] = 0
        write_values[1, -1] = 0
        write = write_values.requires_grad_(True)

        dense_locations = _dense_locations(nodes, mass, valid, n_nodes)
        receiver_kernel = model._receiver_kernel(
            tokens,
            hidden.device,
            causal=causal,
        )
        dense_memory = model._node_memory(
            hidden,
            dense_locations,
            write,
            receiver_kernel=receiver_kernel,
        )
        expected = torch.einsum(
            "btln,btnd->btld",
            dense_locations,
            dense_memory,
        )
        actual, diagnostics = packed_convolution_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            model.relative_node_bias,
            causal=causal,
            feature_chunk_size=3,
            compact_active_nodes=compact_active_nodes,
            convolution_backend=convolution_backend,
            return_diagnostics=True,
        )
        self.assertIsInstance(
            diagnostics,
            PackedConvolutionMemoryDiagnostics,
        )
        expected_active = 4 if compact_active_nodes else n_nodes
        self.assertEqual(diagnostics.active_nodes.numel(), expected_active)
        self.assertEqual(
            tuple(diagnostics.denominator.shape),
            (batch, expected_active, tokens),
        )
        self.assertEqual(
            tuple(diagnostics.kernel.shape),
            (expected_active, 2 * tokens - 1),
        )
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)

        probe = torch.randn_like(expected)
        inputs = (hidden, mass, write, model.relative_node_bias)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            inputs,
            retain_graph=True,
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            inputs,
        )
        for gradient_index, (actual_gradient, expected_gradient) in enumerate(
            zip(actual_gradients, expected_gradients)
        ):
            # Grouped convolution and dense GEMM accumulate the learned
            # relative-kernel gradient in different orders.  State, route,
            # and write gradients remain near machine precision; the kernel
            # parameter uses the same tolerance as production float32 parity.
            tolerance = (
                2e-6
                if convolution_backend == "fft" or gradient_index == 3
                else 3e-10
            )
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=tolerance,
                rtol=tolerance,
            )

    def test_noncausal_all_nodes_matches_dense_values_and_gradients(self) -> None:
        self._compare(
            causal=False,
            compact_active_nodes=False,
            convolution_backend="direct",
        )

    def test_causal_compact_nodes_matches_dense_values_and_gradients(self) -> None:
        self._compare(
            causal=True,
            compact_active_nodes=True,
            convolution_backend="fft",
        )

    def test_fft_reuses_one_kernel_spectrum_across_feature_chunks(self) -> None:
        torch.manual_seed(407)
        batch, tokens, lanes, width, route_width, n_nodes = 2, 5, 3, 7, 2, 6
        feature_chunk_size = 2
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            dtype=torch.float64,
            requires_grad=True,
        )
        nodes = torch.randint(
            0,
            n_nodes,
            (batch, tokens, lanes, route_width),
        )
        mass = torch.rand(
            batch,
            tokens,
            lanes,
            route_width,
            dtype=torch.float64,
            requires_grad=True,
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        write = torch.rand(
            batch,
            tokens,
            dtype=torch.float64,
            requires_grad=True,
        )
        bias = torch.randn(
            n_nodes,
            4,
            dtype=torch.float64,
            requires_grad=True,
        )

        original_rfft = torch.fft.rfft
        call_shapes: list[tuple[int, ...]] = []

        def counted_rfft(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            call_shapes.append(tuple(input.shape))
            return original_rfft(input, *args, **kwargs)

        with mock.patch.object(torch.fft, "rfft", side_effect=counted_rfft):
            context = packed_convolution_node_context(
                hidden,
                nodes,
                mass,
                valid,
                write,
                bias,
                feature_chunk_size=feature_chunk_size,
                convolution_backend="fft",
            )

        kernel_calls = [shape for shape in call_shapes if len(shape) == 2]
        feature_calls = [shape for shape in call_shapes if len(shape) == 4]
        expected_feature_calls = (
            width + feature_chunk_size - 1
        ) // feature_chunk_size
        self.assertEqual(len(kernel_calls), 1)
        self.assertEqual(len(feature_calls), expected_feature_calls)
        self.assertEqual(len(call_shapes), expected_feature_calls + 1)
        self.assertEqual(
            feature_calls[0][1],
            1 + min(width, feature_chunk_size),
        )

        context.square().mean().backward()
        for gradient in (hidden.grad, mass.grad, write.grad, bias.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_fft_crop_is_feature_contiguous_and_owns_only_crop_storage(
        self,
    ) -> None:
        torch.manual_seed(411)
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
            packed_memory_module._fft_kernel_spectrum(
                kernel,
                tokens=tokens,
                feature_dtype=values.dtype,
            )
        )

        result = packed_memory_module._fft_grouped_convolution(
            values,
            kernel_spectrum,
            transform_length=transform_length,
        )

        self.assertEqual(
            tuple(result.shape),
            (batch, active_nodes, tokens, features),
        )
        self.assertTrue(result.is_contiguous())
        self.assertIsNone(result._base)
        self.assertEqual(
            result.untyped_storage().nbytes(),
            result.numel() * result.element_size(),
        )

    def test_direct_fuses_denominator_with_first_feature_chunk(self) -> None:
        torch.manual_seed(410)
        batch, tokens, lanes, width, route_width, n_nodes = 2, 5, 3, 7, 2, 6
        feature_chunk_size = 2
        hidden = torch.randn(batch, tokens, lanes, width, dtype=torch.float64)
        nodes = torch.randint(
            0,
            n_nodes,
            (batch, tokens, lanes, route_width),
        )
        mass = torch.rand(
            batch,
            tokens,
            lanes,
            route_width,
            dtype=torch.float64,
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        write = torch.rand(batch, tokens, dtype=torch.float64)
        bias = torch.randn(n_nodes, 4, dtype=torch.float64)

        original_conv1d = torch.nn.functional.conv1d
        call_shapes: list[tuple[int, ...]] = []

        def counted_conv1d(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            call_shapes.append(tuple(input.shape))
            return original_conv1d(input, *args, **kwargs)

        with mock.patch.object(
            torch.nn.functional,
            "conv1d",
            side_effect=counted_conv1d,
        ):
            context = packed_convolution_node_context(
                hidden,
                nodes,
                mass,
                valid,
                write,
                bias,
                feature_chunk_size=feature_chunk_size,
                convolution_backend="direct",
            )

        expected_calls = (
            width + feature_chunk_size - 1
        ) // feature_chunk_size
        self.assertEqual(len(call_shapes), expected_calls)
        self.assertEqual(
            call_shapes[0][0],
            batch * (1 + min(width, feature_chunk_size)),
        )
        self.assertTrue(torch.isfinite(context).all())

    def test_hot_route_helpers_have_no_python_per_route_loops(self) -> None:
        for helper in (
            packed_memory_module._writer_mass,
            packed_memory_module._writer_state_chunk,
            packed_memory_module._reader_context_chunk,
        ):
            source = textwrap.dedent(inspect.getsource(helper))
            tree = ast.parse(source)
            loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
            self.assertEqual(loops, [], helper.__name__)

    def test_unique_route_fast_path_matches_duplicate_safe_oracle(self) -> None:
        torch.manual_seed(408)
        hidden = torch.randn(
            2, 5, 3, 6, dtype=torch.float64, requires_grad=True
        )
        nodes = torch.tensor([0, 2, 5], dtype=torch.long).view(
            1, 1, 1, 3
        ).expand(2, 5, 3, 3)
        mass = torch.rand(
            2, 5, 3, 3, dtype=torch.float64, requires_grad=True
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, -1, 1, -1] = False
        write = torch.rand(2, 5, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)

        oracle = packed_convolution_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            feature_chunk_size=2,
            convolution_backend="direct",
            assume_unique_routes=False,
        )
        fast = packed_convolution_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            feature_chunk_size=2,
            convolution_backend="direct",
            assume_unique_routes=True,
        )
        torch.testing.assert_close(fast, oracle, atol=1e-12, rtol=1e-12)

        probe = torch.randn_like(fast)
        inputs = (hidden, mass, write, bias)
        oracle_gradients = torch.autograd.grad(
            (oracle * probe).sum(), inputs, retain_graph=True
        )
        fast_gradients = torch.autograd.grad((fast * probe).sum(), inputs)
        for actual, expected in zip(fast_gradients, oracle_gradients):
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)

    def test_full_node_table_skips_identity_searchsorted(self) -> None:
        torch.manual_seed(412)
        hidden = torch.randn(2, 5, 3, 6, dtype=torch.float64)
        nodes = torch.randint(0, 7, (2, 5, 3, 3))
        mass = torch.rand(2, 5, 3, 3, dtype=torch.float64)
        valid = torch.ones_like(nodes, dtype=torch.bool)
        write = torch.rand(2, 5, dtype=torch.float64)
        bias = torch.randn(7, 4, dtype=torch.float64)

        with mock.patch.object(
            torch,
            "searchsorted",
            side_effect=AssertionError("identity node table was searched"),
        ):
            result = packed_convolution_node_context(
                hidden,
                nodes,
                mass,
                valid,
                write,
                bias,
                feature_chunk_size=3,
                convolution_backend="fft",
                compact_active_nodes=False,
                assume_unique_routes=True,
            )

        self.assertTrue(torch.isfinite(result).all())

    def test_zero_reader_mass_returns_differentiable_zero(self) -> None:
        torch.manual_seed(409)
        hidden = torch.randn(1, 4, 2, 5, dtype=torch.float64, requires_grad=True)
        nodes = torch.zeros(1, 4, 2, 2, dtype=torch.long)
        mass = torch.zeros(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
        valid = torch.zeros_like(nodes, dtype=torch.bool)
        write = torch.ones(1, 4, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
        context = packed_convolution_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            bias,
            compact_active_nodes=True,
        )
        self.assertTrue(torch.isfinite(context).all())
        self.assertEqual(float(context.detach().abs().sum()), 0.0)
        gradients = torch.autograd.grad(
            context.sum(),
            (hidden, mass, write, bias),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
