from __future__ import annotations

import unittest

import torch

from gator.models.dense_kernels import (
    cell_linear,
    cell_node_linear,
    folded_cell_node_linear,
    padded_linear,
)
from gator.models.language import GatorLanguageConfig, GatorLanguageModel


class DenseKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    @staticmethod
    def _gradient_tuple(
        output: torch.Tensor,
        operands: tuple[torch.Tensor, ...],
        probe: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad((output * probe).sum(), operands)

    def test_grouped_cell_bmm_matches_einsum_values_and_gradients(self) -> None:
        base = (
            torch.randn(2, 3, 5, 7, dtype=torch.float64),
            torch.randn(3, 11, 7, dtype=torch.float64),
            torch.randn(3, 11, dtype=torch.float64),
        )
        reference_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        grouped_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        reference = cell_linear(
            *reference_inputs, backend="einsum"
        )
        grouped = cell_linear(
            *grouped_inputs,
            backend="grouped_bmm",
            pad_to_multiple=8,
        )
        torch.testing.assert_close(grouped, reference, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(reference)
        reference_gradients = self._gradient_tuple(
            reference, reference_inputs, probe
        )
        grouped_gradients = self._gradient_tuple(
            grouped, grouped_inputs, probe
        )
        for actual, expected in zip(grouped_gradients, reference_gradients):
            torch.testing.assert_close(
                actual, expected, rtol=1e-12, atol=1e-12
            )

    def test_grouped_cell_node_bmm_matches_einsum_and_gradients(self) -> None:
        base = (
            torch.randn(2, 3, 4, 5, 7, dtype=torch.float64),
            torch.randn(3, 5, 9, 7, dtype=torch.float64),
            torch.randn(3, 5, 9, dtype=torch.float64),
        )
        reference_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        grouped_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        reference = cell_node_linear(
            *reference_inputs, backend="einsum"
        )
        grouped = cell_node_linear(
            *grouped_inputs,
            backend="grouped_bmm",
            pad_to_multiple=8,
        )
        torch.testing.assert_close(grouped, reference, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(reference)
        reference_gradients = self._gradient_tuple(
            reference, reference_inputs, probe
        )
        grouped_gradients = self._gradient_tuple(
            grouped, grouped_inputs, probe
        )
        for actual, expected in zip(grouped_gradients, reference_gradients):
            torch.testing.assert_close(
                actual, expected, rtol=1e-12, atol=1e-12
            )

    def test_folded_node_projection_matches_expanded_node_inputs(self) -> None:
        base = (
            torch.randn(2, 3, 4, 7, dtype=torch.float64),
            torch.randn(3, 5, 7, dtype=torch.float64),
            torch.randn(3, 5, 9, 7, dtype=torch.float64),
            torch.randn(3, 5, 9, dtype=torch.float64),
        )
        reference_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        folded_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        reference = folded_cell_node_linear(
            *reference_inputs, backend="einsum"
        )
        folded = folded_cell_node_linear(
            *folded_inputs,
            backend="folded_bmm",
            pad_to_multiple=8,
        )
        torch.testing.assert_close(folded, reference, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(reference)
        reference_gradients = self._gradient_tuple(
            reference, reference_inputs, probe
        )
        folded_gradients = self._gradient_tuple(
            folded, folded_inputs, probe
        )
        for actual, expected in zip(folded_gradients, reference_gradients):
            torch.testing.assert_close(
                actual, expected, rtol=1e-12, atol=1e-12
            )

    def test_padded_shared_linear_matches_unpadded_values_and_gradients(self) -> None:
        base = (
            torch.randn(2, 3, 4, 7, dtype=torch.float64),
            torch.randn(5, 7, dtype=torch.float64),
            torch.randn(5, dtype=torch.float64),
        )
        reference_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        padded_inputs = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        reference = padded_linear(*reference_inputs)
        padded = padded_linear(*padded_inputs, pad_to_multiple=8)
        torch.testing.assert_close(padded, reference, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(reference)
        reference_gradients = self._gradient_tuple(
            reference, reference_inputs, probe
        )
        padded_gradients = self._gradient_tuple(
            padded, padded_inputs, probe
        )
        for actual, expected in zip(padded_gradients, reference_gradients):
            torch.testing.assert_close(
                actual, expected, rtol=1e-12, atol=1e-12
            )

    def test_invalid_shapes_and_padding_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            cell_linear(
                torch.zeros(2, 3, 4),
                torch.zeros(3, 5, 4),
                torch.zeros(3, 5),
            )
        with self.assertRaises(ValueError):
            padded_linear(
                torch.zeros(2, 4),
                torch.zeros(3, 4),
                pad_to_multiple=0,
            )

    def test_full_dense_gator_matches_einsum_backend_and_gradients(self) -> None:
        common = dict(
            vocab_size=64,
            d_model=4,
            embedding_dim=4,
            n_cells=4,
            n_nodes=4,
            relays=2,
            neighbors=2,
            lexical_mode="communal",
            node_topology="symmetric",
            cell_transport_mode="stewardship",
        )
        reference = GatorLanguageModel(
            GatorLanguageConfig(
                **common,
                dense_router_backend="einsum",
                dense_node_backend="einsum",
                dense_kernel_padding=1,
            )
        )
        grouped = GatorLanguageModel(
            GatorLanguageConfig(
                **common,
                dense_router_backend="grouped_bmm",
                dense_node_backend="folded_bmm",
                dense_kernel_padding=8,
            )
        )
        grouped.load_state_dict(reference.state_dict())
        context = torch.randint(0, 64, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        probe = torch.randn(2, 3, 64)

        expected = reference(
            context,
            mask,
            output_slots=3,
            temperature=0.9,
            stewardship_temperature=1.1,
            stewardship_prune=False,
        )
        actual = grouped(
            context,
            mask,
            output_slots=3,
            temperature=0.9,
            stewardship_temperature=1.1,
            stewardship_prune=False,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (expected * probe).sum().backward()
        (actual * probe).sum().backward()
        for name in (
            "address_to_state",
            "address_to_edges",
            "step_projection",
            "router_weight",
            "node_weight",
            "handoff_weight",
            "address_to_readout",
        ):
            expected_gradient = dict(reference.named_parameters())[name].grad
            actual_gradient = dict(grouped.named_parameters())[name].grad
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=5e-5,
                atol=5e-6,
                msg=lambda message, parameter=name: (
                    f"{parameter} gradient mismatch: {message}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
