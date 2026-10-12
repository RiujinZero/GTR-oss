from __future__ import annotations

import unittest

import torch

from gator.models.packed_memory import (
    PackedConvergentMemoryDiagnostics,
    packed_convergent_node_context,
    packed_convergent_writer_features,
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


class PackedConvergentMemoryTests(unittest.TestCase):
    def _compare_case(
        self,
        *,
        causal: bool,
        duplicates_and_tiny_mass: bool,
    ) -> None:
        torch.manual_seed(211 if causal else 199)
        batch, tokens, lanes, width, route_width, n_nodes = 2, 5, 3, 4, 4, 7
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
        nodes = torch.randint(
            0,
            n_nodes,
            (batch, tokens, lanes, route_width),
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, -1, 1, -1] = False
        valid[1, 1, 2, -2:] = False
        raw_mass = torch.rand(
            batch,
            tokens,
            lanes,
            route_width,
            dtype=torch.float64,
        )
        if duplicates_and_tiny_mass:
            nodes[..., 1] = nodes[..., 0]
            raw_mass[..., 0] = 2e-10
            raw_mass[..., 1] = 3e-10
            raw_mass[0, 2, 0, 2] = 7e-10
        mass = raw_mass.requires_grad_(True)
        write_values = torch.rand(batch, tokens, dtype=torch.float64)
        write_values[0, 1] = 0.0
        write_values[1, -2:] = 0.0
        write = write_values.requires_grad_(True)
        receiver_kernel = model._receiver_kernel(
            tokens,
            hidden.device,
            causal=causal,
        )

        dense_locations = _dense_locations(nodes, mass, valid, n_nodes)
        expected_memory = model._node_memory(
            hidden,
            dense_locations,
            write,
            receiver_kernel=receiver_kernel,
        )
        expected = torch.einsum(
            "btln,btnd->btld",
            dense_locations,
            expected_memory,
        )
        actual, diagnostics = packed_convergent_node_context(
            hidden,
            nodes,
            mass,
            valid,
            write,
            receiver_kernel,
            return_diagnostics=True,
        )

        self.assertIsInstance(diagnostics, PackedConvergentMemoryDiagnostics)
        self.assertEqual(
            tuple(diagnostics.writer_features.shape),
            (batch, n_nodes, tokens, width + 1),
        )
        self.assertEqual(
            tuple(diagnostics.node_memory.shape),
            (batch, tokens, n_nodes, width),
        )
        torch.testing.assert_close(
            diagnostics.node_memory,
            expected_memory,
            atol=2e-12,
            rtol=2e-12,
        )
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

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
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=2e-11,
                rtol=2e-11,
            )

    def test_noncausal_values_and_gradients_match_dense_memory(self) -> None:
        self._compare_case(causal=False, duplicates_and_tiny_mass=False)

    def test_causal_zero_write_duplicate_and_sub_epsilon_routes(self) -> None:
        self._compare_case(causal=True, duplicates_and_tiny_mass=True)

    def test_writer_features_match_dense_equation_with_padding(self) -> None:
        torch.manual_seed(223)
        batch, tokens, lanes, width, route_width, n_nodes = 1, 4, 2, 3, 3, 5
        hidden = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            dtype=torch.float64,
        )
        nodes = torch.tensor(
            [[[[1, 1, 4], [2, 3, 0]]] * tokens],
            dtype=torch.long,
        )
        mass = torch.tensor(
            [[[[2e-10, 3e-10, 1.0], [0.4, 0.6, 9.0]]] * tokens],
            dtype=torch.float64,
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[..., 1, 2] = False
        write = torch.tensor(
            [[1.0, 0.0, 0.5, 1.0]],
            dtype=torch.float64,
        )
        dense = _dense_locations(nodes, mass, valid, n_nodes)
        dense_writer_mass = dense.clamp_min(1e-9) * write[:, :, None, None]
        expected_mass = dense_writer_mass.sum(2)
        expected_state = torch.einsum(
            "btln,btld->btnd",
            dense_writer_mass,
            hidden,
        )
        expected = torch.cat(
            (expected_mass.unsqueeze(-1), expected_state),
            -1,
        ).permute(0, 2, 1, 3)

        actual = packed_convergent_writer_features(
            hidden,
            nodes,
            mass,
            valid,
            write,
            n_nodes=n_nodes,
        )
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
