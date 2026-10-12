from __future__ import annotations

import unittest

import torch

from gator.models.packed_frontier import (
    PackedNodeFrontier,
    dense_to_packed_frontier,
    merge_packed_destinations,
    packed_frontier_to_dense,
    packed_node_occupancy,
    transport_packed_frontier,
)


def dense_top_k_oracle(
    dense: torch.Tensor,
    *,
    top_k: int,
    mass_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mass, nodes = dense.topk(top_k, dim=-1)
    valid = mass.gt(0)
    if mass_threshold > 0:
        valid = valid & mass.ge(mass_threshold)
    strongest = mass.argmax(dim=-1, keepdim=True)
    valid = valid.scatter(-1, strongest, True)
    mass = torch.where(valid, mass, torch.zeros_like(mass))
    mass = mass / mass.sum(dim=-1, keepdim=True)
    nodes = torch.where(valid, nodes, torch.zeros_like(nodes))
    selected = torch.zeros_like(dense).scatter_add(-1, nodes, mass)
    return nodes, mass, valid, selected


def dense_transport_oracle(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    destinations: torch.Tensor,
    transition: torch.Tensor,
    *,
    n_nodes: int,
    top_k: int,
    mass_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_nodes = destinations[nodes]
    candidate_mass = (
        torch.where(valid, mass, torch.zeros_like(mass)).unsqueeze(-1)
        * transition
    )
    dense = mass.new_zeros(mass.shape[:-1] + (n_nodes,))
    dense = dense.scatter_add(
        -1,
        candidate_nodes.flatten(-2),
        candidate_mass.flatten(-2),
    )
    return dense_top_k_oracle(
        dense,
        top_k=top_k,
        mass_threshold=mass_threshold,
    )


def dense_occupancy_oracle(
    frontier: PackedNodeFrontier,
    *,
    position_valid: torch.Tensor,
    causal: bool,
    query_nodes: torch.Tensor,
    query_valid: torch.Tensor,
) -> torch.Tensor:
    dense = packed_frontier_to_dense(frontier)
    lanes = dense.shape[2]
    position_mass = (
        dense * position_valid[:, :, None, None].to(dense.dtype)
    ).sum(dim=2)
    if causal:
        occupancy = position_mass.cumsum(dim=1)
        denominator = position_valid.cumsum(dim=1).to(dense.dtype)
    else:
        occupancy = position_mass.sum(dim=1, keepdim=True).expand(
            -1, dense.shape[1], -1
        )
        denominator = position_valid.sum(dim=1, keepdim=True).to(
            dense.dtype
        ).expand(-1, dense.shape[1])
    occupancy = occupancy / (denominator * lanes).clamp_min(1).unsqueeze(-1)
    gathered = occupancy.gather(
        -1, query_nodes.reshape(*query_nodes.shape[:2], -1)
    ).reshape(query_nodes.shape)
    live = query_valid
    return torch.where(live, gathered, torch.zeros_like(gathered))


class PackedFrontierConversionTests(unittest.TestCase):
    def test_dense_top_k_threshold_matches_oracle_and_gradients(self) -> None:
        base = torch.tensor(
            [
                [[0.42, 0.03, 0.31, 0.19, 0.05],
                 [0.08, 0.74, 0.06, 0.07, 0.05]],
                [[0.11, 0.17, 0.29, 0.04, 0.39],
                 [0.04, 0.09, 0.12, 0.69, 0.06]],
            ],
            dtype=torch.float64,
        )
        actual_input = base.clone().requires_grad_(True)
        oracle_input = base.clone().requires_grad_(True)

        packed = dense_to_packed_frontier(
            actual_input, top_k=3, mass_threshold=0.10
        )
        expected_nodes, expected_mass, expected_valid, expected_dense = (
            dense_top_k_oracle(
                oracle_input, top_k=3, mass_threshold=0.10
            )
        )
        self.assertTrue(torch.equal(packed.nodes, expected_nodes))
        self.assertTrue(torch.equal(packed.valid, expected_valid))
        torch.testing.assert_close(packed.mass, expected_mass)
        torch.testing.assert_close(packed.to_dense(), expected_dense)
        torch.testing.assert_close(
            packed.masked_mass.sum(dim=-1),
            torch.ones_like(packed.mass[..., 0]),
        )

        probe = torch.randn_like(expected_dense)
        actual_gradient = torch.autograd.grad(
            (packed.to_dense() * probe).sum(), actual_input
        )[0]
        expected_gradient = torch.autograd.grad(
            (expected_dense * probe).sum(), oracle_input
        )[0]
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )

    def test_dense_diagnostic_merges_duplicates_and_preserves_gradient(self) -> None:
        base_mass = torch.tensor(
            [[0.20, 0.30, 0.50, 9.0]], dtype=torch.float64
        )
        mass = base_mass.clone().requires_grad_(True)
        frontier = PackedNodeFrontier(
            nodes=torch.tensor([[1, 1, 3, 4]]),
            mass=mass,
            valid=torch.tensor([[True, True, True, False]]),
            n_nodes=5,
        )
        dense = packed_frontier_to_dense(frontier)
        torch.testing.assert_close(
            dense,
            torch.tensor(
                [[0.0, 0.5, 0.0, 0.5, 0.0]], dtype=torch.float64
            ),
        )
        probe = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64
        )
        gradient = torch.autograd.grad((dense * probe).sum(), mass)[0]
        torch.testing.assert_close(
            gradient,
            torch.tensor([[2.0, 2.0, 4.0, 0.0]], dtype=torch.float64),
        )

    def test_record_rejects_structural_shape_and_dtype_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            PackedNodeFrontier(
                nodes=torch.zeros(2, 2, dtype=torch.long),
                mass=torch.zeros(2, 3),
                valid=torch.zeros(2, 2, dtype=torch.bool),
                n_nodes=4,
            )
        with self.assertRaisesRegex(ValueError, "int64"):
            PackedNodeFrontier(
                nodes=torch.zeros(2, dtype=torch.int32),
                mass=torch.ones(2),
                valid=torch.ones(2, dtype=torch.bool),
                n_nodes=4,
            )


class PackedFrontierTransportTests(unittest.TestCase):
    def test_collisions_merge_before_top_k(self) -> None:
        frontier = PackedNodeFrontier(
            nodes=torch.tensor([[0, 1]]),
            mass=torch.tensor([[0.55, 0.45]], dtype=torch.float64),
            valid=torch.ones(1, 2, dtype=torch.bool),
            n_nodes=5,
        )
        destinations = torch.tensor(
            [[2, 3], [2, 4], [0, 1], [1, 2], [2, 3]],
            dtype=torch.long,
        )
        transition = torch.tensor(
            [[[0.60, 0.40], [0.60, 0.40]]], dtype=torch.float64
        )
        transported = transport_packed_frontier(
            frontier, destinations, transition, top_k=2
        )
        expected = torch.tensor(
            [[0.0, 0.0, 0.60 / 0.82, 0.22 / 0.82, 0.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(transported.to_dense(), expected)
        self.assertEqual(transported.capacity, 2)
        self.assertEqual(int(transported.valid.sum()), 2)

    def test_transport_matches_dense_scatter_topk_and_gradients(self) -> None:
        nodes = torch.tensor(
            [[[0, 2, 4], [1, 3, 5]], [[5, 1, 2], [4, 0, 3]]],
            dtype=torch.long,
        )
        valid = torch.tensor(
            [
                [[True, True, True], [True, False, True]],
                [[True, True, False], [True, True, True]],
            ]
        )
        base_mass = torch.tensor(
            [
                [[0.51, 0.31, 0.18], [0.62, 0.0, 0.38]],
                [[0.58, 0.42, 0.0], [0.46, 0.34, 0.20]],
            ],
            dtype=torch.float64,
        )
        destinations = torch.tensor(
            [
                [1, 2, 4],
                [2, 3, 5],
                [1, 4, 5],
                [0, 2, 4],
                [0, 1, 5],
                [1, 2, 3],
            ],
            dtype=torch.long,
        )
        base_transition = torch.tensor(
            [
                [
                    [[0.51, 0.31, 0.18], [0.22, 0.53, 0.25], [0.61, 0.24, 0.15]],
                    [[0.17, 0.29, 0.54], [0.30, 0.31, 0.39], [0.55, 0.27, 0.18]],
                ],
                [
                    [[0.43, 0.35, 0.22], [0.19, 0.57, 0.24], [0.20, 0.30, 0.50]],
                    [[0.27, 0.45, 0.28], [0.48, 0.33, 0.19], [0.38, 0.21, 0.41]],
                ],
            ],
            dtype=torch.float64,
        )
        actual_mass = base_mass.clone().requires_grad_(True)
        actual_transition = base_transition.clone().requires_grad_(True)
        oracle_mass = base_mass.clone().requires_grad_(True)
        oracle_transition = base_transition.clone().requires_grad_(True)
        frontier = PackedNodeFrontier(
            nodes=nodes,
            mass=actual_mass,
            valid=valid,
            n_nodes=6,
        )
        actual = transport_packed_frontier(
            frontier,
            destinations,
            actual_transition,
            top_k=3,
            mass_threshold=0.06,
        )
        expected_nodes, expected_mass, expected_valid, expected_dense = (
            dense_transport_oracle(
                nodes,
                oracle_mass,
                valid,
                destinations,
                oracle_transition,
                n_nodes=6,
                top_k=3,
                mass_threshold=0.06,
            )
        )
        self.assertTrue(torch.equal(actual.nodes, expected_nodes))
        self.assertTrue(torch.equal(actual.valid, expected_valid))
        torch.testing.assert_close(actual.mass, expected_mass)
        torch.testing.assert_close(actual.to_dense(), expected_dense)
        torch.testing.assert_close(
            actual.masked_mass.sum(dim=-1),
            torch.ones_like(actual.mass[..., 0]),
        )

        probe = torch.randn_like(expected_dense)
        actual_gradients = torch.autograd.grad(
            (actual.to_dense() * probe).sum(),
            (actual_mass, actual_transition),
        )
        expected_gradients = torch.autograd.grad(
            (expected_dense * probe).sum(),
            (oracle_mass, oracle_transition),
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1e-11,
                atol=1e-12,
            )

    def test_threshold_retains_one_node_and_fixed_capacity(self) -> None:
        candidate_nodes = torch.tensor([[3, 3, 1, 2, 4, 5]])
        candidate_mass = torch.tensor(
            [[0.13, 0.14, 0.12, 0.11, 0.10, 0.08]],
            dtype=torch.float64,
        )
        merged = merge_packed_destinations(
            candidate_nodes,
            candidate_mass,
            torch.ones_like(candidate_nodes, dtype=torch.bool),
            n_nodes=6,
            top_k=4,
            mass_threshold=0.20,
        )
        self.assertEqual(merged.nodes.shape, (1, 4))
        self.assertEqual(int(merged.valid.sum()), 1)
        torch.testing.assert_close(
            merged.to_dense(),
            torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        )


class PackedFrontierOccupancyTests(unittest.TestCase):
    def _case(self) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        nodes = torch.tensor(
            [
                [
                    [[0, 2], [1, 2]],
                    [[1, 3], [1, 4]],
                    [[2, 4], [0, 4]],
                    [[3, 4], [2, 3]],
                ],
                [
                    [[1, 4], [0, 1]],
                    [[2, 3], [2, 4]],
                    [[0, 3], [1, 3]],
                    [[0, 2], [2, 4]],
                ],
            ],
            dtype=torch.long,
        )
        mass = torch.tensor(
            [
                [
                    [[0.7, 0.3], [0.4, 0.6]],
                    [[0.6, 0.4], [0.8, 0.2]],
                    [[0.2, 0.8], [0.3, 0.7]],
                    [[0.55, 0.45], [0.65, 0.35]],
                ],
                [
                    [[0.25, 0.75], [0.35, 0.65]],
                    [[0.45, 0.55], [0.58, 0.42]],
                    [[0.62, 0.38], [0.33, 0.67]],
                    [[0.51, 0.49], [0.73, 0.27]],
                ],
            ],
            dtype=torch.float64,
        )
        position_valid = torch.tensor(
            [[True, True, True, True], [True, False, True, True]]
        )
        query_nodes = torch.tensor(
            [
                [[0, 2, 4], [1, 2, 3], [4, 0, 2], [3, 1, 4]],
                [[1, 0, 4], [2, 3, 1], [3, 0, 1], [2, 4, 0]],
            ],
            dtype=torch.long,
        )
        query_valid = torch.tensor(
            [
                [[True, True, True], [True, False, True], [True, True, True], [True, True, True]],
                [[True, True, True], [True, True, True], [False, True, True], [True, True, True]],
            ]
        )
        return nodes, mass, position_valid, query_nodes, query_valid

    def _assert_mode(self, causal: bool) -> None:
        nodes, base_mass, position_valid, query_nodes, query_valid = self._case()
        actual_mass = base_mass.clone().requires_grad_(True)
        oracle_mass = base_mass.clone().requires_grad_(True)
        actual_frontier = PackedNodeFrontier(
            nodes=nodes,
            mass=actual_mass,
            valid=torch.ones_like(nodes, dtype=torch.bool),
            n_nodes=5,
        )
        oracle_frontier = PackedNodeFrontier(
            nodes=nodes,
            mass=oracle_mass,
            valid=torch.ones_like(nodes, dtype=torch.bool),
            n_nodes=5,
        )
        actual = packed_node_occupancy(
            actual_frontier,
            position_valid=position_valid,
            causal=causal,
            query_nodes=query_nodes,
            query_valid=query_valid,
        )
        expected = dense_occupancy_oracle(
            oracle_frontier,
            position_valid=position_valid,
            causal=causal,
            query_nodes=query_nodes,
            query_valid=query_valid,
        )
        torch.testing.assert_close(actual, expected)
        probe = torch.randn_like(actual)
        actual_gradient = torch.autograd.grad(
            (actual * probe).sum(), actual_mass
        )[0]
        expected_gradient = torch.autograd.grad(
            (expected * probe).sum(), oracle_mass
        )[0]
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-11,
            atol=1e-12,
        )

    def test_causal_occupancy_matches_dense_values_and_gradients(self) -> None:
        self._assert_mode(causal=True)

    def test_noncausal_occupancy_matches_dense_values_and_gradients(self) -> None:
        self._assert_mode(causal=False)

    def test_default_queries_return_frontier_aligned_occupancy(self) -> None:
        nodes, mass, position_valid, _, _ = self._case()
        frontier = PackedNodeFrontier(
            nodes=nodes,
            mass=mass,
            valid=torch.ones_like(nodes, dtype=torch.bool),
            n_nodes=5,
        )
        actual = packed_node_occupancy(
            frontier, position_valid=position_valid, causal=True
        )
        expected = dense_occupancy_oracle(
            frontier,
            position_valid=position_valid,
            causal=True,
            query_nodes=nodes,
            query_valid=frontier.valid,
        )
        self.assertEqual(actual.shape, nodes.shape)
        torch.testing.assert_close(actual, expected)

    def test_scatter_occupancy_branch_matches_dense_oracle(self) -> None:
        # N <= lanes * capacity selects the node-only scatter/cumsum path.
        nodes = torch.tensor(
            [[[[0, 1], [1, 2]], [[2, 3], [0, 3]], [[1, 3], [2, 0]]]],
            dtype=torch.long,
        )
        base_mass = torch.tensor(
            [[[[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.8], [0.5, 0.5]], [[0.9, 0.1], [0.65, 0.35]]]],
            dtype=torch.float64,
        )
        position_valid = torch.tensor([[True, True, False]])
        query_nodes = torch.tensor(
            [[[[0, 3], [2, 1]], [[3, 1], [0, 2]], [[1, 2], [3, 0]]]],
            dtype=torch.long,
        )
        query_valid = torch.ones_like(query_nodes, dtype=torch.bool)
        actual_mass = base_mass.clone().requires_grad_(True)
        expected_mass = base_mass.clone().requires_grad_(True)
        actual_frontier = PackedNodeFrontier(
            nodes=nodes,
            mass=actual_mass,
            valid=torch.ones_like(nodes, dtype=torch.bool),
            n_nodes=4,
        )
        expected_frontier = PackedNodeFrontier(
            nodes=nodes,
            mass=expected_mass,
            valid=torch.ones_like(nodes, dtype=torch.bool),
            n_nodes=4,
        )
        actual = packed_node_occupancy(
            actual_frontier,
            position_valid=position_valid,
            causal=True,
            query_nodes=query_nodes,
            query_valid=query_valid,
        )
        expected = dense_occupancy_oracle(
            expected_frontier,
            position_valid=position_valid,
            causal=True,
            query_nodes=query_nodes,
            query_valid=query_valid,
        )
        torch.testing.assert_close(actual, expected)
        probe = torch.randn_like(actual)
        actual_grad = torch.autograd.grad((actual * probe).sum(), actual_mass)[0]
        expected_grad = torch.autograd.grad((expected * probe).sum(), expected_mass)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
