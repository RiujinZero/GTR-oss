from __future__ import annotations

import unittest

import torch

from gator.models.hierarchical_ingress import (
    HierarchicalIngressLayout,
    estimate_hierarchical_ingress_cost,
    hierarchical_packed_ingress,
)


def dense_hierarchical_oracle(
    packet_features: torch.Tensor,
    node_keys: torch.Tensor,
    node_bias: torch.Tensor,
    group_keys: torch.Tensor,
    group_bias: torch.Tensor,
    layout: HierarchicalIngressLayout,
    *,
    top_k: int,
    temperature: float,
    mass_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_logits = (
        torch.einsum("...r,gr->...g", packet_features, group_keys)
        + group_bias
    )
    node_logits = (
        torch.einsum("...r,nr->...n", packet_features, node_keys)
        + node_bias
    )
    logits = (
        node_logits + group_logits[..., layout.node_to_group]
    ) / temperature
    probabilities = logits.softmax(dim=-1)
    _, nodes = logits.topk(top_k, dim=-1)
    mass = probabilities.gather(-1, nodes)
    valid = torch.ones_like(nodes, dtype=torch.bool)
    if mass_threshold > 0.0:
        valid &= mass.ge(mass_threshold)
        valid[..., 0] = True
        mass = torch.where(valid, mass, torch.zeros_like(mass))
    mass = mass / mass.sum(dim=-1, keepdim=True)
    dense = torch.zeros_like(probabilities).scatter_add(-1, nodes, mass)
    return nodes, mass, valid, dense


def derived_group_parameters(
    node_keys: torch.Tensor,
    node_bias: torch.Tensor,
    layout: HierarchicalIngressLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = layout.valid.to(node_keys.dtype)
    count = valid.sum(dim=-1, keepdim=True)
    keys = (
        node_keys[layout.nodes] * valid[..., None]
    ).sum(dim=1) / count
    bias = (
        node_bias[layout.nodes] * valid
    ).sum(dim=1) / count.squeeze(-1)
    return keys, bias


class HierarchicalIngressLayoutTests(unittest.TestCase):
    def test_balanced_layout_is_deterministic_complete_and_padded(self) -> None:
        first = HierarchicalIngressLayout.balanced(
            n_nodes=10, n_groups=3
        )
        second = HierarchicalIngressLayout.balanced(
            n_nodes=10, n_groups=3
        )
        self.assertTrue(torch.equal(first.nodes, second.nodes))
        self.assertTrue(torch.equal(first.valid, second.valid))
        self.assertTrue(
            torch.equal(first.node_to_group, second.node_to_group)
        )
        self.assertEqual(first.nodes.shape, (3, 4))
        self.assertEqual(first.valid.sum(dim=-1).tolist(), [4, 3, 3])
        self.assertEqual(
            first.nodes[first.valid].tolist(), list(range(10))
        )
        self.assertTrue(bool(first.nodes[~first.valid].eq(0).all()))
        self.assertEqual(
            first.node_to_group.tolist(), [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        )

    def test_layout_rejects_missing_duplicate_and_empty_groups(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            HierarchicalIngressLayout.from_padded(
                torch.tensor([[0, 1], [1, 2]]), n_nodes=4
            )
        with self.assertRaisesRegex(RuntimeError, "at least one node"):
            HierarchicalIngressLayout.from_padded(
                torch.tensor([[0, 1], [-1, -1]]), n_nodes=2
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            HierarchicalIngressLayout.balanced(n_nodes=3, n_groups=4)


class HierarchicalPackedIngressTests(unittest.TestCase):
    def _all_group_equivalence(self, *, learned_groups: bool) -> None:
        torch.manual_seed(521 if learned_groups else 522)
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=7, n_groups=3
        )
        base_features = torch.randn(2, 3, 2, 4, dtype=torch.float64)
        base_node_keys = torch.randn(7, 4, dtype=torch.float64)
        base_node_bias = torch.randn(7, dtype=torch.float64)
        base_group_keys = torch.randn(3, 4, dtype=torch.float64)
        base_group_bias = torch.randn(3, dtype=torch.float64)

        actual_features = base_features.clone().requires_grad_(True)
        actual_node_keys = base_node_keys.clone().requires_grad_(True)
        actual_node_bias = base_node_bias.clone().requires_grad_(True)
        oracle_features = base_features.clone().requires_grad_(True)
        oracle_node_keys = base_node_keys.clone().requires_grad_(True)
        oracle_node_bias = base_node_bias.clone().requires_grad_(True)
        if learned_groups:
            actual_group_keys = base_group_keys.clone().requires_grad_(True)
            actual_group_bias = base_group_bias.clone().requires_grad_(True)
            oracle_group_keys = base_group_keys.clone().requires_grad_(True)
            oracle_group_bias = base_group_bias.clone().requires_grad_(True)
        else:
            actual_group_keys = None
            actual_group_bias = None
            oracle_group_keys, oracle_group_bias = derived_group_parameters(
                oracle_node_keys, oracle_node_bias, layout
            )

        actual = hierarchical_packed_ingress(
            actual_features,
            actual_node_keys,
            actual_node_bias,
            layout=layout,
            top_groups=layout.n_groups,
            top_k=4,
            group_keys=actual_group_keys,
            group_bias=actual_group_bias,
            temperature=0.73,
        )
        expected_nodes, expected_mass, expected_valid, expected_dense = (
            dense_hierarchical_oracle(
                oracle_features,
                oracle_node_keys,
                oracle_node_bias,
                oracle_group_keys,
                oracle_group_bias,
                layout,
                top_k=4,
                temperature=0.73,
            )
        )
        self.assertTrue(torch.equal(actual.nodes, expected_nodes))
        self.assertTrue(torch.equal(actual.valid, expected_valid))
        torch.testing.assert_close(actual.mass, expected_mass)
        torch.testing.assert_close(actual.to_dense(), expected_dense)

        probe = torch.randn_like(expected_dense)
        actual_inputs = [
            actual_features,
            actual_node_keys,
            actual_node_bias,
        ]
        oracle_inputs = [
            oracle_features,
            oracle_node_keys,
            oracle_node_bias,
        ]
        if learned_groups:
            assert actual_group_keys is not None
            assert actual_group_bias is not None
            actual_inputs.extend([actual_group_keys, actual_group_bias])
            oracle_inputs.extend([oracle_group_keys, oracle_group_bias])
        actual_gradients = torch.autograd.grad(
            (actual.to_dense() * probe).sum(), actual_inputs
        )
        oracle_gradients = torch.autograd.grad(
            (expected_dense * probe).sum(), oracle_inputs
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, oracle_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1e-10,
                atol=1e-11,
            )
        if learned_groups:
            self.assertGreater(float(actual_gradients[-2].abs().sum()), 0.0)
            self.assertGreater(float(actual_gradients[-1].abs().sum()), 0.0)

    def test_all_groups_matches_dense_hierarchical_values_and_gradients(self) -> None:
        self._all_group_equivalence(learned_groups=True)

    def test_derived_group_parameters_match_dense_values_and_gradients(self) -> None:
        self._all_group_equivalence(learned_groups=False)

    def test_all_groups_with_zero_group_prior_recovers_dense_node_ingress(self) -> None:
        torch.manual_seed(524)
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=9, n_groups=3
        )
        base_features = torch.randn(2, 4, dtype=torch.float64)
        base_keys = torch.randn(9, 4, dtype=torch.float64)
        base_bias = torch.randn(9, dtype=torch.float64)
        actual_features = base_features.clone().requires_grad_(True)
        actual_keys = base_keys.clone().requires_grad_(True)
        actual_bias = base_bias.clone().requires_grad_(True)
        expected_features = base_features.clone().requires_grad_(True)
        expected_keys = base_keys.clone().requires_grad_(True)
        expected_bias = base_bias.clone().requires_grad_(True)

        actual = hierarchical_packed_ingress(
            actual_features,
            actual_keys,
            actual_bias,
            layout=layout,
            top_groups=layout.n_groups,
            top_k=3,
            group_keys=torch.zeros(3, 4, dtype=torch.float64),
            group_bias=torch.zeros(3, dtype=torch.float64),
            temperature=0.9,
        )
        logits = (
            torch.einsum("...r,nr->...n", expected_features, expected_keys)
            + expected_bias
        ) / 0.9
        _, nodes = logits.topk(3, dim=-1)
        mass = logits.softmax(dim=-1).gather(-1, nodes)
        mass = mass / mass.sum(dim=-1, keepdim=True)
        expected = torch.zeros_like(logits).scatter_add(-1, nodes, mass)
        torch.testing.assert_close(actual.to_dense(), expected)

        probe = torch.randn_like(expected)
        actual_gradients = torch.autograd.grad(
            (actual.to_dense() * probe).sum(),
            (actual_features, actual_keys, actual_bias),
        )
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (expected_features, expected_keys, expected_bias),
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1e-10,
                atol=1e-11,
            )

    def test_padded_selected_group_returns_fixed_capacity_frontier(self) -> None:
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=8, n_groups=3
        )
        features = torch.zeros(2, 3)
        node_keys = torch.zeros(8, 3)
        node_bias = torch.arange(8, dtype=torch.float32) / 10
        group_keys = torch.zeros(3, 3)
        group_bias = torch.tensor([-20.0, -10.0, 10.0])
        frontier = hierarchical_packed_ingress(
            features,
            node_keys,
            node_bias,
            layout=layout,
            top_groups=1,
            top_k=3,
            group_keys=group_keys,
            group_bias=group_bias,
        )
        self.assertEqual(frontier.nodes.shape, (2, 3))
        self.assertEqual(frontier.valid.sum(dim=-1).tolist(), [2, 2])
        self.assertEqual(
            set(frontier.nodes[0, frontier.valid[0]].tolist()), {6, 7}
        )
        self.assertTrue(bool(frontier.mass[~frontier.valid].eq(0).all()))
        self.assertTrue(bool(frontier.nodes[~frontier.valid].eq(0).all()))
        torch.testing.assert_close(
            frontier.mass.sum(dim=-1), torch.ones(2)
        )

    def test_each_node_is_reachable_through_its_group(self) -> None:
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=11, n_groups=4
        )
        features = torch.zeros(1, 5)
        node_keys = torch.zeros(11, 5)
        node_bias = torch.arange(11, dtype=torch.float32) / 100
        group_keys = torch.zeros(4, 5)
        reached: set[int] = set()
        for group in range(layout.n_groups):
            group_bias = torch.full((layout.n_groups,), -100.0)
            group_bias[group] = 100.0
            frontier = hierarchical_packed_ingress(
                features,
                node_keys,
                node_bias,
                layout=layout,
                top_groups=1,
                top_k=layout.group_capacity,
                group_keys=group_keys,
                group_bias=group_bias,
            )
            reached.update(frontier.nodes[frontier.valid].tolist())
        self.assertEqual(reached, set(range(layout.n_nodes)))

    def test_threshold_retains_the_strongest_route(self) -> None:
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=6, n_groups=2
        )
        frontier = hierarchical_packed_ingress(
            torch.zeros(1, 2),
            torch.zeros(6, 2),
            torch.zeros(6),
            layout=layout,
            top_groups=2,
            top_k=4,
            group_keys=torch.zeros(2, 2),
            group_bias=torch.zeros(2),
            mass_threshold=0.99,
        )
        self.assertEqual(int(frontier.valid.sum()), 1)
        torch.testing.assert_close(frontier.mass.sum(), torch.tensor(1.0))

    def test_fullgraph_compile_matches_eager(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        torch.manual_seed(523)
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=8, n_groups=2
        )

        def run(
            features: torch.Tensor,
            node_keys: torch.Tensor,
            node_bias: torch.Tensor,
            group_keys: torch.Tensor,
            group_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            frontier = hierarchical_packed_ingress(
                features,
                node_keys,
                node_bias,
                layout=layout,
                top_groups=1,
                top_k=3,
                group_keys=group_keys,
                group_bias=group_bias,
                temperature=0.8,
                mass_threshold=0.01,
            )
            return frontier.nodes, frontier.mass, frontier.valid

        inputs = (
            torch.randn(2, 3, 4),
            torch.randn(8, 4),
            torch.randn(8),
            torch.randn(2, 4),
            torch.randn(2),
        )
        eager = run(*inputs)
        compiled = torch.compile(
            run, backend="eager", fullgraph=True
        )(*inputs)
        self.assertTrue(torch.equal(compiled[0], eager[0]))
        self.assertTrue(torch.equal(compiled[2], eager[2]))
        torch.testing.assert_close(compiled[1], eager[1])

    def test_cost_estimate_exposes_compute_and_parameter_tradeoff(self) -> None:
        cost = estimate_hierarchical_ingress_cost(
            packets=1024,
            n_nodes=308,
            rank=96,
            n_groups=28,
            group_capacity=11,
            top_groups=2,
        )
        self.assertEqual(cost.candidate_slots, 22)
        self.assertEqual(cost.node_parameters, 308 * 97)
        self.assertEqual(cost.learned_group_parameters, 28 * 97)
        self.assertEqual(
            cost.dense_score_macs, 1024 * 308 * 96
        )
        self.assertEqual(
            cost.hierarchical_score_macs, 1024 * (28 + 22) * 96
        )
        self.assertLess(cost.score_ratio, 0.17)


if __name__ == "__main__":
    unittest.main()
