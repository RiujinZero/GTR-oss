from __future__ import annotations

import unittest

import torch

from gator.models.adaptive_frontier import adaptive_uncertainty_top_k
from gator.models.packed_frontier import PackedNodeFrontier


def dense_adaptive_oracle(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    *,
    n_nodes: int,
    min_k: int,
    max_k: int,
    mode: str,
    entropy_mix: float,
    mass_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent dense-node reference for adaptive packed selection."""

    active_mass = torch.where(valid & mass.gt(0), mass, torch.zeros_like(mass))
    dense = mass.new_zeros(mass.shape[:-1] + (n_nodes,)).scatter_add(
        -1, nodes, active_mass
    )
    probabilities = dense / dense.sum(dim=-1, keepdim=True)
    positive = probabilities.gt(0)
    support = positive.sum(dim=-1)
    entropy = torch.where(
        positive,
        -probabilities
        * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    normalized_entropy = torch.where(
        support.gt(1),
        entropy / support.clamp_min(2).to(mass.dtype).log(),
        torch.zeros_like(entropy),
    ).clamp(0.0, 1.0)

    # Dense node storage is naturally in ascending node-ID order. Stable
    # sorting therefore matches the production tie break.
    order = probabilities.argsort(dim=-1, descending=True, stable=True)
    ranked = probabilities.gather(-1, order)
    top1_margin = (ranked[..., 0] - ranked[..., 1]).clamp(0.0, 1.0)
    margin_uncertainty = 1.0 - top1_margin
    if mode == "entropy":
        uncertainty = normalized_entropy
    elif mode == "margin":
        uncertainty = margin_uncertainty
    else:
        uncertainty = (
            entropy_mix * normalized_entropy
            + (1.0 - entropy_mix) * margin_uncertainty
        )
    uncertainty = uncertainty.clamp(0.0, 1.0)
    selected_k = min_k + torch.floor(
        uncertainty * (max_k - min_k) + 0.5
    ).to(torch.long)
    selected_k = selected_k.clamp(min=min_k, max=max_k)

    ranked = ranked[..., :max_k]
    ranked_nodes = order[..., :max_k]
    rank = torch.arange(max_k, device=mass.device)
    retained = rank < selected_k.unsqueeze(-1)
    retained &= ranked.gt(0)
    if mass_threshold > 0:
        retained &= ranked.ge(mass_threshold)
    strongest = torch.zeros_like(retained)
    strongest[..., 0] = True
    retained |= strongest & ranked.gt(0)
    retained_mass = torch.where(retained, ranked, torch.zeros_like(ranked))
    retained_mass = retained_mass / retained_mass.sum(dim=-1, keepdim=True)
    selected_dense = torch.zeros_like(dense).scatter_add(
        -1,
        torch.where(retained, ranked_nodes, torch.zeros_like(ranked_nodes)),
        retained_mass,
    )
    return (
        selected_dense,
        selected_k,
        normalized_entropy,
        top1_margin,
        uncertainty,
    )


class AdaptiveFrontierTests(unittest.TestCase):
    @staticmethod
    def _frontier(
        mass: torch.Tensor,
        *,
        nodes: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        n_nodes: int | None = None,
    ) -> PackedNodeFrontier:
        if nodes is None:
            nodes = torch.arange(mass.shape[-1]).expand(mass.shape)
        if valid is None:
            valid = torch.ones_like(mass, dtype=torch.bool)
        if n_nodes is None:
            n_nodes = max(mass.shape[-1], int(nodes.max()) + 1)
        return PackedNodeFrontier(
            nodes=nodes.to(dtype=torch.long),
            mass=mass,
            valid=valid,
            n_nodes=n_nodes,
        )

    def test_entropy_expands_uncertain_packets_and_keeps_static_shape(self) -> None:
        frontier = self._frontier(
            torch.tensor(
                [
                    [0.97, 0.01, 0.01, 0.01],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                dtype=torch.float64,
            )
        )
        result = adaptive_uncertainty_top_k(
            frontier, min_k=1, max_k=4, mode="entropy"
        )

        self.assertEqual(tuple(result.frontier.mass.shape), (2, 4))
        self.assertEqual(result.selected_k.tolist(), [1, 4])
        self.assertEqual(result.frontier.valid.sum(-1).tolist(), [1, 4])
        torch.testing.assert_close(
            result.frontier.masked_mass.sum(-1), torch.ones(2, dtype=torch.float64)
        )

    def test_margin_policy_uses_top_two_separation(self) -> None:
        frontier = self._frontier(
            torch.tensor(
                [
                    [0.95, 0.05, 0.00, 0.00],
                    [0.30, 0.29, 0.21, 0.20],
                ],
                dtype=torch.float32,
            )
        )
        result = adaptive_uncertainty_top_k(
            frontier, min_k=1, max_k=4, mode="margin"
        )

        self.assertEqual(result.selected_k.tolist(), [1, 4])
        torch.testing.assert_close(
            result.top1_margin, torch.tensor([0.90, 0.01]), atol=1e-6, rtol=0
        )

    def test_equal_mass_tie_break_is_node_deterministic(self) -> None:
        nodes = torch.tensor([[7, 2, 5, 1]], dtype=torch.long)
        mass = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
        result = adaptive_uncertainty_top_k(
            self._frontier(mass, nodes=nodes, n_nodes=8),
            min_k=2,
            max_k=2,
            mode="hybrid",
        )

        self.assertEqual(result.frontier.nodes.tolist(), [[1, 2]])
        torch.testing.assert_close(
            result.frontier.mass, torch.tensor([[0.5, 0.5]])
        )

    def test_padding_duplicates_and_threshold_keep_one_winner(self) -> None:
        nodes = torch.tensor([[3, 1, 3, 4, 0]], dtype=torch.long)
        mass = torch.tensor(
            [[0.10, 0.10, 0.80, 900.0, 700.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        valid = torch.tensor([[True, True, True, False, False]])
        result = adaptive_uncertainty_top_k(
            self._frontier(mass, nodes=nodes, valid=valid, n_nodes=6),
            min_k=1,
            max_k=4,
            mode="entropy",
            mass_threshold=0.20,
        )

        self.assertEqual(result.selected_k.item(), 2)
        self.assertEqual(result.frontier.nodes.tolist(), [[3, 0, 0, 0]])
        self.assertEqual(result.frontier.valid.tolist(), [[True, False, False, False]])
        torch.testing.assert_close(
            result.frontier.mass, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        )
        result.frontier.mass.sum().backward()
        self.assertEqual(mass.grad[..., 3:].tolist(), [[0.0, 0.0]])

    def test_dense_oracle_matches_values_diagnostics_and_gradients(self) -> None:
        nodes = torch.tensor(
            [
                [[7, 1, 4, 1, 8, 3], [2, 6, 2, 9, 0, 5]],
                [[5, 3, 8, 5, 1, 7], [4, 0, 6, 2, 4, 9]],
            ],
            dtype=torch.long,
        )
        valid = torch.tensor(
            [
                [[True, True, True, True, False, True], [True] * 6],
                [[True, True, True, True, True, False], [True] * 6],
            ]
        )
        base = torch.tensor(
            [
                [[0.08, 0.14, 0.29, 0.17, 99.0, 0.09],
                 [0.34, 0.11, 0.07, 0.20, 0.04, 0.13]],
                [[0.12, 0.22, 0.15, 0.31, 0.06, 77.0],
                 [0.18, 0.07, 0.23, 0.09, 0.16, 0.21]],
            ],
            dtype=torch.float64,
        )
        actual_mass = base.clone().requires_grad_(True)
        oracle_mass = base.clone().requires_grad_(True)

        actual = adaptive_uncertainty_top_k(
            self._frontier(
                actual_mass, nodes=nodes, valid=valid, n_nodes=10
            ),
            min_k=1,
            max_k=4,
            mode="hybrid",
            entropy_mix=0.65,
            mass_threshold=0.08,
        )
        expected = dense_adaptive_oracle(
            nodes,
            oracle_mass,
            valid,
            n_nodes=10,
            min_k=1,
            max_k=4,
            mode="hybrid",
            entropy_mix=0.65,
            mass_threshold=0.08,
        )
        actual_dense = actual.frontier.to_dense()
        expected_dense, expected_k, expected_entropy, expected_margin, expected_u = expected
        torch.testing.assert_close(actual_dense, expected_dense, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(actual.selected_k, expected_k))
        torch.testing.assert_close(actual.normalized_entropy, expected_entropy)
        torch.testing.assert_close(actual.top1_margin, expected_margin)
        torch.testing.assert_close(actual.uncertainty, expected_u)

        probe = torch.randn_like(actual_dense)
        actual_objective = (
            (actual_dense * probe).sum()
            + 0.13 * actual.normalized_entropy.sum()
            + 0.07 * actual.top1_margin.sum()
        )
        expected_objective = (
            (expected_dense * probe).sum()
            + 0.13 * expected_entropy.sum()
            + 0.07 * expected_margin.sum()
        )
        actual_gradient = torch.autograd.grad(actual_objective, actual_mass)[0]
        expected_gradient = torch.autograd.grad(expected_objective, oracle_mass)[0]
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11
        )

    def test_compiles_with_fixed_maximum_shape(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        nodes = torch.tensor([[4, 1, 3, 0]], dtype=torch.long)
        valid = torch.tensor([[True, True, True, False]])

        def select(mass: torch.Tensor) -> tuple[torch.Tensor, ...]:
            result = adaptive_uncertainty_top_k(
                PackedNodeFrontier(
                    nodes=nodes,
                    mass=mass,
                    valid=valid,
                    n_nodes=5,
                ),
                min_k=1,
                max_k=3,
                mode="hybrid",
                entropy_mix=0.6,
                mass_threshold=0.05,
            )
            return (
                result.frontier.nodes,
                result.frontier.mass,
                result.frontier.valid,
                result.selected_k,
                result.uncertainty,
            )

        mass = torch.tensor([[0.62, 0.24, 0.14, 999.0]])
        expected = select(mass)
        compiled = torch.compile(select, backend="eager", fullgraph=True)
        actual = compiled(mass)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_rejects_invalid_policy_configuration(self) -> None:
        frontier = self._frontier(torch.tensor([[0.6, 0.4]]))
        with self.assertRaisesRegex(ValueError, "min_k cannot exceed"):
            adaptive_uncertainty_top_k(frontier, min_k=2, max_k=1)
        with self.assertRaisesRegex(ValueError, "mode must"):
            adaptive_uncertainty_top_k(
                frontier, min_k=1, max_k=2, mode="mystery"  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "entropy_mix"):
            adaptive_uncertainty_top_k(
                frontier, min_k=1, max_k=2, entropy_mix=1.1
            )


if __name__ == "__main__":
    unittest.main()
