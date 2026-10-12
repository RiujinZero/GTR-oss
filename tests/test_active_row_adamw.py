from __future__ import annotations

import unittest

import torch

from gator.active_row_adamw import (
    ActiveRowAdamW,
    active_rows_from_routes,
    estimate_active_row_state,
    sparse_row_gather,
)


class ActiveRowAdamWTests(unittest.TestCase):
    def test_route_activity_filters_validity_and_mass(self) -> None:
        nodes = torch.tensor([[3, 1, 3], [2, 4, 1]])
        valid = torch.tensor(
            [[True, True, False], [True, False, True]]
        )
        mass = torch.tensor([[0.9, 0.2, 0.8], [0.1, 0.7, 0.6]])

        rows = active_rows_from_routes(
            nodes,
            valid=valid,
            mass=mass,
            minimum_mass=0.15,
        )

        torch.testing.assert_close(rows, torch.tensor([1, 3]))

    def test_sparse_row_gather_preserves_values_and_sparse_gradient(self) -> None:
        weight = torch.nn.Parameter(
            torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
        )
        rows = torch.tensor([[3, 1], [3, 4]])

        gathered = sparse_row_gather(weight, rows)
        torch.testing.assert_close(gathered, weight.detach()[rows])
        coefficients = torch.arange(gathered.numel(), dtype=torch.float32)
        (gathered * coefficients.reshape_as(gathered)).sum().backward()

        assert weight.grad is not None
        self.assertTrue(weight.grad.is_sparse)
        self.assertEqual(weight.grad.sparse_dim(), 1)
        self.assertEqual(weight.grad.dense_dim(), 2)
        expected = torch.zeros_like(weight)
        expected.index_add_(
            0,
            rows.flatten(),
            coefficients.reshape_as(gathered).reshape(-1, 2, 3),
        )
        torch.testing.assert_close(weight.grad.to_dense(), expected)

    def test_exact_mode_matches_dense_torch_adamw(self) -> None:
        torch.manual_seed(4)
        initial = torch.randn(5, 3, dtype=torch.float64)
        candidate = torch.nn.Parameter(initial.clone())
        reference = torch.nn.Parameter(initial.clone())
        optimizer = ActiveRowAdamW(
            [candidate],
            mode="exact",
            lr=0.02,
            betas=(0.7, 0.91),
            eps=1e-9,
            weight_decay=0.03,
        )
        reference_optimizer = torch.optim.AdamW(
            [reference],
            lr=0.02,
            betas=(0.7, 0.91),
            eps=1e-9,
            weight_decay=0.03,
        )

        for active in ([1, 3], [0], [4, 1], [2]):
            gradient = torch.zeros_like(candidate)
            gradient[active] = torch.randn(len(active), 3, dtype=torch.float64)
            candidate.grad = gradient.clone()
            reference.grad = gradient.clone()
            # Exact mode deliberately ignores route declarations and advances
            # all rows just like ordinary dense AdamW.
            optimizer.mark_active_rows(candidate, torch.tensor(active))
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)

        candidate_state = optimizer.state[candidate]
        reference_state = reference_optimizer.state[reference]
        torch.testing.assert_close(
            candidate_state["exp_avg"], reference_state["exp_avg"]
        )
        torch.testing.assert_close(
            candidate_state["exp_avg_sq"], reference_state["exp_avg_sq"]
        )

    def test_exact_mode_densifies_sparse_gradient_without_semantic_change(self) -> None:
        torch.manual_seed(14)
        initial = torch.randn(6, 2, 2, dtype=torch.float64)
        candidate = torch.nn.Parameter(initial.clone())
        reference = torch.nn.Parameter(initial.clone())
        optimizer = ActiveRowAdamW(
            [candidate], mode="exact", lr=0.01, weight_decay=0.02
        )
        reference_optimizer = torch.optim.AdamW(
            [reference], lr=0.01, weight_decay=0.02
        )

        for first_rows, second_rows in (([1, 4], [4]), ([2], [1, 2])):
            first_rows = torch.tensor(first_rows)
            second_rows = torch.tensor(second_rows)
            first_values = torch.randn(
                len(first_rows), 2, 2, dtype=torch.float64
            )
            second_values = torch.randn(
                len(second_rows), 2, 2, dtype=torch.float64
            )
            sparse_loss = (
                sparse_row_gather(candidate, first_rows) * first_values
            ).sum() + (
                sparse_row_gather(candidate, second_rows) * second_values
            ).sum()
            dense_loss = (
                reference[first_rows] * first_values
            ).sum() + (
                reference[second_rows] * second_values
            ).sum()
            sparse_loss.backward()
            dense_loss.backward()
            self.assertTrue(candidate.grad.is_sparse)
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            torch.testing.assert_close(
                candidate, reference, rtol=1e-12, atol=1e-12
            )

    def test_lazy_mode_matches_independent_per_row_adamw_clocks(self) -> None:
        torch.manual_seed(8)
        initial = torch.randn(4, 2, dtype=torch.float64)
        candidate = torch.nn.Parameter(initial.clone())
        optimizer = ActiveRowAdamW(
            [candidate],
            mode="lazy",
            lr=0.015,
            betas=(0.6, 0.9),
            eps=1e-10,
            weight_decay=0.04,
        )
        references = [
            torch.nn.Parameter(row.clone()) for row in initial
        ]
        reference_optimizers = [
            torch.optim.AdamW(
                [row],
                lr=0.015,
                betas=(0.6, 0.9),
                eps=1e-10,
                weight_decay=0.04,
            )
            for row in references
        ]

        schedules = ([1, 3], [1], [0, 2], [3], [1, 2])
        for active in schedules:
            gradient = torch.zeros_like(candidate)
            values = torch.randn(len(active), 2, dtype=torch.float64)
            gradient[active] = values
            candidate.grad = gradient
            optimizer.mark_active_rows(candidate, torch.tensor(active))
            optimizer.step()
            for offset, row in enumerate(active):
                references[row].grad = values[offset].clone()
                reference_optimizers[row].step()
                reference_optimizers[row].zero_grad(set_to_none=True)
            torch.testing.assert_close(
                candidate,
                torch.stack([row.detach() for row in references]),
                rtol=2e-10,
                atol=2e-10,
            )

        torch.testing.assert_close(
            optimizer.state[candidate]["row_step"],
            torch.tensor([1, 3, 2, 2]),
        )

    def test_lazy_inactive_row_freezes_old_momentum_and_weight_decay(self) -> None:
        initial = torch.tensor([[1.0], [2.0], [3.0]])
        lazy_parameter = torch.nn.Parameter(initial.clone())
        exact_parameter = torch.nn.Parameter(initial.clone())
        lazy = ActiveRowAdamW(
            [lazy_parameter], mode="lazy", lr=0.1, weight_decay=0.1
        )
        exact = ActiveRowAdamW(
            [exact_parameter], mode="exact", lr=0.1, weight_decay=0.1
        )

        first_gradient = torch.tensor([[1.0], [0.0], [0.0]])
        lazy_parameter.grad = first_gradient.clone()
        exact_parameter.grad = first_gradient.clone()
        lazy.mark_active_rows(lazy_parameter, torch.tensor([0]))
        lazy.step()
        exact.step()
        frozen = lazy_parameter[0].detach().clone()

        second_gradient = torch.tensor([[0.0], [1.0], [0.0]])
        lazy_parameter.grad = second_gradient.clone()
        exact_parameter.grad = second_gradient.clone()
        lazy.mark_active_rows(lazy_parameter, torch.tensor([1]))
        lazy.step()
        exact.step()

        torch.testing.assert_close(lazy_parameter[0], frozen)
        self.assertFalse(torch.equal(exact_parameter[0], frozen))

    def test_sparse_gradient_carries_lazy_activity_without_marking(self) -> None:
        torch.manual_seed(5)
        initial = torch.randn(6, 2, 2)
        sparse_parameter = torch.nn.Parameter(initial.clone())
        dense_parameter = torch.nn.Parameter(initial.clone())
        sparse_optimizer = ActiveRowAdamW(
            [sparse_parameter], mode="lazy", lr=0.01, weight_decay=0.02
        )
        dense_optimizer = ActiveRowAdamW(
            [dense_parameter], mode="lazy", lr=0.01, weight_decay=0.02
        )
        rows = torch.tensor([4, 1, 4, 2])
        coefficients = torch.randn(4, 2, 2)

        (sparse_row_gather(sparse_parameter, rows) * coefficients).sum().backward()
        dense_loss = (dense_parameter[rows] * coefficients).sum()
        dense_loss.backward()
        dense_optimizer.mark_active_rows(dense_parameter, rows)
        sparse_optimizer.step()
        dense_optimizer.step()

        torch.testing.assert_close(sparse_parameter, dense_parameter)
        torch.testing.assert_close(
            sparse_optimizer.state[sparse_parameter]["exp_avg"],
            dense_optimizer.state[dense_parameter]["exp_avg"],
        )
        torch.testing.assert_close(
            sparse_optimizer.state[sparse_parameter]["row_step"],
            dense_optimizer.state[dense_parameter]["row_step"],
        )

    def test_dense_lazy_gradient_requires_explicit_activity(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(3, 2))
        optimizer = ActiveRowAdamW([parameter], mode="lazy")
        parameter.grad = torch.zeros_like(parameter)

        with self.assertRaisesRegex(RuntimeError, "mark_active_rows"):
            optimizer.step()

    def test_strict_activity_rejects_undeclared_dense_gradient(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(3, 2))
        optimizer = ActiveRowAdamW(
            [parameter], mode="lazy", strict_activity=True
        )
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[2] = 1.0
        optimizer.mark_active_rows(parameter, torch.tensor([1]))

        with self.assertRaisesRegex(ValueError, "undeclared nonzero rows"):
            optimizer.step()

    def test_checkpoint_round_trip_and_semantic_guard(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(4, 3))
        optimizer = ActiveRowAdamW([parameter], mode="lazy", lr=0.03)
        parameter.grad = torch.randn_like(parameter)
        optimizer.mark_active_rows(parameter, torch.tensor([1, 3]))
        optimizer.step()
        checkpoint = optimizer.state_dict()

        self.assertEqual(checkpoint["active_row_adamw"]["format_version"], 1)
        self.assertEqual(checkpoint["active_row_adamw"]["mode"], "lazy")
        self.assertFalse(
            checkpoint["active_row_adamw"]["activity_persisted"]
        )
        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = ActiveRowAdamW([resumed_parameter], mode="lazy", lr=0.03)
        resumed.load_state_dict(checkpoint)
        self.assertIn("active_row_adamw", checkpoint)
        torch.testing.assert_close(
            resumed.state[resumed_parameter]["exp_avg"],
            optimizer.state[parameter]["exp_avg"],
        )
        torch.testing.assert_close(
            resumed.state[resumed_parameter]["row_step"],
            optimizer.state[parameter]["row_step"],
        )

        incompatible = ActiveRowAdamW(
            [torch.nn.Parameter(parameter.detach().clone())], mode="exact"
        )
        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            incompatible.load_state_dict(checkpoint)

    def test_state_estimate_makes_dense_moment_cost_explicit(self) -> None:
        parameters = [
            torch.nn.Parameter(torch.empty(10, 8, 8)),
            torch.nn.Parameter(torch.empty(10, 8)),
        ]
        exact = estimate_active_row_state(parameters, mode="exact")
        lazy = estimate_active_row_state(parameters, mode="lazy")

        self.assertEqual(exact.parameter_elements, 720)
        self.assertEqual(lazy.moment_elements, 1440)
        self.assertEqual(exact.moment_bytes, lazy.moment_bytes)
        self.assertEqual(exact.clock_elements, 2)
        self.assertEqual(lazy.clock_elements, 20)
        self.assertEqual(lazy.clock_bytes, 160)


if __name__ == "__main__":
    unittest.main()
