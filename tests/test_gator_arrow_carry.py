from __future__ import annotations

import unittest

import torch

from gator_cells_vectorized import VectorizedGatorCellRing
from parallel_prefix_benchmark import make_prefix_batch


class GatorArrowCarryTests(unittest.TestCase):
    def test_carry_is_parameter_and_checkpoint_compatible_with_arrow(
        self,
    ) -> None:
        reference = VectorizedGatorCellRing(carry_state=False)
        carry = VectorizedGatorCellRing(carry_state=True)

        self.assertEqual(reference.parameter_count, 10_000)
        self.assertEqual(carry.parameter_count, 10_000)
        self.assertEqual(
            tuple(reference.state_dict()),
            tuple(carry.state_dict()),
        )
        carry.load_state_dict(reference.state_dict())

    def test_population_scaling_preserves_cell_capacity(
        self,
    ) -> None:
        reference = VectorizedGatorCellRing(
            n_cells=62,
            carry_state=False,
        )
        carry = VectorizedGatorCellRing(
            n_cells=62,
            carry_state=True,
        )

        self.assertEqual(reference.parameter_count, 61_168)
        self.assertEqual(carry.parameter_count, 61_168)
        self.assertEqual(reference.relay_receptive_field, 17)
        self.assertAlmostEqual(
            reference.relay_population_coverage,
            17 / 62,
        )

    def test_ten_cell_arrow_is_fully_covered_by_four_relays(self) -> None:
        model = VectorizedGatorCellRing(
            n_cells=10,
            relays=4,
            neighbors=4,
        )

        self.assertEqual(model.relay_receptive_field, 10)
        self.assertEqual(model.relay_population_coverage, 1.0)

    def test_relay_depth_restores_population_coverage(self) -> None:
        shallow = VectorizedGatorCellRing(
            n_cells=62,
            relays=4,
            neighbors=4,
        )
        deep = VectorizedGatorCellRing(
            n_cells=62,
            relays=16,
            neighbors=4,
        )

        self.assertEqual(shallow.relay_receptive_field, 17)
        self.assertEqual(deep.relay_receptive_field, 62)

    def test_distant_scaled_cell_has_zero_terminal_lane_gradient(self) -> None:
        torch.manual_seed(5)
        model = VectorizedGatorCellRing(
            n_cells=62,
            relays=4,
            neighbors=4,
            reference_dense=True,
        )
        observed, _ = make_prefix_batch(
            1,
            1,
            torch.device("cpu"),
        )

        _, trace = model(
            observed,
            output_slots=1,
            return_trace=True,
        )
        trace["cell_logits"][0, 0, 0, 0].backward()

        near = model.cells[2].token_embedding.weight.grad
        distant = model.cells[31].token_embedding.weight.grad
        self.assertIsNotNone(near)
        self.assertGreater(float(near.abs().sum()), 0.0)
        if distant is not None:
            self.assertEqual(float(distant.abs().sum()), 0.0)

    def test_twenty_two_node_cells_form_matched_budget_ring(self) -> None:
        model = VectorizedGatorCellRing(
            n_cells=20,
            n_nodes=2,
            neighbors=4,
        )

        self.assertEqual(model.cells[0].parameter_count, 502)
        self.assertEqual(model.parameter_count, 10_280)

        observed, _ = make_prefix_batch(
            3,
            7,
            torch.device("cpu"),
        )
        with torch.no_grad():
            logits = model(observed, output_slots=7)
        self.assertEqual(tuple(logits.shape), (3, 7, 16))
        self.assertTrue(torch.isfinite(logits).all())

    def test_persistent_active_keeps_the_arrow_parameter_budget(
        self,
    ) -> None:
        model = VectorizedGatorCellRing(
            persistent_active=True,
        )

        self.assertEqual(model.parameter_count, 10_000)

    def test_relay_depth_changes_compute_not_parameters(self) -> None:
        four_relays = VectorizedGatorCellRing(
            relays=4,
            persistent_active=True,
        )
        eight_relays = VectorizedGatorCellRing(
            relays=8,
            persistent_active=True,
        )

        self.assertEqual(four_relays.parameter_count, 10_000)
        self.assertEqual(eight_relays.parameter_count, 10_000)

    def test_optimized_dense_matches_locked_reference_forward(self) -> None:
        torch.manual_seed(17)
        optimized = VectorizedGatorCellRing()
        reference = VectorizedGatorCellRing(reference_dense=True)
        reference.load_state_dict(optimized.state_dict())
        observed, _ = make_prefix_batch(
            8,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            optimized_logits = optimized(
                observed,
                output_slots=7,
                temperature=0.71,
            )
            reference_logits = reference(
                observed,
                output_slots=7,
                temperature=0.71,
            )

        torch.testing.assert_close(
            optimized_logits,
            reference_logits,
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    def test_dense_trace_reconstructs_output_and_records_relays(self) -> None:
        torch.manual_seed(19)
        model = VectorizedGatorCellRing()
        observed, _ = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            logits, trace = model(
                observed,
                output_slots=7,
                temperature=0.67,
                return_trace=True,
            )
            reconstructed = torch.einsum(
                "c,bctv->btv",
                trace["output_votes"],
                trace["cell_logits"],
            )

        torch.testing.assert_close(logits, reconstructed)
        self.assertEqual(
            tuple(trace["locations"].shape),
            (4, 5, 10, 15, 4),
        )
        self.assertEqual(
            tuple(trace["transition_entropy"].shape),
            (4,),
        )

    def test_relay_limit_truncates_trained_computation_without_rephasing(
        self,
    ) -> None:
        torch.manual_seed(20)
        model = VectorizedGatorCellRing(relays=4)
        observed, _ = make_prefix_batch(
            3,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            full_logits, full_trace = model(
                observed,
                output_slots=7,
                return_trace=True,
            )
            limited_full_logits, limited_full_trace = model(
                observed,
                output_slots=7,
                return_trace=True,
                relay_limit=4,
            )
            _, two_trace = model(
                observed,
                output_slots=7,
                return_trace=True,
                relay_limit=2,
            )

        torch.testing.assert_close(full_logits, limited_full_logits)
        torch.testing.assert_close(
            full_trace["locations"],
            limited_full_trace["locations"],
        )
        torch.testing.assert_close(
            two_trace["locations"],
            full_trace["locations"][:, :3],
        )
        torch.testing.assert_close(
            two_trace["transition_entropy"],
            full_trace["transition_entropy"][:2],
        )
        with self.assertRaises(ValueError):
            model(observed, relay_limit=0)
        with self.assertRaises(ValueError):
            model(observed, relay_limit=5)

    def test_dense_trace_can_prune_node_support_after_routing(self) -> None:
        torch.manual_seed(21)
        model = VectorizedGatorCellRing()
        observed, _ = make_prefix_batch(
            3,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            logits, trace = model(
                observed,
                output_slots=7,
                return_trace=True,
                node_top_k=1,
            )

        self.assertEqual(tuple(logits.shape), (3, 7, 16))
        retained = (trace["locations"][:, 1:] > 0).sum(-1)
        self.assertTrue(torch.equal(retained, torch.ones_like(retained)))

    def test_dense_trace_can_disable_intermediate_cells(self) -> None:
        torch.manual_seed(22)
        model = VectorizedGatorCellRing()
        observed, _ = make_prefix_batch(
            3,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            logits, trace = model(
                observed,
                output_slots=7,
                return_trace=True,
                active_cells=(0, 4),
            )

        self.assertEqual(tuple(logits.shape), (3, 7, 16))
        self.assertEqual(
            torch.nonzero(
                trace["output_votes"] > 0,
                as_tuple=False,
            ).flatten().tolist(),
            [0, 4],
        )
        inactive_states = trace["output_states"][
            :,
            [1, 2, 3, 5, 6, 7, 8, 9],
        ]
        self.assertTrue(torch.equal(
            inactive_states,
            torch.zeros_like(inactive_states),
        ))

    def test_uniform_initial_gates_make_carry_an_exact_forward_control(
        self,
    ) -> None:
        torch.manual_seed(7)
        reference = VectorizedGatorCellRing(carry_state=False)
        carry = VectorizedGatorCellRing(carry_state=True)
        carry.load_state_dict(reference.state_dict())
        observed, _ = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            reference_logits = reference(
                observed,
                output_slots=7,
                temperature=1.25,
            )
            carry_logits = carry(
                observed,
                output_slots=7,
                temperature=1.25,
            )

        torch.testing.assert_close(
            carry_logits,
            reference_logits,
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    def test_source_specific_gates_diverge_after_learning(
        self,
    ) -> None:
        torch.manual_seed(11)
        reference = VectorizedGatorCellRing(carry_state=False)
        carry = VectorizedGatorCellRing(source_carry=True)
        carry.load_state_dict(reference.state_dict())
        with torch.no_grad():
            pattern = torch.linspace(-2.0, 2.0, 10)
            reference.neighbor_gates[:, 0] = pattern
            carry.neighbor_gates[:, 0] = pattern
        observed, _ = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            reference_logits = reference(observed, output_slots=7)
            carry_logits = carry(observed, output_slots=7)

        self.assertFalse(torch.equal(carry_logits, reference_logits))

    def test_destination_owned_carry_preserves_arrow_with_private_gates(
        self,
    ) -> None:
        torch.manual_seed(23)
        reference = VectorizedGatorCellRing(
            reference_dense=True,
        )
        carry = VectorizedGatorCellRing(
            destination_carry=True,
            reference_dense=True,
        )
        carry.load_state_dict(reference.state_dict())
        with torch.no_grad():
            pattern = torch.linspace(-2.0, 2.0, 10)
            reference.neighbor_gates[:, 0] = pattern
            carry.neighbor_gates[:, 0] = pattern
            reference.neighbor_gates[:, 1] = pattern.flip(0)
            carry.neighbor_gates[:, 1] = pattern.flip(0)
        observed, _ = make_prefix_batch(
            8,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            reference_logits = reference(
                observed,
                output_slots=7,
                temperature=0.63,
            )
            carry_logits = carry(
                observed,
                output_slots=7,
                temperature=0.63,
            )

        torch.testing.assert_close(
            carry_logits,
            reference_logits,
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    def test_destination_owned_carry_preserves_gate_gradient(self) -> None:
        torch.manual_seed(29)
        reference = VectorizedGatorCellRing(
            reference_dense=True,
        )
        carry = VectorizedGatorCellRing(
            destination_carry=True,
            reference_dense=True,
        )
        carry.load_state_dict(reference.state_dict())
        observed, targets = make_prefix_batch(
            16,
            7,
            torch.device("cpu"),
        )
        reference_loss = torch.nn.functional.cross_entropy(
            reference(observed).reshape(-1, 16),
            targets.reshape(-1),
        )
        carry_loss = torch.nn.functional.cross_entropy(
            carry(observed).reshape(-1, 16),
            targets.reshape(-1),
        )
        reference_loss.backward()
        carry_loss.backward()

        torch.testing.assert_close(
            carry.neighbor_gates.grad,
            reference.neighbor_gates.grad,
            atol=2.0e-7,
            rtol=2.0e-6,
        )

    def test_destination_owned_carry_preserves_every_parameter_gradient(
        self,
    ) -> None:
        torch.manual_seed(31)
        reference = VectorizedGatorCellRing(reference_dense=True)
        carry = VectorizedGatorCellRing(
            destination_carry=True,
            reference_dense=True,
        )
        carry.load_state_dict(reference.state_dict())
        observed, targets = make_prefix_batch(
            8,
            7,
            torch.device("cpu"),
        )
        for model in (reference, carry):
            loss = torch.nn.functional.cross_entropy(
                model(
                    observed,
                    output_slots=7,
                    temperature=0.79,
                ).reshape(-1, 16),
                targets.reshape(-1),
            )
            loss.backward()

        reference_parameters = dict(reference.named_parameters())
        carry_parameters = dict(carry.named_parameters())
        self.assertEqual(
            tuple(reference_parameters),
            tuple(carry_parameters),
        )
        for name, reference_parameter in reference_parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(reference_parameter.grad)
                self.assertIsNotNone(carry_parameters[name].grad)
                torch.testing.assert_close(
                    carry_parameters[name].grad,
                    reference_parameter.grad,
                    atol=3.0e-7,
                    rtol=3.0e-6,
                )

    def test_persistent_active_forward_and_gradients_are_finite(
        self,
    ) -> None:
        torch.manual_seed(13)
        model = VectorizedGatorCellRing(
            persistent_active=True,
        )
        observed, targets = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        logits = model(
            observed,
            output_slots=7,
            temperature=1.25,
        )
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        loss.backward()

        self.assertEqual(tuple(logits.shape), (4, 7, 16))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(
            sum(float(value.abs().sum()) for value in gradients),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
