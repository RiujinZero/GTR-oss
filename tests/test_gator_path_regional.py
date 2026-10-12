from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from gator_path import GatorPathPrefixConfig, GatorPathPrefixModel
from gator_path_regional_train import (
    OwnershipRouterSchedule,
    RegionalLayout,
    SplitRegionalOptimizer,
    _task_batch,
    apply_ownership_router_schedule,
    arithmetic_batch,
    build_optimizer,
    evaluate,
    learned_dispatch_access_mask,
    number_batch,
    prefix_batch,
    regional_access_mask,
    regional_call_return_schedule,
    regional_composition_schedule,
    regional_dispatch_mask,
    regional_gate_destination_table,
    regional_return_mask,
    orchestrator_return_loss,
    retain_router_only_gradients,
    should_train_skills,
    regional_skill_access_mask,
    support_width,
)


class GatorPathRegionalTests(unittest.TestCase):
    def test_ownership_router_schedule_decays_at_observed_landmarks(
        self,
    ) -> None:
        schedule = OwnershipRouterSchedule(
            maximum_learning_rate=0.1,
            minimum_learning_rate=0.01,
            start_ownership=0.5,
            target_ownership=0.8,
        )
        ownership = (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
        learning_rates = [
            schedule.learning_rate(value) for value in ownership
        ]

        self.assertAlmostEqual(learning_rates[0], 0.1)
        self.assertAlmostEqual(learning_rates[1], 0.1)
        self.assertAlmostEqual(learning_rates[2], 0.0464158883)
        self.assertAlmostEqual(learning_rates[3], 0.0215443469)
        self.assertAlmostEqual(learning_rates[4], 0.0146779927)
        self.assertAlmostEqual(learning_rates[5], 0.01)
        self.assertAlmostEqual(learning_rates[6], 0.01)
        self.assertTrue(
            all(
                left >= right
                for left, right in zip(
                    learning_rates, learning_rates[1:]
                )
            )
        )

    def test_ownership_schedule_updates_only_split_router_lr(self) -> None:
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=6,
                top_k=2,
            )
        )
        optimizer = build_optimizer(
            model,
            learning_rate=0.001,
            router_learning_rate=0.1,
            weight_decay=1.0e-4,
            mode="split",
        )
        self.assertIsInstance(optimizer, SplitRegionalOptimizer)
        schedule = OwnershipRouterSchedule(0.1, 0.01, 0.5, 0.8)

        applied = apply_ownership_router_schedule(
            optimizer,
            schedule,
            maximum_observed_ownership=0.75,
        )

        self.assertAlmostEqual(applied, 0.0146779927)
        self.assertAlmostEqual(
            optimizer.router_learning_rate,
            0.0146779927,
        )
        self.assertAlmostEqual(
            optimizer.transformation_optimizer.param_groups[0]["lr"],
            0.001,
        )

    def test_split_optimizer_assigns_only_route_heads_to_sgd(self) -> None:
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=6,
                top_k=2,
            )
        )
        optimizer = build_optimizer(
            model,
            learning_rate=0.001,
            router_learning_rate=0.01,
            weight_decay=1.0e-4,
            mode="split",
        )

        self.assertIsInstance(optimizer, SplitRegionalOptimizer)
        router_ids = {
            id(parameter)
            for group in optimizer.router_optimizer.param_groups
            for parameter in group["params"]
        }
        transformation_ids = {
            id(parameter)
            for group in optimizer.transformation_optimizer.param_groups
            for parameter in group["params"]
        }
        expected_router_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if "cell_router_" in name or "node_router_" in name
        }
        all_parameter_ids = {
            id(parameter) for parameter in model.parameters()
        }

        self.assertEqual(router_ids, expected_router_ids)
        self.assertFalse(router_ids & transformation_ids)
        self.assertEqual(
            router_ids | transformation_ids,
            all_parameter_ids,
        )
        self.assertEqual(
            optimizer.router_optimizer.param_groups[0]["momentum"],
            0.0,
        )

    def test_isolated_access_has_three_disconnected_regions(self) -> None:
        mask = regional_access_mask(
            torch.device("cpu"), bridges_open=False
        )

        self.assertTrue(mask[:4, :4].all())
        self.assertTrue(mask[4:8, 4:8].all())
        self.assertTrue(mask[8:, 8:].all())
        self.assertFalse(mask[:4, 4:].any())
        self.assertFalse(mask[4:8, :4].any())
        self.assertFalse(mask[4:8, 8:].any())

    def test_open_access_uses_orchestrator_without_skill_shortcut(self) -> None:
        mask = regional_access_mask(
            torch.device("cpu"), bridges_open=True
        )

        self.assertTrue(mask[:4, 8:].all())
        self.assertTrue(mask[8:, :4].all())
        self.assertTrue(mask[4:8, 8:].all())
        self.assertTrue(mask[8:, 4:8].all())
        self.assertFalse(mask[:4, 4:8].any())
        self.assertFalse(mask[4:8, :4].any())

    def test_two_six_two_layout_reallocates_capacity(self) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=2,
            orchestrator_cells=2,
        )
        mask = regional_access_mask(
            torch.device("cpu"),
            bridges_open=False,
            layout=layout,
        )
        _, _, number_locations = number_batch(
            4, torch.device("cpu"), layout=layout
        )
        _, _, arithmetic_locations = arithmetic_batch(
            4, torch.device("cpu"), layout=layout
        )

        self.assertEqual(layout.number, (0, 1))
        self.assertEqual(layout.arithmetic, (2, 3, 4, 5, 6, 7))
        self.assertTrue(mask[:2, :2].all())
        self.assertTrue(mask[2:8, 2:8].all())
        self.assertTrue((number_locations < 2).all())
        self.assertTrue(
            ((arithmetic_locations >= 2) & (arithmetic_locations < 8)).all()
        )

    def test_orchestrator_can_receive_skill_packets_first(self) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        _, _, number_locations = _task_batch(
            "number",
            4,
            7,
            torch.device("cpu"),
            layout=layout,
            orchestrator_starts=True,
        )
        _, _, arithmetic_locations = _task_batch(
            "arithmetic",
            4,
            7,
            torch.device("cpu"),
            layout=layout,
            orchestrator_starts=True,
        )
        _, _, prefix_locations = _task_batch(
            "prefix",
            4,
            7,
            torch.device("cpu"),
            layout=layout,
            orchestrator_starts=True,
        )

        self.assertEqual(layout.number, (0,))
        self.assertEqual(layout.arithmetic, (1, 2, 3, 4, 5, 6, 7))
        self.assertTrue((number_locations >= 8).all())
        self.assertTrue((arithmetic_locations >= 8).all())
        self.assertTrue((prefix_locations >= 8).all())

    def test_skill_access_activates_only_one_region_plus_orchestrator(
        self,
    ) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        number = regional_skill_access_mask(
            torch.device("cpu"),
            skill="number",
            layout=layout,
        )
        arithmetic = regional_skill_access_mask(
            torch.device("cpu"),
            skill="arithmetic",
            layout=layout,
        )

        self.assertTrue(number[0, 8:].all())
        self.assertTrue(number[8:, 0].all())
        self.assertFalse(number[1:8, 8:].any())
        self.assertTrue(arithmetic[1:8, 8:].all())
        self.assertTrue(arithmetic[8:, 1:8].all())
        self.assertFalse(arithmetic[0, 8:].any())

    def test_learned_dispatch_enforces_ownership_not_call_order(self) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        mask = learned_dispatch_access_mask(
            torch.device("cpu"),
            layout=layout,
        )

        self.assertTrue(mask[8:, 0].all())
        self.assertTrue(mask[8:, 1:8].all())
        self.assertTrue(mask[0, 8:].all())
        self.assertTrue(mask[1:8, 8:].all())
        self.assertFalse(mask[0, 1:8].any())
        self.assertFalse(mask[1:8, 0].any())

    def test_gate_dispatch_has_one_size_independent_entrance_per_skill(
        self,
    ) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        table = regional_gate_destination_table(
            torch.device("cpu"),
            layout=layout,
        )
        number_gate = layout.number[-1]
        arithmetic_gate = layout.arithmetic[-1]

        for orchestrator in layout.orchestrator:
            destinations = table[orchestrator]
            self.assertEqual(
                int(destinations.eq(number_gate).sum()),
                1,
            )
            self.assertEqual(
                int(destinations.eq(arithmetic_gate).sum()),
                1,
            )
            self.assertFalse(
                torch.isin(
                    destinations,
                    torch.tensor(layout.arithmetic[:-1]),
                ).any()
            )

    def test_every_specialist_has_local_work_and_direct_return(self) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        table = regional_gate_destination_table(
            torch.device("cpu"),
            layout=layout,
        )
        orchestrator = torch.tensor(layout.orchestrator)

        for region in (layout.number, layout.arithmetic):
            legal_local = torch.tensor(region)
            for source in region:
                self.assertTrue(
                    torch.isin(table[source, :4], legal_local).all()
                )
                self.assertTrue(
                    torch.isin(table[source, 4:], orchestrator).all()
                )

    def test_gate_dispatch_forward_never_shortcuts_between_skills(
        self,
    ) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        torch.manual_seed(43)
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=6,
                top_k=2,
            )
        )
        metrics = evaluate(
            model,
            task="prefix",
            batch_size=16,
            length=7,
            access_mask=learned_dispatch_access_mask(
                torch.device("cpu"),
                layout=layout,
            ),
            seed=47,
            layout=layout,
            orchestrator_starts=True,
            cell_destination_table=regional_gate_destination_table(
                torch.device("cpu"),
                layout=layout,
            ),
        )

        self.assertEqual(metrics["specialist_shortcuts"], 0.0)

    def test_return_loss_is_differentiable_without_scripting_target_skill(
        self,
    ) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        torch.manual_seed(37)
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=4,
                top_k=2,
            )
        )
        observed, target, locations = _task_batch(
            "arithmetic",
            8,
            7,
            torch.device("cpu"),
            layout=layout,
            orchestrator_starts=True,
        )
        logits, diagnostics = model(
            observed,
            output_slots=target.shape[1],
            route_top_k=2,
            initial_cell_id=locations,
            cell_access_mask=learned_dispatch_access_mask(
                torch.device("cpu"), layout=layout
            ),
            return_diagnostics=True,
        )
        task_loss = F.cross_entropy(logits[:, 0], target[:, 0])
        return_loss = orchestrator_return_loss(
            diagnostics,
            layout=layout,
            return_relays=2,
        )
        (task_loss + 0.1 * return_loss).backward()

        self.assertTrue(torch.isfinite(return_loss))
        self.assertIsNotNone(model.relay.cell_router_weight.grad)
        self.assertGreater(
            float(model.relay.cell_router_weight.grad.abs().sum()),
            0.0,
        )

    def test_router_only_rehearsal_freezes_skill_transformations(self) -> None:
        torch.manual_seed(41)
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=4,
                top_k=2,
            )
        )
        observed, target, locations = _task_batch(
            "number",
            8,
            7,
            torch.device("cpu"),
            layout=RegionalLayout.from_sizes(
                number_cells=1,
                orchestrator_cells=2,
            ),
            orchestrator_starts=True,
        )
        logits = model(
            observed,
            output_slots=1,
            initial_cell_id=locations,
        )
        F.cross_entropy(logits[:, 0], target[:, 0]).backward()
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        retain_router_only_gradients(model, layout=layout)

        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
        }
        self.assertIsNotNone(gradients["relay.cell_router_weight"])
        self.assertIsNotNone(gradients["relay.node_router_weight"])
        transform_gradient = gradients["relay.path_transform.weight"]
        self.assertIsNotNone(transform_gradient)
        self.assertEqual(
            float(transform_gradient[:8].abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(transform_gradient[8:].abs().sum()),
            0.0,
        )
        self.assertIsNone(gradients["token_embedding.weight"])

    def test_loaded_skills_can_be_rehearsed_instead_of_skipped(self) -> None:
        checkpoint = Path("regional-skills.pt")

        self.assertFalse(
            should_train_skills(
                checkpoint,
                rehearse_loaded_skills=False,
            )
        )
        self.assertTrue(
            should_train_skills(
                checkpoint,
                rehearse_loaded_skills=True,
            )
        )
        self.assertTrue(
            should_train_skills(
                None,
                rehearse_loaded_skills=False,
            )
        )

    def test_directional_masks_require_orchestrator_call_and_return(
        self,
    ) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        dispatch = regional_dispatch_mask(
            torch.device("cpu"),
            skill="arithmetic",
            layout=layout,
        )
        returning = regional_return_mask(
            torch.device("cpu"),
            skill="arithmetic",
            layout=layout,
        )

        self.assertFalse(dispatch[8:, 8:].any())
        self.assertTrue(dispatch[8:, 1:8].all())
        self.assertFalse(dispatch[8:, 0].any())
        self.assertFalse(returning[1:8, 1:8].diagonal().any())
        self.assertFalse(returning[1:8, 0].any())

    def test_call_return_schedule_finishes_in_orchestrator(self) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        schedule = regional_call_return_schedule(
            torch.device("cpu"),
            skill="arithmetic",
            relay_steps=8,
            layout=layout,
        )
        torch.manual_seed(23)
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=8,
                top_k=2,
            )
        )
        observed, target, locations = _task_batch(
            "arithmetic",
            8,
            7,
            torch.device("cpu"),
            layout=layout,
            orchestrator_starts=True,
        )
        logits, diagnostics = model(
            observed,
            output_slots=target.shape[1],
            route_top_k=2,
            initial_cell_id=locations,
            cell_access_mask=schedule,
            return_diagnostics=True,
        )

        self.assertEqual(tuple(schedule.shape), (8, 10, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((diagnostics.state.cell_id >= 8).all())

    def test_composition_schedule_never_opens_specialist_shortcut(
        self,
    ) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        schedule = regional_composition_schedule(
            torch.device("cpu"),
            relay_steps=10,
            layout=layout,
        )

        self.assertEqual(tuple(schedule.shape), (10, 10, 10))
        self.assertFalse(schedule[:, 0, 1:8].any())
        self.assertFalse(schedule[:, 1:8, 0].any())

    def test_composed_calls_finish_with_orchestrator_ownership(self) -> None:
        layout = RegionalLayout.from_sizes(
            number_cells=1,
            orchestrator_cells=2,
        )
        schedule = regional_composition_schedule(
            torch.device("cpu"),
            relay_steps=10,
            layout=layout,
        )
        torch.manual_seed(31)
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=10,
                relay_steps=10,
                top_k=2,
            )
        )
        observed, target, locations = _task_batch(
            "prefix",
            2,
            7,
            torch.device("cpu"),
            layout=layout,
            orchestrator_starts=True,
        )
        _, diagnostics = model(
            observed,
            output_slots=target.shape[1],
            route_top_k=2,
            initial_cell_id=locations,
            cell_access_mask=schedule,
            return_diagnostics=True,
        )

        self.assertTrue((diagnostics.state.cell_id >= 8).all())

    def test_skill_batches_have_the_intended_modular_semantics(self) -> None:
        generator = torch.Generator().manual_seed(5)
        observed, target, locations = number_batch(
            32, torch.device("cpu"), generator=generator
        )
        torch.testing.assert_close(target, observed + 4)
        self.assertTrue((locations < 4).all())

        observed, target, locations = arithmetic_batch(
            32, torch.device("cpu"), generator=generator
        )
        expected = 4 + (
            (observed[:, :1] - 4) + observed[:, 1:]
        ).remainder(12)
        torch.testing.assert_close(target, expected)
        self.assertTrue(((locations >= 4) & (locations < 8)).all())

    def test_prefix_places_queries_in_orchestrator_cells(self) -> None:
        observed, target, locations = prefix_batch(
            4, 7, torch.device("cpu")
        )

        self.assertEqual(tuple(observed.shape), (4, 8))
        self.assertEqual(tuple(target.shape), (4, 7))
        self.assertTrue((locations[:, :8] < 4).all())
        self.assertTrue((locations[:, 8:] >= 8).all())

    def test_isolated_forward_cannot_cross_region_boundaries(self) -> None:
        torch.manual_seed(9)
        model = GatorPathPrefixModel(
            GatorPathPrefixConfig(num_cells=10, relay_steps=4, top_k=2)
        )
        observed, target, locations = arithmetic_batch(
            8, torch.device("cpu")
        )
        logits, diagnostics = model(
            observed,
            output_slots=1,
            route_top_k=2,
            initial_cell_id=locations,
            cell_access_mask=regional_access_mask(
                torch.device("cpu"), bridges_open=False
            ),
            return_diagnostics=True,
        )

        self.assertEqual(tuple(logits.shape), (8, 1, 16))
        self.assertTrue(torch.isfinite(logits).all())
        for relay in diagnostics.relay_results:
            self.assertTrue(
                (
                    (relay.route.destination_cell >= 4)
                    & (relay.route.destination_cell < 8)
                ).all()
            )
        loss = F.cross_entropy(logits[:, 0], target[:, 0])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_support_schedule_reaches_deployed_width(self) -> None:
        widths = [support_width(step, 9) for step in range(9)]

        self.assertEqual(widths[0], 6)
        self.assertEqual(widths[-1], 2)
        self.assertTrue(
            all(left >= right for left, right in zip(widths, widths[1:]))
        )


if __name__ == "__main__":
    unittest.main()
