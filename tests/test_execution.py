import unittest

import torch
import torch.nn.functional as F

from gator.execution import (
    StewardshipExecutionController,
    StewardshipExecutionPlan,
    stewardship_active_fraction,
)
from gator.models import GatorLanguageConfig, GatorLanguageModel


class StewardshipExecutionPlanTests(unittest.TestCase):
    def test_plan_selects_relays_at_and_after_latch(self) -> None:
        dense = StewardshipExecutionPlan()
        packed = StewardshipExecutionPlan(packed_from_relay=2)

        self.assertFalse(dense.uses_packed_execution)
        self.assertFalse(dense.uses_packed_at(8))
        self.assertFalse(packed.uses_packed_at(1))
        self.assertTrue(packed.uses_packed_at(2))
        self.assertTrue(packed.uses_packed_at(5))

    def test_plan_rejects_negative_relay(self) -> None:
        with self.assertRaises(ValueError):
            StewardshipExecutionPlan(packed_from_relay=-1)
        with self.assertRaises(ValueError):
            StewardshipExecutionPlan().uses_packed_at(-1)


class StewardshipActiveFractionTests(unittest.TestCase):
    def test_uses_relay_inputs_and_excludes_final_trace_and_padding(self) -> None:
        # [batch=1, traces=4, cells=4, tokens=2] describes three relays.
        ownership = torch.zeros(1, 4, 4, 2)
        ownership[:, 0, :, 0] = 0.25
        ownership[:, 1, :2, 0] = 0.5
        ownership[:, 2, :1, 0] = 1.0
        # This dense post-relay state must not appear in the result.
        ownership[:, 3, :, 0] = 0.25
        # Dense values in an invalid padding position must also be ignored.
        ownership[:, :3, :, 1] = 0.25
        route_valid = torch.tensor([[True, False]])

        fractions = stewardship_active_fraction(ownership, route_valid)

        self.assertEqual(tuple(fractions.shape), (3,))
        self.assertTrue(
            torch.allclose(fractions, torch.tensor([1.0, 0.5, 0.25]))
        )

    def test_defaults_to_all_token_positions(self) -> None:
        ownership = torch.zeros(2, 3, 4, 2)
        ownership[:, 0] = 0.25
        ownership[:, 1, :2] = 0.5

        fractions = stewardship_active_fraction(ownership)

        self.assertTrue(torch.allclose(fractions, torch.tensor([1.0, 0.5])))

    def test_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            stewardship_active_fraction(torch.zeros(2, 3, 4))
        with self.assertRaises(ValueError):
            stewardship_active_fraction(torch.zeros(2, 1, 4, 3))
        with self.assertRaises(ValueError):
            stewardship_active_fraction(
                torch.zeros(2, 3, 4, 3),
                torch.ones(2, 2, dtype=torch.bool),
            )


class StewardshipExecutionControllerTests(unittest.TestCase):
    def test_dense_mode_never_observes_or_latches(self) -> None:
        controller = StewardshipExecutionController(
            mode="dense",
            threshold=0.5,
            relays=4,
            check_every=10,
        )

        self.assertFalse(controller.should_observe(10, pruning=True))
        plan = controller.observe([1.0, 0.4, 0.2, 0.1])
        self.assertIsNone(plan.packed_from_relay)
        self.assertFalse(controller.latched)

    def test_auto_mode_observes_on_schedule_and_latches_earliest_relay(self) -> None:
        controller = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.5,
            relays=4,
            check_every=100,
        )

        self.assertFalse(controller.should_observe(99, pruning=True))
        self.assertFalse(controller.should_observe(100, pruning=False))
        self.assertTrue(controller.should_observe(100, pruning=True))
        plan = controller.observe(torch.tensor([1.0, 0.75, 0.5, 0.25]))

        self.assertEqual(plan.packed_from_relay, 2)
        self.assertEqual(controller.packed_from_relay, 2)
        self.assertTrue(controller.latched)
        self.assertFalse(controller.should_observe(200, pruning=True))
        # A later observation cannot move a one-way latch.
        controller.observe([0.1, 0.1, 0.1, 0.1])
        self.assertEqual(controller.packed_from_relay, 2)

    def test_controller_remains_dense_without_qualifying_relay(self) -> None:
        controller = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.25,
            relays=3,
            check_every=1,
        )

        plan = controller.observe([1.0, 0.6, 0.3])

        self.assertIsNone(plan.packed_from_relay)
        self.assertTrue(controller.should_observe(2, pruning=True))

    def test_static_plan_requires_the_complete_suffix_to_be_sparse(self) -> None:
        controller = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.5,
            relays=4,
            check_every=1,
        )

        plan = controller.observe([1.0, 0.4, 0.8, 0.2])

        self.assertEqual(plan.packed_from_relay, 3)

    def test_threshold_one_can_select_first_relay(self) -> None:
        controller = StewardshipExecutionController(
            mode="packed_auto",
            threshold=1.0,
            relays=2,
            check_every=1,
        )

        self.assertEqual(
            controller.observe([1.0, 1.0]).packed_from_relay,
            0,
        )

    def test_state_round_trip_preserves_latch(self) -> None:
        original = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.5,
            relays=4,
            check_every=25,
        )
        original.observe([1.0, 0.4, 0.2, 0.1])
        restored = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.5,
            relays=4,
            check_every=25,
        )

        restored.load_state_dict(original.state_dict())

        self.assertEqual(restored.plan, original.plan)
        self.assertEqual(restored.state_dict(), original.state_dict())

    def test_state_restore_rejects_incompatible_configuration(self) -> None:
        original = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.5,
            relays=4,
            check_every=25,
        )
        incompatible = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.4,
            relays=4,
            check_every=25,
        )

        with self.assertRaises(ValueError):
            incompatible.load_state_dict(original.state_dict())

    def test_controller_validates_observations_and_schedule(self) -> None:
        controller = StewardshipExecutionController(
            mode="packed_auto",
            threshold=0.5,
            relays=2,
            check_every=1,
        )
        with self.assertRaises(ValueError):
            controller.should_observe(0, pruning=True)
        with self.assertRaises(ValueError):
            controller.observe([0.5])
        with self.assertRaises(ValueError):
            controller.observe([float("nan"), 0.5])
        with self.assertRaises(ValueError):
            StewardshipExecutionController(mode="packed_auto", check_every=0)


class StewardshipExecutionIntegrationTests(unittest.TestCase):
    def test_compiled_dense_and_eager_packed_share_one_model(self) -> None:
        model = GatorLanguageModel(
            GatorLanguageConfig(
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
                stewardship_execution="packed_auto",
            )
        )
        compiled_dense = torch.compile(model, backend="eager")
        context = torch.randint(0, 64, (1, 4))
        mask = torch.ones_like(context, dtype=torch.bool)
        targets = torch.randint(0, 64, (1, 2))

        dense_logits, dense_routes = compiled_dense(
            context,
            mask,
            output_slots=2,
            stewardship_prune=False,
            diagnostics="route_loss",
        )
        F.cross_entropy(
            dense_logits.reshape(-1, 64), targets.reshape(-1)
        ).backward()
        self.assertIsNotNone(model.handoff_weight.grad)

        model.zero_grad(set_to_none=True)
        packed_logits = model(
            context,
            mask,
            output_slots=2,
            stewardship_prune=False,
            stewardship_packed_from_relay=1,
        )
        F.cross_entropy(
            packed_logits.reshape(-1, 64), targets.reshape(-1)
        ).backward()

        self.assertEqual(
            set(dense_routes),
            {
                "route_entropy",
                "route_cell_load",
                "route_active_fraction",
                "route_mean_uncertainty",
                "route_valid",
            },
        )
        self.assertIsNotNone(model.handoff_weight.grad)
        self.assertTrue(torch.isfinite(model.handoff_weight.grad).all())


if __name__ == "__main__":
    unittest.main()
