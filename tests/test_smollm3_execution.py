import unittest

import torch

from gator.execution import StewardshipExecutionController
from gator.models.language import GatorLanguageConfig, GatorLanguageModel
from gator.smollm3_train import (
    _discard_nonfinite_accumulation,
    _gradient_failure_diagnostic,
    _remap_resume_micro_step,
)
from gator.smollm3_recipe import (
    EFFECTIVE_BATCH_SEQUENCES,
    accumulation_steps_for_batch,
)


class SmolLM3ExecutionTests(unittest.TestCase):
    def test_batch_two_preserves_effective_batch(self):
        accumulation = accumulation_steps_for_batch(2)
        self.assertEqual(accumulation, 16)
        self.assertEqual(2 * accumulation, EFFECTIVE_BATCH_SEQUENCES)
        with self.assertRaisesRegex(ValueError, "must divide"):
            accumulation_steps_for_batch(3)

    def test_gradient_failure_diagnostic_names_nonfinite_parameter(self):
        model = torch.nn.Module()
        model.good = torch.nn.Parameter(torch.ones(2))
        model.bad = torch.nn.Parameter(torch.ones(4))
        model.good.grad = torch.tensor([1.0, 3.0])
        model.bad.grad = torch.tensor([2.0, float("nan"), float("inf"), -4.0])

        diagnostic = _gradient_failure_diagnostic(model)
        self.assertIn("nonfinite_parameters=1", diagnostic)
        self.assertIn("name='bad'", diagnostic)
        self.assertIn("nan=1", diagnostic)
        self.assertIn("inf=1", diagnostic)
        self.assertIn("finite_abs_max=4", diagnostic)

    def test_gradient_failure_diagnostic_distinguishes_norm_overflow(self):
        model = torch.nn.Module()
        model.large = torch.nn.Parameter(torch.ones(2))
        model.large.grad = torch.tensor([1.0e30, -1.0e30])

        diagnostic = _gradient_failure_diagnostic(model)
        self.assertIn("all_gradient_elements_finite=true", diagnostic)
        self.assertIn("aggregate_norm_overflow_suspected", diagnostic)
        self.assertIn("largest_finite_name='large'", diagnostic)

    def test_nonfinite_accumulation_is_discarded_at_update_boundary(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter])
        parameter.grad = torch.tensor(float("nan"))
        error = FloatingPointError("nonfinite test gradient")

        micro_step, consecutive = _discard_nonfinite_accumulation(
            optimizer,
            update=125,
            accumulation_steps=4,
            consecutive_failures=0,
            max_consecutive_recoveries=3,
            error=error,
        )
        self.assertEqual(micro_step, 500)
        self.assertEqual(consecutive, 1)
        self.assertIsNone(parameter.grad)

        parameter.grad = torch.tensor(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "budget exhausted"):
            _discard_nonfinite_accumulation(
                optimizer,
                update=125,
                accumulation_steps=4,
                consecutive_failures=3,
                max_consecutive_recoveries=3,
                error=error,
            )
        self.assertIsNone(parameter.grad)

    def test_resume_can_change_physical_batch_geometry(self):
        remapped = _remap_resume_micro_step(
            125,
            1_000,
            checkpoint_batch_size=4,
            checkpoint_accumulation=8,
            requested_batch_size=1,
            requested_accumulation=32,
        )
        self.assertEqual(remapped, 4_000)

        with self.assertRaisesRegex(ValueError, "effective batches remain 32"):
            _remap_resume_micro_step(
                125,
                1_000,
                checkpoint_batch_size=4,
                checkpoint_accumulation=8,
                requested_batch_size=1,
                requested_accumulation=16,
            )
        with self.assertRaisesRegex(ValueError, "update boundary"):
            _remap_resume_micro_step(
                125,
                999,
                checkpoint_batch_size=4,
                checkpoint_accumulation=8,
                requested_batch_size=1,
                requested_accumulation=32,
            )

    def test_density_probe_latches_and_runs_packed_suffix(self):
        torch.manual_seed(4)
        model = GatorLanguageModel(
            GatorLanguageConfig(
                vocab_size=32,
                d_model=8,
                embedding_dim=16,
                lexical_embedding_dim=8,
                n_cells=4,
                n_nodes=4,
                relays=2,
                neighbors=2,
                mask_token_id=3,
                lexical_mode="communal",
                node_topology="asymmetric",
                cell_input_mode="feature_gated",
                cell_output_mode="grouped_concat",
                cell_transport_mode="stewardship",
                stewardship_execution="packed_auto",
                packed_stewardship_backend="persistent_bucketed",
                receiver_memory_backend="linear_scan",
                dense_kernel_padding=8,
            )
        ).train()
        context = torch.randint(4, 32, (2, 8))
        mask = torch.ones_like(context, dtype=torch.bool)

        dense_logits, diagnostics = model(
            context,
            mask,
            output_slots=2,
            diagnostics="route_loss",
        )
        fractions = diagnostics["route_active_fraction"]
        self.assertEqual(fractions.shape, (2,))
        self.assertTrue(torch.isfinite(fractions).all())

        controller = StewardshipExecutionController(
            mode="packed_auto",
            threshold=1.0,
            relays=2,
            check_every=1,
        )
        self.assertTrue(controller.should_observe(1, pruning=True))
        plan = controller.observe(fractions)
        self.assertIsNotNone(plan.packed_from_relay)

        packed_logits = model(
            context,
            mask,
            output_slots=2,
            stewardship_packed_from_relay=plan.packed_from_relay,
        )
        loss = packed_logits.float().square().mean()
        loss.backward()
        self.assertEqual(packed_logits.shape, dense_logits.shape)
        self.assertTrue(torch.isfinite(packed_logits).all())
        self.assertIsNotNone(model.node_weight.grad)
        self.assertTrue(torch.isfinite(model.node_weight.grad).all())


if __name__ == "__main__":
    unittest.main()
