from __future__ import annotations

import unittest

import torch

from gator.tasks import LanguageBatch
from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)
from gator_language_train import (
    AccuracyGatedTeacherHandoff,
    GraduatedTeacherHandoff,
    PrefixDenoisingController,
    TokenwiseTeacherHandoff,
    deterministic_prefix_evaluation_inputs,
    handoff_log_line,
    prefix_denoising_log_line,
    prefix_denoising_output_inputs,
    teacher_forced_output_inputs,
    token_handoff_cross_entropy,
    token_handoff_log_line,
    tokenwise_teacher_output_inputs,
)


class TeacherForcedDecodeTests(unittest.TestCase):
    def test_handoff_loss_ignores_forbidden_padded_labels(self) -> None:
        log_probs = torch.tensor(
            [[[-float("inf"), -0.2, -1.8], [-float("inf"), -0.5, -1.0]]]
        )
        target_ids = torch.tensor([[1, 0]])
        unsupported = torch.tensor([[True, False]])
        target_mask = torch.tensor([[True, False]])

        loss = token_handoff_cross_entropy(
            log_probs,
            target_ids,
            unsupported,
            target_mask,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 0.2, places=6)

    def test_tokenwise_inputs_remove_exact_rotating_support(self) -> None:
        batch = LanguageBatch(
            context_ids=torch.tensor([[3, 4], [5, 6]]),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
            target_ids=torch.tensor(
                [[10, 11, 12, 2], [20, 21, 22, 2]]
            ),
            target_mask=torch.ones(2, 4, dtype=torch.bool),
            target_complete=torch.tensor([True, True]),
        )
        inputs, unsupported, stats = tokenwise_teacher_output_inputs(
            batch,
            blank_token_id=31,
            removed_tokens=1,
            seed=7,
            rotation=3,
        )

        self.assertEqual(inputs.eq(31).sum(dim=1).tolist(), [2, 2])
        self.assertEqual(unsupported.sum(dim=1).tolist(), [2, 2])
        self.assertEqual(stats["removed_tokens"], 2.0)
        self.assertEqual(stats["visible_tokens"], 4.0)

        released, released_mask, _ = tokenwise_teacher_output_inputs(
            batch,
            blank_token_id=31,
            removed_tokens=3,
        )
        self.assertTrue(released.eq(31).all())
        self.assertTrue(released_mask.all())

    def test_tokenwise_controller_advances_retreats_and_round_trips(self) -> None:
        controller = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=1,
            advance_ratio=0.90,
            retreat_ratio=0.70,
            advance_patience=2,
            retreat_patience=2,
            ema_decay=0.0,
            full_probe_fraction=0.0,
        )
        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.45,
                teacher_accuracy=0.50,
            ),
            "hold",
        )
        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.46,
                teacher_accuracy=0.50,
            ),
            "advance",
        )
        self.assertEqual(controller.removed_tokens, 2)
        controller.observe(
            unsupported_accuracy=0.20,
            teacher_accuracy=0.50,
        )
        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.20,
                teacher_accuracy=0.50,
            ),
            "retreat",
        )
        self.assertEqual(controller.removed_tokens, 1)

        restored = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=1,
            advance_ratio=0.90,
            retreat_ratio=0.70,
            advance_patience=2,
            retreat_patience=2,
            ema_decay=0.0,
            full_probe_fraction=0.0,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_tokenwise_controller_requires_absolute_frontier_mastery(self) -> None:
        controller = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=1,
            advance_ratio=0.90,
            retreat_ratio=0.70,
            minimum_teacher_accuracy=0.30,
            minimum_frontier_accuracy=0.30,
            advance_patience=2,
            retreat_patience=2,
            ema_decay=0.0,
            full_probe_fraction=0.0,
        )

        for _ in range(3):
            self.assertEqual(
                controller.observe(
                    unsupported_accuracy=0.29,
                    teacher_accuracy=0.30,
                ),
                "hold",
            )
        self.assertEqual(controller.removed_tokens, 1)
        self.assertEqual(controller.advance_streak, 0)

        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.30,
                teacher_accuracy=0.31,
            ),
            "hold",
        )
        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.30,
                teacher_accuracy=0.31,
            ),
            "advance",
        )
        self.assertEqual(controller.removed_tokens, 2)

        restored = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=1,
            advance_ratio=0.90,
            retreat_ratio=0.70,
            minimum_teacher_accuracy=0.30,
            minimum_frontier_accuracy=0.30,
            advance_patience=2,
            retreat_patience=2,
            ema_decay=0.0,
            full_probe_fraction=0.0,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_tokenwise_controller_can_override_resume_gate_policy(self) -> None:
        original = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=0,
            advance_ratio=0.95,
            retreat_ratio=0.75,
            minimum_teacher_accuracy=0.80,
            minimum_frontier_accuracy=0.80,
            advance_patience=3,
            full_probe_fraction=0.02,
        )
        original.observe(
            unsupported_accuracy=0.79,
            teacher_accuracy=0.82,
        )
        state = original.state_dict()

        tuned = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=0,
            advance_ratio=0.90,
            retreat_ratio=0.75,
            minimum_teacher_accuracy=0.75,
            minimum_frontier_accuracy=0.60,
            advance_patience=3,
            full_probe_fraction=0.02,
        )
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            tuned.load_state_dict(state)
        tuned.load_state_dict(state, allow_policy_override=True)

        self.assertEqual(tuned.removed_tokens, original.removed_tokens)
        self.assertEqual(tuned.advance_streak, original.advance_streak)
        self.assertEqual(tuned.accuracy_ratio_ema, original.accuracy_ratio_ema)
        self.assertEqual(tuned.minimum_teacher_accuracy, 0.75)
        self.assertEqual(tuned.minimum_frontier_accuracy, 0.60)
        self.assertEqual(tuned.advance_ratio, 0.90)

    def test_tokenwise_log_reports_free_probe(self) -> None:
        controller = TokenwiseTeacherHandoff(
            response_slots=8,
            full_probe_fraction=0.0,
        )
        line = token_handoff_log_line(
            controller=controller,
            partial_result={"unsupported_content_accuracy": 0.4},
            teacher_result={
                "unsupported_content_accuracy": 0.5,
                "content_perplexity": 40.0,
            },
            unsupported_result={
                "content_accuracy": 0.1,
                "content_top5_accuracy": 0.3,
                "content_perplexity": 50.0,
                "formation_coverage": 0.2,
            },
            action="hold",
        )
        self.assertIn("teacher_visible=6/7", line)
        self.assertIn("frontier_acc=0.400", line)
        self.assertIn("free_content_acc=0.100", line)
        self.assertIn("free_content_top5=0.300", line)
        self.assertIn("free_coverage=0.200", line)
        self.assertIn("teacher_content_ppl=40.00", line)
        self.assertIn("retention_ratio=disabled", line)

    def test_tokenwise_teacher_ppl_guard_blocks_then_retreats(self) -> None:
        controller = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=1,
            advance_ratio=0.90,
            retreat_ratio=0.70,
            advance_patience=2,
            retreat_patience=2,
            ema_decay=0.0,
            full_probe_fraction=0.0,
            maximum_teacher_perplexity_ratio=1.25,
            teacher_perplexity_retreat_ratio=1.35,
            teacher_perplexity_baseline=40.0,
        )

        for _ in range(2):
            self.assertEqual(
                controller.observe(
                    unsupported_accuracy=0.50,
                    teacher_accuracy=0.50,
                    teacher_content_perplexity=51.0,
                ),
                "hold",
            )
        self.assertEqual(controller.removed_tokens, 1)
        self.assertAlmostEqual(controller.teacher_perplexity_ratio, 1.275)

        for expected in ("hold", "advance"):
            self.assertEqual(
                controller.observe(
                    unsupported_accuracy=0.50,
                    teacher_accuracy=0.50,
                    teacher_content_perplexity=48.0,
                ),
                expected,
            )
        self.assertEqual(controller.removed_tokens, 2)

        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.50,
                teacher_accuracy=0.50,
                teacher_content_perplexity=56.0,
            ),
            "hold",
        )
        self.assertEqual(
            controller.observe(
                unsupported_accuracy=0.50,
                teacher_accuracy=0.50,
                teacher_content_perplexity=56.0,
            ),
            "retreat",
        )
        self.assertEqual(controller.removed_tokens, 1)

        restored = TokenwiseTeacherHandoff(
            response_slots=8,
            start_removed_tokens=1,
            advance_ratio=0.90,
            retreat_ratio=0.70,
            advance_patience=2,
            retreat_patience=2,
            ema_decay=0.0,
            full_probe_fraction=0.0,
            maximum_teacher_perplexity_ratio=1.25,
            teacher_perplexity_retreat_ratio=1.35,
            teacher_perplexity_baseline=40.0,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_tokenwise_teacher_ppl_guard_can_seed_its_baseline(self) -> None:
        controller = TokenwiseTeacherHandoff(
            response_slots=8,
            maximum_teacher_perplexity_ratio=1.25,
            teacher_perplexity_retreat_ratio=1.35,
        )
        controller.observe(
            unsupported_accuracy=0.20,
            teacher_accuracy=0.25,
            teacher_content_perplexity=42.0,
        )
        self.assertEqual(controller.teacher_perplexity_baseline, 42.0)
        self.assertEqual(controller.teacher_perplexity_ratio, 1.0)

    def test_graduated_handoff_alternates_then_adds_unsupported_steps(
        self,
    ) -> None:
        controller = GraduatedTeacherHandoff(
            planned_steps=100,
            baseline_fraction=0.05,
            accuracy_step=0.05,
            release_accuracy=0.90,
            ema_decay=0.0,
        )

        baseline_decisions = [
            controller.choose_teacher_batch(completed_step=step)
            for step in range(1, 6)
        ]
        self.assertEqual(
            baseline_decisions,
            [False, True, False, True, False],
        )
        controller.observe(0.30, completed_step=5)
        self.assertEqual(controller.baseline_accuracy, 0.30)
        self.assertEqual(controller.unsupported_burst, 1)

        controller.observe(0.35, completed_step=10)
        self.assertEqual(controller.unsupported_burst, 2)
        decisions = [
            controller.choose_teacher_batch(completed_step=step)
            for step in range(6, 12)
        ]
        self.assertEqual(
            decisions,
            [False, False, True, False, False, True],
        )

        controller.observe(0.45, completed_step=12)
        self.assertEqual(controller.unsupported_burst, 4)
        self.assertAlmostEqual(controller.teacher_fraction, 0.2)

    def test_graduated_handoff_releases_teacher_and_round_trips(self) -> None:
        controller = GraduatedTeacherHandoff(
            planned_steps=200,
            ema_decay=0.0,
        )
        controller.observe(0.25, completed_step=10)
        controller.observe(0.90, completed_step=20)

        self.assertTrue(controller.teacher_released)
        self.assertEqual(controller.teacher_fraction, 0.0)
        self.assertFalse(
            controller.choose_teacher_batch(completed_step=21)
        )

        restored = GraduatedTeacherHandoff(
            planned_steps=200,
            ema_decay=0.0,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_graduated_handoff_logs_unsupported_gate(self) -> None:
        controller = GraduatedTeacherHandoff(
            planned_steps=100,
            ema_decay=0.0,
        )
        controller.observe(0.30, completed_step=5)
        line = handoff_log_line(
            controller=controller,
            teacher_result={
                "content_accuracy": 0.50,
                "content_perplexity": 20.0,
            },
            parallel_result={
                "content_accuracy": 0.30,
                "content_perplexity": 40.0,
            },
        )
        self.assertIn("baseline_acc=0.300", line)
        self.assertIn("unsupported_burst=1", line)
        self.assertIn("parallel_content_acc=0.300", line)

    def test_accuracy_gated_handoff_matches_requested_curve(self) -> None:
        controller = AccuracyGatedTeacherHandoff(
            start_accuracy=0.60,
            end_accuracy=0.90,
            ema_decay=0.0,
        )
        self.assertEqual(controller.teacher_fraction, 1.0)

        controller.observe(0.59)
        self.assertEqual(controller.teacher_fraction, 1.0)
        controller.observe(0.75)
        self.assertAlmostEqual(controller.teacher_fraction, 0.5)
        controller.observe(0.90)
        self.assertEqual(controller.teacher_fraction, 0.0)

    def test_handoff_uses_whole_batches_at_fractional_support(self) -> None:
        controller = AccuracyGatedTeacherHandoff(
            start_accuracy=0.60,
            end_accuracy=0.90,
            ema_decay=0.0,
        )
        controller.observe(0.75)

        decisions = [
            controller.choose_teacher_batch()
            for _ in range(10)
        ]

        self.assertEqual(sum(decisions), 5)
        self.assertEqual(controller.teacher_batches, 5)
        self.assertEqual(controller.parallel_batches, 5)

    def test_handoff_state_round_trip_and_log(self) -> None:
        controller = AccuracyGatedTeacherHandoff(ema_decay=0.5)
        controller.observe(0.70)
        controller.choose_teacher_batch()
        restored = AccuracyGatedTeacherHandoff(ema_decay=0.5)
        restored.load_state_dict(controller.state_dict())

        self.assertEqual(restored.state_dict(), controller.state_dict())
        line = handoff_log_line(
            controller=restored,
            teacher_result={
                "content_accuracy": 0.7,
                "content_perplexity": 12.5,
            },
            parallel_result={
                "content_accuracy": 0.2,
                "content_perplexity": 50.0,
            },
        )
        self.assertIn("teacher_fraction=0.667", line)
        self.assertIn("teacher_content_acc=0.700", line)
        self.assertIn("parallel_content_acc=0.200", line)
        self.assertIn("batches=0:1", line)

    def test_shifted_outputs_start_with_blank(self) -> None:
        batch = LanguageBatch(
            context_ids=torch.tensor([[3, 4]]),
            context_mask=torch.tensor([[True, True]]),
            target_ids=torch.tensor([[10, 11, 2, 0]]),
            target_mask=torch.tensor([[True, True, True, False]]),
            target_complete=torch.tensor([True]),
        )

        shifted = teacher_forced_output_inputs(
            batch,
            blank_token_id=31,
        )

        self.assertEqual(shifted.tolist(), [[31, 10, 11, 2]])

    def test_prefix_denoising_can_preserve_or_remove_all_support(self) -> None:
        batch = LanguageBatch(
            context_ids=torch.tensor([[3, 4], [5, 6]]),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
            target_ids=torch.tensor(
                [[10, 11, 2, 0], [12, 13, 14, 2]]
            ),
            target_mask=torch.tensor(
                [[True, True, True, False], [True, True, True, True]]
            ),
            target_complete=torch.tensor([True, True]),
        )
        preserved, preserved_stats = prefix_denoising_output_inputs(
            batch,
            blank_token_id=31,
            retention=1.0,
            zero_prefix_fraction=0.0,
            short_prefix_fraction=0.0,
            short_prefix_max_tokens=2,
            span_fraction=0.0,
        )
        removed, removed_stats = prefix_denoising_output_inputs(
            batch,
            blank_token_id=31,
            retention=0.0,
            zero_prefix_fraction=0.0,
            short_prefix_fraction=0.0,
            short_prefix_max_tokens=2,
            span_fraction=0.0,
        )

        self.assertEqual(
            preserved.tolist(),
            [[31, 10, 11, 2], [31, 12, 13, 14]],
        )
        self.assertEqual(preserved_stats["actual_retention"], 1.0)
        self.assertTrue(removed.eq(31).all())
        self.assertEqual(removed_stats["actual_retention"], 0.0)

    def test_short_and_zero_prefix_rows_never_expose_future_tokens(self) -> None:
        torch.manual_seed(3)
        batch = LanguageBatch(
            context_ids=torch.tensor([[3, 4], [5, 6]]),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
            target_ids=torch.tensor(
                [[10, 11, 12, 2], [20, 21, 22, 2]]
            ),
            target_mask=torch.ones(2, 4, dtype=torch.bool),
            target_complete=torch.tensor([True, True]),
        )
        zeroed, _ = prefix_denoising_output_inputs(
            batch,
            blank_token_id=31,
            retention=1.0,
            zero_prefix_fraction=1.0,
            short_prefix_fraction=0.0,
            short_prefix_max_tokens=2,
        )
        shortened, statistics = prefix_denoising_output_inputs(
            batch,
            blank_token_id=31,
            retention=1.0,
            zero_prefix_fraction=0.0,
            short_prefix_fraction=1.0,
            short_prefix_max_tokens=2,
        )

        self.assertTrue(zeroed.eq(31).all())
        self.assertEqual(statistics["short_rows"], 2.0)
        for row in shortened:
            visible = row.ne(31).nonzero(as_tuple=False).flatten().tolist()
            self.assertIn(visible, ([1], [1, 2]))

    def test_prefix_controller_requires_accuracy_and_context_utility(self) -> None:
        controller = PrefixDenoisingController(
            start_retention=0.95,
            floor_retention=0.50,
            warmup_steps=100,
            start_accuracy=0.60,
            end_accuracy=0.90,
            start_context_utility=0.02,
            end_context_utility=0.10,
            ema_decay=0.0,
        )
        self.assertAlmostEqual(controller.retention(0), 0.95)
        self.assertAlmostEqual(controller.retention(100), 0.50)

        controller.observe(
            content_accuracy=0.75,
            shuffled_context_accuracy=0.75,
        )
        self.assertEqual(controller.readiness, 0.0)
        self.assertAlmostEqual(controller.retention(100), 0.50)

        controller.observe(
            content_accuracy=0.75,
            shuffled_context_accuracy=0.70,
            supported_context_accuracy=0.75,
        )
        self.assertAlmostEqual(controller.readiness, 0.375)
        self.assertAlmostEqual(controller.retention(100), 0.3125)

        restored = PrefixDenoisingController(
            start_retention=0.95,
            floor_retention=0.50,
            warmup_steps=100,
            start_accuracy=0.60,
            end_accuracy=0.90,
            start_context_utility=0.02,
            end_context_utility=0.10,
            ema_decay=0.0,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

        line = prefix_denoising_log_line(
            controller=restored,
            completed_step=100,
            corruption={
                "actual_retention": 0.4,
                "zero_rows": 1.0,
                "short_rows": 1.0,
                "span_rows": 2.0,
                "token_rows": 4.0,
            },
            teacher_result={
                "content_accuracy": 0.75,
                "content_perplexity": 10.0,
            },
            supported_context_result={
                "content_accuracy": 0.75,
                "content_perplexity": 11.0,
            },
            shuffled_context_result={
                "content_accuracy": 0.70,
                "content_perplexity": 12.0,
            },
        )
        self.assertIn("retention=0.312", line)
        self.assertIn("context_utility=+0.050", line)

    def test_paired_context_probe_uses_identical_prefix_mask(self) -> None:
        batch = LanguageBatch(
            context_ids=torch.tensor([[3, 4], [5, 6]]),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
            target_ids=torch.tensor(
                [[10, 11, 12, 2], [20, 21, 22, 2]]
            ),
            target_mask=torch.ones(2, 4, dtype=torch.bool),
            target_complete=torch.tensor([True, True]),
        )
        first = deterministic_prefix_evaluation_inputs(
            batch,
            blank_token_id=31,
            retention=0.5,
            row_offset=7,
            seed=17,
        )
        torch.manual_seed(999)
        second = deterministic_prefix_evaluation_inputs(
            batch,
            blank_token_id=31,
            retention=0.5,
            row_offset=7,
            seed=17,
        )

        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(first[:, 0].eq(31).all())

    def test_future_teacher_token_cannot_change_earlier_logits(self) -> None:
        torch.manual_seed(7)
        model = ConvergentLanguageModel(
            ConvergentLanguageConfig(
                vocab_size=32,
                d_model=8,
                embedding_dim=8,
                n_nodes=8,
                degree=3,
                relays=2,
                lanes=4,
                mask_token_id=3,
                blank_token_id=31,
                connection_layout="small_world",
                connection_seed=0,
                connection_rewire=0.5,
                terminal_vote_mode="slot",
            )
        ).eval()
        context = torch.tensor([[3, 5, 6]])
        context_mask = torch.ones_like(context, dtype=torch.bool)
        output_mask = torch.ones(1, 4, dtype=torch.bool)
        teacher_a = torch.tensor([[31, 10, 11, 12]])
        teacher_b = torch.tensor([[31, 10, 11, 20]])

        with torch.no_grad():
            causal_a = model(
                context,
                context_mask,
                output_slots=4,
                output_token_ids=teacher_a,
                output_mask=output_mask,
                causal=True,
            )
            causal_b = model(
                context,
                context_mask,
                output_slots=4,
                output_token_ids=teacher_b,
                output_mask=output_mask,
                causal=True,
            )
            dense_a = model(
                context,
                context_mask,
                output_slots=4,
                output_token_ids=teacher_a,
                output_mask=output_mask,
                causal=False,
            )
            dense_b = model(
                context,
                context_mask,
                output_slots=4,
                output_token_ids=teacher_b,
                output_mask=output_mask,
                causal=False,
            )

        torch.testing.assert_close(
            causal_a[:, :3],
            causal_b[:, :3],
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            float((dense_a[:, :3] - dense_b[:, :3]).abs().max()),
            1e-8,
        )

    def test_teacher_forced_backward_is_finite(self) -> None:
        torch.manual_seed(11)
        model = ConvergentLanguageModel(
            ConvergentLanguageConfig(
                vocab_size=32,
                d_model=8,
                embedding_dim=8,
                n_nodes=8,
                degree=3,
                relays=2,
                lanes=4,
                mask_token_id=3,
                blank_token_id=31,
                terminal_vote_mode="slot",
            )
        )
        context = torch.tensor(
            [
                [3, 4, 5],
                [3, 6, 7],
            ]
        )
        context_mask = torch.ones_like(context, dtype=torch.bool)
        targets = torch.tensor(
            [
                [8, 9, 2, 0],
                [10, 11, 12, 2],
            ]
        )
        target_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        batch = LanguageBatch(
            context_ids=context,
            context_mask=context_mask,
            target_ids=targets,
            target_mask=target_mask,
            target_complete=torch.tensor([True, True]),
        )
        logits = model(
            context,
            context_mask,
            output_slots=4,
            output_token_ids=teacher_forced_output_inputs(
                batch,
                blank_token_id=31,
            ),
            output_mask=target_mask,
            causal=True,
        )
        loss = torch.nn.functional.cross_entropy(
            logits[target_mask],
            targets[target_mask],
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
