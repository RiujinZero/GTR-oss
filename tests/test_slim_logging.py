from __future__ import annotations

import math
import unittest

import torch

from gator_language_train import (
    slim_routing_log_line,
    slim_step_log_line,
    slim_terminal_vote_log_line,
    terminal_vote_summary,
)


class SlimLoggingTests(unittest.TestCase):
    def test_aligned_step_uses_selected_loss_and_required_accuracy(self) -> None:
        line = slim_step_log_line(
            step=50,
            steps_per_second=12.345,
            result={
                "formation_objective": "monotonic",
                "selection_loss": math.log(4.0),
                "aligned_required_accuracy": 0.375,
                "aligned_exact": 0.125,
                "accuracy": 0.9,
                "top5_accuracy": 0.8,
                "content_accuracy": 0.5,
                "content_top5_accuracy": 0.75,
                "formation_coverage": 0.625,
                "whitespace_omission_rate": 0.375,
                "exact": 0.8,
            },
        )

        self.assertEqual(
            line,
            "step=50 steps_per_second=12.35 loss=1.3863 "
            "ppl=4.00 acc=0.375 top5=0.800 "
            "content_acc=0.500 content_top5=0.750 "
            "coverage=0.625 ws_omission=0.375 exact=0.125",
        )

    def test_rigid_step_uses_rigid_accuracy(self) -> None:
        line = slim_step_log_line(
            step=7,
            steps_per_second=2.0,
            result={
                "formation_objective": "slot_ce",
                "selection_loss": 1.0,
                "aligned_required_accuracy": 0.1,
                "aligned_exact": 0.0,
                "accuracy": 0.625,
                "top5_accuracy": 0.875,
                "content_accuracy": 0.5,
                "content_top5_accuracy": 0.75,
                "formation_coverage": 0.625,
                "whitespace_omission_rate": 0.375,
                "exact": 0.25,
            },
        )

        self.assertIn(
            "acc=0.625 top5=0.875 content_acc=0.500 "
            "content_top5=0.750 coverage=0.625 "
            "ws_omission=0.375 exact=0.250",
            line,
        )

    def test_routing_line_keeps_only_operator_metrics(self) -> None:
        line = slim_routing_log_line(
            {
                "support_by_relay": [8.0, 4.125],
                "effective_cells_by_relay": [7.9, 3.5],
                "handoff_support_by_relay": [2.25],
                "stay_rate_by_relay": [0.2],
                "dead_cell_fraction_by_relay": [0.125],
                "packed_execution_by_relay": [False, True],
            }
        )

        self.assertEqual(
            line,
            "routing support=[8.00,4.12] effective=[7.90,3.50] "
            "beam=[2.25] stay=[0.20] dead=[0.12] packed=[0,1]",
        )

    def test_terminal_summary_normalizes_all_three_jury_shapes(self) -> None:
        global_weights = torch.tensor([0.5, 0.3, 0.2])
        prompt_weights = global_weights.expand(2, -1)
        slot_weights = global_weights.expand(2, 4, -1)

        expected = terminal_vote_summary(
            global_weights,
            batch_size=2,
            output_slots=4,
        )
        self.assertEqual(
            terminal_vote_summary(
                prompt_weights,
                batch_size=2,
                output_slots=4,
            ),
            expected,
        )
        self.assertEqual(
            terminal_vote_summary(
                slot_weights,
                batch_size=2,
                output_slots=4,
            ),
            expected,
        )
        self.assertEqual(expected["adjacent_slot_delta"], 0.0)
        self.assertEqual(expected["winning_lanes_per_response"], 1.0)

    def test_slot_terminal_summary_detects_position_specialization(self) -> None:
        slot_weights = torch.eye(3)[None]
        summary = terminal_vote_summary(
            slot_weights,
            batch_size=1,
            output_slots=3,
        )

        self.assertAlmostEqual(summary["effective_voters"], 1.0)
        self.assertAlmostEqual(summary["top1_mass"], 1.0)
        self.assertAlmostEqual(summary["adjacent_slot_delta"], 2.0 / 3.0)
        self.assertEqual(summary["winning_lanes_per_response"], 3.0)
        self.assertEqual(
            slim_terminal_vote_log_line(summary),
            "terminal effective=1.00 top1=1.000 "
            "slot_delta=0.6667 winning_lanes=3.00 query_rms=0.000000",
        )


if __name__ == "__main__":
    unittest.main()
