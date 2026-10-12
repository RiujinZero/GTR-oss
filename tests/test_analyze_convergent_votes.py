from __future__ import annotations

import unittest

import torch

from analyze_convergent_checkpoint import _fuse
from analyze_convergent_votes import summarize_votes, terminal_vote_logits


class ConvergentVoteAnalysisTests(unittest.TestCase):
    def test_uniform_votes_report_all_effective_lanes(self) -> None:
        weights, summary = summarize_votes(torch.zeros(4))

        torch.testing.assert_close(
            weights,
            torch.full((4,), 0.25, dtype=torch.float64),
        )
        self.assertAlmostEqual(summary.effective_entropy, 4.0)
        self.assertAlmostEqual(summary.effective_l2, 4.0)
        self.assertEqual(summary.top90_count, 4)
        self.assertEqual(summary.top95_count, 4)
        self.assertEqual(summary.top99_count, 4)

    def test_concentrated_vote_reports_one_effective_lane(self) -> None:
        _, summary = summarize_votes(
            torch.tensor([0.0, -100.0, -100.0])
        )

        self.assertAlmostEqual(summary.effective_entropy, 1.0)
        self.assertAlmostEqual(summary.effective_l2, 1.0)
        self.assertEqual(summary.top90_count, 1)

    def test_finds_wrapped_language_terminal_votes(self) -> None:
        payload = {
            "model": {
                "core.terminal_votes": torch.tensor([1.0, 2.0]),
                "core.other": torch.tensor([0.0]),
            }
        }

        name, logits = terminal_vote_logits(payload)

        self.assertEqual(name, "core.terminal_votes")
        torch.testing.assert_close(logits, torch.tensor([1.0, 2.0], dtype=torch.float64))

    def test_rejects_ambiguous_terminal_votes(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            terminal_vote_logits(
                {
                    "model": {
                        "a.terminal_votes": torch.zeros(2),
                        "b.terminal_votes": torch.zeros(2),
                    }
                }
            )

    def test_fuse_supports_global_prompt_and_slot_juries(self) -> None:
        torch.manual_seed(3)
        lane_logits = torch.randn(2, 4, 3, 7)
        global_weights = torch.softmax(torch.randn(3), dim=-1)
        prompt_weights = global_weights.expand(2, -1)
        slot_weights = global_weights.expand(2, 4, -1)

        expected = _fuse(lane_logits, global_weights)

        torch.testing.assert_close(
            _fuse(lane_logits, prompt_weights),
            expected,
        )
        torch.testing.assert_close(
            _fuse(lane_logits, slot_weights),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
