from __future__ import annotations

from itertools import combinations
import math
import unittest

import torch
import torch.nn.functional as F

from gator.alignment import (
    best_aligned_diagnostics,
    collapse_blank_sequences,
    monotonic_forward_sum_nll,
    sequence_quality_loss,
    viterbi_aligned_cross_entropy,
    viterbi_expand_targets,
)


BLANK = 0
EOS = 1
A = 2
B = 3
X = 4
VOCABULARY = 6


def peaked_logits(rows: list[list[int]], vocabulary: int = VOCABULARY) -> torch.Tensor:
    logits = torch.full((len(rows), len(rows[0]), vocabulary), -8.0)
    for batch, row in enumerate(rows):
        for slot, token in enumerate(row):
            logits[batch, slot, token] = 8.0
    return logits


class AlignmentTests(unittest.TestCase):
    def test_forward_sum_matches_brute_force(self) -> None:
        probabilities = torch.tensor(
            [
                [0.40, 0.05, 0.50, 0.05],
                [0.20, 0.25, 0.50, 0.05],
                [0.30, 0.55, 0.10, 0.05],
            ],
            dtype=torch.float64,
        )
        logits = probabilities.log()[None]
        reference = torch.tensor([[A, EOS]])
        mask = torch.tensor([[True, True]])
        actual = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
            normalize=False,
            reduction="none",
        )
        total = 0.0
        for emitted_slots in combinations(range(3), 2):
            path = 1.0
            target_index = 0
            for slot in range(3):
                if slot in emitted_slots:
                    path *= float(probabilities[slot, reference[0, target_index]])
                    target_index += 1
                else:
                    path *= float(probabilities[slot, BLANK])
            total += path
        expected = -math.log(total)
        self.assertAlmostEqual(float(actual[0]), expected, places=12)

    def test_full_reference_reduces_to_rigid_cross_entropy(self) -> None:
        torch.manual_seed(3)
        logits = torch.randn(2, 3, VOCABULARY, dtype=torch.float64)
        reference = torch.tensor([[A, B, EOS], [B, A, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        rigid = F.cross_entropy(logits.transpose(1, 2), reference)
        marginal = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.ones(2, dtype=torch.bool),
        )
        aligned = viterbi_aligned_cross_entropy(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.ones(2, dtype=torch.bool),
        )
        torch.testing.assert_close(marginal, rigid)
        torch.testing.assert_close(aligned.loss, rigid)
        torch.testing.assert_close(aligned.alignment.expanded_ids, reference)

    def test_collapse_preserves_repetition_and_whitespace_like_tokens(self) -> None:
        whitespace_piece = 5
        tokens = torch.tensor([[A, BLANK, A, whitespace_piece, A, EOS, B]])
        collapsed = collapse_blank_sequences(tokens, blank_id=BLANK, eos_id=EOS)
        self.assertEqual(collapsed.tokens, ((A, A, whitespace_piece, A),))
        self.assertTrue(bool(collapsed.terminated[0]))
        self.assertEqual(int(collapsed.eos_positions[0]), 5)

    def test_repeated_required_tokens_are_emitted_independently(self) -> None:
        logits = peaked_logits([[A, BLANK, A, EOS]])
        reference = torch.tensor([[A, A, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        alignment = viterbi_expand_targets(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        self.assertEqual(alignment.positions.tolist(), [[0, 2, 3]])
        self.assertEqual(alignment.expanded_ids.tolist(), [[A, BLANK, A, EOS]])

    def test_max_right_shift_limits_viterbi_path(self) -> None:
        logits = torch.full((1, 3, VOCABULARY), -8.0)
        logits[0, 0, BLANK] = 8.0
        logits[0, 0, A] = 0.0
        logits[0, 1, A] = 8.0
        logits[0, 1, EOS] = 1.0
        logits[0, 2, EOS] = 8.0
        reference = torch.tensor([[A, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        unrestricted = viterbi_expand_targets(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        anchored = viterbi_expand_targets(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
            max_right_shift=0,
        )
        self.assertEqual(unrestricted.positions.tolist(), [[1, 2]])
        self.assertEqual(anchored.positions.tolist(), [[0, 1]])

    def test_padded_reference_values_are_ignored(self) -> None:
        torch.manual_seed(5)
        logits = torch.randn(1, 4, VOCABULARY, dtype=torch.float64)
        mask = torch.tensor([[True, True, False, False]])
        first = monotonic_forward_sum_nll(
            logits,
            torch.tensor([[A, EOS, B, X]]),
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        second = monotonic_forward_sum_nll(
            logits,
            torch.tensor([[A, EOS, X, A]]),
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        torch.testing.assert_close(first, second)

    def test_valid_token_mask_removes_invalid_high_logit(self) -> None:
        logits = torch.zeros(1, 3, VOCABULARY, dtype=torch.float64)
        logits[..., X] = 100.0
        valid = torch.ones(VOCABULARY, dtype=torch.bool)
        valid[X] = False
        reference = torch.tensor([[A, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        first = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            valid_token_mask=valid,
        )
        logits[..., X] = -100.0
        second = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            valid_token_mask=valid,
        )
        torch.testing.assert_close(first, second)

    def test_precomputed_log_probs_match_internal_normalization(self) -> None:
        torch.manual_seed(23)
        logits = torch.randn(2, 4, VOCABULARY, dtype=torch.float64)
        reference = torch.tensor([[A, EOS], [B, A]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        complete = torch.tensor([True, False])
        valid = torch.ones(VOCABULARY, dtype=torch.bool)
        valid[X] = False
        log_probs = F.log_softmax(
            logits.masked_fill(~valid[None, None], -torch.inf), dim=-1
        )

        internal = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            valid_token_mask=valid,
            reduction="none",
        )
        shared = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            valid_token_mask=valid,
            precomputed_log_probs=log_probs,
            reduction="none",
        )
        torch.testing.assert_close(internal, shared)

        viterbi_internal = viterbi_aligned_cross_entropy(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            valid_token_mask=valid,
        )
        viterbi_shared = viterbi_aligned_cross_entropy(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            valid_token_mask=valid,
            precomputed_log_probs=log_probs,
        )
        torch.testing.assert_close(viterbi_internal.loss, viterbi_shared.loss)
        torch.testing.assert_close(
            viterbi_internal.required_token_loss,
            viterbi_shared.required_token_loss,
        )

    def test_impossible_reference_has_infinite_loss_and_invalid_path(self) -> None:
        logits = torch.zeros(1, 2, VOCABULARY)
        reference = torch.tensor([[A, B, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        loss = monotonic_forward_sum_nll(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            reduction="none",
        )
        alignment = viterbi_expand_targets(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
        )
        self.assertTrue(torch.isinf(loss[0]))
        self.assertFalse(bool(alignment.valid[0]))

    def test_eos_only_reference_is_well_defined(self) -> None:
        logits = peaked_logits([[EOS, BLANK, BLANK]])
        reference = torch.tensor([[EOS]])
        mask = torch.tensor([[True]])
        diagnostics = best_aligned_diagnostics(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        self.assertEqual(float(diagnostics.required_token_accuracy[0]), 1.0)
        self.assertEqual(float(diagnostics.content_accuracy[0]), 1.0)
        self.assertTrue(bool(diagnostics.rendered_exact[0]))
        self.assertEqual(int(diagnostics.reference_lengths[0]), 0)

    def test_blank_tail_cannot_dominate_formation_accuracy(self) -> None:
        logits = peaked_logits([[BLANK] * 8])
        reference = torch.tensor([[A, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        diagnostics = best_aligned_diagnostics(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        self.assertAlmostEqual(float(diagnostics.formation_accuracy[0]), 0.5, places=6)
        self.assertEqual(float(diagnostics.required_token_accuracy[0]), 0.0)

    def test_edit_counts_and_rendered_exact(self) -> None:
        logits = peaked_logits(
            [
                [A, X, B, EOS],  # one insertion after collapse
                [A, EOS, BLANK, BLANK],  # one deletion after collapse
                [A, X, EOS, BLANK],  # one substitution after collapse
                [A, B, EOS, BLANK],
            ]
        )
        reference = torch.tensor([[A, B, EOS]] * 4)
        mask = torch.ones_like(reference, dtype=torch.bool)
        diagnostics = best_aligned_diagnostics(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.ones(4, dtype=torch.bool),
        )
        self.assertEqual(diagnostics.insertions.tolist(), [1, 0, 0, 0])
        self.assertEqual(diagnostics.deletions.tolist(), [0, 1, 0, 0])
        self.assertEqual(diagnostics.substitutions.tolist(), [0, 0, 1, 0])
        self.assertEqual(diagnostics.edit_distance.tolist(), [1, 1, 1, 0])
        self.assertEqual(diagnostics.rendered_exact.tolist(), [False, False, False, True])
        self.assertAlmostEqual(float(diagnostics.edit_accuracy[2]), 0.5, places=6)

    def test_censored_examples_do_not_receive_eos_count_pressure(self) -> None:
        logits = peaked_logits([[A, B, X]])
        reference = torch.tensor([[A, B]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        quality = sequence_quality_loss(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([False]),
            reduction="none",
        )
        diagnostics = best_aligned_diagnostics(
            logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([False]),
        )
        self.assertEqual(float(quality.complete_eos_count[0]), 0.0)
        self.assertEqual(float(quality.semantic_composition[0]), 0.0)
        self.assertLess(float(quality.expected_length_huber[0]), 1e-6)
        self.assertFalse(bool(diagnostics.rendered_exact_mask[0]))
        self.assertFalse(bool(diagnostics.rendered_exact[0]))

    def test_censored_tail_is_a_free_unobserved_continuation(self) -> None:
        torch.manual_seed(17)
        first = torch.randn(1, 4, VOCABULARY, dtype=torch.float64)
        second = first.clone()
        second[:, 2:] = torch.randn_like(second[:, 2:]) * 20.0
        reference = torch.tensor([[A, B]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        complete = torch.tensor([False])

        first_loss = monotonic_forward_sum_nll(
            first,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            max_right_shift=0,
            normalize=False,
            reduction="none",
        )
        second_loss = monotonic_forward_sum_nll(
            second,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            max_right_shift=0,
            normalize=False,
            reduction="none",
        )
        torch.testing.assert_close(first_loss, second_loss)

        aligned = viterbi_aligned_cross_entropy(
            first,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            max_right_shift=0,
        )
        self.assertEqual(aligned.alignment.score_mask.tolist(), [[True, True, False, False]])
        torch.testing.assert_close(
            aligned.required_token_loss,
            aligned.loss * 2.0,
        )

    def test_edit_accuracy_and_content_prf_follow_documented_counts(self) -> None:
        diagnostics = best_aligned_diagnostics(
            peaked_logits([[B, X, A, EOS, BLANK]]),
            torch.tensor([[A, B, X, EOS]]),
            torch.ones(1, 4, dtype=torch.bool),
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
        )
        self.assertEqual(diagnostics.insertions.tolist(), [1])
        self.assertEqual(diagnostics.deletions.tolist(), [1])
        self.assertEqual(diagnostics.substitutions.tolist(), [0])
        self.assertAlmostEqual(float(diagnostics.edit_accuracy[0]), 0.5, places=6)
        self.assertAlmostEqual(
            float(diagnostics.content_precision[0]), 2.0 / 3.0, places=6
        )
        self.assertAlmostEqual(
            float(diagnostics.content_recall[0]), 2.0 / 3.0, places=6
        )
        self.assertAlmostEqual(float(diagnostics.content_f1[0]), 2.0 / 3.0, places=6)

    def test_sequence_quality_prefers_correct_global_formation(self) -> None:
        reference = torch.tensor([[A, B, EOS]])
        mask = torch.ones_like(reference, dtype=torch.bool)
        complete = torch.tensor([True])
        semantic = torch.zeros(VOCABULARY, dtype=torch.bool)
        semantic[A] = semantic[B] = True
        good = sequence_quality_loss(
            peaked_logits([[A, B, EOS, BLANK]]),
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            semantic_token_mask=semantic,
        )
        bad = sequence_quality_loss(
            peaked_logits([[A, A, A, A]]),
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
            semantic_token_mask=semantic,
        )
        self.assertLess(float(good.loss), float(bad.loss))
        self.assertLess(float(good.semantic_composition), float(bad.semantic_composition))
        self.assertLess(float(good.complete_eos_count), float(bad.complete_eos_count))

    def test_sequence_quality_aggregate_uses_configured_weights(self) -> None:
        result = sequence_quality_loss(
            peaked_logits([[A, A, EOS, BLANK]]),
            torch.tensor([[A, B, EOS]]),
            torch.tensor([[True, True, True]]),
            blank_id=BLANK,
            eos_id=EOS,
            complete=torch.tensor([True]),
            composition_weight=0.7,
            length_weight=0.2,
            eos_weight=0.05,
            reduction="none",
        )
        expected = (
            0.7 * result.semantic_composition
            + 0.2 * result.expected_length_huber
            + 0.05 * result.complete_eos_count
        )
        torch.testing.assert_close(result.loss, expected)

    def test_all_losses_have_finite_gradients(self) -> None:
        torch.manual_seed(9)
        reference = torch.tensor([[A, B, EOS], [B, EOS, X]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        complete = torch.tensor([True, True])

        marginal_logits = torch.randn(2, 5, VOCABULARY, requires_grad=True)
        marginal = monotonic_forward_sum_nll(
            marginal_logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
        )
        marginal.backward()
        self.assertTrue(torch.isfinite(marginal_logits.grad).all())
        self.assertGreater(float(marginal_logits.grad.abs().sum()), 0.0)

        aligned_logits = torch.randn(2, 5, VOCABULARY, requires_grad=True)
        aligned = viterbi_aligned_cross_entropy(
            aligned_logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
        )
        aligned.loss.backward()
        self.assertTrue(torch.isfinite(aligned_logits.grad).all())
        self.assertGreater(float(aligned_logits.grad.abs().sum()), 0.0)

        quality_logits = torch.randn(2, 5, VOCABULARY, requires_grad=True)
        quality = sequence_quality_loss(
            quality_logits,
            reference,
            mask,
            blank_id=BLANK,
            eos_id=EOS,
            complete=complete,
        )
        quality.loss.backward()
        self.assertTrue(torch.isfinite(quality_logits.grad).all())
        self.assertGreater(float(quality_logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
