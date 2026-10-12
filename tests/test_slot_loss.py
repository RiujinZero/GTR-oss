from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from gator.optimizers.slot_loss import (
    SlotLossResult,
    batch_mean_slot_loss,
    batch_centered_cross_entropy_slot_loss,
    cross_entropy_slot_loss,
)


class SlotLossTests(unittest.TestCase):
    def test_per_slot_cross_entropy_and_valid_mask(self) -> None:
        logits = torch.tensor(
            [
                [[3.0, 0.0, -1.0, 0.5], [0.0, 2.0, 0.0, -1.0], [9, 0, 0, 0]],
                [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 3.0, 0.0], [0, 9, 0, 0]],
            ]
        )
        targets = torch.tensor([[0, 1, -100], [3, -100, -100]])
        valid = torch.tensor([[True, True, False], [True, False, False]])

        result = cross_entropy_slot_loss(logits, targets, valid)
        safe = torch.where(valid, targets, torch.zeros_like(targets))
        expected = F.cross_entropy(
            logits.reshape(-1, 4),
            safe.reshape(-1),
            reduction="none",
        ).reshape(2, 3)

        torch.testing.assert_close(result.losses[valid], expected[valid])
        self.assertTrue(torch.equal(result.losses[~valid], torch.zeros(3)))
        torch.testing.assert_close(result.aggregate, expected[valid].mean())
        self.assertTrue(torch.equal(result.valid_mask, valid))
        self.assertFalse(result.losses.requires_grad)

    def test_padding_never_changes_valid_losses_or_aggregate(self) -> None:
        logits = torch.randn(2, 3, 5)
        targets = torch.tensor([[1, 2, -100], [3, 4, -100]])
        valid = torch.tensor([[True, True, False], [True, True, False]])
        first = cross_entropy_slot_loss(logits, targets, valid)

        changed = logits.clone()
        changed[:, 2] = torch.tensor([1000, -1000, 500, -500, 250.0])
        changed_targets = targets.clone()
        changed_targets[:, 2] = 999
        second = cross_entropy_slot_loss(changed, changed_targets, valid)

        torch.testing.assert_close(first.losses, second.losses)
        torch.testing.assert_close(first.aggregate, second.aggregate)

    def test_contract_rejects_invalid_shapes_and_dtypes(self) -> None:
        logits = torch.randn(2, 3, 4)
        targets = torch.zeros(2, 3, dtype=torch.long)
        cases = (
            ("logit rank", logits[:, 0], targets, None),
            ("target shape", logits, targets[:, :2], None),
            ("target dtype", logits, targets.float(), None),
            ("mask shape", logits, targets, torch.ones(2, 2, dtype=torch.bool)),
            ("mask dtype", logits, targets, torch.ones(2, 3)),
            ("empty mask", logits, targets, torch.zeros(2, 3, dtype=torch.bool)),
        )
        for label, candidate_logits, candidate_targets, candidate_mask in cases:
            with self.subTest(label=label), self.assertRaises(
                (TypeError, ValueError)
            ):
                cross_entropy_slot_loss(
                    candidate_logits,
                    candidate_targets,
                    candidate_mask,
                )

    def test_invalid_slot_losses_must_be_finite_exact_zeros(self) -> None:
        valid = torch.tensor([[True, False]])
        for label, invalid_loss in (
            ("nonzero", 1.0),
            ("nan", float("nan")),
            ("infinity", float("inf")),
        ):
            losses = torch.tensor([[0.5, invalid_loss]])
            result = SlotLossResult(losses, valid, losses[0, 0].clone())
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                "invalid slot losses must be finite exact zeros",
            ):
                result.validate()

        valid_result = SlotLossResult(
            torch.tensor([[0.5, 0.0]]),
            valid,
            torch.tensor(0.5),
        )
        self.assertIs(valid_result.validate(), valid_result)

    def test_float64_cross_entropy_retains_dtype(self) -> None:
        logits = torch.randn(2, 3, 5, dtype=torch.float64)
        targets = torch.tensor([[1, 2, 3], [4, 0, 1]])

        result = cross_entropy_slot_loss(logits, targets)

        self.assertEqual(result.losses.dtype, torch.float64)
        self.assertEqual(result.aggregate.dtype, torch.float64)
        expected = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        torch.testing.assert_close(result.losses, expected)
        torch.testing.assert_close(result.aggregate, expected.mean())

    def test_batch_centering_removes_constant_token_shortcut(self) -> None:
        targets = torch.arange(12).view(12, 1).repeat(1, 2)
        constant = torch.randn(1, 2, 12).expand(12, -1, -1).clone()

        result = batch_centered_cross_entropy_slot_loss(
            constant,
            targets,
        )

        expected = torch.full_like(result.losses, torch.log(torch.tensor(12.0)))
        torch.testing.assert_close(result.losses, expected)
        ordinary = F.cross_entropy(
            constant.reshape(-1, 12),
            targets.reshape(-1),
        )
        assert result.guard_aggregate is not None
        torch.testing.assert_close(result.guard_aggregate, ordinary)

    def test_batch_centering_rewards_target_correlated_variation(self) -> None:
        targets = torch.arange(12).view(12, 1).repeat(1, 2)
        logits = torch.zeros(12, 2, 12)
        logits.scatter_(
            2,
            targets.unsqueeze(-1),
            2.0,
        )

        result = batch_centered_cross_entropy_slot_loss(logits, targets)

        self.assertLess(float(result.aggregate), math.log(12.0))

    def test_batch_centering_rejects_single_valid_example(self) -> None:
        logits = torch.zeros(2, 2, 4)
        targets = torch.zeros(2, 2, dtype=torch.long)
        valid = torch.tensor(
            [
                [True, False],
                [False, False],
            ]
        )

        with self.assertRaisesRegex(ValueError, "at least two"):
            batch_centered_cross_entropy_slot_loss(
                logits,
                targets,
                valid,
            )

    def test_batch_mean_preserves_external_global_guard(self) -> None:
        source = SlotLossResult(
            losses=torch.ones(2, 1),
            valid_mask=torch.ones(2, 1, dtype=torch.bool),
            aggregate=torch.tensor(1.0),
            guard_aggregate=torch.tensor(3.0),
        ).validate()

        reduced = batch_mean_slot_loss(source)

        torch.testing.assert_close(reduced.guard_aggregate, torch.tensor(3.0))

    def test_batch_mean_preserves_physical_slots_and_padding(self) -> None:
        source = SlotLossResult(
            losses=torch.tensor(
                [
                    [1.0, 2.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [5.0, 4.0, 0.0],
                ]
            ),
            valid_mask=torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                    [True, True, False],
                ]
            ),
            aggregate=torch.tensor(3.0),
        ).validate()

        reduced = batch_mean_slot_loss(source)

        self.assertEqual(tuple(reduced.losses.shape), (1, 3))
        torch.testing.assert_close(
            reduced.losses,
            torch.tensor([[3.0, 3.0, 0.0]]),
        )
        self.assertTrue(
            torch.equal(
                reduced.valid_mask,
                torch.tensor([[True, True, False]]),
            )
        )
        torch.testing.assert_close(reduced.aggregate, torch.tensor(3.0))


if __name__ == "__main__":
    unittest.main()
