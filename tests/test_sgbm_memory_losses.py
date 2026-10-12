"""Tests for SGBM Phase G: training losses."""

from __future__ import annotations

import pytest
import torch

from gator.continual.memory_losses import (
    contrastive_key_loss,
    predictor_loss,
    retrieval_utility,
    write_calibration_loss,
)


class TestPredictorLoss:
    def test_identical_summaries_give_near_zero_loss(self) -> None:
        summary = torch.randn(3, 6)
        loss = predictor_loss(summary, summary.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_dissimilar_summaries_give_higher_loss(self) -> None:
        predicted = torch.zeros(2, 6)
        close = torch.full((2, 6), 0.01)
        far = torch.full((2, 6), 5.0)
        assert predictor_loss(predicted, far) > predictor_loss(predicted, close)

    def test_rejects_non_detached_actual_summary(self) -> None:
        predicted = torch.randn(2, 4)
        actual = torch.randn(2, 4, requires_grad=True)
        with pytest.raises(ValueError):
            predictor_loss(predicted, actual)


class TestRetrievalUtility:
    def test_is_plain_subtraction(self) -> None:
        utility = retrieval_utility(torch.tensor([2.0]), torch.tensor([0.5]))
        assert utility.item() == pytest.approx(1.5)

    def test_negative_when_memory_hurts(self) -> None:
        utility = retrieval_utility(torch.tensor([0.5]), torch.tensor([2.0]))
        assert utility.item() < 0


class TestWriteCalibrationLoss:
    def test_matching_the_target_gate_minimizes_the_loss(self) -> None:
        # Binary cross-entropy is minimized exactly at write_gate == target,
        # not at zero (its minimum equals the target's own entropy) -- check
        # against nearby mismatched gates instead of expecting near-zero.
        utility = torch.tensor([2.0])
        target_gate = torch.sigmoid(utility)
        matched_loss = write_calibration_loss(target_gate, utility)
        for offset in (0.05, -0.05, 0.2, -0.2):
            mismatched_gate = (target_gate + offset).clamp(1e-4, 1 - 1e-4)
            assert matched_loss <= write_calibration_loss(mismatched_gate, utility)

    def test_miscalibration_gives_higher_loss(self) -> None:
        utility = torch.tensor([5.0])  # target gate close to 1
        wrong_gate = torch.tensor([0.01])
        right_gate = torch.sigmoid(utility)
        assert write_calibration_loss(wrong_gate, utility) > write_calibration_loss(
            right_gate, utility
        )


class TestContrastiveKeyLoss:
    def test_strong_positive_match_gives_low_loss(self) -> None:
        query = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        positive = query.clone()
        negatives = torch.nn.functional.normalize(torch.randn(4, 5, 8), dim=-1)
        loss = contrastive_key_loss(query, positive, negatives, temperature=0.1)
        assert loss.item() < 0.5

    def test_rejects_non_positive_temperature(self) -> None:
        query = torch.randn(2, 4)
        with pytest.raises(ValueError):
            contrastive_key_loss(query, query, torch.randn(2, 3, 4), temperature=0.0)
