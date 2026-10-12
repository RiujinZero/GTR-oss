"""Tests for SGBM Phase C: reliability estimation."""

from __future__ import annotations

import math

import pytest
import torch

from gator.memory.reliability import (
    combined_reliability,
    consensus_reliability,
    entropy_reliability,
    quality_reliability,
)


class TestConsensusReliability:
    def test_zero_disagreement_gives_reliability_one(self) -> None:
        reliability = consensus_reliability(torch.zeros(3))
        assert torch.allclose(reliability, torch.ones(3))

    def test_high_disagreement_gives_reliability_near_zero(self) -> None:
        reliability = consensus_reliability(torch.tensor([20.0]))
        assert reliability.item() < 1e-6


class TestEntropyReliability:
    def test_one_hot_distribution_has_reliability_near_one(self) -> None:
        probs = torch.zeros(2, 5)
        probs[:, 0] = 1.0
        reliability = entropy_reliability(probs, vocab_size=5)
        assert bool((reliability > 0.99).all())

    def test_uniform_distribution_has_reliability_near_zero(self) -> None:
        probs = torch.full((2, 5), 0.2)
        reliability = entropy_reliability(probs, vocab_size=5)
        assert bool((reliability.abs() < 1e-4).all())

    def test_averages_over_middle_dimensions(self) -> None:
        confident = torch.zeros(1, 3, 5)
        confident[:, :, 0] = 1.0
        reliability = entropy_reliability(confident, vocab_size=5)
        assert reliability.shape == (1,)
        assert reliability.item() > 0.99

    def test_rejects_vocab_size_of_one(self) -> None:
        with pytest.raises(ValueError):
            entropy_reliability(torch.ones(1, 1), vocab_size=1)


class TestQualityReliability:
    def test_all_zero_rates_give_full_quality(self) -> None:
        quality = quality_reliability(torch.zeros(2), torch.zeros(2), torch.zeros(2))
        assert torch.allclose(quality, torch.ones(2))

    def test_any_rate_at_one_zeroes_quality(self) -> None:
        quality = quality_reliability(torch.ones(1), torch.zeros(1), torch.zeros(1))
        assert quality.item() == pytest.approx(0.0)

    def test_partial_rates_multiply(self) -> None:
        quality = quality_reliability(
            torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.0])
        )
        assert quality.item() == pytest.approx(0.25)


class TestCombinedReliability:
    def test_product_of_high_components_is_clipped_to_one(self) -> None:
        reliability = combined_reliability(
            torch.tensor([1.5]), torch.tensor([1.5]), torch.tensor([1.5])
        )
        assert reliability.item() == pytest.approx(1.0)

    def test_product_of_components_below_bounds(self) -> None:
        reliability = combined_reliability(
            torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5])
        )
        assert reliability.item() == pytest.approx(0.125)

    def test_never_negative(self) -> None:
        reliability = combined_reliability(
            torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([1.0])
        )
        assert reliability.item() >= 0.0
