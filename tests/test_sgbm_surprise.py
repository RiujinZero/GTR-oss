"""Tests for SGBM Phase C: surprise feature computation."""

from __future__ import annotations

import math

import pytest
import torch

from gator.memory.surprise import (
    SurpriseCombinationWeights,
    SurpriseNormalizer,
    combine_surprise,
    context_continuity,
    predictive_surprise,
    route_surprise,
    semantic_novelty,
    stream_disagreement,
    surprise_momentum_update,
    update_route_baseline,
)


class TestPredictiveSurprise:
    def test_identical_summaries_have_near_zero_surprise(self) -> None:
        summary = torch.randn(3, 8)
        surprise = predictive_surprise(summary, summary)
        assert torch.allclose(surprise, torch.zeros(3), atol=1e-5)

    def test_dissimilar_summaries_score_higher_than_identical(self) -> None:
        predicted = torch.zeros(2, 8)
        actual_same = torch.zeros(2, 8)
        actual_different = torch.ones(2, 8) * 5.0
        low = predictive_surprise(predicted, actual_same)
        high = predictive_surprise(predicted, actual_different)
        assert bool((high > low).all())


class TestStreamDisagreement:
    def test_identical_streams_have_zero_disagreement(self) -> None:
        states = torch.ones(2, 4, 6)
        assert torch.allclose(stream_disagreement(states), torch.zeros(2))

    def test_divergent_streams_have_positive_disagreement(self) -> None:
        states = torch.randn(2, 4, 6) * 10
        assert bool((stream_disagreement(states) > 0).all())


class TestRouteSurprise:
    def test_identical_to_baseline_has_zero_kl(self) -> None:
        distribution = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        surprise = route_surprise(distribution, distribution)
        assert torch.allclose(surprise, torch.zeros(2), atol=1e-5)

    def test_divergent_distribution_has_positive_kl(self) -> None:
        baseline = torch.full((1, 1, 4), 0.25)
        distribution = torch.tensor([[[0.97, 0.01, 0.01, 0.01]]])
        surprise = route_surprise(distribution, baseline)
        assert bool((surprise > 0).all())

    def test_update_route_baseline_moves_toward_new_distribution(self) -> None:
        baseline = torch.tensor([[[1.0, 0.0]]])
        observed = torch.tensor([[[0.0, 1.0]]])
        updated = update_route_baseline(baseline, observed, decay=0.5)
        assert torch.allclose(updated, torch.tensor([[[0.5, 0.5]]]))

    def test_update_route_baseline_rejects_bad_decay(self) -> None:
        baseline = torch.zeros(1, 1, 2)
        with pytest.raises(ValueError):
            update_route_baseline(baseline, baseline, decay=1.5)


class TestSemanticNovelty:
    def test_no_valid_memory_is_maximally_novel(self) -> None:
        candidate = torch.randn(2, 4)
        keys = torch.zeros(2, 3, 4)
        valid = torch.zeros(2, 3, dtype=torch.bool)
        novelty = semantic_novelty(candidate, keys, valid)
        assert torch.allclose(novelty, torch.ones(2))

    def test_identical_valid_key_has_near_zero_novelty(self) -> None:
        candidate = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        keys = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        novelty = semantic_novelty(candidate, keys, valid)
        assert novelty[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_valid_key_has_novelty_near_one(self) -> None:
        candidate = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        keys = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        novelty = semantic_novelty(candidate, keys, valid)
        assert novelty[0].item() == pytest.approx(1.0, abs=1e-5)


class TestContextContinuity:
    def test_identical_summaries_have_continuity_one(self) -> None:
        summary = torch.randn(2, 6)
        continuity = context_continuity(summary, summary)
        assert torch.allclose(continuity, torch.ones(2), atol=1e-5)

    def test_orthogonal_summaries_have_zero_continuity(self) -> None:
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        assert context_continuity(a, b)[0].item() == pytest.approx(0.0, abs=1e-5)


class TestSurpriseNormalizer:
    def test_converges_to_near_zero_mean_on_constant_input(self) -> None:
        normalizer = SurpriseNormalizer(decay=0.9)
        x = torch.full((4,), 5.0)
        for _ in range(200):
            z = normalizer.normalize(x)
        assert z.abs().max().item() < 0.5

    def test_extreme_outlier_is_clamped(self) -> None:
        normalizer = SurpriseNormalizer(decay=0.99, clamp=5.0)
        normalizer.normalize(torch.zeros(4))
        z = normalizer.normalize(torch.full((4,), 1e6))
        assert bool((z <= 5.0).all())
        assert bool((z >= -5.0).all())

    def test_rejects_bad_decay(self) -> None:
        with pytest.raises(ValueError):
            SurpriseNormalizer(decay=1.0)

    def test_state_dict_round_trip(self) -> None:
        normalizer = SurpriseNormalizer()
        normalizer.normalize(torch.randn(4))
        restored = SurpriseNormalizer()
        restored.load_state_dict(normalizer.state_dict())
        assert torch.equal(restored._mean, normalizer._mean)

    def test_update_false_does_not_move_statistics(self) -> None:
        normalizer = SurpriseNormalizer(decay=0.5)
        normalizer.normalize(torch.zeros(4))
        mean_before = normalizer._mean.clone()
        normalizer.normalize(torch.full((4,), 100.0), update=False)
        assert torch.equal(normalizer._mean, mean_before)


class TestSurpriseMomentum:
    def test_high_continuity_and_beta_mostly_preserves_old_momentum(self) -> None:
        updated = surprise_momentum_update(
            torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]), beta=0.9
        )
        assert updated.item() == pytest.approx(0.9)

    def test_zero_continuity_drops_momentum_regardless_of_beta(self) -> None:
        updated = surprise_momentum_update(
            torch.tensor([1.0]), torch.tensor([0.7]), torch.tensor([0.0]), beta=0.9
        )
        assert updated.item() == pytest.approx(0.1 * 0.7)

    def test_rejects_beta_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            surprise_momentum_update(torch.zeros(1), torch.zeros(1), torch.zeros(1), beta=1.5)


class TestCombineSurprise:
    def test_matches_manual_weighted_sum(self) -> None:
        weights = SurpriseCombinationWeights(
            prediction=0.4, streams=0.1, route=0.2, novelty=0.3, momentum=1.0
        )
        raw, blended = combine_surprise(
            prediction=torch.tensor([1.0]),
            streams=torch.tensor([1.0]),
            route=torch.tensor([1.0]),
            novelty=torch.tensor([1.0]),
            momentum=torch.tensor([0.0]),
            weights=weights,
        )
        assert raw.item() == pytest.approx(1.0)
        assert blended.item() == pytest.approx(1.0)  # momentum weight 1.0 -> ignores momentum term

    def test_momentum_weight_zero_uses_only_momentum(self) -> None:
        weights = SurpriseCombinationWeights(momentum=0.0)
        raw, blended = combine_surprise(
            prediction=torch.tensor([1.0]),
            streams=torch.tensor([1.0]),
            route=torch.tensor([1.0]),
            novelty=torch.tensor([1.0]),
            momentum=torch.tensor([0.42]),
            weights=weights,
        )
        assert blended.item() == pytest.approx(0.42)
