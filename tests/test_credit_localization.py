"""Tests for Procoim-S Phase 5: credit localization."""

from __future__ import annotations

import pytest

from gator.optimizers.credit_localization import (
    GroupEligibilityTrace,
    SurpriseFeatures,
    localize_groups,
)


def _features(**overrides) -> SurpriseFeatures:
    defaults = dict(
        route_surprise=0.0, stream_disagreement=0.0, predictive_surprise=0.0, semantic_novelty=0.0
    )
    defaults.update(overrides)
    return SurpriseFeatures(**defaults)


class TestLocalizeGroups:
    def test_dominant_route_surprise_ranks_routing_first(self) -> None:
        groups = localize_groups(_features(route_surprise=5.0), top_k=1)
        assert groups == ("routing",)

    def test_dominant_semantic_novelty_ranks_embedding_first(self) -> None:
        groups = localize_groups(_features(semantic_novelty=5.0), top_k=1)
        assert groups == ("embedding",)

    def test_dominant_stream_disagreement_ranks_receiver_memory_first(self) -> None:
        groups = localize_groups(_features(stream_disagreement=5.0), top_k=1)
        assert groups == ("receiver_memory",)

    def test_dominant_predictive_surprise_ranks_node_compute_first(self) -> None:
        groups = localize_groups(_features(predictive_surprise=5.0), top_k=1)
        assert groups == ("node_compute",)

    def test_all_zero_features_still_returns_one_group(self) -> None:
        groups = localize_groups(_features())
        assert len(groups) == 1

    def test_top_k_controls_returned_count(self) -> None:
        groups = localize_groups(_features(route_surprise=1.0, semantic_novelty=1.0), top_k=3)
        assert len(groups) == 3

    def test_threshold_filters_but_keeps_at_least_one(self) -> None:
        groups = localize_groups(_features(route_surprise=0.01), top_k=2, threshold=10.0)
        assert len(groups) == 1

    def test_rejects_non_positive_top_k(self) -> None:
        with pytest.raises(ValueError):
            localize_groups(_features(), top_k=0)


class TestGroupEligibilityTrace:
    def test_unseen_group_has_zero_eligibility(self) -> None:
        trace = GroupEligibilityTrace()
        assert trace.eligibility("routing") == 0.0

    def test_update_moves_eligibility_toward_new_activity(self) -> None:
        trace = GroupEligibilityTrace(decay=0.5)
        trace.update("routing", 1.0)
        first = trace.eligibility("routing")
        trace.update("routing", 1.0)
        second = trace.eligibility("routing")
        assert second >= first
        assert second <= 1.0

    def test_priority_is_product_of_inputs(self) -> None:
        trace = GroupEligibilityTrace()
        trace.update("routing", 1.0)
        # first update seeds eligibility at exactly the given activity
        assert trace.priority("routing", 2.0, plasticity=0.5) == pytest.approx(1.0)

    def test_rank_groups_orders_by_priority_descending(self) -> None:
        trace = GroupEligibilityTrace()
        trace.update("a", 0.1)
        trace.update("b", 0.9)
        trace.update("c", 0.5)
        ranked = trace.rank_groups(("a", "b", "c"), weight_eligibility=1.0)
        assert ranked == ("b", "c", "a")

    def test_rejects_bad_decay(self) -> None:
        with pytest.raises(ValueError):
            GroupEligibilityTrace(decay=1.0)
