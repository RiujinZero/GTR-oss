"""Tests for the Procoim-S stratified replay reservoir (Phase 2)."""

from __future__ import annotations

import random

import pytest
import torch

from gator.continual.replay_reservoir import (
    DEFAULT_CATEGORY_FRACTIONS,
    ReplayCategory,
    ReplayItem,
    ReplayReservoir,
)


def _item(importance: float = 0.0, task_id: str = "t") -> ReplayItem:
    return ReplayItem(
        input_ids=torch.zeros(1),
        target_ids=torch.zeros(1),
        valid_mask=torch.ones(1, dtype=torch.bool),
        reference_loss=1.0,
        importance=importance,
        task_id=task_id,
    )


class TestConstruction:
    def test_default_fractions_sum_to_one(self) -> None:
        assert sum(DEFAULT_CATEGORY_FRACTIONS.values()) == pytest.approx(1.0)

    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError):
            ReplayReservoir(capacity=0)

    def test_rejects_fractions_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError):
            ReplayReservoir(
                capacity=10,
                category_fractions={ReplayCategory.CONFIRMED: 0.5},
            )


class TestImportanceEviction:
    def test_low_importance_item_evicted_by_higher_importance_arrival(self) -> None:
        reservoir = ReplayReservoir(
            capacity=10,
            category_fractions={ReplayCategory.FOUNDATION: 1.0},
            rng=random.Random(0),
        )
        # category capacity = round(10 * 1.0) = 10, force a tiny slice instead
        reservoir._category_capacity[ReplayCategory.FOUNDATION] = 2
        low = _item(importance=0.1)
        mid = _item(importance=0.5)
        high = _item(importance=0.9)
        reservoir.add(low, ReplayCategory.FOUNDATION)
        reservoir.add(mid, ReplayCategory.FOUNDATION)
        reservoir.add(high, ReplayCategory.FOUNDATION)

        remaining_ids = {item.item_id for item in reservoir.items_in(ReplayCategory.FOUNDATION)}
        assert low.item_id not in remaining_ids
        assert mid.item_id in remaining_ids
        assert high.item_id in remaining_ids

    def test_lower_importance_arrival_does_not_evict_when_full(self) -> None:
        reservoir = ReplayReservoir(
            capacity=10,
            category_fractions={ReplayCategory.FOUNDATION: 1.0},
        )
        reservoir._category_capacity[ReplayCategory.FOUNDATION] = 1
        resident = _item(importance=0.9)
        challenger = _item(importance=0.1)
        reservoir.add(resident, ReplayCategory.FOUNDATION)
        reservoir.add(challenger, ReplayCategory.FOUNDATION)

        remaining = reservoir.items_in(ReplayCategory.FOUNDATION)
        assert len(remaining) == 1
        assert remaining[0].item_id == resident.item_id


class TestRandomCoverage:
    def test_random_category_stays_within_capacity(self) -> None:
        reservoir = ReplayReservoir(
            capacity=10,
            category_fractions={ReplayCategory.RANDOM: 1.0},
            rng=random.Random(1),
        )
        reservoir._category_capacity[ReplayCategory.RANDOM] = 3
        for _ in range(50):
            reservoir.add(_item(), ReplayCategory.RANDOM)
        assert len(reservoir.items_in(ReplayCategory.RANDOM)) == 3


class TestSampling:
    def test_sample_stratified_respects_requested_counts(self) -> None:
        reservoir = ReplayReservoir(capacity=100, rng=random.Random(2))
        for _ in range(20):
            reservoir.add(_item(), ReplayCategory.CONFIRMED)
        for _ in range(20):
            reservoir.add(_item(), ReplayCategory.FOUNDATION)

        sample = reservoir.sample_stratified(confirmed=2, foundation=1)
        assert len(sample) == 3

    def test_sample_from_empty_category_returns_nothing_for_that_slice(self) -> None:
        reservoir = ReplayReservoir(capacity=10)
        sample = reservoir.sample_stratified(confirmed=5)
        assert sample == []

    def test_sample_with_replacement_when_request_exceeds_population(self) -> None:
        reservoir = ReplayReservoir(capacity=100, rng=random.Random(3))
        reservoir.add(_item(), ReplayCategory.RECENT)
        sample = reservoir.sample_stratified(recent=5)
        assert len(sample) == 5

    def test_sampling_increments_uses(self) -> None:
        reservoir = ReplayReservoir(capacity=10)
        item = _item()
        reservoir.add(item, ReplayCategory.CONFIRMED)
        reservoir.sample_stratified(confirmed=1)
        assert item.uses == 1

    def test_repeated_calls_rotate_rather_than_returning_a_fixed_batch(self) -> None:
        reservoir = ReplayReservoir(capacity=100, rng=random.Random(4))
        for _ in range(30):
            reservoir.add(_item(), ReplayCategory.CONFIRMED)

        first = {item.item_id for item in reservoir.sample_stratified(confirmed=5)}
        second = {item.item_id for item in reservoir.sample_stratified(confirmed=5)}
        # With 30 candidates and draws of 5, an identical draw twice in a row
        # is astronomically unlikely — this is a rotation smoke test, not a
        # strict guarantee.
        assert first != second


class TestBookkeeping:
    def test_mark_regression_increments_counter(self) -> None:
        reservoir = ReplayReservoir(capacity=10)
        item = _item()
        reservoir.add(item, ReplayCategory.FRAGILE)
        reservoir.mark_regression(item.item_id)
        assert item.regressions == 1

    def test_update_reference_changes_loss(self) -> None:
        reservoir = ReplayReservoir(capacity=10)
        item = _item()
        reservoir.add(item, ReplayCategory.FOUNDATION)
        reservoir.update_reference(item.item_id, reference_loss=0.42)
        assert item.reference_loss == pytest.approx(0.42)

    def test_category_of_finds_the_right_bucket(self) -> None:
        reservoir = ReplayReservoir(capacity=10)
        item = _item()
        reservoir.add(item, ReplayCategory.RECENT)
        assert reservoir.category_of(item.item_id) is ReplayCategory.RECENT
        assert reservoir.category_of("does-not-exist") is None

    def test_tick_ages_advances_every_resident(self) -> None:
        reservoir = ReplayReservoir(capacity=10)
        a = _item()
        b = _item()
        reservoir.add(a, ReplayCategory.CONFIRMED)
        reservoir.add(b, ReplayCategory.RECENT)
        reservoir.tick_ages()
        reservoir.tick_ages()
        assert a.age == 2
        assert b.age == 2

    def test_len_counts_across_all_categories(self) -> None:
        reservoir = ReplayReservoir(capacity=100)
        reservoir.add(_item(), ReplayCategory.CONFIRMED)
        reservoir.add(_item(), ReplayCategory.FRAGILE)
        assert len(reservoir) == 2
