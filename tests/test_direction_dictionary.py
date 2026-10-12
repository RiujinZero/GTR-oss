"""Tests for Procoim-S Phase 6: the learned direction dictionary."""

from __future__ import annotations

import pytest

from gator.optimizers.direction_dictionary import (
    DirectionCategory,
    DirectionDictionary,
    DirectionRecord,
    RefreshSignals,
    should_refresh_basis,
)


class TestConstruction:
    def test_default_fractions_sum_to_one(self) -> None:
        DirectionDictionary()  # no raise

    def test_rejects_fractions_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError):
            DirectionDictionary(anchor_fraction=0.5, recent_fraction=0.5, exploration_fraction=0.5)

    def test_rejects_negative_fraction(self) -> None:
        with pytest.raises(ValueError):
            DirectionDictionary(anchor_fraction=-0.1, recent_fraction=0.6, exploration_fraction=0.5)

    def test_rejects_utility_ema_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            DirectionDictionary(utility_ema=1.5)


class TestRegisterAndGet:
    def test_register_creates_a_new_record(self) -> None:
        dictionary = DirectionDictionary()
        record = dictionary.register("routing", "dir-0", DirectionCategory.EXPLORATION)
        assert record.direction_id == "dir-0"
        assert record.category is DirectionCategory.EXPLORATION
        assert dictionary.get("routing", "dir-0") is record

    def test_reregister_updates_category_not_utility(self) -> None:
        dictionary = DirectionDictionary()
        dictionary.register("routing", "dir-0", DirectionCategory.EXPLORATION)
        record = dictionary.get("routing", "dir-0")
        record.utility = 0.5
        re_registered = dictionary.register("routing", "dir-0", DirectionCategory.ANCHOR)
        assert re_registered.category is DirectionCategory.ANCHOR
        assert re_registered.utility == pytest.approx(0.5)

    def test_get_missing_direction_returns_none(self) -> None:
        dictionary = DirectionDictionary()
        assert dictionary.get("routing", "missing") is None

    def test_groups_are_independent(self) -> None:
        dictionary = DirectionDictionary()
        dictionary.register("routing", "dir-0", DirectionCategory.ANCHOR)
        assert dictionary.get("readout", "dir-0") is None


class TestRecordTrial:
    def test_updates_ema_using_configured_decay(self) -> None:
        dictionary = DirectionDictionary(utility_ema=0.5)
        dictionary.register("routing", "dir-0", DirectionCategory.RECENT)
        # gain=1.0, forwards=1 -> per_forward_gain ~= 1.0; utility = 0.5*0 + 0.5*1.0 = 0.5
        record = dictionary.record_trial(
            "routing", "dir-0", validated_gain=1.0, forwards_used=1
        )
        assert record.utility == pytest.approx(0.5, abs=1e-3)

    def test_raises_for_unregistered_direction(self) -> None:
        dictionary = DirectionDictionary()
        with pytest.raises(KeyError):
            dictionary.record_trial("routing", "dir-0", validated_gain=1.0, forwards_used=1)

    def test_tracks_contexts_and_catastrophic_regressions(self) -> None:
        dictionary = DirectionDictionary()
        dictionary.register("routing", "dir-0", DirectionCategory.ANCHOR)
        dictionary.record_trial(
            "routing", "dir-0", validated_gain=1.0, forwards_used=1,
            context="ctx-a", catastrophic=True,
        )
        record = dictionary.get("routing", "dir-0")
        assert record.contexts == {"ctx-a"}
        assert record.catastrophic_regressions == 1
        assert record.trials == 1


class TestDirectionRecordValidation:
    def test_rejects_utility_ema_out_of_range(self) -> None:
        record = DirectionRecord(direction_id="d", category=DirectionCategory.ANCHOR)
        with pytest.raises(ValueError):
            record.record_trial(validated_gain=1.0, forwards_used=1, utility_ema=2.0)

    def test_rejects_negative_forwards_used(self) -> None:
        record = DirectionRecord(direction_id="d", category=DirectionCategory.ANCHOR)
        with pytest.raises(ValueError):
            record.record_trial(validated_gain=1.0, forwards_used=-1, utility_ema=0.5)


class TestTickAge:
    def test_increments_only_the_target_group(self) -> None:
        dictionary = DirectionDictionary()
        dictionary.register("routing", "dir-0", DirectionCategory.ANCHOR)
        dictionary.register("readout", "dir-1", DirectionCategory.ANCHOR)
        dictionary.tick_age("routing")
        assert dictionary.get("routing", "dir-0").age == 1
        assert dictionary.get("readout", "dir-1").age == 0


class TestComposition:
    def test_groups_records_by_category(self) -> None:
        dictionary = DirectionDictionary()
        dictionary.register("routing", "a", DirectionCategory.ANCHOR)
        dictionary.register("routing", "b", DirectionCategory.ANCHOR)
        dictionary.register("routing", "c", DirectionCategory.EXPLORATION)
        composition = dictionary.composition("routing")
        assert set(composition[DirectionCategory.ANCHOR]) == {"a", "b"}
        assert composition[DirectionCategory.EXPLORATION] == ("c",)
        assert composition[DirectionCategory.RECENT] == ()

    def test_empty_group_returns_all_empty_categories(self) -> None:
        dictionary = DirectionDictionary()
        composition = dictionary.composition("nonexistent")
        assert all(ids == () for ids in composition.values())


class TestRetireLowestUtility:
    def test_keeps_highest_utility_directions(self) -> None:
        dictionary = DirectionDictionary(utility_ema=0.0)
        for name, gain in [("low", 0.1), ("mid", 0.5), ("high", 1.0)]:
            dictionary.register("routing", name, DirectionCategory.EXPLORATION)
            dictionary.record_trial("routing", name, validated_gain=gain, forwards_used=1)

        retired = dictionary.retire_lowest_utility("routing", keep=1)

        assert set(retired) == {"low", "mid"}
        assert dictionary.get("routing", "high") is not None
        assert dictionary.get("routing", "low") is None

    def test_keep_zero_retires_everything(self) -> None:
        dictionary = DirectionDictionary()
        dictionary.register("routing", "a", DirectionCategory.ANCHOR)
        retired = dictionary.retire_lowest_utility("routing", keep=0)
        assert retired == ("a",)

    def test_rejects_negative_keep(self) -> None:
        dictionary = DirectionDictionary()
        with pytest.raises(ValueError):
            dictionary.retire_lowest_utility("routing", keep=-1)


class TestShouldRefreshBasis:
    def test_no_signals_never_refreshes(self) -> None:
        assert not should_refresh_basis(RefreshSignals())

    def test_stall_threshold_triggers(self) -> None:
        signals = RefreshSignals(consecutive_stalls=5, stall_threshold=5)
        assert should_refresh_basis(signals)

    def test_stall_below_threshold_does_not_trigger(self) -> None:
        signals = RefreshSignals(consecutive_stalls=4, stall_threshold=5)
        assert not should_refresh_basis(signals)

    def test_low_gain_per_forward_triggers(self) -> None:
        signals = RefreshSignals(gain_per_forward=0.001, gain_per_forward_floor=0.01)
        assert should_refresh_basis(signals)

    def test_sustained_surprise_triggers(self) -> None:
        signals = RefreshSignals(sustained_surprise=3.0, surprise_threshold=2.0)
        assert should_refresh_basis(signals)

    def test_protected_debt_triggers(self) -> None:
        signals = RefreshSignals(protected_debt=5.0, protected_debt_threshold=4.0)
        assert should_refresh_basis(signals)

    def test_low_span_novelty_triggers(self) -> None:
        signals = RefreshSignals(span_novelty=0.01, span_novelty_floor=0.05)
        assert should_refresh_basis(signals)

    def test_none_thresholds_are_ignored(self) -> None:
        signals = RefreshSignals(consecutive_stalls=1000, stall_threshold=None)
        assert not should_refresh_basis(signals)
