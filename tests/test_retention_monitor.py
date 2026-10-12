"""Tests for Procoim-S Phase 7: forgetting-surprise monitoring."""

from __future__ import annotations

import pytest

from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.retention_monitor import (
    ForgettingSeverity,
    RetentionMonitor,
    RetentionProbe,
    classify_forgetting,
    forgetting_surprise,
)


def _config(**overrides) -> ProcoimSConfig:
    defaults = dict(forgetting_surprise_warning=2.0, forgetting_surprise_repair=3.0)
    defaults.update(overrides)
    return ProcoimSConfig(**defaults)


class TestForgettingSurprise:
    def test_matches_manual_formula(self) -> None:
        value = forgetting_surprise(current_loss=5.0, expected_loss=2.0, scale=1.0)
        assert value == pytest.approx(3.0, abs=1e-6)

    def test_zero_when_current_matches_expected(self) -> None:
        assert forgetting_surprise(1.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_negative_when_current_is_better_than_expected(self) -> None:
        assert forgetting_surprise(0.5, 1.0, 1.0) < 0


class TestClassifyForgetting:
    def test_below_warning_is_none(self) -> None:
        assert classify_forgetting(1.0, _config()) is ForgettingSeverity.NONE

    def test_at_warning_threshold(self) -> None:
        assert classify_forgetting(2.0, _config()) is ForgettingSeverity.WARNING

    def test_at_repair_threshold(self) -> None:
        assert classify_forgetting(3.0, _config()) is ForgettingSeverity.REPAIR

    def test_at_critical_threshold(self) -> None:
        # repair=3.0, default critical_multiplier=1.5 -> critical at 4.5
        assert classify_forgetting(4.5, _config()) is ForgettingSeverity.CRITICAL

    def test_custom_critical_multiplier(self) -> None:
        assert classify_forgetting(6.0, _config(), critical_multiplier=2.0) is ForgettingSeverity.CRITICAL
        assert classify_forgetting(5.0, _config(), critical_multiplier=2.0) is ForgettingSeverity.REPAIR

    def test_rejects_multiplier_at_or_below_one(self) -> None:
        with pytest.raises(ValueError):
            classify_forgetting(3.0, _config(), critical_multiplier=1.0)


class TestRetentionMonitor:
    def test_evaluate_classifies_a_batch(self) -> None:
        # Values are chosen clearly inside each band, not on the exact
        # boundary -- forgetting_surprise's (scale + eps) denominator makes
        # boundary-exact values fall a hair short due to floating point,
        # which the direct classify_forgetting() boundary tests above
        # already cover without that division.
        monitor = RetentionMonitor(_config())
        probes = [
            RetentionProbe(item_id="a", current_loss=0.5, expected_loss=0.5, scale=1.0),
            RetentionProbe(item_id="b", current_loss=3.0, expected_loss=0.5, scale=1.0),
            RetentionProbe(item_id="c", current_loss=8.5, expected_loss=0.5, scale=1.0),
        ]
        events = monitor.evaluate(probes)
        severities = {event.item_id: event.severity for event in events}
        assert severities["a"] is ForgettingSeverity.NONE
        assert severities["b"] is ForgettingSeverity.WARNING
        assert severities["c"] is ForgettingSeverity.CRITICAL

    def test_events_at_or_above_filters_by_severity(self) -> None:
        monitor = RetentionMonitor(_config())
        probes = [
            RetentionProbe(item_id="a", current_loss=0.5, expected_loss=0.5, scale=1.0),
            RetentionProbe(item_id="b", current_loss=4.0, expected_loss=0.5, scale=1.0),
        ]
        events = monitor.evaluate(probes)
        repair_or_worse = monitor.events_at_or_above(events, ForgettingSeverity.REPAIR)
        assert [event.item_id for event in repair_or_worse] == ["b"]
