"""Tests for Procoim-S Phase 8: A-B-A continual-learning metrics arithmetic."""

from __future__ import annotations

import pytest

from gator.continual.metrics import compute_aba_metrics


class TestComputeABAMetrics:
    def test_matches_manual_arithmetic(self) -> None:
        metrics = compute_aba_metrics(
            accuracy_a_after_phase_a=0.9,
            accuracy_a_after_phase_b=0.4,
            accuracy_a_after_return=0.8,
            accuracy_b_before_phase_b=0.15,
            accuracy_b_after_phase_b=0.85,
            chance_accuracy=0.1,
        )
        assert metrics.a_acquisition == pytest.approx(0.9)
        assert metrics.b_acquisition == pytest.approx(0.85)
        assert metrics.a_forgetting == pytest.approx(0.5)  # 0.9 - 0.4
        assert metrics.a_recovery == pytest.approx(0.4)  # 0.8 - 0.4
        assert metrics.backward_transfer == pytest.approx(-0.1)  # 0.8 - 0.9
        assert metrics.forward_transfer == pytest.approx(0.05)  # 0.15 - 0.1

    def test_zero_forgetting_when_a_accuracy_unchanged(self) -> None:
        metrics = compute_aba_metrics(
            accuracy_a_after_phase_a=0.7,
            accuracy_a_after_phase_b=0.7,
            accuracy_a_after_return=0.7,
            accuracy_b_before_phase_b=0.1,
            accuracy_b_after_phase_b=0.9,
            chance_accuracy=0.1,
        )
        assert metrics.a_forgetting == pytest.approx(0.0)
        assert metrics.a_recovery == pytest.approx(0.0)
        assert metrics.backward_transfer == pytest.approx(0.0)

    def test_positive_backward_transfer_when_return_exceeds_original(self) -> None:
        metrics = compute_aba_metrics(
            accuracy_a_after_phase_a=0.6,
            accuracy_a_after_phase_b=0.3,
            accuracy_a_after_return=0.75,
            accuracy_b_before_phase_b=0.1,
            accuracy_b_after_phase_b=0.9,
            chance_accuracy=0.1,
        )
        assert metrics.backward_transfer > 0
