"""Tests for Procoim-S aggregate candidate utility (Phase 2).

Covers Procoim-S design spec unit tests #8-9 (ordinary replay regression
does not individually veto a candidate; catastrophic canaries still veto)
plus a direct reproduction of the old hard-AND-gate collapse this mechanism
is meant to fix (Papers/GTRL_PROCOIM_TRACE.md section 5.2).
"""

from __future__ import annotations

import math

import pytest

from gator.optimizers.candidate_utility import (
    CandidateUtilityResult,
    ReplayEvaluation,
    catastrophic_violation,
    evaluate_candidate_utility,
    soft_replay_regression_cost,
)
from gator.optimizers.procoim_s import ProcoimSConfig


def _old_hard_and_gate(current_gain: float, evaluations: list[ReplayEvaluation]) -> bool:
    """A stand-in for the old anchor/consolidation AND-gate: every retained
    example must individually not regress beyond its tolerance."""

    if current_gain <= 0:
        return False
    return all(evaluation.regression <= evaluation.tolerance for evaluation in evaluations)


class TestSoftReplayRegressionCost:
    def test_zero_cost_when_all_within_tolerance(self) -> None:
        evaluations = [
            ReplayEvaluation(item_id="a", loss_after=1.0, reference_loss=1.0, tolerance=0.05),
            ReplayEvaluation(item_id="b", loss_after=0.98, reference_loss=1.0, tolerance=0.05),
        ]
        assert soft_replay_regression_cost(evaluations) == 0.0

    def test_quadratic_cost_above_tolerance(self) -> None:
        evaluations = [
            ReplayEvaluation(item_id="a", loss_after=1.2, reference_loss=1.0, tolerance=0.05)
        ]
        # overage = 0.2 - 0.05 = 0.15; cost = weight * overage^2
        assert soft_replay_regression_cost(evaluations) == pytest.approx(0.15**2)

    def test_canaries_excluded_from_soft_cost(self) -> None:
        evaluations = [
            ReplayEvaluation(
                item_id="canary", loss_after=5.0, reference_loss=1.0, tolerance=0.0, is_canary=True
            )
        ]
        assert soft_replay_regression_cost(evaluations) == 0.0


class TestCatastrophicViolation:
    def test_ordinary_item_past_catastrophic_ceiling_vetoes(self) -> None:
        evaluations = [
            ReplayEvaluation(
                item_id="a",
                loss_after=10.0,
                reference_loss=1.0,
                tolerance=0.05,
                catastrophic_ceiling=1.0,
            )
        ]
        assert catastrophic_violation(evaluations) is True

    def test_ordinary_item_within_ceiling_does_not_veto(self) -> None:
        evaluations = [
            ReplayEvaluation(
                item_id="a",
                loss_after=1.5,
                reference_loss=1.0,
                tolerance=0.05,
                catastrophic_ceiling=1.0,
            )
        ]
        assert catastrophic_violation(evaluations) is False


class TestEvaluateCandidateUtility:
    def test_ordinary_replay_regression_does_not_individually_veto(self) -> None:
        """Spec unit test #8."""

        evaluations = [
            ReplayEvaluation(
                item_id=f"item{i}",
                loss_after=1.2,
                reference_loss=1.0,
                tolerance=0.05,
                catastrophic_ceiling=5.0,
                weight=0.01,
            )
            for i in range(3)
        ]
        result = evaluate_candidate_utility(
            current_gain=1.0,
            replay_evaluations=evaluations,
            config=ProcoimSConfig(replay_weight=1.0, minimum_utility=0.0),
        )
        assert result.catastrophic is False
        assert result.accepted is True

    def test_catastrophic_canary_still_vetoes(self) -> None:
        """Spec unit test #9."""

        evaluations = [
            ReplayEvaluation(
                item_id="canary",
                loss_after=10.0,
                reference_loss=1.0,
                tolerance=0.0,
                catastrophic_ceiling=0.5,
                is_canary=True,
            )
        ]
        result = evaluate_candidate_utility(
            current_gain=100.0,  # even a huge current gain must not save it
            replay_evaluations=evaluations,
        )
        assert result.catastrophic is True
        assert result.accepted is False
        assert result.rejection_reason == "catastrophic_canary_violation"

    def test_non_positive_current_gain_rejected(self) -> None:
        result = evaluate_candidate_utility(current_gain=0.0)
        assert result.accepted is False
        assert result.rejection_reason == "non_positive_current_gain"

    def test_aggregate_below_minimum_utility_rejected(self) -> None:
        config = ProcoimSConfig(minimum_utility=10.0)
        result = evaluate_candidate_utility(current_gain=1.0, config=config)
        assert result.accepted is False
        assert result.rejection_reason == "aggregate_utility_below_minimum"

    def test_future_utility_and_norm_penalty_shift_the_aggregate(self) -> None:
        config = ProcoimSConfig(
            future_utility_weight=1.0, delta_norm_weight=1.0, minimum_utility=0.0
        )
        boosted = evaluate_candidate_utility(current_gain=0.1, future_utility=1.0, config=config)
        penalized = evaluate_candidate_utility(current_gain=0.1, delta_norm=1.0, config=config)
        assert boosted.aggregate_utility > penalized.aggregate_utility

    def test_non_finite_gain_is_rejected_not_crashed(self) -> None:
        result = evaluate_candidate_utility(current_gain=math.inf)
        assert result.accepted is False
        assert result.rejection_reason == "non_finite_utility"

    def test_aggregate_utility_reproduces_hard_and_gate_collapse_scenario(self) -> None:
        """Direct reproduction of the anchor/consolidation stall pattern
        (Papers/GTRL_PROCOIM_TRACE.md section 5.2): many small, individually
        tolerable regressions across retained examples collectively vetoed
        every candidate under the old AND-gate. The aggregate scorer should
        accept the same candidate the AND-gate rejects, as long as the total
        soft cost stays below the current gain."""

        evaluations = [
            ReplayEvaluation(
                item_id=f"item{i}",
                loss_after=1.0 + 0.01 * (i + 1),  # small regressions, just over tolerance
                reference_loss=1.0,
                tolerance=0.005,
                catastrophic_ceiling=5.0,
                weight=0.05,
            )
            for i in range(8)
        ]
        assert _old_hard_and_gate(current_gain=0.5, evaluations=evaluations) is False

        result = evaluate_candidate_utility(
            current_gain=0.5,
            replay_evaluations=evaluations,
            config=ProcoimSConfig(replay_weight=1.0, minimum_utility=0.0),
        )
        assert result.accepted is True
