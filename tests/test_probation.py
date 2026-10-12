"""Tests for Procoim-S Phase 3: probation and delayed utility tracking."""

from __future__ import annotations

import pytest
import torch

from gator.optimizers.fast_weights import FastWeightDelta, FastWeightState
from gator.optimizers.probation import ProbationOutcome, ProbationTracker
from gator.optimizers.procoim_s import ProcoimSConfig


def _tracker(**overrides) -> ProbationTracker:
    defaults = dict(
        minimum_future_trials=2,
        minimum_contexts_helped=2,
        provisional_expiry_examples=10,
        future_utility_probe_rate=0.1,
        fast_weight_decay=0.5,
    )
    defaults.update(overrides)
    return ProbationTracker(ProcoimSConfig(**defaults))


def _delta() -> FastWeightDelta:
    return FastWeightDelta(group="routing", values={"w": torch.tensor([1.0, 2.0])})


class TestAdd:
    def test_sets_expiry_from_opened_step_and_config(self) -> None:
        tracker = _tracker(provisional_expiry_examples=5)
        update = tracker.add(
            group="routing",
            delta=_delta(),
            current_gain=0.1,
            replay_cost=0.0,
            surprise=0.5,
            reliability=0.5,
            opened_step=100,
        )
        assert update.expires_step == 105
        assert update.update_id in tracker.updates

    def test_ids_increment_across_multiple_adds(self) -> None:
        tracker = _tracker()
        first = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        second = tracker.add(
            group="b", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        assert second.update_id == first.update_id + 1


class TestRecordFutureUtility:
    def test_positive_utility_increments_helped(self) -> None:
        tracker = _tracker()
        update = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(1.0, decay=0.5)
        assert update.contexts_helped == 1
        assert update.contexts_hurt == 0
        assert update.future_trials == 1

    def test_negative_utility_increments_hurt(self) -> None:
        update = _tracker().add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(-1.0, decay=0.5)
        assert update.contexts_hurt == 1
        assert update.contexts_helped == 0

    def test_ema_blends_across_samples(self) -> None:
        # EMA starts at 0.0: after one sample of 1.0 with decay=0.5 it's
        # 0.5*0 + 0.5*1.0 = 0.5; after a second sample it's
        # 0.5*0.5 + 0.5*1.0 = 0.75 -- converging toward 1.0, not jumping
        # straight there.
        update = _tracker().add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(1.0, decay=0.5)
        assert update.future_utility_ema == pytest.approx(0.5)
        update.record_future_utility(1.0, decay=0.5)
        assert update.future_utility_ema == pytest.approx(0.75)

    def test_rejects_bad_decay(self) -> None:
        update = _tracker().add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        with pytest.raises(ValueError):
            update.record_future_utility(1.0, decay=1.5)


class TestIsExpired:
    def test_boundary_is_inclusive(self) -> None:
        update = _tracker(provisional_expiry_examples=10).add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        assert not update.is_expired(9)
        assert update.is_expired(10)


class TestDecide:
    def test_pending_before_minimum_trials(self) -> None:
        tracker = _tracker(minimum_future_trials=3)
        update = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(1.0, decay=0.5)
        assert tracker.decide(update) is ProbationOutcome.PENDING

    def test_consolidate_when_repeatedly_useful(self) -> None:
        tracker = _tracker(minimum_future_trials=2, minimum_contexts_helped=2)
        update = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(1.0, decay=0.5)
        update.record_future_utility(1.0, decay=0.5)
        assert tracker.decide(update) is ProbationOutcome.CONSOLIDATE

    def test_rollback_when_negative(self) -> None:
        tracker = _tracker(minimum_future_trials=1)
        update = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(-1.0, decay=0.5)
        assert tracker.decide(update) is ProbationOutcome.ROLLBACK

    def test_retain_when_useful_in_exactly_one_context(self) -> None:
        tracker = _tracker(minimum_future_trials=1, minimum_contexts_helped=3)
        update = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(1.0, decay=0.5)
        assert tracker.decide(update) is ProbationOutcome.RETAIN

    def test_decay_when_neutral(self) -> None:
        tracker = _tracker(minimum_future_trials=1, minimum_contexts_helped=3)
        update = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        update.record_future_utility(0.0, decay=0.5)
        assert tracker.decide(update) is ProbationOutcome.DECAY


class TestApplyOutcome:
    def test_decay_scales_the_delta(self) -> None:
        tracker = _tracker(fast_weight_decay=0.25)
        state = FastWeightState()
        state.add_group(_delta())
        update = tracker.add(
            group="routing", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        tracker.apply_outcome(ProbationOutcome.DECAY, update, state)
        assert torch.allclose(state.get_group("routing").values["w"], torch.tensor([0.25, 0.5]))

    def test_rollback_removes_group_and_forgets_the_update(self) -> None:
        tracker = _tracker()
        state = FastWeightState()
        state.add_group(_delta())
        update = tracker.add(
            group="routing", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        tracker.apply_outcome(ProbationOutcome.ROLLBACK, update, state)
        assert "routing" not in state
        assert update.update_id not in tracker.updates

    def test_retain_and_pending_leave_state_untouched(self) -> None:
        tracker = _tracker()
        state = FastWeightState()
        state.add_group(_delta())
        before = state.get_group("routing").values["w"].clone()
        update = tracker.add(
            group="routing", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        tracker.apply_outcome(ProbationOutcome.RETAIN, update, state)
        tracker.apply_outcome(ProbationOutcome.PENDING, update, state)
        assert torch.equal(state.get_group("routing").values["w"], before)

    def test_consolidate_is_a_no_op_here_by_design(self) -> None:
        """Committing into theta_slow is Phase 4's job (consolidation.py)."""

        tracker = _tracker()
        state = FastWeightState()
        state.add_group(_delta())
        before = state.get_group("routing").values["w"].clone()
        update = tracker.add(
            group="routing", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        tracker.apply_outcome(ProbationOutcome.CONSOLIDATE, update, state)
        assert torch.equal(state.get_group("routing").values["w"], before)
        assert update.update_id in tracker.updates


class TestSampleAndExpire:
    def test_should_sample_future_utility_respects_extreme_rates(self) -> None:
        always = _tracker(future_utility_probe_rate=1.0)
        never = _tracker(future_utility_probe_rate=0.0)
        generator = torch.Generator().manual_seed(0)
        assert always.should_sample_future_utility(generator=generator)
        assert not never.should_sample_future_utility(generator=generator)

    def test_expire_stale_removes_only_expired_updates(self) -> None:
        tracker = _tracker(provisional_expiry_examples=5)
        early = tracker.add(
            group="a", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=0,
        )
        late = tracker.add(
            group="b", delta=_delta(), current_gain=0.1, replay_cost=0.0,
            surprise=0.5, reliability=0.5, opened_step=100,
        )
        expired = tracker.expire_stale(current_step=10)
        assert [update.update_id for update in expired] == [early.update_id]
        assert early.update_id not in tracker.updates
        assert late.update_id in tracker.updates
