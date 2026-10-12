"""Tests for SGBM Phase G: episodic batch scheduling."""

from __future__ import annotations

import pytest
import torch

from gator.continual.episodic_stream import (
    BatchStep,
    Episode,
    EpisodicBatchScheduler,
    assert_no_cross_episode_leakage,
)


def _episode(episode_id: str, length: int) -> Episode[int]:
    return Episode(episode_id=episode_id, blocks=list(range(length)))


class TestEpisode:
    def test_rejects_empty_blocks(self) -> None:
        with pytest.raises(ValueError):
            Episode(episode_id="empty", blocks=[])


class TestSchedulerConstruction:
    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError):
            EpisodicBatchScheduler([_episode("a", 2)], batch_size=0)


class TestSchedulerAdvancement:
    def test_first_batch_flags_reset_for_every_active_lane(self) -> None:
        scheduler = EpisodicBatchScheduler([_episode("a", 2), _episode("b", 2)], batch_size=2)
        step = scheduler.next_batch()
        assert step.reset_mask.tolist() == [True, True]
        assert step.active_mask.tolist() == [True, True]
        assert step.episode_ids == ["a", "b"]
        assert step.blocks == [0, 0]

    def test_subsequent_batches_advance_without_reset(self) -> None:
        scheduler = EpisodicBatchScheduler([_episode("a", 3)], batch_size=1)
        scheduler.next_batch()
        step = scheduler.next_batch()
        assert step.reset_mask.tolist() == [False]
        assert step.blocks == [1]

    def test_exhausted_lane_pulls_next_episode_with_reset(self) -> None:
        scheduler = EpisodicBatchScheduler([_episode("a", 1), _episode("b", 2)], batch_size=1)
        first = scheduler.next_batch()
        assert first.episode_ids == ["a"]
        second = scheduler.next_batch()
        assert second.episode_ids == ["b"]
        assert second.reset_mask.tolist() == [True]

    def test_lane_goes_inactive_once_queue_is_empty(self) -> None:
        scheduler = EpisodicBatchScheduler([_episode("a", 1)], batch_size=2)
        step = scheduler.next_batch()
        assert step.active_mask.tolist() == [True, False]
        assert step.episode_ids == ["a", None]

    def test_has_more_reflects_remaining_work(self) -> None:
        scheduler = EpisodicBatchScheduler([_episode("a", 1)], batch_size=1)
        assert scheduler.has_more()
        scheduler.next_batch()
        assert not scheduler.has_more()

    def test_lanes_stay_independent_across_calls(self) -> None:
        scheduler = EpisodicBatchScheduler([_episode("a", 1), _episode("b", 3)], batch_size=2)
        scheduler.next_batch()  # a exhausts, b at block 0
        step = scheduler.next_batch()
        # lane 0 has nothing left (queue empty), lane 1 continues episode b
        assert step.active_mask.tolist() == [False, True]
        assert step.episode_ids == [None, "b"]
        assert step.reset_mask.tolist() == [False, False]


class TestNoLeakageAssertion:
    def test_passes_when_reset_flag_matches_episode_change(self) -> None:
        step = BatchStep(
            blocks=[1],
            active_mask=torch.tensor([True]),
            reset_mask=torch.tensor([True]),
            episode_ids=["b"],
        )
        assert_no_cross_episode_leakage(["a"], step)  # no raise

    def test_raises_when_episode_changes_without_reset_flag(self) -> None:
        step = BatchStep(
            blocks=[1],
            active_mask=torch.tensor([True]),
            reset_mask=torch.tensor([False]),
            episode_ids=["b"],
        )
        with pytest.raises(RuntimeError):
            assert_no_cross_episode_leakage(["a"], step)

    def test_same_episode_continuing_never_raises(self) -> None:
        step = BatchStep(
            blocks=[1],
            active_mask=torch.tensor([True]),
            reset_mask=torch.tensor([False]),
            episode_ids=["a"],
        )
        assert_no_cross_episode_leakage(["a"], step)  # no raise

    def test_inactive_lane_with_none_episode_id_is_ignored(self) -> None:
        step = BatchStep(
            blocks=[None],
            active_mask=torch.tensor([False]),
            reset_mask=torch.tensor([False]),
            episode_ids=[None],
        )
        assert_no_cross_episode_leakage(["a"], step)  # no raise

    def test_rejects_mismatched_batch_size(self) -> None:
        step = BatchStep(
            blocks=[1],
            active_mask=torch.tensor([True]),
            reset_mask=torch.tensor([True]),
            episode_ids=["b"],
        )
        with pytest.raises(ValueError):
            assert_no_cross_episode_leakage(["a", "extra"], step)
