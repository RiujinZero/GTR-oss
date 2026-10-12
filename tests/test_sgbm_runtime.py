"""Tests for MemoryRuntime (SGBM Phase F support)."""

from __future__ import annotations

import pytest
import torch

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.runtime import MemoryRuntime


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(fused_dim=6, key_dim=4, value_dim=3, memory_slots=4, quarantine_slots=2, retrieval_topk=2)
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


class TestAllocate:
    def test_shapes(self) -> None:
        runtime = MemoryRuntime.allocate(3, _config())
        assert runtime.context_summary.shape == (3, 6)
        assert runtime.cache_summary.shape == (3, 6)
        assert runtime.state.batch_size == 3

    def test_rejects_mismatched_summary_shapes(self) -> None:
        config = _config()
        state = MemoryRuntime.allocate(2, config).state
        with pytest.raises(ValueError):
            MemoryRuntime(
                state=state,
                context_summary=torch.zeros(2, 6),
                cache_summary=torch.zeros(2, 5),
            )

    def test_rejects_batch_mismatch_with_state(self) -> None:
        config = _config()
        state = MemoryRuntime.allocate(2, config).state
        with pytest.raises(ValueError):
            MemoryRuntime(
                state=state,
                context_summary=torch.zeros(3, 6),
                cache_summary=torch.zeros(3, 6),
            )


class TestUpdateSummaries:
    def test_context_decays_faster_than_cache(self) -> None:
        runtime = MemoryRuntime.allocate(1, _config())
        new_block = torch.ones(1, 6)
        runtime.update_summaries(new_block, context_decay=0.5, cache_decay=0.99)
        assert runtime.context_summary[0, 0].item() > runtime.cache_summary[0, 0].item()

    def test_repeated_updates_converge_toward_the_block(self) -> None:
        runtime = MemoryRuntime.allocate(1, _config())
        new_block = torch.full((1, 6), 2.0)
        for _ in range(500):
            runtime.update_summaries(new_block, context_decay=0.9, cache_decay=0.9)
        assert runtime.cache_summary[0, 0].item() == pytest.approx(2.0, abs=1e-2)


class TestResetLanes:
    def test_reset_clears_only_masked_lanes(self) -> None:
        runtime = MemoryRuntime.allocate(2, _config())
        runtime.context_summary.fill_(1.0)
        runtime.cache_summary.fill_(1.0)
        runtime.state.strength.fill_(1.0)

        runtime.reset_lanes(torch.tensor([True, False]))

        assert torch.equal(runtime.context_summary[0], torch.zeros(6))
        assert torch.equal(runtime.context_summary[1], torch.ones(6))
        assert torch.equal(runtime.state.strength[0], torch.zeros(runtime.state.memory_slots))
        assert torch.equal(runtime.state.strength[1], torch.ones(runtime.state.memory_slots))


class TestDetach:
    def test_detach_removes_grad_from_summaries_and_state(self) -> None:
        runtime = MemoryRuntime.allocate(1, _config())
        leaf = torch.zeros(1, 6, requires_grad=True)
        runtime.context_summary = leaf * 2.0
        assert runtime.context_summary.requires_grad

        runtime.detach_()

        assert not runtime.context_summary.requires_grad
