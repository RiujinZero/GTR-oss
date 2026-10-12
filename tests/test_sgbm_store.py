"""Tests for SGBM Phase B: stable-memory merge/allocation/decay/eviction."""

from __future__ import annotations

import pytest
import torch

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.state import SurpriseMemoryState
from gator.memory.store import compute_slot_utility, decay_stable_memory, write_stable_memory


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(
        fused_dim=8,
        key_dim=4,
        value_dim=3,
        memory_slots=3,
        quarantine_slots=2,
        retrieval_topk=2,
        merge_threshold=0.9,
    )
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


def _unit_key(seed: int, key_dim: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(1, key_dim, generator=generator)
    return torch.nn.functional.normalize(vector, dim=-1)


class TestWriteAllocatesFreshSlot:
    def test_first_write_into_empty_state_allocates_a_slot(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(0)
        value = torch.ones(1, config.value_dim)

        target = write_stable_memory(
            state,
            candidate_key=key,
            candidate_value=value,
            commit_strength=torch.tensor([0.7]),
            reliability=torch.tensor([0.9]),
            surprise=torch.tensor([0.6]),
            config=config,
        )

        index = int(target.item())
        assert bool(state.valid[0, index])
        assert torch.allclose(state.values[0, index], value)
        assert state.strength[0, index].item() == pytest.approx(0.7)
        assert state.usage[0, index].item() == pytest.approx(1.0)
        assert state.age[0, index].item() == pytest.approx(0.0)
        assert int(state.valid.sum().item()) == 1


class TestMergeVsAllocate:
    def test_near_identical_candidate_merges_into_existing_slot(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(1)
        write_stable_memory(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            commit_strength=torch.tensor([0.5]),
            reliability=torch.tensor([0.5]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        assert int(state.valid.sum().item()) == 1

        # Same key again -> should merge into the same slot, not allocate a
        # second one.
        write_stable_memory(
            state,
            candidate_key=key,
            candidate_value=torch.ones(1, config.value_dim),
            commit_strength=torch.tensor([0.5]),
            reliability=torch.tensor([0.8]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        assert int(state.valid.sum().item()) == 1  # still one slot, merged

    def test_dissimilar_candidate_allocates_a_new_slot(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        write_stable_memory(
            state,
            candidate_key=_unit_key(2),
            candidate_value=torch.zeros(1, config.value_dim),
            commit_strength=torch.tensor([0.5]),
            reliability=torch.tensor([0.5]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        write_stable_memory(
            state,
            candidate_key=_unit_key(999),
            candidate_value=torch.ones(1, config.value_dim),
            commit_strength=torch.tensor([0.5]),
            reliability=torch.tensor([0.5]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        assert int(state.valid.sum().item()) == 2

    def test_merge_blends_value_rather_than_overwriting(self) -> None:
        config = _config(merge_threshold=0.0)  # force merge regardless of exact similarity
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(3)
        write_stable_memory(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            commit_strength=torch.tensor([1.0]),
            reliability=torch.tensor([0.5]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        index = int(state.valid[0].nonzero()[0].item())
        write_stable_memory(
            state,
            candidate_key=key,
            candidate_value=torch.full((1, config.value_dim), 10.0),
            commit_strength=torch.tensor([0.3]),
            reliability=torch.tensor([0.5]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        merged_value = state.values[0, index]
        # Blended toward, but not all the way to, the new candidate value.
        assert bool((merged_value > 0).all())
        assert bool((merged_value < 10.0).all())


class TestEviction:
    def test_full_memory_evicts_lowest_utility_slot(self) -> None:
        config = _config(memory_slots=2, merge_threshold=0.999)
        state = SurpriseMemoryState.allocate(1, config)
        # Slot A: low utility (weak, unreliable, unused, old).
        write_stable_memory(
            state,
            candidate_key=_unit_key(10),
            candidate_value=torch.zeros(1, config.value_dim),
            commit_strength=torch.tensor([0.01]),
            reliability=torch.tensor([0.01]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        # Slot B: high utility (strong, reliable).
        write_stable_memory(
            state,
            candidate_key=_unit_key(11),
            candidate_value=torch.ones(1, config.value_dim),
            commit_strength=torch.tensor([0.95]),
            reliability=torch.tensor([0.95]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        strong_key = state.keys[0, 1].clone()

        # A third, unrelated candidate should evict the weak slot (index 0),
        # not the strong one.
        write_stable_memory(
            state,
            candidate_key=_unit_key(12),
            candidate_value=torch.full((1, config.value_dim), 5.0),
            commit_strength=torch.tensor([0.5]),
            reliability=torch.tensor([0.5]),
            surprise=torch.tensor([0.5]),
            config=config,
        )
        assert torch.allclose(state.keys[0, 1], strong_key)


class TestActiveMask:
    def test_inactive_lane_is_completely_untouched(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(2, config)
        before = state.clone()

        write_stable_memory(
            state,
            candidate_key=torch.stack([_unit_key(20).squeeze(0), _unit_key(21).squeeze(0)]),
            candidate_value=torch.ones(2, config.value_dim),
            commit_strength=torch.tensor([0.5, 0.5]),
            reliability=torch.tensor([0.5, 0.5]),
            surprise=torch.tensor([0.5, 0.5]),
            config=config,
            active_mask=torch.tensor([True, False]),
        )

        assert torch.equal(state.keys[1], before.keys[1])
        assert torch.equal(state.valid[1], before.valid[1])
        assert not bool(state.valid[1].any())
        assert bool(state.valid[0].any())


class TestUtility:
    def test_higher_strength_and_reliability_increase_utility(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        state.strength[0, 0] = 0.9
        state.reliability[0, 0] = 0.9
        state.strength[0, 1] = 0.1
        state.reliability[0, 1] = 0.1
        utility = compute_slot_utility(state, eviction_gamma=1e-3)
        assert utility[0, 0] > utility[0, 1]

    def test_older_slot_has_lower_utility_all_else_equal(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        state.strength[0, :2] = 0.5
        state.reliability[0, :2] = 0.5
        state.age[0, 0] = 0.0
        state.age[0, 1] = 1000.0
        utility = compute_slot_utility(state, eviction_gamma=1e-2)
        assert utility[0, 0] > utility[0, 1]


class TestDecay:
    def test_age_advances_only_for_valid_slots(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        state.valid[0, 0] = True
        decay_stable_memory(state, config)
        assert state.age[0, 0].item() == pytest.approx(1.0)
        assert state.age[0, 1].item() == pytest.approx(0.0)

    def test_strength_decays_toward_zero(self) -> None:
        config = _config(base_decay=0.5)
        state = SurpriseMemoryState.allocate(1, config)
        state.valid[0, 0] = True
        state.strength[0, 0] = 1.0
        state.reliability[0, 0] = 0.0  # low reliability -> faster decay
        decay_stable_memory(state, config, time_constant=1.0)
        assert state.strength[0, 0].item() < 1.0

    def test_high_reliability_decays_slower_than_low_reliability(self) -> None:
        config = _config(base_decay=0.5)
        state = SurpriseMemoryState.allocate(1, config)
        state.valid[0, :2] = True
        state.strength[0, :2] = 1.0
        state.age[0, :2] = 5.0
        state.reliability[0, 0] = 0.0
        state.reliability[0, 1] = 0.99
        decay_stable_memory(state, config, time_constant=1.0)
        assert state.strength[0, 1] > state.strength[0, 0]

    def test_strength_below_threshold_frees_the_slot(self) -> None:
        config = _config(base_decay=10.0)
        state = SurpriseMemoryState.allocate(1, config)
        state.valid[0, 0] = True
        state.strength[0, 0] = 1e-3
        state.reliability[0, 0] = 0.0
        state.age[0, 0] = 100.0
        forgotten = decay_stable_memory(state, config, time_constant=1.0, forget_threshold=1e-2)
        assert bool(forgotten[0, 0])
        assert not bool(state.valid[0, 0])
        assert state.strength[0, 0].item() == pytest.approx(0.0)

    def test_decay_returns_no_forgotten_slots_when_nothing_crosses_threshold(self) -> None:
        config = _config(base_decay=1e-6)
        state = SurpriseMemoryState.allocate(1, config)
        state.valid[0, 0] = True
        state.strength[0, 0] = 1.0
        state.reliability[0, 0] = 0.99
        forgotten = decay_stable_memory(state, config)
        assert not bool(forgotten.any())
        assert bool(state.valid[0, 0])
