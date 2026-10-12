"""Tests for SGBM Phase B: quarantine buffer and confirmation/promotion logic.

Covers design-spec unit tests #6-7 (high-surprise low-reliability items do
not enter stable memory; repeated reliable evidence promotes a matching
quarantine item).
"""

from __future__ import annotations

import pytest
import torch

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.quarantine import (
    expire_quarantine,
    promote_confirmed,
    promotion_ready_mask,
    write_or_confirm_quarantine,
)
from gator.memory.state import SurpriseMemoryState
from gator.memory.store import write_stable_memory


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(
        fused_dim=8,
        key_dim=4,
        value_dim=3,
        memory_slots=4,
        quarantine_slots=2,
        retrieval_topk=2,
        merge_threshold=0.9,
        promotion_confirmations=2,
        write_threshold=0.55,
    )
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


def _unit_key(seed: int, key_dim: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(1, key_dim, generator=generator)
    return torch.nn.functional.normalize(vector, dim=-1)


class TestWriteOrConfirm:
    def test_first_write_is_not_a_confirmation(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        _, was_confirmation = write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(0),
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        assert not bool(was_confirmation[0])
        assert int(state.quarantine_valid.sum().item()) == 1
        assert int(state.quarantine_confirmations.sum().item()) == 0

    def test_repeated_similar_candidate_confirms_the_same_slot(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(1)
        target_first, _ = write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        target_second, was_confirmation = write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.6]),
            reliability=torch.tensor([0.6]),
            config=config,
        )
        assert bool(was_confirmation[0])
        assert int(target_first.item()) == int(target_second.item())
        assert int(state.quarantine_confirmations[0, target_second].item()) == 1
        assert int(state.quarantine_valid.sum().item()) == 1

    def test_dissimilar_candidate_allocates_separate_slot(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(2),
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(999),
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        assert int(state.quarantine_valid.sum().item()) == 2


class TestPromotion:
    def test_high_surprise_low_reliability_stays_quarantined(self) -> None:
        """Design spec unit test #6."""

        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(3),
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.1]),  # low reliability/strength
            reliability=torch.tensor([0.1]),
            config=config,
        )
        promoted = promote_confirmed(state, config)
        assert not bool(promoted.any())
        assert int(state.valid.sum().item()) == 0
        assert int(state.quarantine_valid.sum().item()) == 1

    def test_confirmed_reliable_quarantine_item_promotes(self) -> None:
        """Design spec unit test #7."""

        config = _config(promotion_confirmations=2, write_threshold=0.5)
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(4)
        for _ in range(3):
            write_or_confirm_quarantine(
                state,
                candidate_key=key,
                candidate_value=torch.full((1, config.value_dim), 7.0),
                write_strength=torch.tensor([0.9]),
                reliability=torch.tensor([0.9]),
                config=config,
            )
        assert bool(promotion_ready_mask(state, config).any())

        promoted = promote_confirmed(state, config)
        assert bool(promoted[0])
        assert int(state.valid.sum().item()) == 1
        promoted_index = int(state.valid[0].nonzero()[0].item())
        assert torch.allclose(state.values[0, promoted_index], torch.full((config.value_dim,), 7.0))

    def test_promote_with_nothing_ready_is_a_no_op(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        before = state.clone()
        promoted = promote_confirmed(state, config)
        assert not bool(promoted.any())
        assert torch.equal(state.valid, before.valid)
        assert torch.equal(state.quarantine_valid, before.quarantine_valid)

    def test_promotion_clears_the_source_quarantine_slot(self) -> None:
        config = _config(promotion_confirmations=1, write_threshold=0.5)
        state = SurpriseMemoryState.allocate(1, config)
        write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(5),
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.9]),
            reliability=torch.tensor([0.9]),
            config=config,
        )
        # promotion_confirmations=1 means the confirmations counter (0 on
        # first write) is not yet enough — bump it via a second write.
        write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(5),
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.9]),
            reliability=torch.tensor([0.9]),
            config=config,
        )
        promote_confirmed(state, config)
        assert int(state.quarantine_valid.sum().item()) == 0
        assert int(state.quarantine_confirmations.sum().item()) == 0


class TestExpiry:
    def test_unconfirmed_entry_expires_past_max_age(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        write_or_confirm_quarantine(
            state,
            candidate_key=_unit_key(6),
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        for _ in range(5):
            expire_quarantine(state, max_age_unconfirmed=3.0)
        assert int(state.quarantine_valid.sum().item()) == 0

    def test_confirmed_entry_survives_when_no_confirmed_ceiling_given(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(7)
        write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        for _ in range(10):
            expire_quarantine(state, max_age_unconfirmed=3.0)
        assert int(state.quarantine_valid.sum().item()) == 1

    def test_confirmed_entry_expires_past_confirmed_ceiling_when_given(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(8)
        write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.zeros(1, config.value_dim),
            write_strength=torch.tensor([0.2]),
            reliability=torch.tensor([0.2]),
            config=config,
        )
        for _ in range(10):
            expire_quarantine(state, max_age_unconfirmed=3.0, max_age_confirmed=5.0)
        assert int(state.quarantine_valid.sum().item()) == 0
