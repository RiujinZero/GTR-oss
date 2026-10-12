"""Tests for SGBM Phase A: config validation and per-lane memory state.

Covers design-spec unit tests #3 (episode reset clears only selected batch
lanes), #11 (serialization round trip), and #12 (online updates carry no
autograd graph) at the state-container level — the mechanics that produce
real writes/reads are Phase B and are tested separately once built.
"""

from __future__ import annotations

import pytest
import torch

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.state import SurpriseMemoryState


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(fused_dim=8, key_dim=4, value_dim=6, memory_slots=5, quarantine_slots=3)
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


class TestSurpriseMemoryConfig:
    def test_defaults_construct_with_only_fused_dim(self) -> None:
        config = SurpriseMemoryConfig(fused_dim=32)
        assert config.memory_slots == 32
        assert config.retrieval_topk == 4

    @pytest.mark.parametrize(
        "overrides",
        [
            {"fused_dim": 0},
            {"key_dim": 0},
            {"value_dim": -1},
            {"memory_slots": 0},
            {"quarantine_slots": -1},
            {"retrieval_topk": 0},
            {"surprise_ema": 1.0},
            {"surprise_ema": 0.0},
            {"surprise_momentum_beta": 1.5},
            {"merge_threshold": 2.0},
            {"promotion_confirmations": 0},
            {"memory_dropout": -0.1},
            {"base_decay": -1e-3},
        ],
    )
    def test_invalid_fields_rejected(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            _config(**overrides)

    def test_retrieval_topk_cannot_exceed_memory_slots(self) -> None:
        with pytest.raises(ValueError):
            _config(memory_slots=4, retrieval_topk=8)

    def test_quarantine_threshold_must_not_exceed_write_threshold(self) -> None:
        with pytest.raises(ValueError):
            _config(quarantine_threshold=0.9, write_threshold=0.5)


class TestAllocate:
    def test_shapes_match_config(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(2, config)
        assert state.keys.shape == (2, 5, 4)
        assert state.values.shape == (2, 5, 6)
        assert state.strength.shape == (2, 5)
        assert state.valid.shape == (2, 5)
        assert state.quarantine_keys.shape == (2, 3, 4)
        assert state.quarantine_values.shape == (2, 3, 6)
        assert state.quarantine_confirmations.shape == (2, 3)
        assert state.surprise_momentum.shape == (2,)
        assert state.previous_block_summary.shape == (2, 8)

    def test_dtypes(self) -> None:
        state = SurpriseMemoryState.allocate(1, _config())
        assert state.valid.dtype == torch.bool
        assert state.quarantine_valid.dtype == torch.bool
        assert state.quarantine_confirmations.dtype == torch.int64
        assert state.keys.dtype == torch.float32

    def test_freshly_allocated_state_has_no_valid_slots(self) -> None:
        state = SurpriseMemoryState.allocate(2, _config())
        assert not bool(state.valid.any())
        assert not bool(state.quarantine_valid.any())

    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError):
            SurpriseMemoryState.allocate(0, _config())

    def test_batch_size_and_slot_count_properties(self) -> None:
        state = SurpriseMemoryState.allocate(3, _config())
        assert state.batch_size == 3
        assert state.memory_slots == 5
        assert state.quarantine_slots == 3


class TestShapeValidation:
    def test_mismatched_batch_dimension_rejected(self) -> None:
        config = _config()
        good = SurpriseMemoryState.allocate(2, config)
        with pytest.raises(ValueError):
            SurpriseMemoryState(
                keys=good.keys,
                values=good.values,
                strength=good.strength,
                age=good.age,
                usage=good.usage,
                reliability=good.reliability,
                surprise=good.surprise,
                valid=good.valid,
                quarantine_keys=good.quarantine_keys,
                quarantine_values=good.quarantine_values,
                quarantine_strength=good.quarantine_strength,
                quarantine_confirmations=good.quarantine_confirmations,
                quarantine_age=good.quarantine_age,
                quarantine_valid=good.quarantine_valid,
                surprise_momentum=torch.zeros(999),  # wrong batch size
                previous_block_summary=good.previous_block_summary,
            )


class TestResetLanes:
    def test_reset_clears_only_masked_lanes(self) -> None:
        """Design spec unit test #3."""

        state = SurpriseMemoryState.allocate(3, _config())
        state.strength.fill_(1.0)
        state.valid.fill_(True)
        state.surprise_momentum.fill_(0.7)

        mask = torch.tensor([False, True, False])
        state.reset_lanes(mask)

        assert torch.equal(state.strength[0], torch.ones(5))
        assert torch.equal(state.strength[1], torch.zeros(5))
        assert torch.equal(state.strength[2], torch.ones(5))
        assert bool(state.valid[0].all())
        assert not bool(state.valid[1].any())
        assert bool(state.valid[2].all())
        assert state.surprise_momentum.tolist() == pytest.approx([0.7, 0.0, 0.7])

    def test_reset_with_all_false_mask_is_a_no_op(self) -> None:
        state = SurpriseMemoryState.allocate(2, _config())
        state.strength.fill_(1.0)
        state.reset_lanes(torch.tensor([False, False]))
        assert torch.equal(state.strength, torch.ones(2, 5))

    def test_reset_rejects_wrong_shaped_mask(self) -> None:
        state = SurpriseMemoryState.allocate(2, _config())
        with pytest.raises(ValueError):
            state.reset_lanes(torch.tensor([False, False, False]))


class TestDetach:
    def test_detach_removes_autograd_graph(self) -> None:
        """Design spec unit test #12 (state-container half)."""

        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        leaf = torch.zeros(1, config.memory_slots, config.key_dim, requires_grad=True)
        state.keys = leaf * 2.0  # non-leaf tensor with a grad_fn
        assert state.keys.requires_grad
        assert state.keys.grad_fn is not None

        state.detach_()

        assert not state.keys.requires_grad
        assert state.keys.grad_fn is None


class TestCloneAndSerialize:
    def test_clone_is_independent(self) -> None:
        state = SurpriseMemoryState.allocate(1, _config())
        clone = state.clone()
        clone.strength.fill_(5.0)
        assert not torch.equal(state.strength, clone.strength)

    def test_state_dict_round_trip_reproduces_values(self) -> None:
        """Design spec unit test #11 (state-container half)."""

        config = _config()
        state = SurpriseMemoryState.allocate(2, config)
        state.keys.copy_(torch.randn_like(state.keys))
        state.valid[0, 0] = True
        state.quarantine_confirmations[0, 0] = 2

        restored = SurpriseMemoryState.allocate(2, config)
        restored.load_state_dict(state.state_dict())

        assert torch.equal(restored.keys, state.keys)
        assert torch.equal(restored.valid, state.valid)
        assert torch.equal(restored.quarantine_confirmations, state.quarantine_confirmations)

    def test_load_state_dict_rejects_missing_fields(self) -> None:
        state = SurpriseMemoryState.allocate(1, _config())
        with pytest.raises(KeyError):
            state.load_state_dict({"keys": state.keys})

    def test_state_dict_values_are_detached_copies(self) -> None:
        state = SurpriseMemoryState.allocate(1, _config())
        snapshot = state.state_dict()
        snapshot["keys"].fill_(9.0)
        assert not torch.equal(state.keys, snapshot["keys"])
