"""Tests for SGBM Phase E: the top-level SurpriseGatedMemory read/write/maintain API."""

from __future__ import annotations

import pytest
import torch

from gator.memory import SurpriseGatedMemory, SurpriseMemoryConfig, SurpriseMemoryState
from gator.memory.controller import WriteGateCoefficients
from gator.memory.quarantine import write_or_confirm_quarantine


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(
        fused_dim=8,
        key_dim=4,
        value_dim=3,
        memory_slots=4,
        quarantine_slots=2,
        retrieval_topk=2,
        merge_threshold=0.9,
        quarantine_threshold=0.20,
        write_threshold=0.55,
        promotion_confirmations=2,
    )
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


def _unit_key(seed: int, key_dim: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(1, key_dim, generator=generator)
    return torch.nn.functional.normalize(vector, dim=-1)


# Coefficients chosen so write_gate_strength(surprise, novelty=0) lands
# roughly at `surprise` itself: sigmoid(8*s - 4) crosses 0.5 around s=0.5,
# and with reliability=1 that keeps commit_strength close to intuitive
# "high surprise -> high commit" behavior for these tests.
_SIMPLE_GATE = WriteGateCoefficients(surprise_weight=8.0, novelty_weight=0.0, bias=0.0)


class TestWriteRouting:
    def test_low_surprise_is_ignored(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(1, config)

        memory.write(
            candidate_key=_unit_key(0),
            candidate_value=torch.ones(1, config.value_dim),
            surprise=torch.tensor([-10.0]),  # gate ~= 0
            reliability=torch.tensor([1.0]),
            state=state,
        )

        assert int(state.valid.sum().item()) == 0
        assert int(state.quarantine_valid.sum().item()) == 0

    def test_low_reliability_routes_to_quarantine_not_stable(self) -> None:
        """Design spec unit test #6 at the module level: high surprise, low
        reliability must not enter stable memory."""

        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(1, config)

        memory.write(
            candidate_key=_unit_key(1),
            candidate_value=torch.ones(1, config.value_dim),
            surprise=torch.tensor([10.0]),  # write_gate ~= 1
            reliability=torch.tensor([0.3]),  # commit_strength ~= 0.3 -> quarantine band
            state=state,
        )

        assert int(state.valid.sum().item()) == 0
        assert int(state.quarantine_valid.sum().item()) == 1

    def test_high_surprise_high_reliability_routes_to_stable(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(1, config)

        memory.write(
            candidate_key=_unit_key(2),
            candidate_value=torch.ones(1, config.value_dim),
            surprise=torch.tensor([10.0]),
            reliability=torch.tensor([0.9]),
            state=state,
        )

        assert int(state.valid.sum().item()) == 1
        assert int(state.quarantine_valid.sum().item()) == 0

    def test_batched_lanes_get_independent_decisions(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(3, config)
        keys = torch.cat([_unit_key(10), _unit_key(11), _unit_key(12)], dim=0)

        memory.write(
            candidate_key=keys,
            candidate_value=torch.ones(3, config.value_dim),
            surprise=torch.tensor([-10.0, 10.0, 10.0]),
            reliability=torch.tensor([1.0, 0.3, 0.9]),
            state=state,
        )

        assert int(state.valid[0].sum().item()) == 0
        assert int(state.quarantine_valid[0].sum().item()) == 0

        assert int(state.valid[1].sum().item()) == 0
        assert int(state.quarantine_valid[1].sum().item()) == 1

        assert int(state.valid[2].sum().item()) == 1
        assert int(state.quarantine_valid[2].sum().item()) == 0

    def test_write_does_not_require_grad_even_with_grad_carrying_inputs(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(3).requires_grad_(True)
        value = torch.ones(1, config.value_dim, requires_grad=True)

        memory.write(
            candidate_key=key,
            candidate_value=value,
            surprise=torch.tensor([10.0], requires_grad=True),
            reliability=torch.tensor([0.9], requires_grad=True),
            state=state,
        )

        assert not state.keys.requires_grad
        assert not state.values.requires_grad


class TestReadWriteRoundTrip:
    def test_written_stable_memory_is_retrievable(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(4)
        value = torch.tensor([[1.0, 2.0, 3.0]])

        memory.write(
            candidate_key=key,
            candidate_value=value,
            surprise=torch.tensor([10.0]),
            reliability=torch.tensor([0.9]),
            state=state,
        )

        retrieved, metrics = memory.read(key, state)
        assert torch.allclose(retrieved, value, atol=1e-3)
        assert metrics["retrieval_weights"].sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_reading_increments_usage_of_retrieved_slot(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, gate_coefficients=_SIMPLE_GATE)
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(5)
        memory.write(
            candidate_key=key,
            candidate_value=torch.ones(1, config.value_dim),
            surprise=torch.tensor([10.0]),
            reliability=torch.tensor([0.9]),
            state=state,
        )
        assert state.usage.sum().item() == pytest.approx(1.0)  # from the write itself
        memory.read(key, state)
        assert state.usage.sum().item() == pytest.approx(2.0)  # write + read bump


class TestMaintain:
    def test_maintain_decays_and_can_forget(self) -> None:
        config = _config(base_decay=50.0)
        memory = SurpriseGatedMemory(config, decay_forget_threshold=1e-2, decay_time_constant=1.0)
        state = SurpriseMemoryState.allocate(1, config)
        state.valid[0, 0] = True
        state.strength[0, 0] = 1e-3
        state.reliability[0, 0] = 0.0
        state.age[0, 0] = 50.0

        result = memory.maintain(state)

        assert bool(result["forgotten"][0, 0])
        assert not bool(state.valid[0, 0])

    def test_maintain_promotes_confirmed_quarantine_entries(self) -> None:
        """Design spec unit test #7, exercised through the module's maintain()
        wiring. The quarantine confirmation EMA (write_or_confirm_quarantine,
        tested directly in test_sgbm_quarantine.py) can only cross the
        promotion bar if a later observation is itself strong enough — set
        that up directly rather than via write()'s own gate arithmetic,
        which is already covered by TestWriteRouting above."""

        config = _config(promotion_confirmations=1, write_threshold=0.5)
        memory = SurpriseGatedMemory(config)
        state = SurpriseMemoryState.allocate(1, config)
        key = _unit_key(6)
        write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.9]),
            reliability=torch.tensor([0.9]),
            config=config,
        )
        write_or_confirm_quarantine(
            state,
            candidate_key=key,
            candidate_value=torch.ones(1, config.value_dim),
            write_strength=torch.tensor([0.9]),
            reliability=torch.tensor([0.9]),
            config=config,
        )
        assert int(state.quarantine_valid.sum().item()) == 1
        assert int(state.quarantine_confirmations.sum().item()) >= 1

        result = memory.maintain(state)

        assert bool(result["promoted"][0])
        assert int(state.valid.sum().item()) == 1
        assert int(state.quarantine_valid.sum().item()) == 0

    def test_maintain_expires_stale_unconfirmed_quarantine(self) -> None:
        config = _config()
        memory = SurpriseGatedMemory(config, quarantine_max_age_unconfirmed=2.0)
        state = SurpriseMemoryState.allocate(1, config)
        memory.write(
            candidate_key=_unit_key(7),
            candidate_value=torch.ones(1, config.value_dim),
            surprise=torch.tensor([10.0]),
            reliability=torch.tensor([0.3]),
            state=state,
        )
        for _ in range(5):
            memory.maintain(state)
        assert int(state.quarantine_valid.sum().item()) == 0
