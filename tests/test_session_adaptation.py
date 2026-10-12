"""Tests for Procoim-S Phase 9: inference-time session-scoped adaptation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gator.optimizers.fast_weights import FastWeightDelta
from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.session_adaptation import (
    SessionAdaptationPolicy,
    SessionFastWeightRegistry,
    consolidate_session,
)


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("layer1", nn.Linear(4, 3))
    return model


def _policy(**overrides) -> SessionAdaptationPolicy:
    defaults = dict(allowed_groups=("routing", "readout"), max_session_delta_norm=1.0)
    defaults.update(overrides)
    return SessionAdaptationPolicy(**defaults)


class TestSessionAdaptationPolicy:
    def test_rejects_empty_allowed_groups(self) -> None:
        with pytest.raises(ValueError):
            SessionAdaptationPolicy(allowed_groups=(), max_session_delta_norm=1.0)

    def test_rejects_non_positive_max_norm(self) -> None:
        with pytest.raises(ValueError):
            SessionAdaptationPolicy(allowed_groups=("routing",), max_session_delta_norm=0.0)


class TestSessionFastWeightRegistry:
    def test_get_or_create_returns_the_same_object_on_repeat_calls(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        first = registry.get_or_create("user-1")
        second = registry.get_or_create("user-1")
        assert first is second

    def test_different_sessions_get_different_objects(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        state_a = registry.get_or_create("user-1")
        state_b = registry.get_or_create("user-2")
        assert state_a is not state_b

    def test_apply_session_delta_rejects_disallowed_group(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy(allowed_groups=("routing",)))
        with pytest.raises(PermissionError):
            registry.apply_session_delta(
                "user-1",
                FastWeightDelta(group="embedding", values={"w": torch.ones(2)}),
            )

    def test_apply_session_delta_merges_into_the_session_state(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        registry.apply_session_delta(
            "user-1", FastWeightDelta(group="routing", values={"w": torch.tensor([0.1, 0.1])})
        )
        state = registry.get_or_create("user-1")
        assert torch.allclose(state.get_group("routing").values["w"], torch.tensor([0.1, 0.1]))

    def test_apply_session_delta_clamps_to_max_norm(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy(max_session_delta_norm=1.0))
        registry.apply_session_delta(
            "user-1", FastWeightDelta(group="routing", values={"w": torch.tensor([3.0, 4.0])})
        )
        state = registry.get_or_create("user-1")
        assert state.get_group("routing").norm() == pytest.approx(1.0, abs=1e-5)

    def test_sessions_are_fully_isolated(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        registry.apply_session_delta(
            "user-1", FastWeightDelta(group="routing", values={"w": torch.tensor([0.5])})
        )
        user_2_state = registry.get_or_create("user-2")
        assert "routing" not in user_2_state

    def test_end_session_removes_and_returns_state(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        registry.get_or_create("user-1")
        removed = registry.end_session("user-1")
        assert removed is not None
        assert "user-1" not in registry

    def test_end_session_on_unknown_session_returns_none(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        assert registry.end_session("ghost") is None

    def test_no_shared_state_true_for_distinct_sessions(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        registry.get_or_create("user-1")
        registry.get_or_create("user-2")
        assert registry.no_shared_state_across_sessions()

    def test_no_shared_state_detects_an_accidental_alias(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        shared_state = registry.get_or_create("user-1")
        registry._sessions["user-2"] = shared_state  # simulate a bug
        assert not registry.no_shared_state_across_sessions()


class TestConsolidateSession:
    def test_raises_without_explicit_approval(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy())
        model = _tiny_model()
        registry.apply_session_delta(
            "user-1", FastWeightDelta(group="routing", values={"layer1.bias": torch.ones(3)})
        )
        with pytest.raises(PermissionError):
            consolidate_session(
                "user-1", registry, model,
                evaluate=lambda: (1.0, []),
                config=ProcoimSConfig(),
                approved=False,
            )

    def test_commits_when_approved(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy(max_session_delta_norm=10.0))
        model = _tiny_model()
        before = model.layer1.bias.clone()
        registry.apply_session_delta(
            "user-1", FastWeightDelta(group="routing", values={"layer1.bias": torch.ones(3)})
        )

        result = consolidate_session(
            "user-1", registry, model,
            evaluate=lambda: (1.0, []),
            config=ProcoimSConfig(minimum_utility=0.0, replay_weight=1.0),
            approved=True,
        )

        assert result.committed
        assert torch.allclose(model.layer1.bias, before + 1.0)

    def test_does_not_touch_other_sessions(self) -> None:
        registry = SessionFastWeightRegistry(policy=_policy(max_session_delta_norm=10.0))
        model = _tiny_model()
        registry.apply_session_delta(
            "user-1", FastWeightDelta(group="routing", values={"layer1.bias": torch.ones(3)})
        )
        registry.apply_session_delta(
            "user-2", FastWeightDelta(group="routing", values={"layer1.bias": torch.full((3,), 5.0)})
        )

        consolidate_session(
            "user-1", registry, model,
            evaluate=lambda: (1.0, []),
            config=ProcoimSConfig(minimum_utility=0.0),
            approved=True,
        )

        user_2_state = registry.get_or_create("user-2")
        assert torch.allclose(user_2_state.get_group("routing").values["layer1.bias"], torch.full((3,), 5.0))
