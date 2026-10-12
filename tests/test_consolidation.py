"""Tests for Procoim-S Phase 4: consolidation (committing fast weights into
theta_slow via a line search over consolidation_line_search alphas)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gator.optimizers.candidate_utility import ReplayEvaluation
from gator.optimizers.consolidation import (
    apply_scaled_fast_delta,
    consolidate,
    consolidate_componentwise,
    run_consolidation,
)
from gator.optimizers.fast_weights import FastWeightDelta, FastWeightState
from gator.optimizers.procoim_s import ProcoimSConfig


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("layer1", nn.Linear(4, 3))
    model.add_module("layer2", nn.Linear(3, 2))
    return model


def _config(**overrides) -> ProcoimSConfig:
    defaults = dict(
        minimum_utility=0.0,
        replay_weight=1.0,
        consolidation_line_search=(1.0, 0.75, 0.5, 0.25, 0.125),
        allow_componentwise_commit=True,
        fast_weight_decay=0.5,
    )
    defaults.update(overrides)
    return ProcoimSConfig(**defaults)


def _state_with_delta(model: nn.Module, *, group: str = "layer1", magnitude: float = 1.0) -> FastWeightState:
    state = FastWeightState()
    state.add_group(
        FastWeightDelta(
            group=group,
            values={f"{group}.bias": torch.full_like(getattr(model, group).bias, magnitude)},
        )
    )
    return state


class TestApplyScaledFastDelta:
    def test_alpha_zero_is_a_no_op(self) -> None:
        model = _tiny_model()
        before = model.layer1.bias.clone()
        state = _state_with_delta(model)
        with apply_scaled_fast_delta(model, state, 0.0):
            assert torch.equal(model.layer1.bias, before)

    def test_alpha_one_applies_the_full_delta(self) -> None:
        model = _tiny_model()
        before = model.layer1.bias.clone()
        state = _state_with_delta(model, magnitude=2.0)
        with apply_scaled_fast_delta(model, state, 1.0):
            assert torch.allclose(model.layer1.bias, before + 2.0)
        assert torch.equal(model.layer1.bias, before)  # restored after context exits

    def test_partial_alpha_scales_the_delta(self) -> None:
        model = _tiny_model()
        before = model.layer1.bias.clone()
        state = _state_with_delta(model, magnitude=4.0)
        with apply_scaled_fast_delta(model, state, 0.25):
            assert torch.allclose(model.layer1.bias, before + 1.0)

    def test_group_restriction_leaves_other_groups_untouched(self) -> None:
        model = _tiny_model()
        before2 = model.layer2.bias.clone()
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": torch.ones_like(model.layer1.bias)})
        )
        state.add_group(
            FastWeightDelta(group="layer2", values={"layer2.bias": torch.ones_like(model.layer2.bias)})
        )
        with apply_scaled_fast_delta(model, state, 1.0, group="layer1"):
            assert torch.equal(model.layer2.bias, before2)

    def test_scaling_does_not_mutate_the_original_state(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model, magnitude=3.0)
        original = state.get_group("layer1").values["layer1.bias"].clone()
        with apply_scaled_fast_delta(model, state, 0.25):
            pass
        assert torch.equal(state.get_group("layer1").values["layer1.bias"], original)

    def test_rejects_out_of_range_alpha(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model)
        with pytest.raises(ValueError):
            with apply_scaled_fast_delta(model, state, 1.5):
                pass


class TestConsolidate:
    def test_commits_at_the_largest_alpha_when_it_already_passes(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model)

        def evaluate():
            return 1.0, []  # always positive gain, no replay cost

        result = consolidate(model, state, evaluate=evaluate, config=_config())

        assert result.committed
        assert result.alpha == 1.0
        assert result.attempted_alphas == (1.0,)
        assert state.get_group("layer1").is_zero()  # fully committed, nothing left

    def test_falls_back_to_a_smaller_alpha_when_larger_ones_fail(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model)
        calls: list[float] = []

        def evaluate():
            # Reject alpha=1.0 and 0.75 via a catastrophic replay violation,
            # accept everything smaller.
            current_alpha = config_line_search[len(calls)]
            calls.append(current_alpha)
            if current_alpha >= 0.75:
                evaluations = [
                    ReplayEvaluation(
                        item_id="canary",
                        loss_after=10.0,
                        reference_loss=0.0,
                        catastrophic_ceiling=1.0,
                        is_canary=True,
                    )
                ]
            else:
                evaluations = []
            return 1.0, evaluations

        config_line_search = (1.0, 0.75, 0.5, 0.25, 0.125)
        result = consolidate(model, state, evaluate=evaluate, config=_config())

        assert result.committed
        assert result.alpha == 0.5
        assert result.attempted_alphas == (1.0, 0.75, 0.5)

    def test_not_committed_when_every_alpha_is_rejected(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model)
        original = state.get_group("layer1").values["layer1.bias"].clone()

        def evaluate():
            return -1.0, []  # negative gain rejects every alpha

        result = consolidate(model, state, evaluate=evaluate, config=_config())

        assert not result.committed
        assert result.alpha == 0.0
        assert torch.equal(state.get_group("layer1").values["layer1.bias"], original)

    def test_group_restriction_only_commits_the_named_group(self) -> None:
        model = _tiny_model()
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": torch.ones_like(model.layer1.bias)})
        )
        state.add_group(
            FastWeightDelta(group="layer2", values={"layer2.bias": torch.ones_like(model.layer2.bias)})
        )

        def evaluate():
            return 1.0, []

        consolidate(model, state, evaluate=evaluate, config=_config(), group="layer1")

        assert state.get_group("layer1").is_zero()
        assert not state.get_group("layer2").is_zero()

    def test_evaluate_sees_the_actually_scaled_model(self) -> None:
        model = _tiny_model()
        before = model.layer1.bias.clone()
        state = _state_with_delta(model, magnitude=8.0)
        observed_values: list[torch.Tensor] = []

        def evaluate():
            observed_values.append(model.layer1.bias.clone())
            return 1.0, []

        consolidate(model, state, evaluate=evaluate, config=_config())

        assert torch.allclose(observed_values[0], before + 8.0)  # alpha=1.0 seen first


class TestConsolidateComponentwise:
    def test_each_group_gets_an_independent_line_search(self) -> None:
        model = _tiny_model()
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": torch.ones_like(model.layer1.bias)})
        )
        state.add_group(
            FastWeightDelta(group="layer2", values={"layer2.bias": torch.ones_like(model.layer2.bias)})
        )

        def evaluate_group(group: str):
            # layer1 always accepted, layer2 always rejected.
            return (1.0, []) if group == "layer1" else (-1.0, [])

        results = consolidate_componentwise(
            model, state, evaluate_group=evaluate_group, config=_config()
        )

        by_group = {item.group: item.result for item in results}
        assert by_group["layer1"].committed
        assert not by_group["layer2"].committed
        assert state.get_group("layer1").is_zero()
        assert not state.get_group("layer2").is_zero()


class TestRunConsolidation:
    def test_joint_success_skips_componentwise(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model)

        def evaluate():
            return 1.0, []

        joint, componentwise = run_consolidation(
            model, state, evaluate=evaluate, evaluate_group=None, config=_config()
        )

        assert joint.committed
        assert componentwise == ()

    def test_joint_failure_falls_back_to_componentwise_success(self) -> None:
        model = _tiny_model()
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": torch.ones_like(model.layer1.bias)})
        )
        state.add_group(
            FastWeightDelta(group="layer2", values={"layer2.bias": torch.ones_like(model.layer2.bias)})
        )

        def evaluate():
            return -1.0, []  # joint always fails

        def evaluate_group(group: str):
            return (1.0, []) if group == "layer1" else (-1.0, [])

        joint, componentwise = run_consolidation(
            model, state, evaluate=evaluate, evaluate_group=evaluate_group, config=_config()
        )

        assert not joint.committed
        assert len(componentwise) == 2
        assert any(item.result.committed for item in componentwise)
        assert state.get_group("layer1").is_zero()

    def test_total_failure_shrinks_delta_fast(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model, magnitude=4.0)
        original = state.get_group("layer1").values["layer1.bias"].clone()

        def evaluate():
            return -1.0, []

        def evaluate_group(group: str):
            return -1.0, []

        run_consolidation(
            model,
            state,
            evaluate=evaluate,
            evaluate_group=evaluate_group,
            config=_config(fast_weight_decay=0.5),
        )

        assert torch.allclose(state.get_group("layer1").values["layer1.bias"], original * 0.5)

    def test_componentwise_disabled_skips_straight_to_shrink(self) -> None:
        model = _tiny_model()
        state = _state_with_delta(model, magnitude=4.0)
        original = state.get_group("layer1").values["layer1.bias"].clone()
        called_group_eval = False

        def evaluate():
            return -1.0, []

        def evaluate_group(group: str):
            nonlocal called_group_eval
            called_group_eval = True
            return 1.0, []

        joint, componentwise = run_consolidation(
            model,
            state,
            evaluate=evaluate,
            evaluate_group=evaluate_group,
            config=_config(allow_componentwise_commit=False, fast_weight_decay=0.5),
        )

        assert not called_group_eval
        assert componentwise == ()
        assert torch.allclose(state.get_group("layer1").values["layer1.bias"], original * 0.5)
