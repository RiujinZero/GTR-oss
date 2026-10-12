"""Tests for the Procoim-S fast-weight container (Phase 1).

Covers Procoim-S design spec unit tests #1-2 (``theta_slow`` remains
unchanged during ordinary fast-weight adaptation; removing ``delta_fast``
exactly recovers the trusted slow model) plus the surrounding container
mechanics (merge, decay, commit, serialization).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gator.optimizers.fast_weights import (
    FastWeightDelta,
    FastWeightState,
    apply_fast_delta,
)
from gator.optimizers.procoim_s import ProcoimSConfig


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("layer1", nn.Linear(4, 3))
    model.add_module("layer2", nn.Linear(3, 2))
    return model


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _assert_matches(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    current = dict(model.named_parameters())
    assert current.keys() == snapshot.keys()
    for name, value in snapshot.items():
        assert torch.equal(current[name].detach(), value), name


class TestFastWeightDelta:
    def test_add_merges_two_deltas(self) -> None:
        a = FastWeightDelta(group="routing", values={"w": torch.tensor([1.0, 2.0])})
        b = FastWeightDelta(group="routing", values={"w": torch.tensor([0.5, -1.0])})
        a.add_(b)
        assert torch.equal(a.values["w"], torch.tensor([1.5, 1.0]))

    def test_add_with_alpha_scales_incoming_delta(self) -> None:
        a = FastWeightDelta(group="routing", values={"w": torch.tensor([1.0])})
        b = FastWeightDelta(group="routing", values={"w": torch.tensor([2.0])})
        a.add_(b, alpha=0.5)
        assert torch.equal(a.values["w"], torch.tensor([2.0]))

    def test_add_rejects_mismatched_group(self) -> None:
        a = FastWeightDelta(group="routing", values={})
        b = FastWeightDelta(group="readout", values={})
        with pytest.raises(ValueError):
            a.add_(b)

    def test_norm_is_euclidean_across_all_tensors(self) -> None:
        delta = FastWeightDelta(
            group="routing",
            values={"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])},
        )
        assert delta.norm() == pytest.approx(5.0)

    def test_is_zero_true_for_empty_and_all_zero_values(self) -> None:
        assert FastWeightDelta(group="routing").is_zero()
        assert FastWeightDelta(
            group="routing", values={"w": torch.zeros(3)}
        ).is_zero()
        assert not FastWeightDelta(
            group="routing", values={"w": torch.tensor([0.0, 1e-3])}
        ).is_zero()

    def test_state_dict_round_trip(self) -> None:
        delta = FastWeightDelta(group="routing", values={"w": torch.tensor([1.0, 2.0])})
        restored = FastWeightDelta.from_state_dict(delta.state_dict())
        assert restored.group == delta.group
        assert torch.equal(restored.values["w"], delta.values["w"])
        restored.values["w"].add_(1.0)
        assert not torch.equal(restored.values["w"], delta.values["w"])


class TestFastWeightState:
    def test_add_group_then_combined_deltas_sums_across_groups(self) -> None:
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="routing", values={"shared": torch.tensor([1.0])})
        )
        state.add_group(
            FastWeightDelta(group="readout", values={"shared": torch.tensor([2.0])})
        )
        combined = state.combined_deltas()
        assert torch.equal(combined["shared"], torch.tensor([3.0]))

    def test_add_group_merges_repeated_calls_for_same_group(self) -> None:
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="routing", values={"w": torch.tensor([1.0])}))
        state.add_group(FastWeightDelta(group="routing", values={"w": torch.tensor([1.0])}))
        assert torch.equal(state.get_group("routing").values["w"], torch.tensor([2.0]))

    def test_is_empty_true_until_a_nonzero_delta_is_added(self) -> None:
        state = FastWeightState()
        assert state.is_empty
        state.add_group(FastWeightDelta(group="routing", values={"w": torch.zeros(2)}))
        assert state.is_empty
        state.add_group(FastWeightDelta(group="routing", values={"w": torch.tensor([0.1, 0.0])}))
        assert not state.is_empty

    def test_decay_uniform_scales_every_group(self) -> None:
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="a", values={"w": torch.tensor([2.0])}))
        state.add_group(FastWeightDelta(group="b", values={"w": torch.tensor([4.0])}))
        state.decay_(0.5)
        assert state.get_group("a").values["w"].item() == pytest.approx(1.0)
        assert state.get_group("b").values["w"].item() == pytest.approx(2.0)

    def test_decay_per_group_mapping_applies_independently(self) -> None:
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="a", values={"w": torch.tensor([2.0])}))
        state.add_group(FastWeightDelta(group="b", values={"w": torch.tensor([2.0])}))
        state.decay_({"a": 0.0, "b": 1.0})
        assert state.get_group("a").values["w"].item() == pytest.approx(0.0)
        assert state.get_group("b").values["w"].item() == pytest.approx(2.0)

    def test_norm_matches_combined_euclidean_norm(self) -> None:
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="a", values={"w": torch.tensor([3.0])}))
        state.add_group(FastWeightDelta(group="b", values={"w": torch.tensor([4.0])}))
        assert state.norm() == pytest.approx(5.0)
        assert state.norm(group="a") == pytest.approx(3.0)
        assert state.norm(group="missing") == 0.0

    def test_state_dict_round_trip(self) -> None:
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="routing", values={"w": torch.tensor([1.0, 2.0])}))
        restored = FastWeightState()
        restored.load_state_dict(state.state_dict())
        assert restored.groups == state.groups
        assert torch.equal(
            restored.get_group("routing").values["w"],
            state.get_group("routing").values["w"],
        )


class TestApplyFastDelta:
    def test_slow_weights_unchanged_after_context_exits(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(
                group="layer1",
                values={"layer1.weight": torch.ones_like(model.layer1.weight)},
            )
        )
        with apply_fast_delta(model, state):
            pass
        _assert_matches(model, before)

    def test_live_model_reflects_theta_slow_plus_delta_fast_inside_context(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        delta_value = torch.full_like(model.layer1.bias, 0.25)
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": delta_value})
        )
        with apply_fast_delta(model, state):
            live_bias = dict(model.named_parameters())["layer1.bias"].detach()
            assert torch.allclose(live_bias, before["layer1.bias"] + delta_value)

    def test_removing_delta_exactly_recovers_slow_model_bit_for_bit(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(
                group="layer2",
                values={
                    "layer2.weight": torch.randn_like(model.layer2.weight) * 1e-3,
                    "layer2.bias": torch.randn_like(model.layer2.bias) * 1e-3,
                },
            )
        )
        with apply_fast_delta(model, state):
            pass
        _assert_matches(model, before)

    def test_theta_slow_restored_even_if_the_context_body_raises(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.weight": torch.ones_like(model.layer1.weight)})
        )
        with pytest.raises(RuntimeError):
            with apply_fast_delta(model, state):
                raise RuntimeError("boom")
        _assert_matches(model, before)

    def test_empty_state_is_a_no_op(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        with apply_fast_delta(model, FastWeightState()):
            _assert_matches(model, before)
        _assert_matches(model, before)

    def test_unknown_parameter_name_raises(self) -> None:
        model = _tiny_model()
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="ghost", values={"does.not.exist": torch.zeros(1)})
        )
        with pytest.raises(KeyError):
            with apply_fast_delta(model, state):
                pass

    def test_shape_mismatch_raises(self) -> None:
        model = _tiny_model()
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.weight": torch.zeros(1)})
        )
        with pytest.raises(ValueError):
            with apply_fast_delta(model, state):
                pass


class TestCommit:
    def test_full_commit_moves_delta_into_model_and_zeroes_it(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        delta_value = torch.full_like(model.layer1.bias, 0.1)
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="layer1", values={"layer1.bias": delta_value}))

        state.commit_(model, alpha=1.0)

        current = dict(model.named_parameters())["layer1.bias"].detach()
        assert torch.allclose(current, before["layer1.bias"] + delta_value)
        assert state.get_group("layer1").is_zero()

    def test_partial_commit_leaves_remainder_in_delta_fast(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        delta_value = torch.full_like(model.layer1.bias, 1.0)
        state = FastWeightState()
        state.add_group(FastWeightDelta(group="layer1", values={"layer1.bias": delta_value}))

        state.commit_(model, alpha=0.25)

        current = dict(model.named_parameters())["layer1.bias"].detach()
        assert torch.allclose(current, before["layer1.bias"] + 0.25 * delta_value)
        assert torch.allclose(state.get_group("layer1").values["layer1.bias"], 0.75 * delta_value)

    def test_zero_alpha_commit_is_a_no_op(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": torch.ones_like(model.layer1.bias)})
        )
        state.commit_(model, alpha=0.0)
        _assert_matches(model, before)

    def test_commit_rejects_out_of_range_alpha(self) -> None:
        model = _tiny_model()
        state = FastWeightState()
        with pytest.raises(ValueError):
            state.commit_(model, alpha=1.5)

    def test_commit_restricted_to_one_group_leaves_others_untouched(self) -> None:
        model = _tiny_model()
        before = _snapshot(model)
        state = FastWeightState()
        state.add_group(
            FastWeightDelta(group="layer1", values={"layer1.bias": torch.ones_like(model.layer1.bias)})
        )
        state.add_group(
            FastWeightDelta(group="layer2", values={"layer2.bias": torch.ones_like(model.layer2.bias)})
        )
        state.commit_(model, alpha=1.0, group="layer1")

        current = dict(model.named_parameters())
        assert torch.allclose(current["layer1.bias"].detach(), before["layer1.bias"] + 1.0)
        assert torch.equal(current["layer2.bias"].detach(), before["layer2.bias"])
        assert state.get_group("layer1").is_zero()
        assert not state.get_group("layer2").is_zero()


class TestProcoimSConfig:
    def test_defaults_construct_without_error(self) -> None:
        config = ProcoimSConfig()
        assert config.enabled is False

    def test_direction_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            ProcoimSConfig(
                anchor_direction_fraction=0.5,
                recent_direction_fraction=0.5,
                exploration_direction_fraction=0.5,
            )

    def test_invalid_probe_routing_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProcoimSConfig(probe_routing_mode="quantum")

    def test_maximum_below_minimum_candidate_directions_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProcoimSConfig(minimum_candidate_directions=64, maximum_candidate_directions=16)

    def test_empty_consolidation_line_search_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProcoimSConfig(consolidation_line_search=())

    def test_consolidation_line_search_values_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            ProcoimSConfig(consolidation_line_search=(1.0, 1.5))
