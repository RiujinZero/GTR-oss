from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch
from torch import nn

from gator.optimizers.procoim import (
    CandidateResult,
    ProcoimAnchor,
    ProcoimConfig,
    ProcoimOptimizer,
    SlotStatus,
    _CandidateBankResult,
)
from gator.optimizers.slot_loss import SlotLossResult
from gator.training import ProcoimRetentionWindow, ProcoimTrainingEngine


def _result(losses: torch.Tensor, valid: torch.Tensor | None = None) -> SlotLossResult:
    detached = losses.detach().clone()
    mask = (
        torch.ones_like(detached, dtype=torch.bool)
        if valid is None
        else valid.detach().clone()
    )
    detached = torch.where(mask, detached, torch.zeros_like(detached))
    return SlotLossResult(
        detached,
        mask,
        detached[mask].mean(),
    ).validate()


class _ScalarSlotModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float64))
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_counter.add_(1)
        return self.value.expand(inputs.shape[0], 1)


class _MixedModeSlotModel(_ScalarSlotModel):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__(value)
        self.mode_tree = nn.Sequential(
            nn.Dropout(0.5),
            nn.Sequential(nn.ReLU()),
        )


class _MultiTensorGroupModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small = nn.Parameter(torch.ones(4, dtype=torch.float64))
        self.large = nn.Parameter(
            torch.full((16,), 2.0, dtype=torch.float64)
        )

    def procoim_parameter_groups(self):
        return {
            "combined": [self.small, self.large],
        }


class _MatrixGroupModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(
            torch.ones((6, 5), dtype=torch.float64)
        )
        self.vector = nn.Parameter(torch.ones(7, dtype=torch.float64))

    def procoim_parameter_groups(self):
        return {
            "combined": [self.matrix, self.vector],
        }


class _TwoGroupSlotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.second = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.first + self.second).expand(inputs.shape[0], 1)

    def procoim_parameter_groups(self):
        return {
            "first": [self.first],
            "second": [self.second],
        }


class _VectorSlotModel(nn.Module):
    def __init__(self, slots: int = 2) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(slots, dtype=torch.float64))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value.expand(inputs.shape[0], self.value.numel())


class _SelectiveTwoGroupModel(_TwoGroupSlotModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second.expand(inputs.shape[0], 1)


class _WeightedTwoGroupModel(_TwoGroupSlotModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (2.0 * self.first + 0.5 * self.second).expand(
            inputs.shape[0], 1
        )


def _scalar_provider(model, batch, forward_kwargs) -> SlotLossResult:
    del forward_kwargs
    inputs, target = batch
    value = model(inputs)
    return _result((value - target).square())


def _constant_provider(model, batch, forward_kwargs) -> SlotLossResult:
    del batch, forward_kwargs
    model(torch.zeros(1, dtype=torch.float64))
    return _result(torch.ones(1, 1, dtype=torch.float64))


class ProcoimStateTests(unittest.TestCase):
    def test_default_reduction_is_half_percent(self) -> None:
        config = ProcoimConfig()
        self.assertEqual(config.reduction_fraction, 0.005)
        self.assertEqual(config.candidate_directions, 32)
        self.assertEqual(config.max_candidate_banks, 16)
        self.assertEqual(config.max_accepted_updates, 64)
        self.assertTrue(config.verify_accepted_candidate)
        self.assertEqual(config.stability_required, 3)
        self.assertEqual(config.probe_scale, 0.03)
        self.assertFalse(config.reuse_successful_directions)
        self.assertFalse(config.use_antithetic_estimate)
        self.assertFalse(config.antithetic_estimate_only)
        self.assertFalse(config.antithetic_estimate_anchor_aware)
        self.assertFalse(config.antithetic_slot_estimates)
        self.assertIsNone(config.antithetic_estimate_step_scale)
        self.assertFalse(config.directional_adam)
        self.assertEqual(config.directional_adam_learning_rate, 0.03)
        self.assertEqual(config.directional_adam_beta1, 0.9)
        self.assertEqual(config.directional_adam_beta2, 0.99)
        self.assertEqual(config.directional_adam_weight_decay, 0.0)
        self.assertEqual(
            config.directional_basis_mode, "rotating_factorized"
        )
        self.assertEqual(config.directional_factor_rank, 1)
        self.assertEqual(config.directional_basis_rotation_banks, 8)
        self.assertEqual(config.directional_accumulation_batches, 1)
        self.assertIsNone(config.reference_reduction_fraction)
        self.assertEqual(
            config.candidate_backtrack_factors, (1.0, 0.5, 0.25, 0.125)
        )
        self.assertFalse(config.adaptive_probe_scale)
        self.assertEqual(config.parameter_group_granularity, "semantic")
        self.assertFalse(config.adaptive_group_selection)
        self.assertEqual(config.slot_objective_reduction, "per_example")
        self.assertEqual(config.anchor_soft_regression_fraction, 0.03)
        self.assertEqual(config.anchor_hard_regression_fraction, 0.10)
        self.assertIsNone(
            config.anchor_minimum_retained_reduction_fraction
        )

    def test_disabled_adaptive_scale_does_not_restrict_legacy_probe(self) -> None:
        config = ProcoimConfig(probe_scale=0.2)
        self.assertFalse(config.adaptive_probe_scale)
        with self.assertRaisesRegex(ValueError, "maximum_probe_scale"):
            ProcoimConfig(
                probe_scale=0.2,
                adaptive_probe_scale=True,
            )

    def test_disabled_adaptive_reduction_fraction_does_not_restrict_minimum(
        self,
    ) -> None:
        config = ProcoimConfig(reduction_fraction=0.0)
        self.assertFalse(config.adaptive_reduction_fraction)
        with self.assertRaisesRegex(
            ValueError, "minimum_reduction_fraction"
        ):
            ProcoimConfig(
                reduction_fraction=0.0,
                adaptive_reduction_fraction=True,
            )

    def test_effective_reduction_fraction_decays_logarithmically(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.01,
                adaptive_reduction_fraction=True,
                reduction_fraction_decay_rate=0.5,
                minimum_reduction_fraction=1e-6,
                parameter_group_schedule=("all_parameters",),
            )
        )
        # No completed episodes yet: the bar is the full requested value.
        self.assertAlmostEqual(
            optimizer._effective_reduction_fraction(), 0.01
        )
        previous = optimizer._effective_reduction_fraction()
        for episodes in (1, 5, 20, 100):
            optimizer.episodes = episodes
            current = optimizer._effective_reduction_fraction()
            self.assertLess(current, previous)
            self.assertGreaterEqual(
                current, optimizer.config.minimum_reduction_fraction
            )
            previous = current

    def test_fixed_reduction_fraction_does_not_decay_across_episodes(
        self,
    ) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.01,
                parameter_group_schedule=("all_parameters",),
            )
        )
        optimizer.episodes = 50
        self.assertEqual(
            optimizer._effective_reduction_fraction(), 0.01
        )

    def test_fixed_probe_scale_does_not_decay_across_stalls(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.005,
                candidate_directions=1,
                max_candidate_banks=2,
                minimum_active_gain=1.0,
                probe_scale=0.03,
                probe_scale_decay=0.1,
                parameter_group_schedule=("all_parameters",),
            )
        )
        result = optimizer.run_episode(
            _ScalarSlotModel(0.0),
            (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
            _constant_provider,
        )

        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.current_probe_scale, 0.03)
        self.assertEqual(
            result.parameter_group_probe_scales["all_parameters"],
            0.03,
        )

    def test_active_reaching_target_becomes_soft_frozen(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(reduction_fraction=0.10, stability_required=3)
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]]))
        )
        optimizer.update_slot_states(state, torch.tensor([[0.89]]))

        self.assertEqual(state.count(SlotStatus.ACTIVE), 0)
        self.assertEqual(state.count(SlotStatus.SOFT_FROZEN), 1)
        self.assertEqual(int(state.stable_count[0, 0]), 1)
        self.assertAlmostEqual(float(state.best_loss[0, 0]), 0.89, places=6)

    def test_stability_streak_promotes_to_protected(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(reduction_fraction=0.10, stability_required=3)
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]]))
        )
        optimizer.update_slot_states(state, torch.tensor([[0.89]]))
        optimizer.update_slot_states(state, torch.tensor([[0.895]]))
        self.assertEqual(state.count(SlotStatus.SOFT_FROZEN), 1)
        self.assertEqual(int(state.stable_count[0, 0]), 2)
        optimizer.update_slot_states(state, torch.tensor([[0.892]]))
        self.assertEqual(state.count(SlotStatus.PROTECTED), 1)
        self.assertEqual(int(state.stable_count[0, 0]), 3)
        self.assertAlmostEqual(float(state.best_loss[0, 0]), 0.89, places=6)

    def test_regressed_soft_slot_reactivates(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(reduction_fraction=0.10, stability_required=3)
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]]))
        )
        optimizer.update_slot_states(state, torch.tensor([[0.89]]))
        optimizer.update_slot_states(state, torch.tensor([[0.95]]))
        self.assertEqual(state.count(SlotStatus.ACTIVE), 1)
        self.assertEqual(int(state.stable_count[0, 0]), 0)

    def test_hard_protection_violation_is_ineligible(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.10,
                minimum_active_gain=0.0,
                absolute_regression_tolerance=1e-4,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0, 0.5]]))
        )
        state.status[0, 1] = int(SlotStatus.PROTECTED)
        state.best_loss[0, 1] = 0.25
        state.current_loss[0, 1] = 0.25
        before = state.clone()
        scored = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.5, 0.5]])),
            group_name="test",
            direction_index=0,
            sign=1,
        )
        self.assertTrue(scored.hard_violation)
        self.assertFalse(scored.eligible)
        self.assertTrue(torch.equal(state.status, before.status))
        self.assertTrue(torch.equal(state.best_loss, before.best_loss))

    def test_candidate_must_meet_overall_fractional_reduction(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.005,
                minimum_active_gain=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[100.0]]))
        )

        below = optimizer.score_candidate(
            state,
            _result(torch.tensor([[99.6]])),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        clears = optimizer.score_candidate(
            state,
            _result(torch.tensor([[99.4]])),
            group_name="all_parameters",
            direction_index=1,
            sign=1,
        )

        self.assertGreater(below.active_gain, 0.0)
        self.assertFalse(below.eligible)
        self.assertTrue(clears.eligible)

    def test_protected_anchor_prevents_cumulative_drift(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.10,
                minimum_active_gain=0.0,
                absolute_regression_tolerance=0.0,
                soft_regression_fraction=0.02,
                hard_regression_fraction=0.08,
                protection_weight=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0, 1.0, 0.5]]))
        )
        state.status[0, 2] = int(SlotStatus.PROTECTED)
        state.best_loss[0, 2] = 0.25
        state.current_loss[0, 2] = 0.25
        hard_ceiling = 0.25 + 0.08 * (0.5 - 0.25)

        for protected_loss in (0.271, 0.28):
            scored = optimizer.score_candidate(
                state,
                _result(torch.tensor([[0.8, 1.0, protected_loss]])),
                group_name="test",
                direction_index=0,
                sign=1,
            )
            self.assertFalse(scored.eligible)
            self.assertTrue(scored.hard_violation)
            self.assertEqual(float(state.best_loss[0, 2]), 0.25)

        accepted = (
            torch.tensor([[0.8, 1.0, 0.26]]),
            torch.tensor([[0.8, 0.89, hard_ceiling]]),
        )
        for losses in accepted:
            scored = optimizer.score_candidate(
                state,
                _result(losses),
                group_name="test",
                direction_index=0,
                sign=1,
            )
            self.assertTrue(scored.eligible)
            optimizer.update_slot_states(state, scored.losses)
            self.assertLessEqual(
                float(state.current_loss[0, 2]),
                hard_ceiling + 1e-7,
            )
            self.assertEqual(float(state.best_loss[0, 2]), 0.25)

    def test_padded_slots_never_transition_or_update(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(reduction_fraction=0.10, stability_required=1)
        )
        valid = torch.tensor([[True, False]])
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0, 100.0]]), valid)
        )
        invalid_before = state.current_loss[0, 1].clone()
        optimizer.update_slot_states(state, torch.tensor([[0.8, 0.0]]))
        self.assertEqual(int(state.status[0, 1]), int(SlotStatus.ACTIVE))
        self.assertEqual(int(state.stable_count[0, 1]), 0)
        self.assertTrue(torch.equal(state.current_loss[0, 1], invalid_before))

    def test_episode_anchor_never_writes_empty_or_unsolved_slots(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(reduction_fraction=0.1, stability_required=3)
        )
        valid = torch.tensor([[True, False]])
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0, 0.0]]), valid)
        )
        optimizer.update_slot_states(state, torch.tensor([[0.8, 99.0]]))

        anchor = ProcoimAnchor.from_episode(
            batch=None,
            state=state,
            anchor_id="masked",
        )

        self.assertTrue(anchor.slotwise)
        self.assertTrue(bool(anchor.protected_mask[0, 0]))
        self.assertFalse(bool(anchor.protected_mask[0, 1]))
        self.assertFalse(bool(anchor.hard_mask[0, 0]))
        self.assertFalse(bool(anchor.hard_mask[0, 1]))
        self.assertFalse(bool(anchor.valid_mask[0, 1]))

    def test_soft_frozen_anchor_penalizes_without_hard_veto(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.1,
                minimum_active_gain=0.0,
                anchor_soft_regression_fraction=0.0,
                anchor_hard_regression_fraction=0.1,
                anchor_absolute_tolerance=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]]))
        )
        current = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.8]])),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        anchor = ProcoimAnchor(
            batch=None,
            start_losses=torch.tensor([[2.0]]),
            best_losses=torch.tensor([[1.0]]),
            valid_mask=torch.tensor([[True]]),
            protected_mask=torch.tensor([[True]]),
            hard_mask=torch.tensor([[False]]),
        )
        replay = _result(torch.tensor([[1.2]]))

        soft = optimizer._score_anchor_regression(
            current,
            (anchor,),
            (replay,),
        )
        self.assertGreater(soft.anchor_penalty, 0.0)
        self.assertFalse(soft.anchor_violation)

        anchor.hard_mask.fill_(True)
        hard = optimizer._score_anchor_regression(
            current,
            (anchor,),
            (replay,),
        )
        self.assertTrue(hard.anchor_violation)

    def test_anchor_specific_zero_tolerance_guards_distribution_loss(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.1,
                minimum_active_gain=0.0,
                anchor_regression_fraction=0.5,
                anchor_absolute_tolerance=1.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]]))
        )
        current = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.8]])),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        sentinel = ProcoimAnchor(
            batch=None,
            best_aggregate=1.0,
            soft_regression_fraction=0.0,
            hard_regression_fraction=0.0,
            absolute_tolerance=0.0,
        )

        regressed = optimizer._score_anchor_regression(
            current,
            (sentinel,),
            (_result(torch.tensor([[1.001]])),),
        )
        preserved = optimizer._score_anchor_regression(
            current,
            (sentinel,),
            (_result(torch.tensor([[0.999]])),),
        )

        self.assertTrue(regressed.anchor_violation)
        self.assertFalse(regressed.eligible)
        self.assertFalse(preserved.anchor_violation)
        self.assertTrue(preserved.eligible)

    def test_slotwise_anchor_uses_saved_gain_for_hard_tolerance(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.1,
                minimum_active_gain=0.0,
                anchor_soft_regression_fraction=0.03,
                anchor_hard_regression_fraction=0.10,
                anchor_absolute_tolerance=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]]))
        )
        current_candidate = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.8]])),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        valid = torch.tensor([[True, False]])
        anchor = ProcoimAnchor(
            batch=None,
            start_losses=torch.tensor([[1.0, 3.0]]),
            best_losses=torch.tensor([[0.5, 3.0]]),
            valid_mask=valid,
            protected_mask=torch.tensor([[True, False]]),
        )
        # Saved gain is 0.5, so the hard ceiling is 0.55. The invalid
        # second slot is intentionally enormous and must not participate.
        scored = optimizer._score_anchor_regression(
            current_candidate,
            (anchor,),
            (_result(torch.tensor([[0.551, 999.0]]), valid),),
        )

        self.assertTrue(scored.anchor_violation)
        self.assertFalse(scored.eligible)

    def test_anchor_audit_floor_retains_half_of_requested_progress(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.0001,
                minimum_active_gain=0.0,
                anchor_hard_regression_fraction=0.10,
                anchor_minimum_retained_reduction_fraction=0.00005,
                anchor_absolute_tolerance=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]], dtype=torch.float64))
        )
        current = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.9]], dtype=torch.float64)),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        anchor = ProcoimAnchor(
            batch=None,
            start_losses=torch.tensor([[1.0]], dtype=torch.float64),
            best_losses=torch.tensor([[0.9]], dtype=torch.float64),
            valid_mask=torch.tensor([[True]]),
            protected_mask=torch.tensor([[True]]),
            hard_mask=torch.tensor([[True]]),
        )

        preserved = optimizer._score_anchor_regression(
            current,
            (anchor,),
            (_result(torch.tensor([[0.99994]], dtype=torch.float64)),),
            audit=True,
        )
        erased = optimizer._score_anchor_regression(
            current,
            (anchor,),
            (_result(torch.tensor([[0.99996]], dtype=torch.float64)),),
            audit=True,
        )

        self.assertFalse(preserved.anchor_violation)
        self.assertTrue(preserved.eligible)
        self.assertTrue(erased.anchor_violation)
        self.assertFalse(erased.eligible)

    def test_anchor_audit_floor_is_bank_level_with_catastrophe_bound(
        self,
    ) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.0001,
                minimum_active_gain=0.0,
                anchor_hard_regression_fraction=0.10,
                anchor_minimum_retained_reduction_fraction=0.00005,
                anchor_absolute_tolerance=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0]], dtype=torch.float64))
        )
        current = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.9]], dtype=torch.float64)),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        anchors = tuple(
            ProcoimAnchor(
                batch=None,
                start_losses=torch.tensor([[1.0]], dtype=torch.float64),
                best_losses=torch.tensor([[0.9]], dtype=torch.float64),
                valid_mask=torch.tensor([[True]]),
                protected_mask=torch.tensor([[True]]),
                hard_mask=torch.tensor([[True]]),
            )
            for _ in range(2)
        )

        redistributed = optimizer._score_anchor_regression(
            current,
            anchors,
            (
                _result(torch.tensor([[1.00004]], dtype=torch.float64)),
                _result(torch.tensor([[0.99984]], dtype=torch.float64)),
            ),
            audit=True,
        )
        insufficient = optimizer._score_anchor_regression(
            current,
            anchors,
            (
                _result(torch.tensor([[0.99996]], dtype=torch.float64)),
                _result(torch.tensor([[0.99996]], dtype=torch.float64)),
            ),
            audit=True,
        )
        catastrophic = optimizer._score_anchor_regression(
            current,
            anchors,
            (
                _result(torch.tensor([[1.101]], dtype=torch.float64)),
                _result(torch.tensor([[0.8]], dtype=torch.float64)),
            ),
            audit=True,
        )

        self.assertAlmostEqual(
            redistributed.anchor_retained_reduction,
            0.00006,
        )
        self.assertFalse(redistributed.anchor_violation)
        self.assertTrue(redistributed.eligible)
        self.assertTrue(insufficient.anchor_violation)
        self.assertTrue(catastrophic.anchor_violation)

    def test_anchor_audit_floor_cannot_exceed_search_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "anchor_minimum_retained_reduction_fraction",
        ):
            ProcoimConfig(
                reduction_fraction=0.0001,
                anchor_minimum_retained_reduction_fraction=0.0002,
            )

    def test_antithetic_coefficient_uses_smooth_active_loss(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.005,
                protection_weight=0.0,
                global_regression_weight=0.0,
                anchor_weight=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.ones(1, 2, dtype=torch.float64))
        )
        positive = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.8, 1.4]], dtype=torch.float64)),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        negative = optimizer.score_candidate(
            state,
            _result(torch.tensor([[1.2, 0.9]], dtype=torch.float64)),
            group_name="all_parameters",
            direction_index=0,
            sign=-1,
        )

        # Mean positive loss is 1.10 and mean negative loss is 1.05.
        # The clipped target-gap utility is highly saturated at this probe
        # radius, but the finite-difference coefficient remains the smooth
        # raw-loss difference divided by the current mean loss.
        self.assertAlmostEqual(
            optimizer._antithetic_smooth_coefficient(
                state, positive, negative
            ),
            -0.05,
        )

    def test_antithetic_coefficient_can_target_one_physical_slot(self) -> None:
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.005,
                protection_weight=0.0,
                global_regression_weight=0.0,
                anchor_weight=0.0,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.ones(1, 2, dtype=torch.float64))
        )
        positive = optimizer.score_candidate(
            state,
            _result(torch.tensor([[0.8, 1.4]], dtype=torch.float64)),
            group_name="all_parameters",
            direction_index=0,
            sign=1,
        )
        negative = optimizer.score_candidate(
            state,
            _result(torch.tensor([[1.2, 0.9]], dtype=torch.float64)),
            group_name="all_parameters",
            direction_index=0,
            sign=-1,
        )
        slot_zero = torch.tensor([[True, False]])
        slot_one = torch.tensor([[False, True]])

        self.assertAlmostEqual(
            optimizer._antithetic_smooth_coefficient(
                state, positive, negative, active_mask=slot_zero
            ),
            0.4,
        )
        self.assertAlmostEqual(
            optimizer._antithetic_smooth_coefficient(
                state, positive, negative, active_mask=slot_one
            ),
            -0.5,
        )


class ProcoimTransactionTests(unittest.TestCase):
    def _config(self, **overrides) -> ProcoimConfig:
        values = {
            "reduction_fraction": 0.1,
            "candidate_directions": 1,
            "max_candidate_banks": 1,
            "max_accepted_updates": 1,
            "stability_required": 1,
            "verify_accepted_candidate": False,
            "minimum_active_gain": 0.0,
            "probe_scale": 0.1,
            "minimum_probe_scale": 0.1,
            "parameter_rms_floor": 1.0,
            "parameter_group_schedule": ("all_parameters",),
            "seed": 7,
        }
        values.update(overrides)
        return ProcoimConfig(**values)

    def test_rademacher_directions_are_normalized_per_tensor(self) -> None:
        model = _MultiTensorGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                parameter_group_schedule=("combined",),
                parameter_rms_floor=0.0,
            )
        )
        group = optimizer.select_parameter_groups(model)[0]
        snapshot = optimizer._snapshot_group(group)

        direction = optimizer.sample_direction(
            group,
            snapshot,
            probe_scale=0.1,
        )

        # Each tensor receives its own parameter-RMS-relative L2 radius,
        # independent of its element count.
        self.assertAlmostEqual(float(direction[0].norm()), 0.1)
        self.assertAlmostEqual(float(direction[1].norm()), 0.2)

    def test_candidates_restore_exact_base_and_apply_only_winner(self) -> None:
        model = _ScalarSlotModel(0.0)
        seen: list[float] = []

        def provider(candidate, batch, kwargs):
            seen.append(float(candidate.value))
            return _scalar_provider(candidate, batch, kwargs)

        optimizer = ProcoimOptimizer(self._config(candidate_directions=4))
        result = optimizer.run_episode(
            model,
            (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
            provider,
        )

        self.assertEqual(seen, [0.0, 0.1])
        self.assertEqual(float(model.value.detach()), 0.1)
        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.total_forward_calls, 2)
        self.assertEqual(model.forward_counter.item(), 0)

    def test_rejected_bank_restores_parameters_buffers_and_mode_exactly(self) -> None:
        model = _ScalarSlotModel(0.25)
        model.train()
        parameter_before = model.value.detach().clone()
        buffer_before = model.forward_counter.detach().clone()
        optimizer = ProcoimOptimizer(
            self._config(minimum_active_gain=1.0)
        )
        result = optimizer.run_episode(
            model,
            (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
            _constant_provider,
        )
        self.assertEqual(result.accepted_updates, 0)
        self.assertTrue(torch.equal(model.value, parameter_before))
        self.assertTrue(torch.equal(model.forward_counter, buffer_before))
        self.assertTrue(model.training)

    def test_mixed_child_training_modes_are_restored_exactly(self) -> None:
        model = _MixedModeSlotModel(0.25)
        model.train()
        model.mode_tree.eval()
        model.mode_tree[1].train()
        model.mode_tree[1][0].eval()
        modes_before = [module.training for module in model.modules()]

        optimizer = ProcoimOptimizer(self._config(minimum_active_gain=1.0))
        optimizer.run_episode(
            model,
            (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
            _constant_provider,
        )

        self.assertEqual(
            [module.training for module in model.modules()],
            modes_before,
        )

    def test_max_forward_calls_respects_baseline_and_antithetic_boundary(self) -> None:
        expected = {
            1: (1, 0),
            2: (1, 0),
            3: (3, 2),
        }
        for budget, (total_forwards, candidate_forwards) in expected.items():
            with self.subTest(max_forward_calls=budget):
                model = _ScalarSlotModel(0.25)
                optimizer = ProcoimOptimizer(
                    self._config(minimum_active_gain=1.0)
                )
                result = optimizer.run_episode(
                    model,
                    (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
                    _constant_provider,
                    max_forward_calls=budget,
                )
                self.assertEqual(result.total_forward_calls, total_forwards)
                self.assertEqual(result.candidate_forwards, candidate_forwards)
                self.assertLessEqual(result.total_forward_calls, budget)
                self.assertEqual(
                    optimizer.total_forward_calls,
                    result.total_forward_calls,
                )

    def test_directional_accumulation_averages_fresh_batch_objectives(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.05,
                antithetic=False,
                slot_objective_reduction="batch_mean",
                directional_accumulation_batches=2,
            )
        )
        direction = (torch.tensor(0.1, dtype=torch.float64),)
        with mock.patch.object(
            optimizer,
            "sample_direction",
            return_value=direction,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                _scalar_provider,
                accumulation_batches=(
                    (
                        torch.zeros(1, dtype=torch.float64),
                        torch.full((1, 1), 3.0, dtype=torch.float64),
                    ),
                ),
            )

        self.assertAlmostEqual(result.start_aggregate_loss, 5.0)
        self.assertAlmostEqual(result.end_aggregate_loss, 4.61)
        self.assertEqual(result.baseline_forwards, 2)
        self.assertEqual(result.candidate_forwards, 2)
        self.assertEqual(result.total_forward_calls, 4)
        self.assertEqual(result.accepted_updates, 1)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)

    def test_reference_guard_rejects_local_gain_that_harms_reference(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(antithetic=False)
        )
        direction = (torch.tensor(0.1, dtype=torch.float64),)
        with mock.patch.object(
            optimizer,
            "sample_direction",
            return_value=direction,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                _scalar_provider,
                reference_batch=(
                    torch.zeros(1, dtype=torch.float64),
                    -torch.ones(1, 1, dtype=torch.float64),
                ),
            )

        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.reference_forwards, 2)
        self.assertEqual(result.reference_vetoes, 1)
        self.assertEqual(result.total_forward_calls, 4)
        self.assertAlmostEqual(result.reference_start_loss, 1.0)
        self.assertAlmostEqual(result.reference_end_loss, 1.0)
        self.assertAlmostEqual(float(model.value.detach()), 0.0)

    def test_reference_guard_accepts_shared_fractional_gain(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(antithetic=False)
        )
        direction = (torch.tensor(0.1, dtype=torch.float64),)
        with mock.patch.object(
            optimizer,
            "sample_direction",
            return_value=direction,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                _scalar_provider,
                reference_batch=(
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
            )

        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.reference_forwards, 2)
        self.assertEqual(result.reference_vetoes, 0)
        self.assertEqual(result.total_forward_calls, 4)
        self.assertAlmostEqual(result.reference_start_loss, 1.0)
        self.assertAlmostEqual(result.reference_end_loss, 0.81)
        self.assertAlmostEqual(result.reference_reduction, 0.19)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)

    def test_reference_gate_can_allow_bounded_partner_regression(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(antithetic=False)
        )
        direction = (torch.tensor(0.1, dtype=torch.float64),)
        with mock.patch.object(
            optimizer,
            "sample_direction",
            return_value=direction,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                _scalar_provider,
                reference_batch=(
                    torch.zeros(1, dtype=torch.float64),
                    torch.full((1, 1), -100.0, dtype=torch.float64),
                ),
                reference_reduction_fraction=-0.01,
            )

        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.reference_vetoes, 0)
        self.assertLess(result.reference_reduction, 0.0)
        self.assertGreater(result.reference_reduction, -0.01)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)

    def test_reference_partner_rotates_without_resetting_episode_state(
        self,
    ) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(
                antithetic=False,
                max_candidate_banks=3,
            )
        )
        partners = iter(
            (
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.full((1, 1), -2.0, dtype=torch.float64),
                ),
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.full((1, 1), -3.0, dtype=torch.float64),
                ),
            )
        )
        provider = mock.Mock(side_effect=lambda: next(partners))
        direction = (torch.tensor(0.1, dtype=torch.float64),)
        with mock.patch.object(
            optimizer,
            "sample_direction",
            return_value=direction,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                _scalar_provider,
                reference_batch=(
                    torch.zeros(1, dtype=torch.float64),
                    -torch.ones(1, 1, dtype=torch.float64),
                ),
                reference_batch_provider=provider,
                reference_rotation_forwards=2,
            )

        self.assertEqual(provider.call_count, 2)
        self.assertEqual(result.reference_rotations, 2)
        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.candidate_forwards, 3)
        self.assertEqual(result.reference_forwards, 6)
        self.assertEqual(result.total_forward_calls, 10)
        self.assertEqual(result.completion_reason, "candidate_bank_budget")
        self.assertAlmostEqual(result.reference_start_loss, 9.0)
        self.assertAlmostEqual(result.reference_end_loss, 9.0)
        self.assertAlmostEqual(float(model.value.detach()), 0.0)

    def test_reference_rotation_can_replace_incompatible_partner(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(
                antithetic=False,
                max_candidate_banks=2,
            )
        )
        provider = mock.Mock(
            return_value=(
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            )
        )
        direction = (torch.tensor(0.1, dtype=torch.float64),)
        with mock.patch.object(
            optimizer,
            "sample_direction",
            return_value=direction,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                _scalar_provider,
                reference_batch=(
                    torch.zeros(1, dtype=torch.float64),
                    -torch.ones(1, 1, dtype=torch.float64),
                ),
                reference_batch_provider=provider,
                reference_rotation_forwards=2,
            )

        provider.assert_called_once_with()
        self.assertEqual(result.reference_rotations, 1)
        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.reference_vetoes, 1)
        self.assertAlmostEqual(result.reference_start_loss, 1.0)
        self.assertAlmostEqual(result.reference_end_loss, 0.81)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)

    def test_reference_rotation_arguments_are_paired(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(self._config())
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires both a provider",
        ):
            optimizer.run_episode(
                model,
                batch,
                _scalar_provider,
                reference_batch=batch,
                reference_rotation_forwards=1024,
            )
    def test_directional_accumulation_requires_batch_mean_objective(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "directional accumulation requires batch_mean",
        ):
            self._config(directional_accumulation_batches=2)

    def test_minimum_episode_budget_covers_one_atomic_candidate_bank(self) -> None:
        direct = ProcoimOptimizer(self._config())
        self.assertEqual(direct.minimum_episode_forward_calls(), 3)

        directional = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                directional_adam=True,
                verify_accepted_candidate=True,
                slot_objective_reduction="batch_mean",
                directional_accumulation_batches=2,
            )
        )
        self.assertEqual(
            directional.minimum_episode_forward_calls(
                anchor_count=3,
                reference_enabled=True,
            ),
            19,
        )

    def test_terminal_stability_mask_change_is_rejected_before_mutation(self) -> None:
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.0,
                stability_required=3,
            )
        )
        baseline = _result(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[True, False]]),
        )
        changed = _result(
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            torch.tensor([[False, True]]),
        )
        state = optimizer.initialize_episode_state(baseline)
        before = state.clone()
        model = _ScalarSlotModel(0.0)

        with (
            mock.patch.object(optimizer, "_evaluate", return_value=changed),
            self.assertRaisesRegex(ValueError, "stability valid_mask changed"),
        ):
            optimizer._terminal_stability(
                model,
                None,
                _constant_provider,
                {},
                state,
                search_rng=mock.sentinel.search_rng,
                buffer_snapshot={},
            )

        self.assertTrue(torch.equal(state.current_loss, before.current_loss))
        self.assertTrue(torch.equal(state.best_loss, before.best_loss))
        self.assertTrue(torch.equal(state.status, before.status))
        self.assertTrue(torch.equal(state.stable_count, before.stable_count))

    def test_loss_provider_parameter_mutation_is_detected(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(self._config())

        def mutating_provider(candidate, batch, forward_kwargs):
            del batch, forward_kwargs
            candidate.value.add_(0.25)
            return _result(torch.ones(1, 1, dtype=torch.float64))

        with self.assertRaisesRegex(
            RuntimeError,
            "loss_provider mutated model parameters",
        ):
            optimizer.run_episode(
                model,
                (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
                mutating_provider,
            )

    def test_budget_completion_mode_reports_completion(self) -> None:
        model = _ScalarSlotModel(0.25)
        optimizer = ProcoimOptimizer(
            self._config(
                completion_mode="budget",
                minimum_active_gain=1.0,
            )
        )
        result = optimizer.run_episode(
            model,
            (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
            _constant_provider,
        )
        self.assertTrue(result.completed)
        self.assertEqual(result.completion_reason, "budget")

    def test_no_grad_or_backward_is_used(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(self._config())
        with (
            mock.patch.object(
                torch.Tensor,
                "backward",
                side_effect=AssertionError("backward called"),
            ),
            mock.patch(
                "torch.autograd.backward",
                side_effect=AssertionError("autograd.backward called"),
            ),
        ):
            optimizer.run_episode(
                model,
                (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
                _scalar_provider,
            )
        self.assertIsNone(model.value.grad)

    def test_fixed_seed_is_bitwise_deterministic(self) -> None:
        first = _ScalarSlotModel(0.0)
        second = copy.deepcopy(first)
        first_optimizer = ProcoimOptimizer(self._config(seed=123))
        second_optimizer = ProcoimOptimizer(self._config(seed=123))
        batch = (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1))

        first_result = first_optimizer.run_episode(
            first, batch, _scalar_provider
        )
        second_result = second_optimizer.run_episode(
            second, batch, _scalar_provider
        )

        self.assertTrue(torch.equal(first.value, second.value))
        self.assertTrue(
            torch.equal(first_result.state.status, second_result.state.status)
        )
        self.assertTrue(
            torch.equal(
                first_result.state.best_loss,
                second_result.state.best_loss,
            )
        )
        self.assertEqual(
            first_result.accepted_updates,
            second_result.accepted_updates,
        )
        self.assertEqual(
            first_result.candidate_forwards,
            second_result.candidate_forwards,
        )

    def test_successful_banks_reset_consecutive_stall_budget(self) -> None:
        model = _VectorSlotModel(slots=3)
        optimizer = ProcoimOptimizer(
            self._config(
                max_candidate_banks=1,
                max_accepted_updates=3,
            )
        )
        directions = [
            (torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.2, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64),),
        ]
        with mock.patch.object(
            optimizer,
            "sample_direction",
            side_effect=directions,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 3, dtype=torch.float64),
                ),
                _scalar_provider,
            )

        self.assertEqual(result.accepted_updates, 3)
        self.assertEqual(result.stall_count, 0)
        self.assertEqual(result.completion_reason, "all_slots")
        self.assertEqual(result.total_forward_calls, 4)
        self.assertTrue(
            torch.equal(
                model.value.detach(),
                torch.full((3,), 0.2, dtype=torch.float64),
            )
        )

    def test_group_cursor_rotates_across_episodes_and_state_round_trip(self) -> None:
        config = self._config(
            parameter_group_schedule=("first", "second"),
        )
        model = _TwoGroupSlotModel()
        optimizer = ProcoimOptimizer(config)
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )

        first = optimizer.run_episode(model, batch, _scalar_provider)
        second = optimizer.run_episode(model, batch, _scalar_provider)
        self.assertEqual(
            first.parameter_group_acceptance_counts,
            {"first": 1, "second": 0},
        )
        self.assertEqual(
            second.parameter_group_acceptance_counts,
            {"first": 0, "second": 1},
        )

        checkpoint = optimizer.state_dict()
        restored = ProcoimOptimizer(config)
        restored.load_state_dict(checkpoint)
        restored_checkpoint = restored.state_dict()
        for key in (
            "episodes",
            "total_forward_calls",
            "total_candidate_forwards",
            "total_accepted_updates",
            "parameter_group_acceptance_counts",
            "next_parameter_group_index",
            "group_signature",
        ):
            self.assertEqual(restored_checkpoint[key], checkpoint[key])
        self.assertEqual(
            restored_checkpoint["generator_states"].keys(),
            checkpoint["generator_states"].keys(),
        )
        for key, value in checkpoint["generator_states"].items():
            self.assertTrue(
                torch.equal(restored_checkpoint["generator_states"][key], value)
            )

        third = restored.run_episode(model, batch, _scalar_provider)
        self.assertEqual(
            third.parameter_group_acceptance_counts,
            {"first": 1, "second": 0},
        )
        self.assertEqual(
            restored.state_dict()["next_parameter_group_index"],
            checkpoint["next_parameter_group_index"] + 1,
        )

    def test_exception_after_commit_preserves_weights_and_persistent_counters(
        self,
    ) -> None:
        from gator.training import ProcoimTrainingEngine

        model = _VectorSlotModel()
        engine = ProcoimTrainingEngine(
            model,
            self._config(
                max_candidate_banks=2,
                max_accepted_updates=2,
            ),
        )
        provider_calls = 0

        def provider(candidate, batch, kwargs):
            nonlocal provider_calls
            provider_calls += 1
            if provider_calls == 3:
                raise RuntimeError("provider failed after commit")
            return _scalar_provider(candidate, batch, kwargs)

        directions = [
            (torch.tensor([0.2, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.2], dtype=torch.float64),),
        ]
        with (
            mock.patch.object(
                engine.optimizer,
                "sample_direction",
                side_effect=directions,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "provider failed after commit",
            ),
        ):
            engine.run_episode(
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 2, dtype=torch.float64),
                ),
                provider,
            )

        self.assertTrue(
            torch.equal(
                model.value.detach(),
                torch.tensor([0.2, 0.0], dtype=torch.float64),
            )
        )
        self.assertEqual(engine.optimizer.total_accepted_updates, 1)
        self.assertEqual(engine.optimizer.total_candidate_forwards, 2)
        self.assertEqual(engine.optimizer.total_forward_calls, 3)
        self.assertEqual(engine.accepted_updates, 1)
        self.assertEqual(engine.step_number, 3)

    def test_baseline_exception_counts_forward_and_synchronizes_engine(
        self,
    ) -> None:
        from gator.training import ProcoimTrainingEngine

        model = _ScalarSlotModel(0.0)
        engine = ProcoimTrainingEngine(model, self._config())

        def provider(candidate, batch, kwargs):
            candidate(batch[0])
            del kwargs
            raise RuntimeError("baseline failed")

        with self.assertRaisesRegex(RuntimeError, "baseline failed"):
            engine.run_episode(
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64),
                ),
                provider,
            )

        self.assertEqual(engine.optimizer.total_forward_calls, 1)
        self.assertEqual(engine.optimizer.total_candidate_forwards, 0)
        self.assertEqual(engine.optimizer.total_accepted_updates, 0)
        self.assertEqual(engine.step_number, 1)
        self.assertEqual(engine.episode_number, 0)

    def test_forward_only_engine_counts_every_search_forward_as_step(self) -> None:
        from gator.training import ProcoimTrainingEngine

        model = _ScalarSlotModel(0.0)
        engine = ProcoimTrainingEngine(model, self._config())
        result = engine.run_episode(
            (torch.zeros(1, dtype=torch.float64), torch.ones(1, 1)),
            _scalar_provider,
        )
        self.assertEqual(engine.step_number, result.total_forward_calls)
        self.assertEqual(engine.step_number, 2)

    def test_post_accept_verification_is_counted(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(verify_accepted_candidate=True)
        )
        result = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            ),
            _scalar_provider,
        )

        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.candidate_forwards, 1)
        self.assertEqual(result.verification_forwards, 1)
        self.assertEqual(result.total_forward_calls, 3)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_failed_post_accept_verification_restores_winner(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(verify_accepted_candidate=True)
        )
        calls = 0

        def provider(candidate, batch, kwargs):
            nonlocal calls
            calls += 1
            result = _scalar_provider(candidate, batch, kwargs)
            if calls == 3:
                return _result(torch.ones_like(result.losses))
            return result

        result = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            ),
            provider,
        )

        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.verification_forwards, 1)
        self.assertEqual(float(model.value.detach()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


class ProcoimOptimizedSearchTests(unittest.TestCase):
    @staticmethod
    def _config(**overrides) -> ProcoimConfig:
        values = {
            "reduction_fraction": 0.1,
            "candidate_directions": 1,
            "max_candidate_banks": 4,
            "max_accepted_updates": 1,
            "verify_accepted_candidate": False,
            "stability_required": 1,
            "minimum_active_gain": 0.0,
            "probe_scale": 0.1,
            "minimum_probe_scale": 0.1,
            "maximum_probe_scale": 0.2,
            "parameter_rms_floor": 1.0,
            "parameter_group_schedule": ("all_parameters",),
            "seed": 11,
        }
        values.update(overrides)
        return ProcoimConfig(**values)

    def test_directional_adam_rejects_incompatible_search_modes(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "replaces momentum and antithetic estimate",
        ):
            ProcoimConfig(
                directional_adam=True,
                reuse_successful_directions=True,
            )
        with self.assertRaisesRegex(
            ValueError,
            "requires antithetic",
        ):
            ProcoimConfig(
                directional_adam=True,
                antithetic=False,
            )

    def test_persistent_direction_is_seeded_not_rng_ordered(self) -> None:
        model = _MultiTensorGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]
        snapshot = optimizer._snapshot_group(group)
        first = optimizer._persistent_direction(
            group,
            snapshot,
            basis_index=0,
            probe_scale=0.1,
        )
        optimizer.sample_direction(
            group,
            snapshot,
            probe_scale=0.1,
        )
        repeated = optimizer._persistent_direction(
            group,
            snapshot,
            basis_index=0,
            probe_scale=0.1,
        )
        other = optimizer._persistent_direction(
            group,
            snapshot,
            basis_index=1,
            probe_scale=0.1,
        )

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first, repeated, strict=True)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(first, other, strict=True)
            )
        )

    def test_basis_rotation_forwards_overrides_bank_based_rotation(
        self,
    ) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=4,
                directional_basis_mode="rotating_factorized",
                directional_basis_rotation_banks=1,
                directional_basis_rotation_forwards=500,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]

        # No forwards recorded yet: epoch 0 regardless of the (ignored)
        # bank-based rotation cadence.
        self.assertEqual(optimizer._directional_basis_epoch(group), 0)

        optimizer._group_basis_forward_counts[group.name] = 499
        self.assertEqual(optimizer._directional_basis_epoch(group), 0)
        optimizer._group_basis_forward_counts[group.name] = 500
        self.assertEqual(optimizer._directional_basis_epoch(group), 1)
        optimizer._group_basis_forward_counts[group.name] = 1250
        self.assertEqual(optimizer._directional_basis_epoch(group), 2)

        # A large bank count would have rotated many times under the
        # bank-based cadence (rotation_banks=1); confirm it's ignored.
        optimizer._group_bank_counts[group.name] = 999
        self.assertEqual(optimizer._directional_basis_epoch(group), 2)

    def test_basis_forward_count_accumulates_from_bank_results(self) -> None:
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(
                directional_adam=True,
                directional_adam_learning_rate=0.1,
                candidate_backtrack_factors=(1.0,),
                directional_basis_rotation_forwards=1,
            )
        )
        result = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            ),
            _scalar_provider,
        )
        self.assertGreater(
            optimizer._group_basis_forward_counts["all_parameters"], 0
        )
        self.assertEqual(result.accepted_updates, 1)

    def test_basis_rotation_forwards_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "directional_basis_rotation_forwards"
        ):
            ProcoimConfig(directional_basis_rotation_forwards=0)
        with self.assertRaisesRegex(
            ValueError, "directional_basis_rotation_forwards"
        ):
            ProcoimConfig(directional_basis_rotation_forwards=-1)
        # None (disabled) and positive values are both fine.
        ProcoimConfig(directional_basis_rotation_forwards=None)
        ProcoimConfig(directional_basis_rotation_forwards=500)

    def test_basis_rotation_stall_banks_rotates_only_on_consecutive_failures(
        self,
    ) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=4,
                directional_basis_mode="rotating_factorized",
                directional_basis_rotation_stall_banks=3,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]

        def _fail() -> _CandidateBankResult:
            return _CandidateBankResult(
                winner=None,
                winner_direction=None,
                candidate_forwards=10,
                anchor_forwards=0,
                hard_vetoes=0,
                anchor_vetoes=0,
            )

        def _success() -> _CandidateBankResult:
            winner = CandidateResult(
                group_name="combined",
                direction_index=0,
                sign=1,
                losses=torch.zeros(1, dtype=torch.float64),
                aggregate=0.0,
                active_gain=0.1,
                soft_penalty=0.0,
                global_regression=0.0,
                utility=0.1,
                hard_violation=False,
                eligible=True,
            )
            return _CandidateBankResult(
                winner=winner,
                winner_direction=(torch.zeros(1, dtype=torch.float64),),
                candidate_forwards=10,
                anchor_forwards=0,
                hard_vetoes=0,
                anchor_vetoes=0,
            )

        self.assertEqual(optimizer._directional_basis_epoch(group), 0)

        # Two failures: not enough to hit the stall_banks=3 threshold yet.
        optimizer._record_group_bank(group, _fail())
        optimizer._record_group_bank(group, _fail())
        self.assertEqual(optimizer._directional_basis_epoch(group), 0)

        # A success resets the streak; it never rotates on its own.
        optimizer._record_group_bank(group, _success())
        self.assertEqual(optimizer._directional_basis_epoch(group), 0)

        # Three consecutive failures after the reset trips the rotation.
        optimizer._record_group_bank(group, _fail())
        optimizer._record_group_bank(group, _fail())
        optimizer._record_group_bank(group, _fail())
        self.assertEqual(optimizer._directional_basis_epoch(group), 1)

        # The streak resets after rotating, so it takes another full
        # streak of 3 to rotate again.
        optimizer._record_group_bank(group, _fail())
        optimizer._record_group_bank(group, _fail())
        self.assertEqual(optimizer._directional_basis_epoch(group), 1)
        optimizer._record_group_bank(group, _fail())
        self.assertEqual(optimizer._directional_basis_epoch(group), 2)

    def test_basis_rotation_stall_banks_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "directional_basis_rotation_stall_banks"
        ):
            ProcoimConfig(directional_basis_rotation_stall_banks=0)
        with self.assertRaisesRegex(
            ValueError, "directional_basis_rotation_stall_banks"
        ):
            ProcoimConfig(directional_basis_rotation_stall_banks=-1)
        # None (disabled) and positive values are both fine.
        ProcoimConfig(directional_basis_rotation_stall_banks=None)
        ProcoimConfig(directional_basis_rotation_stall_banks=3)
        # Cannot combine with the forwards-based rotation trigger.
        with self.assertRaisesRegex(
            ValueError, "cannot both be set"
        ):
            ProcoimConfig(
                directional_basis_rotation_forwards=500,
                directional_basis_rotation_stall_banks=3,
            )

    def test_consolidation_cosine_margin_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "directional_basis_consolidation_cosine_margin"
        ):
            ProcoimConfig(directional_basis_consolidation_cosine_margin=0.0)
        with self.assertRaisesRegex(
            ValueError, "directional_basis_consolidation_cosine_margin"
        ):
            ProcoimConfig(directional_basis_consolidation_cosine_margin=-0.1)
        with self.assertRaisesRegex(
            ValueError, "requires directional_basis_mode"
        ):
            ProcoimConfig(
                directional_basis_mode="fixed_rademacher",
                directional_basis_consolidation_cosine_margin=0.05,
            )
        # None (disabled) and a positive value under rotating_factorized
        # (the default mode) are both fine.
        ProcoimConfig(directional_basis_consolidation_cosine_margin=None)
        ProcoimConfig(
            directional_basis_mode="rotating_factorized",
            directional_basis_consolidation_cosine_margin=0.05,
        )
        with self.assertRaisesRegex(
            ValueError, "directional_basis_consolidation_ema_beta"
        ):
            ProcoimConfig(directional_basis_consolidation_ema_beta=1.0)
        with self.assertRaisesRegex(
            ValueError, "directional_basis_consolidation_ema_beta"
        ):
            ProcoimConfig(directional_basis_consolidation_ema_beta=-0.1)

    def test_consolidation_cosine_gate_defers_rotation_when_aligned(
        self,
    ) -> None:
        """A window whose accepted-direction vector realigns with the
        previous window (cosine spiking above the group's running
        baseline) should hold the basis in place for one extra window
        instead of rotating on the plain banks-based schedule."""

        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=4,
                directional_basis_mode="rotating_factorized",
                directional_basis_rotation_banks=1,
                directional_basis_consolidation_cosine_margin=0.05,
                directional_basis_consolidation_ema_beta=0.5,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]

        def _accept(value: float) -> _CandidateBankResult:
            winner = CandidateResult(
                group_name="combined",
                direction_index=0,
                sign=1,
                losses=torch.zeros(1, dtype=torch.float64),
                aggregate=0.0,
                active_gain=0.1,
                soft_penalty=0.0,
                global_regression=0.0,
                utility=0.1,
                hard_violation=False,
                eligible=True,
            )
            direction = (
                torch.full((6, 5), value, dtype=torch.float64),
                torch.full((7,), value, dtype=torch.float64),
            )
            return _CandidateBankResult(
                winner=winner,
                winner_direction=direction,
                candidate_forwards=10,
                anchor_forwards=0,
                hard_vetoes=0,
                anchor_vetoes=0,
            )

        def _accept_and_record(value: float) -> None:
            bank = _accept(value)
            assert bank.winner is not None and bank.winner_direction is not None
            optimizer._accumulate_consolidation_window(
                group,
                bank.winner_direction,
                bank.winner.sign,
            )
            optimizer._record_group_bank(group, bank)

        self.assertEqual(optimizer._directional_basis_epoch(group), 0)

        # Window 0 closes with nothing to compare against yet (no
        # previous window vector): the basis rotates normally.
        _accept_and_record(1.0)
        self.assertEqual(optimizer._directional_basis_epoch(group), 1)

        # Window 1 flips direction (cosine -1.0 against window 0). No
        # baseline exists yet either, so this only establishes one; the
        # basis still rotates normally.
        _accept_and_record(-1.0)
        self.assertEqual(optimizer._directional_basis_epoch(group), 2)
        self.assertEqual(optimizer.total_consolidation_freezes, 0)

        # Window 2 repeats the same direction as window 1 (cosine +1.0),
        # far above the -1.0 baseline plus the 0.05 margin: this is the
        # rare aligned window, and rotation should be deferred by one.
        _accept_and_record(-1.0)
        self.assertEqual(optimizer._directional_basis_epoch(group), 2)
        self.assertEqual(optimizer.total_consolidation_freezes, 1)

        # Window 3 flips again (cosine -1.0 against window 2's -1.0
        # baseline update), well below baseline + margin: no new freeze,
        # but the earlier deferral is not undone, so the basis resumes
        # advancing one window behind the natural schedule.
        _accept_and_record(1.0)
        self.assertEqual(optimizer._directional_basis_epoch(group), 3)
        self.assertEqual(optimizer.total_consolidation_freezes, 1)

    def test_consolidation_cosine_gate_state_round_trips(self) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=4,
                directional_basis_mode="rotating_factorized",
                directional_basis_rotation_banks=1,
                directional_basis_consolidation_cosine_margin=0.05,
                directional_basis_consolidation_ema_beta=0.5,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]

        def _accept_and_record(value: float) -> None:
            winner = CandidateResult(
                group_name="combined",
                direction_index=0,
                sign=1,
                losses=torch.zeros(1, dtype=torch.float64),
                aggregate=0.0,
                active_gain=0.1,
                soft_penalty=0.0,
                global_regression=0.0,
                utility=0.1,
                hard_violation=False,
                eligible=True,
            )
            direction = (
                torch.full((6, 5), value, dtype=torch.float64),
                torch.full((7,), value, dtype=torch.float64),
            )
            bank = _CandidateBankResult(
                winner=winner,
                winner_direction=direction,
                candidate_forwards=10,
                anchor_forwards=0,
                hard_vetoes=0,
                anchor_vetoes=0,
            )
            optimizer._accumulate_consolidation_window(group, direction, 1)
            optimizer._record_group_bank(group, bank)

        for value in (1.0, -1.0, -1.0):
            _accept_and_record(value)
        self.assertEqual(optimizer.total_consolidation_freezes, 1)
        self.assertEqual(optimizer._directional_basis_epoch(group), 2)

        restored = ProcoimOptimizer(optimizer.config)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored_group = restored.select_parameter_groups(model)[0]
        self.assertEqual(
            restored.total_consolidation_freezes,
            optimizer.total_consolidation_freezes,
        )
        self.assertEqual(
            restored._directional_basis_epoch(restored_group),
            optimizer._directional_basis_epoch(group),
        )
        self.assertEqual(
            restored._group_window_cosine_baseline[group.name],
            optimizer._group_window_cosine_baseline[group.name],
        )

    def test_rotating_factorized_basis_is_seeded_low_rank_and_rotates(
        self,
    ) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=4,
                directional_basis_mode="rotating_factorized",
                directional_factor_rank=1,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]
        snapshot = optimizer._snapshot_group(group)
        first = optimizer._directional_basis_direction(
            group,
            snapshot,
            basis_index=0,
            basis_epoch=0,
            probe_scale=0.1,
        )
        repeated = optimizer._directional_basis_direction(
            group,
            snapshot,
            basis_index=0,
            basis_epoch=0,
            probe_scale=0.1,
        )
        rotated = optimizer._directional_basis_direction(
            group,
            snapshot,
            basis_index=0,
            basis_epoch=1,
            probe_scale=0.1,
        )

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first, repeated, strict=True)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(first, rotated, strict=True)
            )
        )
        self.assertLessEqual(
            int(torch.linalg.matrix_rank(first[0])),
            1,
        )
        for original, direction in zip(
            snapshot,
            first,
            strict=True,
        ):
            expected = optimizer._tensor_probe_magnitude(original, 0.1)
            self.assertAlmostEqual(
                float(direction.norm()),
                expected,
                places=10,
            )

    def test_fixed_directional_basis_ignores_rotation_epoch(self) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                directional_basis_mode="fixed_rademacher",
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]
        snapshot = optimizer._snapshot_group(group)
        first = optimizer._directional_basis_direction(
            group,
            snapshot,
            basis_index=0,
            basis_epoch=0,
            probe_scale=0.1,
        )
        later = optimizer._directional_basis_direction(
            group,
            snapshot,
            basis_index=0,
            basis_epoch=99,
            probe_scale=0.1,
        )

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first, later, strict=True)
            )
        )

    def test_streamed_basis_combination_matches_materialized_result(
        self,
    ) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=4,
                directional_basis_mode="rotating_factorized",
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]
        snapshot = optimizer._snapshot_group(group)
        magnitudes = tuple(
            optimizer._tensor_probe_magnitude(value, 1.0)
            for value in snapshot
        )
        coefficients = torch.tensor(
            [0.5, -0.25, 0.125, 0.75],
            dtype=torch.float64,
        )
        materialized = [
            optimizer._directional_basis_direction(
                group,
                snapshot,
                basis_index=index,
                basis_epoch=3,
                probe_scale=1.0,
                tensor_magnitudes=magnitudes,
            )
            for index in range(4)
        ]
        expected = optimizer._combine_directions(
            materialized,
            coefficients,
        )
        actual = optimizer._combine_directional_basis(
            group,
            snapshot,
            coefficients,
            basis_epoch=3,
            tensor_magnitudes=magnitudes,
        )

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(expected, actual, strict=True)
            )
        )

    def test_directional_moments_reset_when_basis_rotates(self) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                directional_basis_mode="rotating_factorized",
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]
        gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)
        _, first_m, first_v, first_step = (
            optimizer._directional_adam_proposal(
                group,
                gradient,
                basis_epoch=0,
            )
        )
        optimizer._directional_adam_m[group.name] = first_m
        optimizer._directional_adam_v[group.name] = first_v
        optimizer._directional_adam_steps[group.name] = first_step
        optimizer._directional_adam_basis_epochs[group.name] = 0
        rotated, _, _, rotated_step = optimizer._directional_adam_proposal(
            group,
            gradient,
            basis_epoch=1,
        )
        fresh, _, _, fresh_step = ProcoimOptimizer(
            optimizer.config
        )._directional_adam_proposal(
            group,
            gradient,
            basis_epoch=1,
        )

        self.assertEqual(rotated_step, 1)
        self.assertEqual(fresh_step, 1)
        self.assertTrue(torch.equal(rotated, fresh))

    def test_directional_basis_epoch_uses_group_local_bank_count(
        self,
    ) -> None:
        model = _MatrixGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                directional_basis_mode="rotating_factorized",
                directional_basis_rotation_banks=8,
                parameter_group_schedule=("combined",),
            )
        )
        group = optimizer.select_parameter_groups(model)[0]

        self.assertEqual(optimizer._directional_basis_epoch(group), 0)
        optimizer._group_bank_counts[group.name] = 7
        self.assertEqual(optimizer._directional_basis_epoch(group), 0)
        optimizer._group_bank_counts[group.name] = 8
        self.assertEqual(optimizer._directional_basis_epoch(group), 1)
        optimizer._group_bank_counts[group.name] = 24
        self.assertEqual(optimizer._directional_basis_epoch(group), 3)

    def test_version_thirteen_checkpoint_requires_fixed_legacy_basis(
        self,
    ) -> None:
        model = _MatrixGroupModel()
        fixed_config = self._config(
            candidate_directions=2,
            directional_basis_mode="fixed_rademacher",
            parameter_group_schedule=("combined",),
        )
        optimizer = ProcoimOptimizer(fixed_config)
        optimizer.select_parameter_groups(model)
        legacy = copy.deepcopy(optimizer.state_dict())
        legacy["version"] = 13
        for name in (
            "directional_basis_mode",
            "directional_factor_rank",
            "directional_basis_rotation_banks",
        ):
            legacy["config"].pop(name)
        legacy.pop("directional_adam_basis_epochs")

        restored = ProcoimOptimizer(fixed_config)
        restored.load_state_dict(legacy)
        restored.select_parameter_groups(model)
        self.assertEqual(
            restored.config.directional_basis_mode,
            "fixed_rademacher",
        )
        with self.assertRaisesRegex(ValueError, "cannot enable new"):
            ProcoimOptimizer(
                self._config(
                    candidate_directions=2,
                    directional_basis_mode="rotating_factorized",
                    parameter_group_schedule=("combined",),
                )
            ).load_state_dict(legacy)

    def test_directional_adam_accepts_and_checkpoints_moments(self) -> None:
        model = _ScalarSlotModel(0.0)
        config = self._config(
            directional_adam=True,
            directional_adam_learning_rate=0.1,
            candidate_backtrack_factors=(1.0,),
        )
        optimizer = ProcoimOptimizer(config)
        result = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            ),
            _scalar_provider,
        )

        self.assertEqual(result.candidate_forwards, 3)
        self.assertEqual(result.total_forward_calls, 4)
        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.directional_adam_accepts, 1)
        self.assertEqual(result.total_directional_adam_accepts, 1)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        checkpoint = optimizer.state_dict()
        self.assertEqual(
            checkpoint["directional_adam_steps"]["all_parameters"],
            1,
        )
        self.assertEqual(
            checkpoint["directional_adam_basis_epochs"]["all_parameters"],
            0,
        )
        self.assertEqual(
            tuple(checkpoint["directional_adam_m"]["all_parameters"].shape),
            (1,),
        )
        restored = ProcoimOptimizer(config)
        restored.load_state_dict(checkpoint)
        restored.select_parameter_groups(model)
        restored_state = restored.state_dict()
        self.assertTrue(
            torch.equal(
                restored_state["directional_adam_m"]["all_parameters"],
                checkpoint["directional_adam_m"]["all_parameters"],
            )
        )
        self.assertTrue(
            torch.equal(
                restored_state["directional_adam_v"]["all_parameters"],
                checkpoint["directional_adam_v"]["all_parameters"],
            )
        )

    def test_rejected_directional_adam_does_not_commit_moments(self) -> None:
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        anchor = ProcoimAnchor(
            batch=(
                torch.zeros(1, dtype=torch.float64),
                -torch.ones(1, 1, dtype=torch.float64),
            ),
            best_aggregate=1.0,
        )
        model = _ScalarSlotModel(0.0)
        optimizer = ProcoimOptimizer(
            self._config(
                directional_adam=True,
                directional_adam_learning_rate=0.1,
                candidate_backtrack_factors=(1.0, 0.5, 0.25),
                anchor_regression_fraction=0.0,
                anchor_absolute_tolerance=0.0,
            )
        )
        result = optimizer.run_episode(
            model,
            batch,
            _scalar_provider,
            anchors=(anchor,),
        )

        self.assertEqual(result.accepted_updates, 0)
        self.assertEqual(result.directional_adam_accepts, 0)
        self.assertEqual(float(model.value.detach()), 0.0)
        checkpoint = optimizer.state_dict()
        self.assertEqual(checkpoint["directional_adam_m"], {})
        self.assertEqual(checkpoint["directional_adam_v"], {})
        self.assertEqual(checkpoint["directional_adam_steps"], {})

    def test_directional_projection_cancels_protected_damage(self) -> None:
        optimizer = ProcoimOptimizer(
            self._config(
                directional_adam=True,
                candidate_directions=2,
                minimum_active_gain=0.0,
                absolute_regression_tolerance=0.0,
                directional_adam_learning_rate=0.1,
            )
        )
        state = optimizer.initialize_episode_state(
            _result(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        )
        state.status[0, 1] = int(SlotStatus.PROTECTED)
        state.current_loss[0, 1] = 0.5
        state.best_loss[0, 1] = 0.5
        sensitivity = torch.tensor(
            [[[-1.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float64,
        )
        projected = optimizer._project_directional_coefficients(
            torch.tensor([0.1, 0.1], dtype=torch.float64),
            sensitivity,
            state,
        )

        self.assertIsNotNone(projected)
        assert projected is not None
        active_change = float(sensitivity[0, 0].dot(projected))
        protected_change = float(sensitivity[0, 1].dot(projected))
        self.assertLessEqual(active_change, -0.1 + 1e-7)
        self.assertLessEqual(protected_change, 0.04 + 1e-7)
        self.assertLess(float(projected[1]), 0.0)

    def test_signed_direction_momentum_is_reused_at_adapted_radius(self) -> None:
        model = _ScalarSlotModel(0.0)
        config = self._config(
            max_accepted_updates=2,
            reuse_successful_directions=True,
            direction_momentum=0.0,
            adaptive_probe_scale=True,
            probe_scale_growth=2.0,
        )
        optimizer = ProcoimOptimizer(config)
        first = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                -torch.ones(1, 1, dtype=torch.float64),
            ),
            _scalar_provider,
        )
        result = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                -torch.ones(1, 1, dtype=torch.float64),
            ),
            _scalar_provider,
        )

        self.assertEqual(first.accepted_updates, 1)
        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.reused_direction_accepts, 1)
        self.assertEqual(result.total_reused_direction_accepts, 1)
        self.assertAlmostEqual(float(model.value.detach()), -0.3, places=12)
        self.assertEqual(optimizer.total_forward_calls, 5)
        self.assertEqual(
            result.parameter_group_probe_scales["all_parameters"],
            0.2,
        )
        checkpoint = optimizer.state_dict()
        self.assertAlmostEqual(
            float(checkpoint["direction_momentum"]["all_parameters"][0]),
            -1.0,
            places=12,
        )
        restored = ProcoimOptimizer(config)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.total_reused_direction_accepts, 1)
        self.assertTrue(
            torch.equal(
                restored.state_dict()["direction_momentum"]
                ["all_parameters"][0],
                checkpoint["direction_momentum"]["all_parameters"][0],
            )
        )

    def test_antithetic_estimate_is_explicitly_scored(self) -> None:
        model = _VectorSlotModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                use_antithetic_estimate=True,
                minimum_active_gain=0.15,
            )
        )
        directions = [
            (torch.tensor([0.1, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.1], dtype=torch.float64),),
        ]
        with mock.patch.object(
            optimizer,
            "sample_direction",
            side_effect=directions,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                ),
                _scalar_provider,
            )

        self.assertEqual(result.candidate_forwards, 5)
        self.assertEqual(result.total_forward_calls, 6)
        self.assertEqual(result.antithetic_estimate_accepts, 1)
        self.assertEqual(result.total_antithetic_estimate_accepts, 1)
        self.assertGreater(float(model.value[0].detach()), 0.0)
        self.assertLess(float(model.value[1].detach()), 0.0)
        self.assertLess(result.end_aggregate_loss, 0.905)

    def test_antithetic_estimate_only_ignores_eligible_direct_probe(
        self,
    ) -> None:
        model = _VectorSlotModel()
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.05,
                candidate_directions=2,
                use_antithetic_estimate=True,
                antithetic_estimate_only=True,
                minimum_active_gain=0.0,
            )
        )
        directions = [
            (torch.tensor([0.1, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.1], dtype=torch.float64),),
        ]
        with mock.patch.object(
            optimizer,
            "sample_direction",
            side_effect=directions,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                ),
                _scalar_provider,
            )

        self.assertEqual(result.candidate_forwards, 5)
        self.assertEqual(result.antithetic_estimate_accepts, 1)
        self.assertGreater(float(model.value[0].detach()), 0.0)
        self.assertLess(float(model.value[1].detach()), 0.0)

    def test_antithetic_estimate_commit_scale_is_independent(self) -> None:
        model = _VectorSlotModel()
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.001,
                candidate_directions=2,
                use_antithetic_estimate=True,
                antithetic_estimate_only=True,
                antithetic_estimate_step_scale=0.02,
                minimum_active_gain=0.0,
            )
        )
        directions = [
            (torch.tensor([0.1, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.1], dtype=torch.float64),),
        ]
        with mock.patch.object(
            optimizer,
            "sample_direction",
            side_effect=directions,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                ),
                _scalar_provider,
            )

        self.assertEqual(result.antithetic_estimate_accepts, 1)
        self.assertAlmostEqual(
            float(model.value.detach().norm()),
            0.02,
        )
        self.assertGreater(float(model.value[0].detach()), 0.0)
        self.assertLess(float(model.value[1].detach()), 0.0)

        with self.assertRaisesRegex(
            ValueError,
            "slot estimates require",
        ):
            ProcoimConfig(
                antithetic_slot_estimates=True,
            )

    def test_slot_estimates_add_one_scored_proposal_per_position(self) -> None:
        model = _VectorSlotModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                use_antithetic_estimate=True,
                antithetic_estimate_only=True,
                antithetic_slot_estimates=True,
                candidate_backtrack_factors=(1.0,),
                minimum_active_gain=2.0,
                max_candidate_banks=1,
            )
        )
        directions = [
            (torch.tensor([0.1, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.1], dtype=torch.float64),),
        ]
        with mock.patch.object(
            optimizer,
            "sample_direction",
            side_effect=directions,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                ),
                _scalar_provider,
            )

        # Four +/- measurements, one aggregate proposal, and one proposal
        # for each of the two active physical output positions.
        self.assertEqual(result.candidate_forwards, 7)
        self.assertEqual(result.accepted_updates, 0)

    def test_antithetic_estimate_only_requires_estimate(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires use_antithetic_estimate",
        ):
            ProcoimConfig(antithetic_estimate_only=True)

        with self.assertRaisesRegex(
            ValueError,
            "anchor-aware estimates require",
        ):
            ProcoimConfig(
                antithetic_estimate_anchor_aware=True,
            )

    def test_antithetic_estimate_respects_forward_budget(self) -> None:
        model = _VectorSlotModel()
        optimizer = ProcoimOptimizer(
            self._config(
                candidate_directions=2,
                use_antithetic_estimate=True,
                minimum_active_gain=0.15,
            )
        )
        directions = [
            (torch.tensor([0.1, 0.0], dtype=torch.float64),),
            (torch.tensor([0.0, 0.1], dtype=torch.float64),),
        ]
        with mock.patch.object(
            optimizer,
            "sample_direction",
            side_effect=directions,
        ):
            result = optimizer.run_episode(
                model,
                (
                    torch.zeros(1, dtype=torch.float64),
                    torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                ),
                _scalar_provider,
                max_forward_calls=5,
            )

        self.assertEqual(result.candidate_forwards, 4)
        self.assertEqual(result.total_forward_calls, 5)
        self.assertEqual(result.antithetic_estimate_accepts, 0)
        self.assertEqual(int(model.value.ne(0).sum()), 0)

    def test_adaptive_probe_scale_is_group_local_and_capped(self) -> None:
        model = _SelectiveTwoGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                parameter_group_schedule=("first", "second"),
                max_accepted_updates=2,
                adaptive_probe_scale=True,
                probe_scale_decay=0.5,
                probe_scale_growth=2.0,
                minimum_probe_scale=0.001,
            )
        )
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        optimizer.run_episode(model, batch, _scalar_provider)
        result = optimizer.run_episode(model, batch, _scalar_provider)

        self.assertEqual(optimizer.total_accepted_updates, 2)
        self.assertEqual(
            optimizer.parameter_group_acceptance_counts,
            {"second": 2},
        )
        self.assertAlmostEqual(
            result.parameter_group_probe_scales["first"],
            0.025,
        )
        self.assertAlmostEqual(
            result.parameter_group_probe_scales["second"],
            0.2,
        )

    def test_adaptive_candidate_directions_grows_on_stall_and_resets_on_success(
        self,
    ) -> None:
        model = _SelectiveTwoGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                parameter_group_schedule=("first", "second"),
                candidate_directions=4,
                max_candidate_banks=6,
                max_accepted_updates=1,
                adaptive_candidate_directions=True,
                minimum_candidate_directions=1,
                candidate_directions_growth=2.0,
            )
        )
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        result = None
        for _ in range(4):
            result = optimizer.run_episode(model, batch, _scalar_provider)

        # "first" never affects the loss, so every bank it searches stalls
        # and its probe width doubles until it hits the declared cap.
        self.assertEqual(result.parameter_group_candidate_directions["first"], 4)
        # "second" always finds an accept immediately, so its probe width
        # resets back to the minimum after each success.
        self.assertEqual(
            result.parameter_group_candidate_directions["second"], 1
        )
        self.assertEqual(
            optimizer.parameter_group_acceptance_counts,
            {"second": 4},
        )

    def test_directional_adam_with_adaptive_candidate_directions_narrows_search(
        self,
    ) -> None:
        model = _ScalarSlotModel(0.0)
        config = self._config(
            candidate_directions=4,
            directional_adam=True,
            directional_adam_learning_rate=0.1,
            candidate_backtrack_factors=(1.0,),
            adaptive_candidate_directions=True,
            minimum_candidate_directions=1,
            candidate_directions_growth=2.0,
        )
        optimizer = ProcoimOptimizer(config)
        result = optimizer.run_episode(
            model,
            (
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            ),
            _scalar_provider,
        )

        # Only the minimum probe width (1 direction) should have been
        # spent, not the full declared candidate_directions=4.
        self.assertEqual(result.candidate_forwards, 3)
        self.assertEqual(result.accepted_updates, 1)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)
        self.assertEqual(
            result.parameter_group_candidate_directions["all_parameters"],
            1,
        )

        # The persistent moment buffers stay sized at the full declared
        # width regardless of how narrow any one bank's probe was.
        checkpoint = optimizer.state_dict()
        self.assertEqual(
            tuple(checkpoint["directional_adam_m"]["all_parameters"].shape),
            (4,),
        )

    def test_adaptive_candidate_directions_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "candidate_directions_growth"
        ):
            ProcoimConfig(candidate_directions_growth=1.0)
        with self.assertRaisesRegex(
            ValueError, "minimum_candidate_directions"
        ):
            ProcoimConfig(
                candidate_directions=4,
                minimum_candidate_directions=8,
                adaptive_candidate_directions=True,
            )
        # Disabled adaptive search does not restrict the legacy field.
        ProcoimConfig(
            candidate_directions=4,
            minimum_candidate_directions=8,
        )

    def test_adaptive_group_selection_prefers_gain_per_forward(self) -> None:
        model = _WeightedTwoGroupModel()
        optimizer = ProcoimOptimizer(
            self._config(
                parameter_group_schedule=("first", "second"),
                max_accepted_updates=3,
                adaptive_group_selection=True,
                group_score_beta=0.0,
                group_exploration=0.0,
            )
        )
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        optimizer.run_episode(model, batch, _scalar_provider)
        optimizer.run_episode(model, batch, _scalar_provider)
        result = optimizer.run_episode(model, batch, _scalar_provider)

        self.assertEqual(
            optimizer.parameter_group_acceptance_counts,
            {"first": 2, "second": 1},
        )
        self.assertGreaterEqual(
            result.parameter_group_gain_per_forward["first"],
            result.parameter_group_gain_per_forward["second"],
        )

    def test_failed_current_gate_does_not_evaluate_anchors(self) -> None:
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        anchor = ProcoimAnchor(batch=batch, best_aggregate=1.0)
        optimizer = ProcoimOptimizer(
            self._config(minimum_active_gain=2.0)
        )

        result = optimizer.run_episode(
            _ScalarSlotModel(0.0),
            batch,
            _scalar_provider,
            anchors=(anchor,),
        )

        self.assertGreater(result.candidate_forwards, 0)
        self.assertEqual(result.anchor_forwards, 0)
        self.assertEqual(result.accepted_updates, 0)

    def test_estimate_only_probes_replay_only_the_composite(self) -> None:
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        anchor = ProcoimAnchor(batch=batch, best_aggregate=1.0)
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.05,
                candidate_directions=2,
                use_antithetic_estimate=True,
                antithetic_estimate_only=True,
                antithetic_estimate_step_scale=0.1,
            )
        )

        result = optimizer.run_episode(
            _ScalarSlotModel(0.0),
            batch,
            _scalar_provider,
            anchors=(anchor,),
        )

        self.assertEqual(result.candidate_forwards, 5)
        self.assertEqual(result.anchor_forwards, 1)
        self.assertEqual(result.accepted_updates, 1)
        self.assertEqual(result.antithetic_estimate_accepts, 1)

    def test_anchor_aware_estimate_scores_every_probe_on_replay(self) -> None:
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        anchor = ProcoimAnchor(batch=batch, best_aggregate=1.0)
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.05,
                candidate_directions=2,
                use_antithetic_estimate=True,
                antithetic_estimate_only=True,
                antithetic_estimate_anchor_aware=True,
                antithetic_estimate_step_scale=0.1,
            )
        )

        result = optimizer.run_episode(
            _ScalarSlotModel(0.0),
            batch,
            _scalar_provider,
            anchors=(anchor,),
        )

        # Four +/- probes and the synthesized commit candidate each score
        # the replay anchor. The default estimate-only mode scores only the
        # final composite and remains covered by the preceding test.
        self.assertEqual(result.candidate_forwards, 5)
        self.assertEqual(result.anchor_forwards, 5)
        self.assertEqual(result.accepted_updates, 1)

    def test_estimate_backtracks_until_anchor_protection_is_safe(self) -> None:
        batch = (
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        anchor = ProcoimAnchor(
            batch=(
                torch.zeros(1, dtype=torch.float64),
                -torch.ones(1, 1, dtype=torch.float64),
            ),
            start_losses=torch.tensor([[2.0]], dtype=torch.float64),
            best_losses=torch.tensor([[1.0]], dtype=torch.float64),
            valid_mask=torch.tensor([[True]]),
            protected_mask=torch.tensor([[True]]),
        )
        optimizer = ProcoimOptimizer(
            self._config(
                reduction_fraction=0.005,
                candidate_directions=2,
                use_antithetic_estimate=True,
                antithetic_estimate_only=True,
                antithetic_estimate_step_scale=0.1,
                anchor_hard_regression_fraction=0.1,
                anchor_absolute_tolerance=0.005,
                candidate_backtrack_factors=(1.0, 0.5, 0.25),
            )
        )
        model = _ScalarSlotModel(0.0)

        result = optimizer.run_episode(
            model,
            batch,
            _scalar_provider,
            anchors=(anchor,),
        )

        # The full +0.1 step damages the opposite-target anchor. Half of
        # that step still clears the current 0.5% target and fits exactly
        # inside the saved-gain-relative hard bound.
        self.assertAlmostEqual(float(model.value.detach()), 0.05)
        self.assertEqual(result.candidate_forwards, 6)
        self.assertEqual(result.anchor_forwards, 2)
        self.assertEqual(result.anchor_vetoes, 1)
        self.assertEqual(result.accepted_updates, 1)

    def test_hard_anchor_veto_and_atomic_forward_budget(self) -> None:
        batch = (
            torch.zeros(1, dtype=torch.float64),
            -torch.ones(1, 1, dtype=torch.float64),
        )
        anchor = ProcoimAnchor(
            batch=(
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            ),
            best_aggregate=0.81,
        )
        config = self._config(
            reduction_fraction=0.1,
            max_candidate_banks=1,
            anchor_regression_fraction=0.0,
            anchor_absolute_tolerance=0.0,
        )

        model = _ScalarSlotModel(0.1)
        optimizer = ProcoimOptimizer(config)
        result = optimizer.run_episode(
            model,
            batch,
            _scalar_provider,
            anchors=(anchor,),
        )
        self.assertEqual(result.accepted_updates, 0)
        self.assertAlmostEqual(float(model.value.detach()), 0.1)
        self.assertEqual(result.candidate_forwards, 2)
        self.assertEqual(result.anchor_forwards, 1)
        self.assertEqual(result.anchor_vetoes, 1)
        self.assertEqual(result.total_anchor_vetoes, 1)
        self.assertEqual(result.total_forward_calls, 4)
        self.assertAlmostEqual(anchor.best_aggregate, 0.81)

        budget_model = _ScalarSlotModel(0.1)
        budget_anchor = ProcoimAnchor(
            batch=anchor.batch,
            best_aggregate=0.81,
        )
        budget_result = ProcoimOptimizer(config).run_episode(
            budget_model,
            batch,
            _scalar_provider,
            anchors=(budget_anchor,),
            max_forward_calls=5,
        )
        self.assertEqual(budget_result.total_forward_calls, 4)
        self.assertEqual(budget_result.candidate_forwards, 2)
        self.assertEqual(budget_result.anchor_forwards, 1)
        self.assertAlmostEqual(float(budget_model.value.detach()), 0.1)

        control = _ScalarSlotModel(0.1)
        control_result = ProcoimOptimizer(config).run_episode(
            control,
            batch,
            _scalar_provider,
        )
        self.assertEqual(control_result.accepted_updates, 1)
        self.assertAlmostEqual(float(control.value.detach()), 0.0)
        self.assertEqual(control_result.total_forward_calls, 3)

    def test_version_one_checkpoint_loads_only_with_new_features_off(self) -> None:
        optimizer = ProcoimOptimizer(ProcoimConfig())
        legacy = copy.deepcopy(optimizer.state_dict())
        legacy["version"] = 1
        new_config_fields = {
            "reuse_successful_directions",
            "direction_momentum",
            "use_antithetic_estimate",
            "antithetic_estimate_anchor_aware",
            "verify_accepted_candidate",
            "directional_adam",
            "directional_adam_learning_rate",
            "directional_adam_beta1",
            "directional_adam_beta2",
            "directional_adam_epsilon",
            "directional_adam_weight_decay",
            "directional_adam_projection_iterations",
            "candidate_backtrack_factors",
            "adaptive_probe_scale",
            "probe_scale_growth",
            "maximum_probe_scale",
            "adaptive_group_selection",
            "group_score_beta",
            "group_exploration",
            "anchor_regression_fraction",
            "anchor_soft_regression_fraction",
            "anchor_hard_regression_fraction",
            "anchor_minimum_retained_reduction_fraction",
            "anchor_absolute_tolerance",
            "anchor_weight",
            "anchor_hard_protection",
        }
        for name in new_config_fields:
            legacy["config"].pop(name)
        for name in (
            "total_anchor_forwards",
            "total_reused_direction_accepts",
            "total_antithetic_estimate_accepts",
            "total_anchor_vetoes",
            "group_probe_scales",
            "group_gain_ema",
            "group_bank_counts",
            "direction_momentum",
            "directional_adam_m",
            "directional_adam_v",
            "directional_adam_steps",
            "directional_adam_last_decay_step",
            "total_directional_adam_accepts",
        ):
            legacy.pop(name)

        restored = ProcoimOptimizer(ProcoimConfig())
        restored.load_state_dict(legacy)
        self.assertEqual(restored.total_forward_calls, 0)
        with self.assertRaisesRegex(ValueError, "cannot enable new"):
            ProcoimOptimizer(
                ProcoimConfig(reuse_successful_directions=True)
            ).load_state_dict(legacy)


class ProcoimRetentionWindowTests(unittest.TestCase):
    @staticmethod
    def _engine(
        model: nn.Module,
        *,
        anchor_soft_regression_fraction: float = 0.10,
        anchor_hard_regression_fraction: float = 0.10,
    ) -> ProcoimTrainingEngine:
        return ProcoimTrainingEngine(
            model,
            ProcoimConfig(
                reduction_fraction=0.005,
                candidate_directions=1,
                max_candidate_banks=1,
                max_accepted_updates=1,
                stability_required=1,
                minimum_active_gain=0.0,
                probe_scale=0.1,
                minimum_probe_scale=0.1,
                parameter_rms_floor=1.0,
                parameter_group_schedule=("all_parameters",),
                anchor_soft_regression_fraction=(
                    anchor_soft_regression_fraction
                ),
                anchor_hard_regression_fraction=anchor_hard_regression_fraction,
                anchor_absolute_tolerance=0.005,
            ),
        )

    @staticmethod
    def _opposite_anchor() -> ProcoimAnchor:
        return ProcoimAnchor(
            batch=(
                torch.zeros(1, dtype=torch.float64),
                -torch.ones(1, 1, dtype=torch.float64),
            ),
            start_losses=torch.tensor([[2.0]], dtype=torch.float64),
            best_losses=torch.tensor([[1.0]], dtype=torch.float64),
            valid_mask=torch.tensor([[True]]),
            protected_mask=torch.tensor([[True]]),
        )

    def test_consolidation_selects_largest_safe_alpha(self) -> None:
        model = _ScalarSlotModel(0.0)
        engine = self._engine(model)
        window = ProcoimRetentionWindow(
            engine,
            interval_examples=1,
            commit_alphas=(1.0, 0.5, 0.25),
            shrink_factor=0.5,
        )
        with torch.no_grad():
            model.value.fill_(0.1)
        window.record_episode()

        result = window.consolidate(
            (self._opposite_anchor(),),
            _scalar_provider,
        )

        self.assertEqual(result.committed_alpha, 0.5)
        self.assertFalse(result.rolled_back)
        self.assertEqual(result.audit_forwards, 2)
        self.assertEqual(engine.step_number, 2)
        self.assertAlmostEqual(float(model.value.detach()), 0.05)
        self.assertEqual(window.windows_committed, 1)

    def test_consolidation_allows_bounded_soft_regression(self) -> None:
        model = _ScalarSlotModel(0.0)
        engine = self._engine(
            model,
            anchor_soft_regression_fraction=0.03,
        )
        window = ProcoimRetentionWindow(
            engine,
            interval_examples=1,
            commit_alphas=(1.0, 0.5, 0.25),
            shrink_factor=0.5,
        )
        anchor = self._opposite_anchor()
        # The candidate anchor loss is 1.0609. It exceeds the configured
        # soft boundary (1.035) but remains below the hard boundary (1.105).
        # Soft replay regression is a utility penalty, not a consolidation
        # veto, so the complete live update must remain eligible.
        with torch.no_grad():
            model.value.fill_(0.03)
        window.record_episode()

        result = window.consolidate(
            (anchor,),
            _scalar_provider,
        )

        self.assertEqual(result.committed_alpha, 1.0)
        self.assertFalse(result.rolled_back)
        self.assertFalse(result.hard_violation)
        self.assertGreater(result.audit_penalty, 0.0)
        self.assertEqual(result.audit_forwards, 1)
        self.assertAlmostEqual(float(model.value.detach()), 0.03)

    def test_unsafe_window_restores_theta_base_exactly(self) -> None:
        model = _ScalarSlotModel(0.0)
        engine = self._engine(model)
        window = ProcoimRetentionWindow(
            engine,
            interval_examples=1,
            commit_alphas=(1.0, 0.75),
            shrink_factor=0.75,
        )
        base = model.value.detach().clone()
        with torch.no_grad():
            model.value.fill_(1.0)
        window.record_episode()

        result = window.consolidate(
            (self._opposite_anchor(),),
            _scalar_provider,
        )

        self.assertTrue(result.rolled_back)
        self.assertEqual(result.committed_alpha, 0.0)
        self.assertTrue(torch.equal(model.value, base))
        self.assertEqual(window.windows_rolled_back, 1)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_budgetless_partial_window_rolls_back_without_forward(self) -> None:
        model = _ScalarSlotModel(0.0)
        engine = self._engine(model)
        window = ProcoimRetentionWindow(engine, interval_examples=2)
        base = model.value.detach().clone()
        with torch.no_grad():
            model.value.fill_(0.25)
        window.record_episode()
        window.rollback_unconsolidated()

        self.assertTrue(torch.equal(model.value, base))
        self.assertEqual(engine.step_number, 0)
        self.assertFalse(window.has_provisional_updates)


class ProcoimGTRLCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _model():
        from gator_linear.core import GatorLinearConfig
        from gator_linear.prefix import (
            GatorLinearPrefixConfig,
            GatorLinearPrefixModel,
        )

        core = GatorLinearConfig(
            vocab_size=16,
            streams=1,
            nodes=2,
            width=2,
            relays=1,
            degree=2,
            buses=1,
            mailbox_channels=1,
            mailbox_width=2,
            basis_rank=1,
            address_width=2,
            router_rank=1,
            conv_kernel=3,
            context_length=3,
            private_stream_io=False,
            graph_layout="dense",
        )
        return GatorLinearPrefixModel(GatorLinearPrefixConfig(core))

    def test_gtrl_groups_are_disjoint_complete_and_freeze_embeddings_by_default(
        self,
    ) -> None:
        model = self._model()
        exposed = model.procoim_parameter_groups()
        self.assertEqual(
            set(exposed),
            {
                "routing",
                "receiver_memory",
                "node_compute",
                "stream_mix",
                "readout",
                "embedding",
            },
        )
        identifiers = [
            id(parameter)
            for parameters in exposed.values()
            for parameter in parameters
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(
            set(identifiers),
            {id(parameter) for parameter in model.parameters()},
        )

        optimizer = ProcoimOptimizer(ProcoimConfig())
        selected = optimizer.select_parameter_groups(model)
        self.assertEqual(
            [group.name for group in selected],
            [
                "routing",
                "receiver_memory",
                "stream_mix",
                "node_compute",
                "readout",
            ],
        )
        selected_ids = {
            id(parameter)
            for group in selected
            for parameter in group.parameters
        }
        embedding_ids = {id(parameter) for parameter in exposed["embedding"]}
        self.assertTrue(selected_ids.isdisjoint(embedding_ids))

        names = {
            id(parameter): name
            for name, parameter in model.named_parameters()
        }
        receiver_names = {
            names[id(parameter)]
            for parameter in exposed["receiver_memory"]
        }
        self.assertTrue(receiver_names)
        self.assertTrue(
            all(
                ".mixer." in name
                or ".write_" in name
                or ".read_" in name
                or ".context_norm." in name
                for name in receiver_names
            )
        )

    def test_gtrl_tensor_granularity_splits_selected_semantic_groups(
        self,
    ) -> None:
        model = self._model()
        selected_semantics = ("routing", "node_compute")
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                parameter_group_granularity="tensor",
                parameter_group_schedule=selected_semantics,
            )
        )
        selected = optimizer.select_parameter_groups(model)
        expected_ids = {
            id(parameter)
            for name in selected_semantics
            for parameter in model.procoim_parameter_groups()[name]
        }

        self.assertTrue(selected)
        self.assertTrue(
            all(len(group.parameters) == 1 for group in selected)
        )
        self.assertTrue(
            all("::" in group.name for group in selected)
        )
        self.assertEqual(
            {id(group.parameters[0]) for group in selected},
            expected_ids,
        )

    def test_gtrl_relay_granularity_is_complete_and_intermediate(self) -> None:
        model = self._model()
        semantics = ("routing", "receiver_memory", "node_compute")
        semantic = ProcoimOptimizer(
            ProcoimConfig(parameter_group_schedule=semantics)
        ).select_parameter_groups(model)
        relay = ProcoimOptimizer(
            ProcoimConfig(
                parameter_group_granularity="relay",
                parameter_group_schedule=semantics,
            )
        ).select_parameter_groups(model)
        tensor = ProcoimOptimizer(
            ProcoimConfig(
                parameter_group_granularity="tensor",
                parameter_group_schedule=semantics,
            )
        ).select_parameter_groups(model)

        expected_ids = {
            id(parameter)
            for name in semantics
            for parameter in model.procoim_parameter_groups()[name]
        }
        relay_ids = {
            id(parameter)
            for group in relay
            for parameter in group.parameters
        }
        self.assertEqual(relay_ids, expected_ids)
        self.assertLess(len(semantic), len(relay))
        self.assertLess(len(relay), len(tensor))
        self.assertTrue(
            all("::relay" in group.name or "::shared" in group.name for group in relay)
        )

    def test_prefix_evaluation_reports_finite_per_position_metrics(self) -> None:
        from gator_linear_prefix_train import evaluate_linear_prefix

        metrics = evaluate_linear_prefix(
            self._model(),
            length=1,
            device=torch.device("cpu"),
            batch_size=8,
            seed=31,
            temperature=1.0,
        )
        self.assertEqual(len(metrics["by_position"]), 1)
        self.assertEqual(len(metrics["valid_by_position"]), 1)
        self.assertEqual(len(metrics["loss_by_position"]), 1)
        for name in (
            "stream_agreement",
            "stream_accuracy_spread",
            "valid_accuracy",
            "valid_exact",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])))

    def test_one_forward_only_bank_runs_through_gtrl(self) -> None:
        from gator.optimizers.slot_loss import cross_entropy_slot_loss
        from parallel_prefix_benchmark import make_prefix_batch

        model = self._model()
        generator = torch.Generator(device="cpu").manual_seed(7)
        batch = make_prefix_batch(
            1,
            1,
            torch.device("cpu"),
            generator=generator,
            sampling_device=torch.device("cpu"),
        )

        def provider(candidate, current_batch, kwargs):
            observed, targets = current_batch
            return cross_entropy_slot_loss(
                candidate(observed, **kwargs),
                targets,
            )

        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                candidate_directions=1,
                max_candidate_banks=1,
                max_accepted_updates=1,
                stability_required=1,
                minimum_active_gain=0.0,
                probe_scale=0.01,
                minimum_probe_scale=0.01,
            )
        )
        result = optimizer.run_episode(
            model,
            batch,
            provider,
            forward_kwargs={"output_slots": 1},
        )
        self.assertGreaterEqual(result.total_forward_calls, 2)
        self.assertLessEqual(result.total_forward_calls, 3)
        self.assertTrue(torch.isfinite(result.state.current_loss).all())
        self.assertEqual(model(*batch[:1], output_slots=1).shape, (1, 1, 16))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
