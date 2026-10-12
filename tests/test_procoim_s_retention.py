"""Tests for the aggregate-utility Procoim retention window integration.

Orchestration correctness is tested against a tiny synthetic model with a
fully controllable scalar_loss_fn (mirroring test_consolidation.py's
approach); a separate smoke test exercises the real integration path
against an actual GatorLinearPrefixModel + prefix_slot_loss_provider to
confirm the pieces actually fit together end to end.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gator.continual.replay_reservoir import ReplayReservoir
from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.procoim_s_retention import ProcoimSRetentionWindow
from gator.optimizers.procoim import ProcoimConfig, ProcoimOptimizer
from gator.training import ProcoimTrainingEngine


class _TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, observed: torch.Tensor, **_: object) -> torch.Tensor:
        return self.linear(observed)

    def procoim_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"routing": [self.linear.weight, self.linear.bias]}


def _engine() -> ProcoimTrainingEngine:
    model = _TinyLinearModel()
    config = ProcoimConfig(parameter_group_schedule=("routing",))
    return ProcoimTrainingEngine(model, config)


def _window(engine: ProcoimTrainingEngine, **overrides) -> ProcoimSRetentionWindow:
    defaults = dict(
        interval_examples=1,
        commit_alphas=(1.0, 0.5, 0.25),
        replay_reservoir=ReplayReservoir(capacity=16),
        config=ProcoimSConfig(minimum_utility=0.0, replay_weight=1.0, replay_soft_tolerance=0.1),
        replay_sample_size=0,  # keep tests focused on the current-batch gain alone
    )
    defaults.update(overrides)
    return ProcoimSRetentionWindow(engine, **defaults)


def _constant_loss_fn(losses_by_alpha_call: list[float]):
    """A scalar_loss_fn stub that returns successive values from a list,
    letting a test dictate exactly what "loss" the window observes at each
    forward call regardless of the model's actual parameters."""

    calls = iter(losses_by_alpha_call)

    def fn(model, batch, forward_kwargs):
        return next(calls)

    return fn


class TestConstruction:
    def test_rejects_non_positive_interval(self) -> None:
        with pytest.raises(ValueError):
            _window(_engine(), interval_examples=0)

    def test_rejects_empty_commit_alphas(self) -> None:
        with pytest.raises(ValueError):
            _window(_engine(), commit_alphas=())

    def test_rejects_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            _window(_engine(), commit_alphas=(1.5,))

    def test_rejects_negative_replay_sample_size(self) -> None:
        with pytest.raises(ValueError):
            _window(_engine(), replay_sample_size=-1)


class TestRecordEpisodeAndDue:
    def test_due_becomes_true_after_interval(self) -> None:
        window = _window(_engine(), interval_examples=2)
        assert not window.due
        window.record_episode()
        assert not window.due
        window.record_episode()
        assert window.due

    def test_has_provisional_updates_tracks_examples_in_window(self) -> None:
        window = _window(_engine())
        assert not window.has_provisional_updates
        window.record_episode()
        assert window.has_provisional_updates


class TestConsolidateWithoutABatch:
    def test_trivially_commits_at_alpha_one(self) -> None:
        window = _window(_engine())
        window.record_episode()  # no batch supplied
        result = window.consolidate(scalar_loss_fn=lambda *args: 0.0)
        assert result.committed_alpha == pytest.approx(1.0)
        assert not result.rolled_back
        assert window.windows_committed == 1


class TestConsolidateWithABatch:
    def test_improving_delta_commits_at_largest_alpha(self) -> None:
        engine = _engine()
        window = _window(engine, commit_alphas=(1.0,))
        batch = (torch.zeros(1, 4), torch.zeros(1, 4))
        window.record_episode(batch=batch)

        with torch.no_grad():
            engine.model.linear.bias.add_(1.0)  # a "provisional update" already applied live

        # base loss (theta_base, before the +1.0) = 2.0; at alpha=1.0
        # (fully live) = 0.5 -- current_gain = 1.5 > 0, no replay items
        # (replay_sample_size=0) -> accepted.
        loss_fn = _constant_loss_fn([2.0, 0.5])
        result = window.consolidate(scalar_loss_fn=loss_fn)

        assert result.committed_alpha == pytest.approx(1.0)
        assert not result.rolled_back
        assert window.windows_committed == 1

    def test_degrading_delta_rolls_back_to_base(self) -> None:
        engine = _engine()
        window = _window(engine, commit_alphas=(1.0,))
        batch = (torch.zeros(1, 4), torch.zeros(1, 4))
        before = engine.model.linear.bias.clone()
        window.record_episode(batch=batch)

        with torch.no_grad():
            engine.model.linear.bias.add_(1.0)

        # current_gain = base(1.0) - after(3.0) = -2.0 -> rejected at every alpha
        loss_fn = _constant_loss_fn([1.0, 3.0])
        result = window.consolidate(scalar_loss_fn=loss_fn)

        assert result.rolled_back
        assert result.committed_alpha == pytest.approx(0.0)
        assert torch.allclose(engine.model.linear.bias, before)
        assert window.windows_rolled_back == 1

    def test_falls_back_to_smaller_alpha_when_larger_one_regresses_replay(self) -> None:
        engine = _engine()
        reservoir = ReplayReservoir(capacity=16)
        window = _window(
            engine,
            commit_alphas=(1.0, 0.5),
            replay_reservoir=reservoir,
            replay_sample_size=1,
            config=ProcoimSConfig(
                minimum_utility=0.0,
                replay_weight=1.0,
                replay_soft_tolerance=0.0,
                replay_catastrophic_tolerance=0.05,
            ),
        )
        batch = (torch.zeros(1, 4), torch.zeros(1, 4))
        window.record_episode(batch=batch)  # also seeds the reservoir with this batch
        with torch.no_grad():
            engine.model.linear.bias.add_(1.0)

        # Sequence of scalar_loss_fn calls inside consolidate():
        # 1) base current loss
        # 2) base replay loss (the one sampled item, same batch as current)
        # 3) alpha=1.0 current loss (gain positive)
        # 4) alpha=1.0 replay loss (catastrophic regression -> rejected)
        # 5) alpha=0.5 current loss (gain positive)
        # 6) alpha=0.5 replay loss (small regression, within tolerance -> accepted)
        loss_fn = _constant_loss_fn([1.0, 1.0, 0.2, 10.0, 0.5, 1.02])
        result = window.consolidate(scalar_loss_fn=loss_fn)

        assert result.committed_alpha == pytest.approx(0.5)
        assert not result.rolled_back


class TestRollbackUnconsolidated:
    def test_restores_base_and_clears_window(self) -> None:
        engine = _engine()
        window = _window(engine)
        before = engine.model.linear.bias.clone()
        window.record_episode(batch=(torch.zeros(1, 4), torch.zeros(1, 4)))
        with torch.no_grad():
            engine.model.linear.bias.add_(5.0)

        window.rollback_unconsolidated()

        assert torch.allclose(engine.model.linear.bias, before)
        assert not window.has_provisional_updates
        assert window.windows_rolled_back == 1

    def test_is_a_no_op_with_no_provisional_updates(self) -> None:
        window = _window(_engine())
        window.rollback_unconsolidated()
        assert window.windows_rolled_back == 0


class TestRealModelSmoke:
    def test_integration_with_a_real_gator_linear_prefix_model(self) -> None:
        from gator_linear import GatorLinearConfig, GatorLinearPrefixConfig, GatorLinearPrefixModel
        from gator_linear_prefix_train import prefix_slot_loss_provider

        core = GatorLinearConfig(vocab_size=16, streams=3, nodes=8, width=6, relays=2, degree=4)
        model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core))
        procoim_config = ProcoimConfig(
            parameter_group_schedule=("routing", "receiver_memory", "stream_mix", "node_compute", "readout")
        )
        engine = ProcoimTrainingEngine(model, procoim_config)
        window = ProcoimSRetentionWindow(
            engine,
            interval_examples=1,
            commit_alphas=(1.0, 0.5),
            replay_reservoir=ReplayReservoir(capacity=8),
            config=ProcoimSConfig(minimum_utility=-1e9, replay_weight=1.0),  # accept unconditionally
            replay_sample_size=1,
        )

        observed = torch.randint(0, 16, (2, 4))
        targets = torch.randint(0, 16, (2, 2))
        forward_kwargs = {"output_slots": 2}

        def scalar_loss_fn(model, batch, forward_kwargs):
            return float(prefix_slot_loss_provider(model, batch, forward_kwargs).aggregate)

        window.record_episode(batch=(observed, targets), forward_kwargs=forward_kwargs)
        with torch.no_grad():
            for parameter in model.linear.parameters() if hasattr(model, "linear") else model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)

        result = window.consolidate(scalar_loss_fn=scalar_loss_fn)
        assert result.committed_alpha in (0.0, 0.5, 1.0)
