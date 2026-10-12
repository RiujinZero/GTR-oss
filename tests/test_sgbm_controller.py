"""Tests for SGBM Phase D: write-gate controller."""

from __future__ import annotations

import pytest
import torch

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.controller import (
    LearnedWriteGate,
    WriteDecision,
    WriteGateCoefficients,
    commit_strength_from_gate,
    decide_write,
    write_gate_strength,
)


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(fused_dim=8, quarantine_threshold=0.20, write_threshold=0.55)
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


class TestWriteGateStrength:
    def test_low_surprise_and_novelty_gate_near_zero(self) -> None:
        gate = write_gate_strength(torch.tensor([0.0]), torch.tensor([0.0]))
        assert gate.item() < 0.1

    def test_high_surprise_and_novelty_gate_near_one(self) -> None:
        gate = write_gate_strength(torch.tensor([5.0]), torch.tensor([5.0]))
        assert gate.item() > 0.9

    def test_custom_coefficients_shift_the_gate(self) -> None:
        weak = WriteGateCoefficients(surprise_weight=0.1, novelty_weight=0.1, bias=3.0)
        strong = WriteGateCoefficients(surprise_weight=10.0, novelty_weight=10.0, bias=3.0)
        low = write_gate_strength(torch.tensor([1.0]), torch.tensor([1.0]), coefficients=weak)
        high = write_gate_strength(torch.tensor([1.0]), torch.tensor([1.0]), coefficients=strong)
        assert high.item() > low.item()


class TestCommitStrength:
    def test_is_product_of_gate_and_reliability(self) -> None:
        commit = commit_strength_from_gate(torch.tensor([0.8]), torch.tensor([0.5]))
        assert commit.item() == pytest.approx(0.4)

    def test_zero_reliability_zeroes_commit_regardless_of_gate(self) -> None:
        commit = commit_strength_from_gate(torch.tensor([1.0]), torch.tensor([0.0]))
        assert commit.item() == pytest.approx(0.0)


class TestDecideWrite:
    def test_low_commit_strength_ignores(self) -> None:
        decision = decide_write(torch.tensor([0.05]), _config())
        assert int(decision.item()) == WriteDecision.IGNORE

    def test_mid_commit_strength_quarantines(self) -> None:
        decision = decide_write(torch.tensor([0.35]), _config())
        assert int(decision.item()) == WriteDecision.QUARANTINE

    def test_high_commit_strength_goes_stable(self) -> None:
        decision = decide_write(torch.tensor([0.9]), _config())
        assert int(decision.item()) == WriteDecision.STABLE

    def test_boundaries_are_inclusive(self) -> None:
        config = _config()
        at_quarantine = decide_write(torch.tensor([config.quarantine_threshold]), config)
        at_write = decide_write(torch.tensor([config.write_threshold]), config)
        assert int(at_quarantine.item()) == WriteDecision.QUARANTINE
        assert int(at_write.item()) == WriteDecision.STABLE

    def test_batched_decisions_are_independent(self) -> None:
        config = _config()
        decisions = decide_write(torch.tensor([0.0, 0.3, 0.9]), config)
        assert decisions.tolist() == [
            WriteDecision.IGNORE,
            WriteDecision.QUARANTINE,
            WriteDecision.STABLE,
        ]


class TestLearnedWriteGate:
    def test_output_shape_and_range(self) -> None:
        gate = LearnedWriteGate(feature_count=9)
        output = gate(torch.randn(4, 9))
        assert output.shape == (4,)
        assert bool((output > 0).all())
        assert bool((output < 1).all())

    def test_rejects_non_positive_feature_count(self) -> None:
        with pytest.raises(ValueError):
            LearnedWriteGate(feature_count=0)
