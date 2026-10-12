"""Tests for Procoim-S Phase 8: A-B-A / interleaved benchmark evaluation harness."""

from __future__ import annotations

import torch

from gator.continual.benchmarks import evaluate_interleaved_accuracy, evaluate_task_accuracy
from gator.continual.task_schedule import ModularArithmeticTask, generate_interleaved_stream
from gator_linear import GatorLinearConfig, GatorLinearPrefixConfig, GatorLinearPrefixModel


def _tiny_model(vocab_size: int = 16) -> GatorLinearPrefixModel:
    core = GatorLinearConfig(
        vocab_size=vocab_size,
        streams=3,
        nodes=8,
        width=6,
        relays=2,
        degree=4,
    )
    return GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core))


class TestEvaluateTaskAccuracy:
    def test_returns_a_value_in_unit_interval(self) -> None:
        model = _tiny_model()
        task = ModularArithmeticTask(name="mod12", modulus=12)
        generator = torch.Generator().manual_seed(0)
        accuracy = evaluate_task_accuracy(model, task, count=8, input_length=3, generator=generator)
        assert 0.0 <= accuracy <= 1.0

    def test_runs_without_error_for_a_second_task(self) -> None:
        model = _tiny_model()
        task = ModularArithmeticTask(name="mod7", modulus=7, offset=2)
        generator = torch.Generator().manual_seed(1)
        accuracy = evaluate_task_accuracy(model, task, count=6, input_length=2, generator=generator)
        assert 0.0 <= accuracy <= 1.0


class TestEvaluateInterleavedAccuracy:
    def test_returns_per_task_accuracy_dict(self) -> None:
        model = _tiny_model()
        task_a = ModularArithmeticTask(name="a", modulus=12)
        task_b = ModularArithmeticTask(name="b", modulus=7)
        generator = torch.Generator().manual_seed(2)
        stream = generate_interleaved_stream(
            [task_a, task_b], count=12, input_length=3, generator=generator
        )
        result = evaluate_interleaved_accuracy(model, stream)
        assert set(result.keys()) <= {"a", "b"}
        assert all(0.0 <= value <= 1.0 for value in result.values())

    def test_empty_stream_returns_empty_dict(self) -> None:
        model = _tiny_model()
        assert evaluate_interleaved_accuracy(model, []) == {}
