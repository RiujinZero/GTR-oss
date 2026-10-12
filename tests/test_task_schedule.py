"""Tests for Procoim-S Phase 8: synthetic task families and A-B-A/interleaved scheduling."""

from __future__ import annotations

import pytest
import torch

from gator.continual.task_schedule import (
    ModularArithmeticTask,
    build_aba_schedule,
    generate_interleaved_stream,
)


class TestModularArithmeticTask:
    def test_rejects_modulus_at_or_below_one(self) -> None:
        with pytest.raises(ValueError):
            ModularArithmeticTask(name="bad", modulus=1)

    def test_target_matches_manual_formula(self) -> None:
        task = ModularArithmeticTask(name="a", modulus=12, offset=3, scale=2)
        inputs = torch.tensor([1, 2, 3])
        target = task.target(inputs)
        assert int(target.item()) == (6 * 2 + 3) % 12

    def test_generate_produces_requested_count_and_shapes(self) -> None:
        task = ModularArithmeticTask(name="a", modulus=12)
        generator = torch.Generator().manual_seed(0)
        examples = task.generate(5, input_length=4, generator=generator)
        assert len(examples) == 5
        for inputs, target in examples:
            assert inputs.shape == (4,)
            assert target.shape == ()
            assert bool((inputs >= 0).all()) and bool((inputs < 12).all())

    def test_generate_targets_match_task_rule(self) -> None:
        task = ModularArithmeticTask(name="a", modulus=7, offset=1)
        generator = torch.Generator().manual_seed(1)
        examples = task.generate(3, input_length=2, generator=generator)
        for inputs, target in examples:
            assert torch.equal(target, task.target(inputs))

    def test_generate_rejects_non_positive_count_or_length(self) -> None:
        task = ModularArithmeticTask(name="a", modulus=12)
        generator = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError):
            task.generate(0, input_length=2, generator=generator)
        with pytest.raises(ValueError):
            task.generate(1, input_length=0, generator=generator)


class TestBuildABASchedule:
    def test_composes_three_phases_in_order(self) -> None:
        task_a = ModularArithmeticTask(name="mod12", modulus=12)
        task_b = ModularArithmeticTask(name="mod7", modulus=7)
        schedule = build_aba_schedule(task_a, task_b, examples_per_phase=10)
        assert schedule.phase_a.task is task_a
        assert schedule.phase_b.task is task_b
        assert schedule.phase_a_return.task is task_a
        assert schedule.phase_a.name == "A"
        assert schedule.phase_b.name == "B"
        assert schedule.phase_a_return.name == "A-return"
        assert all(
            phase.example_count == 10
            for phase in (schedule.phase_a, schedule.phase_b, schedule.phase_a_return)
        )

    def test_rejects_same_name_tasks(self) -> None:
        task_a = ModularArithmeticTask(name="same", modulus=12)
        task_b = ModularArithmeticTask(name="same", modulus=7)
        with pytest.raises(ValueError):
            build_aba_schedule(task_a, task_b, examples_per_phase=10)

    def test_rejects_non_positive_examples_per_phase(self) -> None:
        task_a = ModularArithmeticTask(name="a", modulus=12)
        task_b = ModularArithmeticTask(name="b", modulus=7)
        with pytest.raises(ValueError):
            build_aba_schedule(task_a, task_b, examples_per_phase=0)


class TestGenerateInterleavedStream:
    def test_produces_requested_count(self) -> None:
        task_a = ModularArithmeticTask(name="a", modulus=12)
        task_b = ModularArithmeticTask(name="b", modulus=7)
        generator = torch.Generator().manual_seed(0)
        stream = generate_interleaved_stream(
            [task_a, task_b], count=20, input_length=3, generator=generator
        )
        assert len(stream) == 20
        assert set(example.task_name for example in stream) <= {"a", "b"}

    def test_rejects_empty_task_list(self) -> None:
        generator = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError):
            generate_interleaved_stream([], count=5, input_length=3, generator=generator)

    def test_rejects_non_positive_count(self) -> None:
        task_a = ModularArithmeticTask(name="a", modulus=12)
        generator = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError):
            generate_interleaved_stream([task_a], count=0, input_length=3, generator=generator)

    def test_examples_carry_correct_targets_for_their_task(self) -> None:
        task_a = ModularArithmeticTask(name="a", modulus=12)
        task_b = ModularArithmeticTask(name="b", modulus=7, offset=2)
        generator = torch.Generator().manual_seed(2)
        stream = generate_interleaved_stream(
            [task_a, task_b], count=10, input_length=3, generator=generator
        )
        tasks_by_name = {"a": task_a, "b": task_b}
        for example in stream:
            expected = tasks_by_name[example.task_name].target(example.inputs)
            assert torch.equal(example.target, expected)
