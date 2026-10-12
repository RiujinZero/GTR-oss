from __future__ import annotations

import argparse
import math
import unittest

import torch

from gator import (
    OptimizationConfig,
    OptimizationEngine,
    SchedulerConfig,
)
from gator.cli import (
    add_optimization_arguments,
    optimization_config_from_args,
)


class _NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = value.shape
        return value.sum() * 0.0

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (
            torch.full(
                ctx.input_shape,
                float("nan"),
                dtype=gradient.dtype,
                device=gradient.device,
            ),
        )


class _HugeFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = value.shape
        return value.sum() * 0.0

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (
            torch.full(
                ctx.input_shape,
                1e30,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
        )


class OptimizationSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(2, 1, bias=False)
        self.engine = OptimizationEngine(
            self.model,
            OptimizationConfig(learning_rate=0.01, weight_decay=0.0),
        )

    def test_nonfinite_loss_cannot_mutate_model_or_step_counter(self) -> None:
        before = self.model.weight.detach().clone()
        loss = self.model.weight.sum() * float("nan")

        with self.assertRaisesRegex(FloatingPointError, "non-finite loss"):
            self.engine.step(loss)

        torch.testing.assert_close(self.model.weight, before)
        self.assertEqual(self.engine.step_number, 0)

    def test_nonfinite_gradient_cannot_mutate_model_or_step_counter(self) -> None:
        before = self.model.weight.detach().clone()
        loss = _NonFiniteGradient.apply(self.model.weight)

        with self.assertRaises(FloatingPointError) as raised:
            self.engine.step(loss)

        message = str(raised.exception)
        self.assertIn("optimizer step 1", message)
        self.assertIn("gradient norm", message)
        self.assertIn("weight", message)
        self.assertIn("nan=2", message)
        torch.testing.assert_close(self.model.weight, before)
        self.assertEqual(self.engine.step_number, 0)

    def test_finite_fp32_norm_overflow_is_clipped_and_updated(self) -> None:
        before = self.model.weight.detach().clone()
        loss = _HugeFiniteGradient.apply(self.model.weight)
        self.assertTrue(
            torch.isinf(torch.full((2,), 1e30, dtype=torch.float32).norm())
        )

        result = self.engine.step(loss)

        self.assertTrue(math.isfinite(result.gradient_norm))
        self.assertGreater(result.gradient_norm, 1e30)
        self.assertTrue(torch.isfinite(self.model.weight).all())
        self.assertFalse(torch.equal(self.model.weight, before))
        for state in self.engine.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.isfinite(value).all())

    def test_continuation_keeps_moments_but_restarts_scheduler_and_lr(self) -> None:
        first_model = torch.nn.Linear(2, 1, bias=False)
        first_engine = OptimizationEngine(
            first_model,
            OptimizationConfig(
                learning_rate=0.01,
                weight_decay=0.0,
                scheduler=SchedulerConfig(name="cosine", total_steps=10),
            ),
        )
        first_engine.step(first_model.weight.square().sum())
        state = first_engine.state_dict()

        resumed_model = torch.nn.Linear(2, 1, bias=False)
        resumed_engine = OptimizationEngine(
            resumed_model,
            OptimizationConfig(
                learning_rate=0.002,
                weight_decay=0.0,
                scheduler=SchedulerConfig(name="cosine", total_steps=100),
            ),
        )
        resumed_engine.load_state_dict(
            state,
            load_optimizer=True,
            load_scheduler=False,
            learning_rate=0.002,
        )

        self.assertEqual(resumed_engine.step_number, 1)
        self.assertAlmostEqual(resumed_engine.learning_rate, 0.002)
        self.assertTrue(resumed_engine.optimizer.state)
        assert resumed_engine.scheduler is not None
        self.assertEqual(resumed_engine.scheduler.T_max, 100)
        self.assertNotEqual(
            resumed_engine.scheduler.state_dict(),
            state["scheduler"],
        )

    def test_cli_exposes_adam_epsilon(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--steps", type=int, default=20_000)
        add_optimization_arguments(parser)
        arguments = parser.parse_args(
            ["--optimizer-epsilon", "0.0001"]
        )

        config = optimization_config_from_args(arguments)

        self.assertEqual(config.epsilon, 1e-4)

    def test_cli_can_preserve_a_longer_scheduler_horizon(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--steps", type=int, default=10_000)
        add_optimization_arguments(parser)
        arguments = parser.parse_args(
            [
                "--scheduler",
                "cosine",
                "--scheduler-total-steps",
                "50000",
            ]
        )

        config = optimization_config_from_args(arguments)

        self.assertEqual(config.scheduler.total_steps, 50_000)


if __name__ == "__main__":
    unittest.main()
