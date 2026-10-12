import unittest

import torch

from gator.smollm3_train import (
    _assert_training_state_finite,
    _clip_gradients_or_raise,
    _require_finite_loss,
)


class SmolLM3FiniteGuardTests(unittest.TestCase):
    @staticmethod
    def model_and_optimizer():
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_finite_loss_guard_reports_exact_location_and_source(self):
        with self.assertRaisesRegex(FloatingPointError, "next_update=8") as raised:
            _require_finite_loss(
                torch.tensor(float("nan")),
                update=7,
                micro_step=231,
                sources=("fineweb-edu",),
            )
        message = str(raised.exception)
        self.assertIn("micro_step=231", message)
        self.assertIn("fineweb-edu", message)
        self.assertIn("optimizer step and checkpoint were skipped", message)

    def test_gradient_guard_rejects_nonfinite_gradient(self):
        model, _ = self.model_and_optimizer()
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        model.bias.grad = torch.zeros_like(model.bias)
        with self.assertRaisesRegex(
            FloatingPointError, "non-finite SmolLM3 gradient"
        ):
            _clip_gradients_or_raise(
                model,
                update=2,
                micro_step=96,
                sources=("dclm", "fineweb-edu"),
            )

    def test_checkpoint_guard_rejects_nonfinite_model_parameter(self):
        model, optimizer = self.model_and_optimizer()
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(
            FloatingPointError, "model parameter 'weight'"
        ):
            _assert_training_state_finite(
                model,
                optimizer,
                context="before checkpoint",
            )

    def test_checkpoint_guard_rejects_nonfinite_optimizer_state(self):
        model, optimizer = self.model_and_optimizer()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        state = next(iter(optimizer.state.values()))
        state["exp_avg"].view(-1)[0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "optimizer state"):
            _assert_training_state_finite(
                model,
                optimizer,
                context="before checkpoint",
            )

    def test_finite_training_state_and_gradient_pass(self):
        model, optimizer = self.model_and_optimizer()
        model(torch.ones(1, 2)).sum().backward()
        norm = _clip_gradients_or_raise(
            model,
            update=0,
            micro_step=32,
            sources=("dclm",),
        )
        self.assertGreater(norm, 0)
        optimizer.step()
        _assert_training_state_finite(
            model,
            optimizer,
            context="before checkpoint",
        )


if __name__ == "__main__":
    unittest.main()
