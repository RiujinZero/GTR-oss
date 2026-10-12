from __future__ import annotations

import unittest

import torch

from gator_sparse import SparseGator
from parallel_prefix_benchmark import make_prefix_batch


class SparseGatorTests(unittest.TestCase):
    def test_broad_start_is_uniform_without_changing_budget(self) -> None:
        model = SparseGator(broad_start=True)

        self.assertEqual(model.parameter_count, 10_010)
        probabilities = model.start_logits.softmax(-1)
        torch.testing.assert_close(
            probabilities,
            torch.full_like(probabilities, 1.0 / 40.0),
        )

    def test_broad_start_forward_and_gradient_are_finite(self) -> None:
        torch.manual_seed(41)
        model = SparseGator(broad_start=True)
        observed, targets = make_prefix_batch(
            3,
            7,
            torch.device("cpu"),
        )

        logits = model(observed, output_slots=7, temperature=1.25)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        loss.backward()

        self.assertEqual(tuple(logits.shape), (3, 7, 16))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
