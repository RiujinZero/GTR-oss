from __future__ import annotations

import unittest

import torch

from gator.optimizers.procoim import ProcoimAnchor
from gator_linear_prefix_train import _select_anchor_sentinels
from parallel_prefix_benchmark import (
    make_balanced_prefix_batch,
    make_prefix_batch,
)


class PortablePrefixBatchTests(unittest.TestCase):
    def test_dedicated_generator_reproduces_batch_stream(self) -> None:
        first = torch.Generator(device="cpu").manual_seed(1_000_003)
        second = torch.Generator(device="cpu").manual_seed(1_000_003)

        first_batches = [
            make_prefix_batch(
                8,
                7,
                torch.device("cpu"),
                generator=first,
                sampling_device=torch.device("cpu"),
            )
            for _ in range(3)
        ]
        second_batches = [
            make_prefix_batch(
                8,
                7,
                torch.device("cpu"),
                generator=second,
                sampling_device=torch.device("cpu"),
            )
            for _ in range(3)
        ]

        for (observed_a, targets_a), (observed_b, targets_b) in zip(
            first_batches,
            second_batches,
            strict=True,
        ):
            self.assertTrue(torch.equal(observed_a, observed_b))
            self.assertTrue(torch.equal(targets_a, targets_b))

    def test_saved_generator_state_resumes_at_next_batch(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(77)
        make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
            generator=generator,
            sampling_device=torch.device("cpu"),
        )
        state = generator.get_state()
        expected = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
            generator=generator,
            sampling_device=torch.device("cpu"),
        )

        resumed = torch.Generator(device="cpu")
        resumed.set_state(state)
        actual = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
            generator=resumed,
            sampling_device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))


class BalancedPrefixBatchTests(unittest.TestCase):
    def test_targets_are_exactly_balanced_and_prefix_is_valid(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(71)
        observed, targets = make_balanced_prefix_batch(
            24,
            7,
            torch.device("cpu"),
            generator=generator,
            sampling_device=torch.device("cpu"),
        )

        for slot in range(targets.shape[1]):
            counts = torch.bincount(targets[:, slot] - 4, minlength=12)
            self.assertTrue(torch.equal(counts, torch.full((12,), 2)))
        start = observed[:, 0] - 4
        reconstructed = (
            start[:, None] + observed[:, 1:].cumsum(dim=1)
        ).remainder(12) + 4
        self.assertTrue(torch.equal(targets, reconstructed))

    def test_requires_a_positive_multiple_of_twelve(self) -> None:
        for batch_size in (0, 8, 13):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "multiple of 12"):
                    make_balanced_prefix_batch(
                        batch_size,
                        2,
                        torch.device("cpu"),
                    )


class ProcoimSentinelSelectionTests(unittest.TestCase):
    def test_pending_anchor_replaces_same_id_and_can_be_excluded(self) -> None:
        permanent = ProcoimAnchor(
            batch="permanent",
            best_aggregate=1.0,
            anchor_id=("fixed", 0),
        )
        other = ProcoimAnchor(
            batch="other",
            best_aggregate=1.0,
            anchor_id=("fixed", 1),
        )
        pending = ProcoimAnchor(
            batch="pending",
            best_aggregate=0.9,
            anchor_id=("fixed", 0),
        )

        selected = _select_anchor_sentinels(
            (permanent, other, pending),
            count=2,
            episode=0,
        )
        self.assertIn(pending, selected)
        self.assertNotIn(permanent, selected)

        excluded = _select_anchor_sentinels(
            (permanent, other, pending),
            count=2,
            episode=0,
            exclude_id=("fixed", 1),
        )
        self.assertEqual(excluded, (pending,))


if __name__ == "__main__":
    unittest.main()
