from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from gator.optimizers.procoim import ProcoimConfig, ProcoimOptimizer
from parallel_prefix_benchmark import (
    PrefixTransformer980,
    make_prefix_batch,
    prefix_slot_loss_provider,
)


def _exhaustive_length_one_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return every start/delta pair for the 12-way length-one task."""

    starts = torch.arange(12).repeat_interleave(12)
    deltas = torch.arange(12).repeat(12)
    observed = torch.stack((starts + 4, deltas), dim=1)
    targets = (4 + (starts + deltas).remainder(12)).unsqueeze(1)
    return observed, targets


@torch.no_grad()
def _validation_metrics(
    model: PrefixTransformer980,
) -> tuple[float, float]:
    observed, targets = _exhaustive_length_one_batch()
    logits = model(observed, output_slots=1)
    loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
    accuracy = logits.argmax(dim=-1).eq(targets).float().mean()
    return float(loss), float(accuracy)


class ProcoimGeneratedPrefixIntegrationTests(unittest.TestCase):
    def test_forward_only_updates_learn_above_chance_on_fresh_prefixes(
        self,
    ) -> None:
        """A small fixed-seed proof, not a claim of prefix-task mastery."""

        torch.manual_seed(0)
        model = PrefixTransformer980()
        optimizer = ProcoimOptimizer(
            ProcoimConfig(
                reduction_fraction=0.005,
                candidate_directions=4,
                max_candidate_banks=1,
                max_accepted_updates=1,
                stability_required=1,
                minimum_active_gain=0.0,
                probe_scale=0.02,
                minimum_probe_scale=0.02,
                seed=0,
            )
        )
        initial_parameters = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]
        initial_loss, initial_accuracy = _validation_metrics(model)
        saw_protected_slot = False

        episode_count = 1000
        for episode in range(episode_count):
            batch = make_prefix_batch(
                batch_size=64,
                length=1,
                device=torch.device("cpu"),
            )
            result = optimizer.run_episode(
                model,
                batch,
                prefix_slot_loss_provider,
                forward_kwargs={"output_slots": 1},
                example_id=episode,
            )
            saw_protected_slot |= result.protected_slots > 0

        final_loss, final_accuracy = _validation_metrics(model)
        parameters_changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                initial_parameters,
                model.parameters(),
                strict=True,
            )
        )

        self.assertEqual(optimizer.episodes, episode_count)
        self.assertGreaterEqual(optimizer.total_forward_calls, episode_count)
        self.assertLess(
            optimizer.total_forward_calls,
            episode_count * (1 + 2 * optimizer.config.candidate_directions),
        )
        self.assertGreater(optimizer.total_accepted_updates, 0)
        self.assertTrue(saw_protected_slot)
        self.assertTrue(parameters_changed)
        self.assertLess(final_loss, initial_loss)
        self.assertGreater(final_accuracy, initial_accuracy)
        self.assertGreater(final_accuracy, 1.0 / 12.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
