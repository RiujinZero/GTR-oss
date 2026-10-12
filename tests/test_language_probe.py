from __future__ import annotations

import unittest

import torch

from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)
from language_probe import model_from_checkpoint


class LanguageProbeTests(unittest.TestCase):
    def test_loads_legacy_convergent_checkpoint_as_global_vote_model(
        self,
    ) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
        )
        reference = ConvergentLanguageModel(config)
        legacy_config = config.to_dict()
        legacy_config.pop("terminal_vote_mode")
        loaded = model_from_checkpoint(
            {
                "model_config": legacy_config,
                "model": reference.state_dict(),
            }
        )

        self.assertIsInstance(loaded, ConvergentLanguageModel)
        self.assertEqual(loaded.config.terminal_vote_mode, "global")
        context = torch.randint(2, 63, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        torch.testing.assert_close(
            loaded(context, mask, output_slots=4),
            reference(context, mask, output_slots=4),
        )

    def test_loads_prompt_conditioned_convergent_checkpoint(self) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="prompt",
        )
        reference = ConvergentLanguageModel(config)
        with torch.no_grad():
            reference.core.terminal_vote_query.normal_()
        loaded = model_from_checkpoint(
            {
                "model_config": config.to_dict(),
                "model": reference.state_dict(),
            }
        )

        self.assertIsInstance(loaded, ConvergentLanguageModel)
        self.assertEqual(loaded.config.terminal_vote_mode, "prompt")
        torch.testing.assert_close(
            loaded.core.terminal_vote_query,
            reference.core.terminal_vote_query,
        )

    def test_loads_slot_conditioned_convergent_checkpoint(self) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="slot",
        )
        reference = ConvergentLanguageModel(config)
        with torch.no_grad():
            reference.core.terminal_vote_query.normal_()
        loaded = model_from_checkpoint(
            {
                "model_config": config.to_dict(),
                "model": reference.state_dict(),
            }
        )

        self.assertIsInstance(loaded, ConvergentLanguageModel)
        self.assertEqual(loaded.config.terminal_vote_mode, "slot")
        torch.testing.assert_close(
            loaded.core.terminal_vote_query,
            reference.core.terminal_vote_query,
        )


if __name__ == "__main__":
    unittest.main()
