from __future__ import annotations

import unittest

import torch

from gator.models import GatorLanguageConfig, GatorLanguageModel


def _config(*, lexical_embedding_dim: int | None) -> GatorLanguageConfig:
    return GatorLanguageConfig(
        vocab_size=31,
        d_model=8,
        embedding_dim=16,
        lexical_embedding_dim=lexical_embedding_dim,
        n_cells=4,
        n_nodes=4,
        relays=2,
        neighbors=2,
        mask_token_id=3,
        node_topology="symmetric",
        lexical_mode="communal",
        cell_input_mode="feature_gated",
        cell_output_mode="grouped_concat",
        receiver_memory_backend="linear_scan",
    )


class FactorizedGatorLanguageTests(unittest.TestCase):
    def test_legacy_default_keeps_the_original_tied_table_shape(self) -> None:
        model = GatorLanguageModel(_config(lexical_embedding_dim=None))

        self.assertEqual(model.config.language_width, 16)
        self.assertEqual(model.config.lexical_width, 16)
        self.assertEqual(tuple(model.token_embedding.weight.shape), (31, 16))
        self.assertIsInstance(model.lexical_projection, torch.nn.Identity)

    def test_factorization_reduces_parameters_and_serializes_width(self) -> None:
        legacy = GatorLanguageModel(_config(lexical_embedding_dim=None))
        factorized = GatorLanguageModel(_config(lexical_embedding_dim=4))

        self.assertEqual(tuple(factorized.token_embedding.weight.shape), (31, 4))
        self.assertEqual(tuple(factorized.lexical_projection.weight.shape), (16, 4))
        self.assertEqual(factorized.config.to_dict()["lexical_embedding_dim"], 4)
        self.assertEqual(legacy.parameter_count - factorized.parameter_count, 308)

    def test_tied_factorized_input_and_readout_both_receive_gradients(self) -> None:
        torch.manual_seed(7)
        model = GatorLanguageModel(_config(lexical_embedding_dim=4))
        context = torch.tensor([[4, 5, 6]], dtype=torch.long)
        context_mask = torch.ones_like(context, dtype=torch.bool)
        targets = torch.tensor([[7, 8]], dtype=torch.long)

        logits = model(context, context_mask, output_slots=2)
        self.assertEqual(tuple(logits.shape), (1, 2, 31))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()

        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertIsNotNone(model.lexical_projection.weight.grad)
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.lexical_projection.weight.grad.abs().sum()), 0.0)
        parameter_names = dict(model.named_parameters())
        self.assertIn("token_embedding.weight", parameter_names)
        self.assertIn("lexical_projection.weight", parameter_names)
        self.assertNotIn("lm_head.weight", parameter_names)


if __name__ == "__main__":
    unittest.main()
