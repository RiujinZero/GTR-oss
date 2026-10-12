from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from gator.tasks import legal_response_token_mask, whitespace_token_mask
from gator.tokenization import CompactTokenizer
from gator_language_train import formation_loss_components


class FormationObjectiveIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CompactTokenizer(["word", "next", " "])
        self.blank = self.tokenizer.blank_token_id
        assert self.blank is not None
        self.word = self.tokenizer.encode("word")[0]
        self.next_token = self.tokenizer.encode("next")[0]
        self.valid = legal_response_token_mask(self.tokenizer)
        whitespace = whitespace_token_mask(self.tokenizer)
        self.semantic = self.valid & ~whitespace
        self.semantic = self.semantic.clone()
        self.semantic[self.tokenizer.EOS_ID] = False
        self.semantic[self.blank] = False
        self.batch = SimpleNamespace(
            target_ids=torch.tensor(
                [
                    [self.word, self.next_token, self.tokenizer.EOS_ID, 0],
                    [self.next_token, self.word, self.next_token, self.word],
                ]
            ),
            target_mask=torch.tensor(
                [[True, True, True, False], [True, True, True, True]]
            ),
            target_complete=torch.tensor([True, False]),
        )

    def test_every_objective_mode_is_finite_and_backpropagates(self) -> None:
        for objective in ("slot_ce", "blank_ce", "viterbi", "monotonic"):
            raw_logits = torch.randn(
                2, 4, self.tokenizer.vocab_size, requires_grad=True
            )
            logits = raw_logits.masked_fill(
                ~self.valid[None, None], float("-inf")
            )
            components = formation_loss_components(
                logits,
                self.batch,
                self.tokenizer,
                valid_token_mask=self.valid,
                semantic_token_mask=self.semantic,
                objective=objective,
                alignment_mix=1.0,
                max_right_shift=2,
                sequence_composition_weight=1.0,
                sequence_length_weight=0.25,
                sequence_eos_weight=0.1,
                compute_all=True,
                compute_sequence=True,
            )
            self.assertTrue(torch.isfinite(components["selected"]))
            assert components["sequence"] is not None
            total = components["selected"] + 0.01 * components["sequence"].loss
            total.backward()
            assert raw_logits.grad is not None
            self.assertTrue(torch.isfinite(raw_logits.grad).all())
            self.assertGreater(float(raw_logits.grad.abs().sum()), 0.0)

    def test_alignment_warmup_interpolates_from_blank_ce(self) -> None:
        logits = torch.randn(2, 4, self.tokenizer.vocab_size)
        components = formation_loss_components(
            logits,
            self.batch,
            self.tokenizer,
            valid_token_mask=self.valid,
            semantic_token_mask=self.semantic,
            objective="monotonic",
            alignment_mix=0.0,
            max_right_shift=2,
            sequence_composition_weight=1.0,
            sequence_length_weight=0.25,
            sequence_eos_weight=0.1,
            compute_all=False,
            compute_sequence=False,
        )
        torch.testing.assert_close(components["selected"], components["blank_ce"])
        self.assertIsNone(components["sequence"])


if __name__ == "__main__":
    unittest.main()
