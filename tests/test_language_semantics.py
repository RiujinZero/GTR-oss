from __future__ import annotations

import unittest

import torch

from gator.data import AssistantExample, Conversation, DialogueTurn
from gator.models import (
    GatorLanguageConfig,
    GatorLanguageModel,
    OneShotTransformerLanguageConfig,
    OneShotTransformerLanguageModel,
)
from gator.tasks import (
    EncodedAssistantExample,
    LanguageBatchConfig,
    LanguageBatcher,
    StreamingLanguageSource,
    WeightedLanguageSource,
    legal_response_token_mask,
    masked_cross_entropy,
    rigid_blank_tail_targets,
    whitespace_token_mask,
)
from gator.tokenization import CompactTokenizer


class CompactTokenizerV2Tests(unittest.TestCase):
    def test_v1_round_trip_and_ids_remain_exact(self) -> None:
        state = {
            "format": "gator-compact-v1",
            "learned_pieces": ["hello", " ", "world"],
        }
        v1 = CompactTokenizer.from_dict(state)
        v2 = CompactTokenizer(state["learned_pieces"])

        self.assertFalse(v1.has_blank)
        self.assertIsNone(v1.blank_token_id)
        self.assertEqual(v1.to_dict(), state)
        self.assertEqual(v1.vocab_size, CompactTokenizer.LEARNED_OFFSET + 3)
        self.assertEqual(v1.encode("hello world"), v2.encode("hello world"))
        self.assertEqual(
            v1.encode("unseen"),
            [CompactTokenizer.BYTE_OFFSET + byte for byte in b"unseen"],
        )
        self.assertEqual(
            v1.decode(v1.encode("hello world")),
            v2.decode(v2.encode("hello world")),
        )
        self.assertTrue(v2.has_blank)
        self.assertEqual(v2.blank_token_id, v1.vocab_size)

    def test_training_reserves_final_requested_row_for_blank(self) -> None:
        requested_size = CompactTokenizer.LEARNED_OFFSET + 8
        conversation = Conversation(
            conversation_id=0,
            turns=(
                DialogueTurn(
                    "user",
                    " ".join(f"piece{index}" for index in range(32)),
                ),
            ),
        )
        tokenizer = CompactTokenizer.train(
            [conversation], vocab_size=requested_size
        )

        self.assertEqual(tokenizer.format_version, CompactTokenizer.FORMAT_V2)
        self.assertEqual(tokenizer.vocab_size, requested_size)
        self.assertEqual(tokenizer.blank_token_id, requested_size - 1)
        self.assertEqual(len(tokenizer.learned_pieces), 7)
        self.assertEqual(tokenizer.learned_end, tokenizer.blank_token_id)

    def test_v2_training_rejects_an_unfillable_requested_vocabulary(self) -> None:
        conversation = Conversation(
            conversation_id=0,
            turns=(DialogueTurn("user", "only"),),
        )
        with self.assertRaisesRegex(ValueError, "lower vocab_size"):
            CompactTokenizer.train(
                [conversation],
                vocab_size=CompactTokenizer.LEARNED_OFFSET + 4,
            )

    def test_blank_is_hidden_normally_and_visible_diagnostically(self) -> None:
        tokenizer = CompactTokenizer(["hello", "world"])
        hello = tokenizer.encode("hello")[0]
        world = tokenizer.encode("world")[0]
        blank = tokenizer.blank_token_id
        assert blank is not None
        ids = [tokenizer.PAD_ID, hello, blank, tokenizer.EOS_ID, world]

        self.assertEqual(tokenizer.decode(ids), "hello")
        self.assertEqual(
            tokenizer.decode(ids, diagnostic=True),
            "<pad>hello<blank><eos>world",
        )
        self.assertEqual(
            tokenizer.decode(
                [tokenizer.USER_ID, tokenizer.ASSISTANT_ID],
                show_roles=False,
            ),
            "",
        )

    def test_blank_does_not_split_utf8_byte_fallback(self) -> None:
        tokenizer = CompactTokenizer([])
        encoded = tokenizer.encode("é")
        self.assertGreater(len(encoded), 1)
        blank = tokenizer.blank_token_id
        assert blank is not None

        interleaved = [encoded[0], blank, *encoded[1:]]
        self.assertEqual(tokenizer.decode(interleaved), "é")


class LanguageTokenSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CompactTokenizer([" ", "  ", "\n", "\t", "word", "next"])

    def test_response_vocabulary_censors_control_tokens(self) -> None:
        legal = legal_response_token_mask(self.tokenizer)
        for token_id in (
            self.tokenizer.PAD_ID,
            self.tokenizer.BOS_ID,
            self.tokenizer.MASK_ID,
            self.tokenizer.USER_ID,
            self.tokenizer.ASSISTANT_ID,
        ):
            self.assertFalse(bool(legal[token_id]), token_id)

        self.assertTrue(bool(legal[self.tokenizer.EOS_ID]))
        self.assertTrue(bool(legal[self.tokenizer.BYTE_OFFSET + ord("x")]))
        self.assertTrue(bool(legal[self.tokenizer.encode("word")[0]]))
        assert self.tokenizer.blank_token_id is not None
        self.assertTrue(bool(legal[self.tokenizer.blank_token_id]))

    def test_whitespace_mask_excludes_all_structural_tokens(self) -> None:
        whitespace = whitespace_token_mask(self.tokenizer)
        for text in (" ", "  ", "\n", "\t"):
            self.assertTrue(bool(whitespace[self.tokenizer.encode(text)[0]]), repr(text))
        self.assertTrue(
            bool(whitespace[self.tokenizer.BYTE_OFFSET + ord(" ")])
        )
        self.assertFalse(bool(whitespace[self.tokenizer.encode("word")[0]]))
        for token_id in (
            self.tokenizer.PAD_ID,
            self.tokenizer.BOS_ID,
            self.tokenizer.EOS_ID,
            self.tokenizer.MASK_ID,
            self.tokenizer.USER_ID,
            self.tokenizer.ASSISTANT_ID,
            self.tokenizer.blank_token_id,
        ):
            assert token_id is not None
            self.assertFalse(bool(whitespace[token_id]), token_id)

    def test_collation_marks_only_windows_where_eos_survives_complete(self) -> None:
        word = self.tokenizer.encode("word")[0]
        next_token = self.tokenizer.encode("next")[0]
        examples = [
            EncodedAssistantExample(0, 1, (10,), (word, self.tokenizer.EOS_ID)),
            EncodedAssistantExample(
                1,
                1,
                (11,),
                (word, next_token, word, next_token, self.tokenizer.EOS_ID),
            ),
            EncodedAssistantExample(
                2,
                1,
                (12,),
                (word, next_token, word, self.tokenizer.EOS_ID),
            ),
        ]
        batch = LanguageBatcher(
            examples,
            LanguageBatchConfig(context_tokens=2, response_tokens=4),
        ).collate(examples, "cpu")

        self.assertEqual(
            batch.target_ids[0].tolist(),
            [word, self.tokenizer.EOS_ID, self.tokenizer.PAD_ID, self.tokenizer.PAD_ID],
        )
        self.assertEqual(batch.target_mask[0].tolist(), [True, True, False, False])
        self.assertEqual(batch.target_mask[1].tolist(), [True, True, True, True])
        self.assertEqual(batch.target_complete.tolist(), [True, False, True])

    def test_rigid_blank_tail_scores_only_known_complete_tails(self) -> None:
        blank = self.tokenizer.blank_token_id
        assert blank is not None
        target_ids = torch.tensor(
            [
                [20, self.tokenizer.EOS_ID, self.tokenizer.PAD_ID, self.tokenizer.PAD_ID],
                [21, 22, self.tokenizer.PAD_ID, self.tokenizer.PAD_ID],
            ]
        )
        target_mask = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )
        target_complete = torch.tensor([True, False])

        labels, loss_mask = rigid_blank_tail_targets(
            target_ids,
            target_mask,
            target_complete,
            blank_token_id=blank,
        )

        self.assertEqual(
            labels.tolist(),
            [
                [20, self.tokenizer.EOS_ID, blank, blank],
                [21, 22, self.tokenizer.PAD_ID, self.tokenizer.PAD_ID],
            ],
        )
        self.assertEqual(
            loss_mask.tolist(),
            [[True, True, True, True], [True, True, False, False]],
        )
        self.assertEqual(
            target_ids[0].tolist(),
            [20, self.tokenizer.EOS_ID, self.tokenizer.PAD_ID, self.tokenizer.PAD_ID],
        )

    def test_reference_cap_reserves_physical_alignment_slack(self) -> None:
        word = self.tokenizer.encode("word")[0]
        next_token = self.tokenizer.encode("next")[0]
        example = EncodedAssistantExample(
            0,
            1,
            (word,),
            (word, next_token, word, self.tokenizer.EOS_ID),
        )
        batch = LanguageBatcher(
            [example],
            LanguageBatchConfig(
                context_tokens=2,
                response_tokens=4,
                reference_tokens=2,
            ),
        ).collate([example], "cpu")
        self.assertEqual(batch.target_mask.tolist(), [[True, True, False, False]])
        self.assertEqual(batch.target_complete.tolist(), [False])
        blank = self.tokenizer.blank_token_id
        assert blank is not None
        labels, mask = rigid_blank_tail_targets(
            batch.target_ids,
            batch.target_mask,
            batch.target_complete,
            blank_token_id=blank,
            score_censored_tail=True,
        )
        self.assertEqual(labels.tolist(), [[word, next_token, blank, blank]])
        self.assertEqual(mask.tolist(), [[True, True, True, True]])

    def _model_configs(self):
        legal = legal_response_token_mask(self.tokenizer)
        forbidden = tuple((~legal).nonzero(as_tuple=False).flatten().tolist())
        blank = self.tokenizer.blank_token_id
        assert blank is not None
        yield GatorLanguageModel(
            GatorLanguageConfig(
                vocab_size=self.tokenizer.vocab_size,
                d_model=4,
                embedding_dim=4,
                n_cells=4,
                n_nodes=4,
                relays=2,
                neighbors=2,
                node_topology="symmetric",
                lexical_mode="communal",
                mask_token_id=self.tokenizer.MASK_ID,
                blank_token_id=blank,
                forbidden_output_token_ids=forbidden,
            )
        )
        yield OneShotTransformerLanguageModel(
            OneShotTransformerLanguageConfig(
                vocab_size=self.tokenizer.vocab_size,
                d_model=8,
                n_heads=2,
                ff_width=16,
                n_layers=2,
                context_tokens=3,
                response_tokens=2,
                dropout=0.0,
                mask_token_id=self.tokenizer.MASK_ID,
                blank_token_id=blank,
                forbidden_output_token_ids=forbidden,
            )
        )

    def test_response_models_mask_forbidden_logits_and_gradients(self) -> None:
        legal = legal_response_token_mask(self.tokenizer)
        word = self.tokenizer.encode("word")[0]
        context = torch.tensor([[self.tokenizer.PAD_ID, word, word]])
        context_mask = torch.tensor([[False, True, True]])
        for model in self._model_configs():
            model.train()
            model.zero_grad(set_to_none=True)
            logits = model(context, context_mask, output_slots=2)
            self.assertTrue(torch.isneginf(logits[..., ~legal]).all())
            self.assertTrue(logits.softmax(-1)[..., ~legal].eq(0).all())
            loss = -logits.log_softmax(-1)[..., word].mean()
            loss.backward()
            assert model.lm_bias.grad is not None
            self.assertTrue(model.lm_bias.grad[~legal].eq(0).all())

    def test_padding_values_cannot_affect_broadcast_or_transformer_outputs(self) -> None:
        word = self.tokenizer.encode("word")[0]
        next_token = self.tokenizer.encode("next")[0]
        context_a = torch.tensor([[self.tokenizer.PAD_ID, word, next_token]])
        context_b = torch.tensor([[next_token, word, next_token]])
        context_mask = torch.tensor([[False, True, True]])
        for model in self._model_configs():
            model.eval()
            with torch.no_grad():
                logits_a = model(context_a, context_mask, output_slots=2)
                logits_b = model(context_b, context_mask, output_slots=2)
            torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)

    def test_masked_ce_never_evaluates_impossible_padding_targets(self) -> None:
        vocabulary = self.tokenizer.vocab_size
        word = self.tokenizer.encode("word")[0]
        logits = torch.zeros(1, 2, vocabulary, requires_grad=True)
        masked_logits = logits.masked_fill(
            torch.nn.functional.one_hot(
                torch.tensor(self.tokenizer.PAD_ID), vocabulary
            ).bool()[None, None],
            float("-inf"),
        )
        targets = torch.tensor([[word, self.tokenizer.PAD_ID]])
        target_mask = torch.tensor([[True, False]])
        loss = masked_cross_entropy(masked_logits, targets, target_mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        assert logits.grad is not None
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_fully_masked_ce_is_finite_with_forbidden_logits(self) -> None:
        logits = torch.randn(
            1, 2, self.tokenizer.vocab_size, requires_grad=True
        )
        masked_logits = logits.masked_fill(
            ~legal_response_token_mask(self.tokenizer)[None, None, :],
            float("-inf"),
        )
        target_ids = torch.full(
            (1, 2), self.tokenizer.PAD_ID, dtype=torch.long
        )
        target_mask = torch.zeros((1, 2), dtype=torch.bool)

        loss = masked_cross_entropy(masked_logits, target_ids, target_mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        assert logits.grad is not None
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_weighted_sources_interleave_and_report_each_corpus(self) -> None:
        example_a = EncodedAssistantExample(1, 1, (10,), (20,), "a")
        example_b = EncodedAssistantExample(2, 1, (11,), (21,), "b")

        class FixedSource:
            def __init__(self, example):
                self.example = example
                self.restarts = 0

            def next(self):
                return self.example

        source = WeightedLanguageSource(
            [
                ("tulu", FixedSource(example_a)),
                ("preferences", FixedSource(example_b)),
            ],
            [1.0, 3.0],
            seed=7,
        )
        observed = [source.next().source for _ in range(1_000)]

        self.assertEqual(source.items_emitted, 1_000)
        self.assertEqual(sum(source.sample_counts.values()), 1_000)
        self.assertGreater(
            source.sample_counts["preferences"],
            source.sample_counts["tulu"],
        )
        self.assertIn("a", observed)
        self.assertIn("b", observed)
        self.assertEqual(source.restarts, {"tulu": 0, "preferences": 0})

    def test_nonrepeating_streams_drop_at_eof_without_duplicate_examples(
        self,
    ) -> None:
        tokenizer = CompactTokenizer(
            ["prompt", "response", "zero", "one", "two"]
        )

        def stream_factory(
            offset: int,
            count: int,
            source: str,
        ):
            examples = tuple(
                AssistantExample(
                    conversation_id=offset + index,
                    turn_index=1,
                    context=(DialogueTurn("user", f"prompt {index}"),),
                    response=f"{source} response {index}",
                    source=source,
                )
                for index in range(count)
            )
            return lambda: iter(examples)

        first = StreamingLanguageSource(
            stream_factory(0, 2, "first"),
            tokenizer,
            shuffle_buffer=8,
            seed=3,
            repeat=False,
        )
        second = StreamingLanguageSource(
            stream_factory(100, 4, "second"),
            tokenizer,
            shuffle_buffer=8,
            seed=5,
            repeat=False,
        )
        weighted = WeightedLanguageSource(
            [("first", first), ("second", second)],
            [1.0, 1.0],
            seed=7,
        )

        collator = LanguageBatcher(
            (
                EncodedAssistantExample(
                    999,
                    1,
                    (tokenizer.BOS_ID,),
                    (tokenizer.EOS_ID,),
                    "collator",
                ),
            ),
            LanguageBatchConfig(
                context_tokens=16,
                response_tokens=16,
            ),
            seed=11,
        )
        batch = weighted.sample_cpu(6, collator)

        self.assertEqual(batch.target_ids.shape, (6, 16))
        self.assertEqual(
            len(
                {
                    tuple(row.tolist())
                    for row in batch.target_ids
                }
            ),
            6,
        )
        self.assertEqual(weighted.sample_counts, {"first": 2, "second": 4})
        self.assertEqual(first.restarts, 0)
        self.assertEqual(second.restarts, 0)
        with self.assertRaisesRegex(ValueError, "all streaming"):
            weighted.next()
        self.assertEqual(
            weighted.exhausted_sources,
            {"first", "second"},
        )

        cycling = StreamingLanguageSource(
            stream_factory(200, 3, "cycling"),
            tokenizer,
            shuffle_buffer=8,
            seed=13,
            repeat=True,
        )
        first_epoch = [cycling.next() for _ in range(3)]
        self.assertEqual(
            {example.conversation_id for example in first_epoch},
            {200, 201, 202},
        )
        self.assertEqual(cycling.restarts, 0)

        second_epoch_first = cycling.next()
        self.assertIn(
            second_epoch_first.conversation_id,
            {200, 201, 202},
        )
        self.assertEqual(cycling.restarts, 1)

    def test_streaming_source_exact_state_round_trip(self) -> None:
        tokenizer = CompactTokenizer(["prompt", "response"])
        examples = tuple(
            AssistantExample(
                conversation_id=index,
                turn_index=1,
                context=(DialogueTurn("user", f"prompt {index}"),),
                response=f"response {index}",
                source="stream",
            )
            for index in range(17)
        )

        original = StreamingLanguageSource(
            lambda: iter(examples),
            tokenizer,
            shuffle_buffer=5,
            seed=19,
            repeat=True,
        )
        for _ in range(23):
            original.next()
        state = original.state_dict()
        expected = [original.next() for _ in range(30)]

        restored = StreamingLanguageSource(
            lambda: iter(examples),
            tokenizer,
            shuffle_buffer=5,
            seed=19,
            repeat=True,
        )
        restored.load_state_dict(state)
        observed = [restored.next() for _ in range(30)]

        self.assertTrue(original.exact_state_supported)
        self.assertTrue(state["exact"])
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
