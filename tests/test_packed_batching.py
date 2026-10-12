from __future__ import annotations

import random
import unittest

import torch

from gator.tasks import (
    EncodedAssistantExample,
    LanguageBatch,
    LanguageBatchConfig,
    LanguageBatcher,
    MixtureLanguageBatcher,
    PackedLanguageSource,
    WeightedLanguageSource,
)
from gator.tokenization import CompactTokenizer


def assert_batch_equal(
    case: unittest.TestCase,
    first: LanguageBatch,
    second: LanguageBatch,
) -> None:
    for name in (
        "context_ids",
        "context_mask",
        "target_ids",
        "target_mask",
        "target_complete",
    ):
        case.assertTrue(
            torch.equal(getattr(first, name), getattr(second, name)),
            name,
        )


def finite_examples() -> tuple[EncodedAssistantExample, ...]:
    eos = CompactTokenizer.EOS_ID
    return (
        EncodedAssistantExample(0, 1, (10,), (20, eos), "finite-a"),
        EncodedAssistantExample(
            1, 1, (11, 12, 13, 14), (21, 22, 23, eos), "finite-b"
        ),
        EncodedAssistantExample(2, 1, (15, 16), (24,), "finite-c"),
    )


def packed_source(seed: int = 0, offset: int = 0) -> PackedLanguageSource:
    pad = CompactTokenizer.PAD_ID
    eos = CompactTokenizer.EOS_ID
    return PackedLanguageSource(
        torch.tensor(
            [
                [pad, pad, 30 + offset],
                [pad, 31 + offset, 32 + offset],
                [33 + offset, 34 + offset, 35 + offset],
            ],
            dtype=torch.int32,
        ),
        torch.tensor(
            [
                [40 + offset, eos, pad, pad],
                [41 + offset, 42 + offset, 43 + offset, 44 + offset],
                [45 + offset, 46 + offset, eos, pad],
            ],
            dtype=torch.int32,
        ),
        seed=seed,
    )


class PackedBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LanguageBatchConfig(
            context_tokens=3,
            response_tokens=4,
        )

    def test_finite_row_gather_matches_legacy_collation(self) -> None:
        examples = finite_examples()
        batcher = LanguageBatcher(examples, self.config, seed=17)
        indices = torch.tensor([2, 0, 2, 1], dtype=torch.long)

        packed = batcher.batch_rows_cpu(indices)
        legacy = batcher.collate(
            [examples[index] for index in indices.tolist()],
            "cpu",
        )

        assert_batch_equal(self, packed, legacy)
        self.assertEqual(packed.context_ids.dtype, torch.long)
        self.assertEqual(packed.target_mask.dtype, torch.bool)

    def test_finite_sampling_is_deterministic_and_restorable(self) -> None:
        examples = finite_examples()
        first = LanguageBatcher(examples, self.config, seed=23)
        same_seed = LanguageBatcher(examples, self.config, seed=23)
        assert_batch_equal(
            self, first.sample_cpu(11), same_seed.sample_cpu(11)
        )

        state = first.state_dict()
        expected = first.sample_cpu(13)
        restored = LanguageBatcher(examples, self.config, seed=999)
        restored.load_state_dict(state)
        assert_batch_equal(self, expected, restored.sample_cpu(13))

    def test_finite_state_rejects_same_shape_different_corpus(self) -> None:
        original = LanguageBatcher(
            finite_examples(),
            self.config,
            seed=23,
        )
        changed = list(finite_examples())
        changed[0] = EncodedAssistantExample(
            0,
            1,
            (99,),
            changed[0].response_ids,
            "finite-a",
        )
        restored = LanguageBatcher(changed, self.config, seed=23)

        with self.assertRaisesRegex(ValueError, "corpus_fingerprint"):
            restored.load_state_dict(original.state_dict())

    def test_finite_fingerprint_includes_active_padding_semantics(self) -> None:
        eos = CompactTokenizer.EOS_ID
        empty_context = LanguageBatcher(
            [EncodedAssistantExample(0, 1, (), (9, eos), "a")],
            self.config,
            seed=3,
        )
        active_pad_context = LanguageBatcher(
            [
                EncodedAssistantExample(
                    0,
                    1,
                    (CompactTokenizer.PAD_ID,),
                    (9, eos),
                    "a",
                )
            ],
            self.config,
            seed=3,
        )
        self.assertTrue(
            torch.equal(
                empty_context._packed_batch.context_ids,
                active_pad_context._packed_batch.context_ids,
            )
        )
        self.assertNotEqual(
            empty_context.corpus_fingerprint,
            active_pad_context.corpus_fingerprint,
        )
        with self.assertRaisesRegex(ValueError, "corpus_fingerprint"):
            active_pad_context.load_state_dict(
                empty_context.state_dict()
            )

    def test_finite_sampling_preserves_legacy_python_rng_order(self) -> None:
        examples = finite_examples()
        seed = 71
        oracle = random.Random(seed)
        expected_indices = [
            oracle.randrange(len(examples)) for _ in range(17)
        ]
        batcher = LanguageBatcher(examples, self.config, seed=seed)

        actual = batcher.sample_cpu(17)
        expected = batcher.collate(
            [examples[index] for index in expected_indices],
            "cpu",
        )

        assert_batch_equal(self, actual, expected)

    def test_packed_row_gather_builds_masks_without_encoded_objects(self) -> None:
        source = packed_source()
        batch = source.batch_rows_cpu(torch.tensor([2, 0, 1, 2]))

        self.assertEqual(
            batch.context_ids.tolist(),
            [
                [33, 34, 35],
                [0, 0, 30],
                [0, 31, 32],
                [33, 34, 35],
            ],
        )
        self.assertEqual(
            batch.context_mask.tolist(),
            [
                [True, True, True],
                [False, False, True],
                [False, True, True],
                [True, True, True],
            ],
        )
        self.assertEqual(
            batch.target_complete.tolist(),
            [True, True, False, True],
        )

    def test_build_encoded_packs_without_retokenizing(self) -> None:
        examples = finite_examples()
        source = PackedLanguageSource.build_encoded(
            examples,
            self.config,
            chunk_size=2,
            seed=13,
            progress_every=0,
        )
        expected = LanguageBatcher(
            examples,
            self.config,
            seed=13,
        )._packed_batch
        actual = source.batch_rows_cpu(
            torch.arange(len(examples), dtype=torch.long)
        )

        assert_batch_equal(self, actual, expected)

    def test_packed_sampling_state_round_trip_is_exact(self) -> None:
        source = packed_source(seed=31)
        source.sample_cpu(7)
        state = source.state_dict()
        expected = source.sample_cpu(19)

        restored = packed_source(seed=999)
        restored.load_state_dict(state)
        assert_batch_equal(self, expected, restored.sample_cpu(19))

    def test_packed_state_rejects_same_shape_different_corpus(self) -> None:
        original = packed_source(seed=31, offset=0)
        changed = packed_source(seed=31, offset=1)

        with self.assertRaisesRegex(ValueError, "corpus_fingerprint"):
            changed.load_state_dict(original.state_dict())

    def test_packed_source_clones_caller_storage(self) -> None:
        context = torch.tensor([[0, 0, 7]], dtype=torch.int32)
        target = torch.tensor([[8, CompactTokenizer.EOS_ID, 0, 0]])
        source = PackedLanguageSource(context, target, seed=1)
        context.fill_(99)
        target.fill_(99)

        batch = source.batch_rows_cpu(torch.tensor([0]))
        self.assertEqual(batch.context_ids.tolist(), [[0, 0, 7]])
        self.assertEqual(
            batch.target_ids.tolist(),
            [[8, CompactTokenizer.EOS_ID, 0, 0]],
        )

    def test_mixture_never_calls_packed_encoded_object_oracle(self) -> None:
        class TensorOnlyPackedSource(PackedLanguageSource):
            def next(self) -> EncodedAssistantExample:
                raise AssertionError("packed rows must stay tensorized")

        base = packed_source(seed=5)
        source = TensorOnlyPackedSource(
            base.context_ids,
            base.target_ids,
            seed=5,
        )
        mixture = MixtureLanguageBatcher(
            finite_examples(),
            source,
            self.config,
            streaming_weight=1.0,
            seed=7,
        )

        batch = mixture.sample_cpu(32)

        self.assertEqual(batch.context_ids.shape, (32, 3))
        self.assertEqual(batch.target_ids.shape, (32, 4))
        self.assertEqual(mixture.streaming_samples, 32)
        self.assertEqual(mixture.finite_samples, 0)

    def test_mixture_preserves_legacy_selector_and_row_rng_order(self) -> None:
        mixture_seed = 83
        packed_seed = 89
        source = packed_source(seed=packed_seed)
        mixture = MixtureLanguageBatcher(
            finite_examples(),
            source,
            self.config,
            streaming_weight=0.6,
            seed=mixture_seed,
        )
        actual = mixture.sample_cpu(41)

        mixture_oracle = random.Random(mixture_seed)
        packed_oracle = random.Random(packed_seed)
        expected_rows = []
        finite = mixture.finite_batcher._packed_batch.target_ids
        packed = source.target_ids.to(torch.long)
        for _ in range(41):
            if mixture_oracle.random() < 0.6:
                expected_rows.append(
                    packed[packed_oracle.randrange(source.examples)]
                )
            else:
                expected_rows.append(
                    finite[
                        mixture_oracle.randrange(
                            len(mixture.finite_batcher.examples)
                        )
                    ]
                )

        self.assertTrue(
            torch.equal(actual.target_ids, torch.stack(expected_rows))
        )

    def test_weighted_composite_mixture_state_round_trip(self) -> None:
        def hierarchy(seed_delta: int):
            weighted = WeightedLanguageSource(
                [
                    ("packed-a", packed_source(41 + seed_delta, 0)),
                    ("packed-b", packed_source(43 + seed_delta, 50)),
                ],
                [1.0, 2.0],
                seed=47 + seed_delta,
            )
            return MixtureLanguageBatcher(
                finite_examples(),
                weighted,
                self.config,
                streaming_weight=0.75,
                seed=53 + seed_delta,
            )

        original = hierarchy(0)
        self.assertTrue(original.exact_resume_supported)
        original.sample_cpu(17)
        state = original.state_dict()
        self.assertTrue(state["exact"])
        expected_first = original.sample_cpu(29)
        expected_second = original.sample_cpu(31)

        restored = hierarchy(1000)
        restored.load_state_dict(state)
        assert_batch_equal(self, expected_first, restored.sample_cpu(29))
        assert_batch_equal(self, expected_second, restored.sample_cpu(31))
        self.assertEqual(
            restored.streaming_samples, original.streaming_samples
        )
        self.assertEqual(restored.finite_samples, original.finite_samples)
        self.assertEqual(
            restored.streaming_source.sample_counts,
            original.streaming_source.sample_counts,
        )

    def test_batch_transfer_api_keeps_all_fields_together(self) -> None:
        batch = LanguageBatcher(
            finite_examples(), self.config
        ).sample_cpu(2)

        moved = batch.to("cpu", non_blocking=True)
        assert_batch_equal(self, batch, moved)
        pinned = batch.pin_memory()
        if torch.cuda.is_available():
            for name in (
                "context_ids",
                "context_mask",
                "target_ids",
                "target_mask",
                "target_complete",
            ):
                self.assertTrue(getattr(pinned, name).is_pinned(), name)
        else:
            self.assertIs(pinned, batch)


if __name__ == "__main__":
    unittest.main()
