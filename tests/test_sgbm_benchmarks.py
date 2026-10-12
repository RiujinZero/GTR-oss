"""Tests for SGBM Phase G: the delayed-recall benchmark harness.

These tests validate the harness mechanics (generation shapes, the model
walkthrough running end to end without error) — not that an untrained
model demonstrates real recall, which requires actual training (see
benchmarks.py's module docstring).
"""

from __future__ import annotations

import pytest
import torch

from gator.continual.benchmarks import DelayedRecallBenchmark, evaluate_delayed_recall
from gator.memory import SurpriseMemoryConfig
from gator_linear import GatorLinearConfig, GatorLinearPrefixConfig, GatorLinearPrefixModel


class TestDelayedRecallBenchmarkConstruction:
    def test_rejects_non_positive_block_size(self) -> None:
        with pytest.raises(ValueError):
            DelayedRecallBenchmark(vocab_size=16, block_size=0)

    def test_rejects_filler_start_not_above_query_marker(self) -> None:
        with pytest.raises(ValueError):
            DelayedRecallBenchmark(vocab_size=16, block_size=2, query_marker=1, filler_start=1)

    def test_rejects_vocab_too_small_for_content(self) -> None:
        with pytest.raises(ValueError):
            DelayedRecallBenchmark(vocab_size=1, block_size=2, query_marker=0, filler_start=1)


class TestGenerate:
    def test_produces_requested_count_and_filler_length(self) -> None:
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=2, seed=0)
        episodes = benchmark.generate(3, filler_blocks=4)
        assert len(episodes) == 3
        assert all(len(episode.filler_blocks) == 4 for episode in episodes)
        assert all(episode.fact_tokens.shape == (2,) for episode in episodes)

    def test_content_tokens_stay_above_filler_start(self) -> None:
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=8, filler_start=3, seed=0)
        episodes = benchmark.generate(2, filler_blocks=3)
        for episode in episodes:
            assert bool((episode.fact_tokens >= 3).all())
            for filler in episode.filler_blocks:
                assert bool((filler >= 3).all())

    def test_query_block_is_filled_with_the_marker(self) -> None:
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=4, query_marker=0, seed=0)
        query = benchmark.query_block()
        assert torch.equal(query, torch.zeros(4, dtype=torch.long))

    def test_rejects_non_positive_count_or_negative_fillers(self) -> None:
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=2, seed=0)
        with pytest.raises(ValueError):
            benchmark.generate(0, filler_blocks=1)
        with pytest.raises(ValueError):
            benchmark.generate(1, filler_blocks=-1)

    def test_to_episode_packs_fact_then_fillers_in_order(self) -> None:
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=2, seed=0)
        recall_episode = benchmark.generate(1, filler_blocks=2)[0]
        episode = recall_episode.to_episode()
        assert len(episode.blocks) == 3
        assert torch.equal(episode.blocks[0], recall_episode.fact_tokens)
        assert torch.equal(episode.blocks[1], recall_episode.filler_blocks[0])
        assert torch.equal(episode.blocks[2], recall_episode.filler_blocks[1])


class TestEvaluateDelayedRecall:
    def test_runs_end_to_end_without_error(self) -> None:
        core = GatorLinearConfig(
            vocab_size=16,
            streams=3,
            nodes=8,
            width=6,
            relays=2,
            degree=4,
            block_causal=True,
            hard_routing=True,
            controlled_writing=True,
        )
        memory_config = SurpriseMemoryConfig(
            fused_dim=6, key_dim=4, value_dim=5, memory_slots=8, quarantine_slots=2, retrieval_topk=2
        )
        model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core, memory=memory_config))
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=2, filler_start=1, seed=1)
        episode = benchmark.generate(1, filler_blocks=3)[0]

        result = evaluate_delayed_recall(model, episode, memory_config=memory_config)

        assert result["predicted_with_memory"].shape == (2,)
        assert result["predicted_without_memory"].shape == (2,)
        assert torch.equal(result["target"], episode.fact_tokens)
        assert 0.0 <= result["accuracy_with_memory"].item() <= 1.0
        assert 0.0 <= result["accuracy_without_memory"].item() <= 1.0

    def test_runs_with_zero_filler_blocks(self) -> None:
        core = GatorLinearConfig(
            vocab_size=16,
            streams=3,
            nodes=8,
            width=6,
            relays=2,
            degree=4,
            block_causal=True,
            hard_routing=True,
            controlled_writing=True,
        )
        memory_config = SurpriseMemoryConfig(
            fused_dim=6, key_dim=4, value_dim=5, memory_slots=8, quarantine_slots=2, retrieval_topk=2
        )
        model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core, memory=memory_config))
        benchmark = DelayedRecallBenchmark(vocab_size=16, block_size=2, filler_start=1, seed=2)
        episode = benchmark.generate(1, filler_blocks=0)[0]

        result = evaluate_delayed_recall(model, episode, memory_config=memory_config)
        assert result["predicted_with_memory"].shape == (2,)
