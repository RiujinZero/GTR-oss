"""Tests for SGBM Phase B: memory read (scoring, top-k retrieval, stream injection)."""

from __future__ import annotations

import pytest
import torch

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.retrieval import (
    StreamMemoryInjector,
    bump_retrieval_usage,
    retrieve_top_k,
    score_memory_slots,
)
from gator.memory.state import SurpriseMemoryState


def _config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(fused_dim=8, key_dim=4, value_dim=3, memory_slots=4, quarantine_slots=2, retrieval_topk=2)
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


class TestScoreMemorySlots:
    def test_invalid_slots_score_negative_infinity(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        state.keys[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        state.valid[0, 0] = True
        query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        scores = score_memory_slots(query, state)
        assert torch.isfinite(scores[0, 0])
        assert torch.isinf(scores[0, 1]) and scores[0, 1] < 0


class TestRetrieveTopK:
    def test_empty_memory_returns_zero_vector_no_nan(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        query = torch.randn(1, config.key_dim)
        retrieved, metrics = retrieve_top_k(query, state, top_k=config.retrieval_topk)
        assert torch.equal(retrieved, torch.zeros(1, config.value_dim))
        assert not torch.isnan(retrieved).any()
        assert not torch.isnan(metrics["retrieval_weights"]).any()

    def test_single_valid_slot_is_retrieved_exactly(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        state.keys[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        state.values[0, 0] = torch.tensor([1.0, 2.0, 3.0])
        state.valid[0, 0] = True
        query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        retrieved, _ = retrieve_top_k(query, state, top_k=config.retrieval_topk, temperature=0.1)
        assert torch.allclose(retrieved[0], state.values[0, 0], atol=1e-4)

    def test_weights_sum_to_one_when_any_valid_slot_exists(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        state.keys[0, :2] = torch.eye(4)[:2]
        state.valid[0, :2] = True
        query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        _, metrics = retrieve_top_k(query, state, top_k=config.retrieval_topk)
        assert metrics["retrieval_weights"].sum(dim=1)[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_rejects_non_positive_top_k(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        with pytest.raises(ValueError):
            retrieve_top_k(torch.randn(1, 4), state, top_k=0)


class TestBumpUsage:
    def test_only_above_threshold_weights_increment_usage(self) -> None:
        config = _config()
        state = SurpriseMemoryState.allocate(1, config)
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.9, 0.01]])
        bump_retrieval_usage(state, indices, weights, threshold=0.05)
        assert state.usage[0, 0].item() == pytest.approx(1.0)
        assert state.usage[0, 1].item() == pytest.approx(0.0)


class TestStreamMemoryInjector:
    def test_output_shape_matches_input(self) -> None:
        injector = StreamMemoryInjector(stream_count=3, stream_width=5, value_dim=4)
        stream_states = torch.randn(2, 3, 5)
        retrieved = torch.randn(2, 4)
        output = injector(stream_states, retrieved)
        assert output.shape == stream_states.shape

    def test_rejects_wrong_stream_count(self) -> None:
        injector = StreamMemoryInjector(stream_count=3, stream_width=5, value_dim=4)
        with pytest.raises(ValueError):
            injector(torch.randn(2, 4, 5), torch.randn(2, 4))

    def test_full_dropout_in_training_mode_is_identity(self) -> None:
        injector = StreamMemoryInjector(stream_count=2, stream_width=3, value_dim=4, dropout=1.0)
        injector.train()
        stream_states = torch.randn(5, 2, 3)
        retrieved = torch.randn(5, 4)
        output = injector(stream_states, retrieved)
        assert torch.equal(output, stream_states)

    def test_eval_mode_ignores_dropout(self) -> None:
        injector = StreamMemoryInjector(stream_count=2, stream_width=3, value_dim=4, dropout=1.0)
        injector.eval()
        stream_states = torch.randn(3, 2, 3)
        retrieved = torch.randn(3, 4)
        output = injector(stream_states, retrieved)
        # In eval mode dropout must not apply, so output should generally
        # differ from the raw input (the gated injection is active).
        assert not torch.equal(output, stream_states)

    def test_rejects_invalid_dropout(self) -> None:
        with pytest.raises(ValueError):
            StreamMemoryInjector(stream_count=1, stream_width=2, value_dim=2, dropout=1.5)
