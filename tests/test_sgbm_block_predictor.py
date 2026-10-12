"""Tests for SGBM Phase D: block expectation predictor and summarizer."""

from __future__ import annotations

import pytest
import torch

from gator.memory.block_predictor import BlockExpectationPredictor
from gator.memory.block_summarizer import BlockSummarizer


class TestBlockExpectationPredictor:
    def test_output_shape_matches_output_dim(self) -> None:
        predictor = BlockExpectationPredictor(
            context_dim=6, cache_dim=4, memory_dim=5, output_dim=8
        )
        output = predictor(torch.randn(3, 6), torch.randn(3, 4), torch.randn(3, 5))
        assert output.shape == (3, 8)

    def test_rejects_non_positive_dims(self) -> None:
        with pytest.raises(ValueError):
            BlockExpectationPredictor(context_dim=0, cache_dim=4, memory_dim=5, output_dim=8)

    def test_gradients_flow_to_all_three_inputs(self) -> None:
        predictor = BlockExpectationPredictor(
            context_dim=3, cache_dim=3, memory_dim=3, output_dim=4
        )
        context = torch.randn(2, 3, requires_grad=True)
        cache = torch.randn(2, 3, requires_grad=True)
        memory = torch.randn(2, 3, requires_grad=True)
        output = predictor(context, cache, memory)
        output.sum().backward()
        assert context.grad is not None and bool((context.grad != 0).any())
        assert cache.grad is not None and bool((cache.grad != 0).any())
        assert memory.grad is not None and bool((memory.grad != 0).any())


class TestBlockSummarizer:
    def test_output_shape_matches_output_dim(self) -> None:
        summarizer = BlockSummarizer(
            token_embedding_dim=4,
            hidden_dim=6,
            stream_width=5,
            route_stat_dim=3,
            indicator_dim=2,
            output_dim=8,
        )
        output = summarizer(
            token_embeddings=torch.randn(2, 4, 4),
            fused_hidden=torch.randn(2, 6),
            stream_states=torch.randn(2, 3, 5),
            route_stats=torch.randn(2, 3),
            indicators=torch.randn(2, 2),
        )
        assert output.shape == (2, 8)

    def test_rejects_non_positive_dims(self) -> None:
        with pytest.raises(ValueError):
            BlockSummarizer(
                token_embedding_dim=4,
                hidden_dim=6,
                stream_width=5,
                route_stat_dim=3,
                indicator_dim=0,
                output_dim=8,
            )

    def test_gradients_flow_to_token_and_stream_inputs(self) -> None:
        summarizer = BlockSummarizer(
            token_embedding_dim=3,
            hidden_dim=3,
            stream_width=3,
            route_stat_dim=2,
            indicator_dim=2,
            output_dim=4,
        )
        tokens = torch.randn(2, 4, 3, requires_grad=True)
        streams = torch.randn(2, 3, 3, requires_grad=True)
        output = summarizer(
            token_embeddings=tokens,
            fused_hidden=torch.randn(2, 3),
            stream_states=streams,
            route_stats=torch.randn(2, 2),
            indicators=torch.randn(2, 2),
        )
        output.sum().backward()
        assert tokens.grad is not None and bool((tokens.grad != 0).any())
        assert streams.grad is not None and bool((streams.grad != 0).any())
