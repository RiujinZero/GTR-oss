"""Tests for SGBM Phase G: memory-on/off evaluation utility."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gator.continual.memory_eval import compare_with_without_memory
from gator.memory import SurpriseMemoryConfig
from gator.memory.runtime import MemoryRuntime
from gator_linear import GatorLinearConfig, GatorLinearPrefixConfig, GatorLinearPrefixModel


def _model_and_cache():
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
    cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
    return model, cache, memory_config


def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


class TestCompareWithWithoutMemory:
    def test_memory_none_skips_the_with_memory_arm(self) -> None:
        model, cache, _ = _model_and_cache()
        targets = torch.zeros(1, 2, dtype=torch.long)
        result = compare_with_without_memory(
            model, cache, targets=targets, memory=None, loss_fn=_loss_fn
        )
        assert result["loss_with_memory"] is None
        assert result["utility"] is None
        assert result["loss_without_memory"] is not None

    def test_memory_provided_computes_both_arms_and_utility(self) -> None:
        model, cache, memory_config = _model_and_cache()
        runtime = MemoryRuntime.allocate(1, memory_config)
        targets = torch.zeros(1, 2, dtype=torch.long)

        result = compare_with_without_memory(
            model, cache, targets=targets, memory=runtime, loss_fn=_loss_fn
        )

        assert result["loss_with_memory"] is not None
        assert torch.allclose(
            result["utility"], result["loss_without_memory"] - result["loss_with_memory"]
        )

    def test_cache_is_not_mutated_by_the_comparison(self) -> None:
        model, cache, memory_config = _model_and_cache()
        runtime = MemoryRuntime.allocate(1, memory_config)
        targets = torch.zeros(1, 2, dtype=torch.long)
        next_position_before = cache.next_position.item()

        compare_with_without_memory(model, cache, targets=targets, memory=runtime, loss_fn=_loss_fn)

        assert cache.next_position.item() == next_position_before
        # cache is still usable afterward
        logits = model.forward_cached(cache)
        assert logits.shape == (1, 2, 16)
