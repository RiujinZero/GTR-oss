"""SGBM Phase F: GTR-L forward-pass integration tests.

Covers the wiring added to ``gator_linear/prefix.py``'s cached generation
path (``forward_cached``/``_cached_proposal_hidden``): reading and
injecting Surprise-Gated Block Memory before relay 0, and writing a
candidate back from the just-formed proposal afterward. ``append_cache``
is deliberately never touched by this integration (design spec section
6.2's "no target leakage" — see the invariant test below).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from gator.memory import SurpriseMemoryConfig
from gator.memory.runtime import MemoryRuntime
from gator_linear import GatorLinearConfig, GatorLinearPrefixConfig, GatorLinearPrefixModel


def _core_config(**overrides) -> GatorLinearConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return GatorLinearConfig(**defaults)


def _memory_config(**overrides) -> SurpriseMemoryConfig:
    defaults = dict(
        fused_dim=6,
        key_dim=4,
        value_dim=5,
        memory_slots=8,
        quarantine_slots=2,
        retrieval_topk=2,
    )
    defaults.update(overrides)
    return SurpriseMemoryConfig(**defaults)


def _seeded_runtime(batch: int, config: SurpriseMemoryConfig, *, seed: int = 0) -> MemoryRuntime:
    """A runtime with two populated stable slots and nonzero running
    summaries, so retrieval/query behavior isn't degenerate (see Phase F's
    own diagnostic: an all-zero fresh runtime makes the query a constant)."""

    generator = torch.Generator().manual_seed(seed)
    runtime = MemoryRuntime.allocate(batch, config)
    runtime.state.keys[:, 0] = torch.nn.functional.normalize(
        torch.randn(batch, config.key_dim, generator=generator), dim=-1
    )
    runtime.state.keys[:, 1] = torch.nn.functional.normalize(
        torch.randn(batch, config.key_dim, generator=generator), dim=-1
    )
    runtime.state.values[:, 0] = torch.randn(batch, config.value_dim, generator=generator)
    runtime.state.values[:, 1] = torch.randn(batch, config.value_dim, generator=generator)
    runtime.state.valid[:, 0] = True
    runtime.state.valid[:, 1] = True
    runtime.context_summary.copy_(torch.randn(batch, config.fused_dim, generator=generator))
    runtime.cache_summary.copy_(torch.randn(batch, config.fused_dim, generator=generator))
    return runtime


class TestConfigValidation:
    def test_memory_none_by_default(self) -> None:
        config = GatorLinearPrefixConfig(core=_core_config())
        assert config.memory is None

    def test_fused_dim_must_match_core_width(self) -> None:
        with pytest.raises(ValueError):
            GatorLinearPrefixConfig(
                core=_core_config(width=6), memory=_memory_config(fused_dim=8)
            )

    def test_memory_must_be_correct_type(self) -> None:
        with pytest.raises(TypeError):
            GatorLinearPrefixConfig(core=_core_config(), memory="not a config")


class TestDisabledIsUnchanged:
    def test_default_model_has_no_memory_submodules(self) -> None:
        model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=_core_config()))
        assert model.memory_config is None
        assert not hasattr(model, "surprise_gated_memory")
        assert not any(name.startswith("memory_") for name, _ in model.named_parameters())

    def test_forward_cached_without_memory_kwarg_works_as_before(self) -> None:
        model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=_core_config()))
        cache = model.prefill_cache(torch.zeros(2, 2, dtype=torch.long), block_size=2)
        logits = model.forward_cached(cache)
        assert logits.shape == (2, 2, 16)

    def test_passing_memory_to_a_memory_disabled_model_raises(self) -> None:
        model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=_core_config()))
        cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
        config = _memory_config()
        runtime = MemoryRuntime.allocate(1, config)
        with pytest.raises(ValueError):
            model.forward_cached(cache, memory=runtime)

    def test_disabled_model_is_deterministic_given_a_fixed_seed(self) -> None:
        torch.manual_seed(0)
        model_a = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=_core_config()))
        torch.manual_seed(0)
        model_b = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=_core_config()))
        observed = torch.zeros(2, 2, dtype=torch.long)
        cache_a = model_a.prefill_cache(observed, block_size=2)
        cache_b = model_b.prefill_cache(observed, block_size=2)
        logits_a = model_a.forward_cached(cache_a)
        logits_b = model_b.forward_cached(cache_b)
        assert torch.allclose(logits_a, logits_b)


class TestEnabledPath:
    def test_forward_cached_with_memory_returns_correct_shape(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(2, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(2, memory_config)
        logits = model.forward_cached(cache, memory=runtime)
        assert logits.shape == (2, 2, 16)

    def test_diagnostics_include_memory_metrics(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(1, memory_config)
        _, diagnostics = model.forward_cached(cache, memory=runtime, return_diagnostics=True)
        expected_keys = {
            "predictive_surprise",
            "stream_disagreement",
            "route_surprise",
            "semantic_novelty",
            "context_continuity",
            "combined_surprise",
            "blended_surprise",
            "reliability",
            "retrieval_weights",
            "retrieval_indices",
            "retrieval_entropy",
            "retrieval_top_similarity",
        }
        assert expected_keys.issubset(diagnostics["memory"].keys())

    def test_running_summaries_update_after_a_call(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(1, memory_config)
        before_context = runtime.context_summary.clone()
        before_cache = runtime.cache_summary.clone()

        model.forward_cached(cache, memory=runtime)

        assert not torch.equal(runtime.context_summary, before_context)
        assert not torch.equal(runtime.cache_summary, before_cache)

    def test_surprise_momentum_and_previous_summary_update(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(1, memory_config)
        assert torch.equal(runtime.state.previous_block_summary, torch.zeros(1, 6))

        model.forward_cached(cache, memory=runtime)

        assert not torch.equal(runtime.state.previous_block_summary, torch.zeros(1, 6))

    def test_repeated_calls_do_not_crash_and_keep_stable_shapes(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(2, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(2, memory_config)
        for _ in range(5):
            logits = model.forward_cached(cache, memory=runtime)
            assert logits.shape == (2, 2, 16)
            assert torch.isfinite(logits).all()

    def test_gradients_flow_to_memory_submodules(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(2, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(2, memory_config)

        logits = model.forward_cached(cache, memory=runtime)
        logits.sum().backward()

        assert model.memory_query_projection.weight.grad is not None
        assert bool((model.memory_query_projection.weight.grad != 0).any())
        assert bool(
            (model.memory_stream_injector.gate_projections[0].weight.grad != 0).any()
        )

    def test_core_relay_parameters_still_receive_gradients(self) -> None:
        """A regression guard: the memory branch must not sever gradient
        flow to the underlying GTR-L core it's injecting into."""

        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(2, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(2, memory_config)

        logits = model.forward_cached(cache, memory=runtime)
        logits.sum().backward()

        # relay 1 (not relay 0) has a nonzero controlled-writing gate at
        # every position in this 2-relay config, so it is a reliable place
        # to check that ordinary gradient flow through the core survives.
        relay1_params = dict(model.core.relays[1].named_parameters())
        assert bool((relay1_params["write_down"].grad != 0).any())

    def test_write_does_not_require_grad_on_memory_state(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(1, memory_config)

        model.forward_cached(cache, memory=runtime)

        assert not runtime.state.keys.requires_grad
        assert not runtime.state.values.requires_grad


class TestNoTargetLeakage:
    def test_append_cache_signature_has_no_memory_parameter(self) -> None:
        """Design spec section 6.2: memory written from block t may
        influence block t+1, never block t itself — append_cache commits
        the already-decided accepted block and must stay untouched by any
        of this integration."""

        signature = inspect.signature(GatorLinearPrefixModel.append_cache)
        assert "memory" not in signature.parameters

    def test_append_cache_still_works_unchanged_alongside_a_memory_enabled_model(self) -> None:
        memory_config = _memory_config()
        model = GatorLinearPrefixModel(
            GatorLinearPrefixConfig(core=_core_config(), memory=memory_config)
        )
        cache = model.prefill_cache(torch.zeros(1, 2, dtype=torch.long), block_size=2)
        runtime = _seeded_runtime(1, memory_config)
        logits = model.forward_cached(cache, memory=runtime)
        accepted = logits.argmax(dim=-1)
        next_cache = model.append_cache(cache, accepted)
        assert next_cache.next_position.item() == 4
