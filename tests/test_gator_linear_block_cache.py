from __future__ import annotations

import unittest

import torch

from gator_linear import (
    EvoTokenRefiner,
    GatorLinearBlockCache,
    GatorLinearConfig,
    GatorLinearPrefixConfig,
    GatorLinearPrefixModel,
    LinearSequenceMixer,
    fixed_address_code,
)


def build_model(**overrides: object) -> GatorLinearPrefixModel:
    settings: dict[str, object] = {
        "vocab_size": 16,
        "streams": 3,
        "nodes": 8,
        "width": 6,
        "relays": 2,
        "degree": 3,
        "buses": 2,
        "mailbox_channels": 2,
        "mailbox_width": 4,
        "basis_rank": 3,
        "address_width": 4,
        "router_rank": 4,
        "conv_kernel": 3,
        "context_length": 32,
        "controlled_writing": True,
        "block_causal": True,
    }
    settings.update(overrides)
    torch.manual_seed(19)
    return GatorLinearPrefixModel(
        GatorLinearPrefixConfig(core=GatorLinearConfig(**settings))
    )


def valid_mask() -> torch.Tensor:
    result = torch.zeros(16, dtype=torch.bool)
    result[4:] = True
    return result


def cache_tensors(cache: GatorLinearBlockCache) -> list[torch.Tensor]:
    tensors = [cache.next_position]
    for relay in cache.relays:
        tensors.extend(
            [
                relay.mixer.local_tail,
                relay.mixer.forward_state,
                relay.mixer.node_sum,
            ]
        )
        if relay.attention is not None:
            tensors.extend([relay.attention.kv_state, relay.attention.key_state])
        if relay.occupancy_sum is not None:
            tensors.append(relay.occupancy_sum)
    return tensors


class BlockCausalMixerTest(unittest.TestCase):
    def test_later_block_is_isolated_but_same_block_is_bidirectional(self) -> None:
        config = GatorLinearConfig(
            vocab_size=16,
            streams=2,
            nodes=4,
            width=4,
            relays=1,
            degree=2,
            buses=2,
            mailbox_channels=1,
            mailbox_width=3,
            basis_rank=2,
            address_width=2,
            router_rank=2,
            conv_kernel=3,
            context_length=12,
            block_causal=True,
        )
        torch.manual_seed(3)
        mixer = LinearSequenceMixer(config).double()
        source = torch.randn(1, 12, 4, 1, 3, dtype=torch.float64)
        baseline = mixer(source, block_size=4)

        later = source.clone()
        later[:, 8:] += 7.0
        later_result = mixer(later, block_size=4)
        torch.testing.assert_close(later_result[:, :8], baseline[:, :8])
        self.assertGreater(
            float((later_result[:, 8:] - baseline[:, 8:]).abs().max()),
            1e-8,
        )

        within = source.clone()
        within[:, 7] += 5.0
        within_result = mixer(within, block_size=4)
        self.assertGreater(
            float((within_result[:, 4:7] - baseline[:, 4:7]).abs().max()),
            1e-8,
        )

        differentiable = source.clone().requires_grad_(True)
        mixer(differentiable, block_size=4)[:, 4].sum().backward()
        self.assertEqual(
            float(differentiable.grad[:, 8:].abs().max()), 0.0
        )
        self.assertGreater(float(differentiable.grad[:, 7].abs().sum()), 0.0)
        self.assertIsNotNone(mixer.reverse_decay.grad)
        self.assertGreater(float(mixer.reverse_decay.grad.abs().sum()), 0.0)

    def test_address_codes_are_append_stable(self) -> None:
        kwargs = {
            "width": 6,
            "device": torch.device("cpu"),
            "dtype": torch.float64,
            "reference_length": 32,
        }
        complete = fixed_address_code(12, offset=0, **kwargs)
        pieces = torch.cat(
            [fixed_address_code(4, offset=start, **kwargs) for start in (0, 4, 8)]
        )
        torch.testing.assert_close(complete, pieces)

    def test_bf16_addresses_keep_every_slot_distinct_near_2048(self) -> None:
        kwargs = {
            "length": 8,
            "width": 6,
            "device": torch.device("cpu"),
            "offset": torch.tensor(2040, dtype=torch.long),
            "reference_length": 2048,
        }
        bf16 = fixed_address_code(dtype=torch.bfloat16, **kwargs)
        reference = fixed_address_code(dtype=torch.float32, **kwargs).to(
            torch.bfloat16
        )
        torch.testing.assert_close(bf16, reference, rtol=0.0, atol=0.0)
        self.assertEqual(torch.unique(bf16, dim=0).shape[0], 8)

    def test_address_range_is_bounded_for_python_and_tensor_offsets(self) -> None:
        kwargs = {
            "length": 4,
            "width": 4,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "reference_length": 8,
        }
        with self.assertRaisesRegex(ValueError, "reference_length"):
            fixed_address_code(offset=6, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            fixed_address_code(offset=torch.tensor(-1), **kwargs)
        with self.assertRaisesRegex(RuntimeError, "reference_length"):
            fixed_address_code(offset=torch.tensor(6), **kwargs)

    def test_bf16_scan_scales_stay_finite_near_2048(self) -> None:
        config = GatorLinearConfig(
            vocab_size=16,
            streams=2,
            nodes=4,
            width=4,
            relays=1,
            degree=2,
            buses=2,
            mailbox_channels=1,
            mailbox_width=3,
            basis_rank=2,
            address_width=2,
            router_rank=2,
            conv_kernel=3,
            context_length=2048,
            block_causal=True,
        )
        mixer = LinearSequenceMixer(config)
        reference = torch.zeros(1, 8, 4, 1, 3, dtype=torch.bfloat16)
        decay = torch.full((3,), 0.9995, dtype=torch.float32)
        normaliser = mixer._scan_normaliser(
            decay, 8, reference, offset=torch.tensor(2040)
        )
        scale = mixer._asymptotic_scale(decay, reference)
        self.assertTrue(torch.isfinite(normaliser).all())
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue(bool((normaliser > 0).all()))
        self.assertTrue(bool((scale > 0).all()))


class BlockCacheEquivalenceTest(unittest.TestCase):
    def _assert_reference_match(self, **overrides: object) -> None:
        model = build_model(**overrides).double().eval()
        observed = torch.randint(4, 16, (2, 8))
        queries = torch.randn(2, 4, 6, dtype=torch.float64)
        with torch.no_grad():
            reference = model(
                observed,
                output_slots=4,
                output_embeddings=queries,
            )
            cache = model.prefill_cache(observed, block_size=4)
            cached = model.forward_cached(cache, output_embeddings=queries)
        torch.testing.assert_close(cached, reference, atol=2e-8, rtol=2e-8)

    def test_cached_query_matches_vectorized_training_reference(self) -> None:
        for scan_normalise in (False, True):
            for linear_attention in (False, True):
                for router_occupancy in (False, True):
                    with self.subTest(
                        scan_normalise=scan_normalise,
                        linear_attention=linear_attention,
                        router_occupancy=router_occupancy,
                    ):
                        self._assert_reference_match(
                            scan_normalise=scan_normalise,
                            linear_attention=linear_attention,
                            router_occupancy=router_occupancy,
                        )

    def test_multiple_appends_match_vectorized_reference(self) -> None:
        model = build_model(
            linear_attention=True, router_occupancy=True
        ).double().eval()
        blocks = [torch.randint(4, 16, (2, 4)) for _ in range(4)]
        queries = torch.randn(2, 4, 6, dtype=torch.float64)
        cache = model.prefill_cache(torch.cat(blocks[:2], dim=1), block_size=4)
        original = [tensor.clone() for tensor in cache_tensors(cache)]
        cache = model.append_cache(cache, blocks[2])
        cache = model.append_cache(cache, blocks[3])
        cached = model.forward_cached(cache, output_embeddings=queries)
        reference = model(
            torch.cat(blocks, dim=1),
            output_slots=4,
            output_embeddings=queries,
        )
        torch.testing.assert_close(cached, reference, atol=3e-8, rtol=3e-8)
        self.assertEqual(int(cache.next_position), 16)
        self.assertEqual(len(original), len(cache_tensors(cache)))

    def test_proposal_is_read_only_and_append_is_functional(self) -> None:
        model = build_model(linear_attention=True, router_occupancy=True).eval()
        observed = torch.randint(4, 16, (2, 8))
        accepted = torch.randint(4, 16, (2, 4))
        cache = model.prefill_cache(observed, block_size=4)
        snapshots = [tensor.clone() for tensor in cache_tensors(cache)]
        model.forward_cached(cache)
        for tensor, snapshot in zip(cache_tensors(cache), snapshots, strict=True):
            torch.testing.assert_close(tensor, snapshot, rtol=0.0, atol=0.0)
        next_cache = model.append_cache(cache, accepted)
        self.assertIsNot(next_cache, cache)
        self.assertEqual(int(cache.next_position), 8)
        self.assertEqual(int(next_cache.next_position), 12)
        for tensor, snapshot in zip(cache_tensors(cache), snapshots, strict=True):
            torch.testing.assert_close(tensor, snapshot, rtol=0.0, atol=0.0)

    def test_cache_payload_does_not_grow_with_context(self) -> None:
        model = build_model(linear_attention=True, router_occupancy=True).eval()
        one = model.prefill_cache(torch.randint(4, 16, (2, 4)), block_size=4)
        many = model.prefill_cache(torch.randint(4, 16, (2, 24)), block_size=4)
        self.assertEqual(
            [tensor.numel() for tensor in cache_tensors(one)],
            [tensor.numel() for tensor in cache_tensors(many)],
        )
        self.assertTrue(all(tensor.grad_fn is None for tensor in cache_tensors(many)))

    def test_malformed_cache_positions_are_rejected(self) -> None:
        model = build_model().eval()
        cache = model.prefill_cache(torch.randint(4, 16, (2, 8)), block_size=4)
        for position, message in ((-4, "negative"), (6, "align")):
            with self.subTest(position=position):
                malformed = cache._replace(
                    next_position=torch.tensor(position, dtype=torch.long)
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    model.forward_cached(malformed)

    def test_context_overflow_is_rejected_eager_and_compiled(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        model = build_model().eval()
        full = model.prefill_cache(
            torch.randint(4, 16, (2, 32)), block_size=4
        )
        queries = torch.randn(2, 4, 6)

        def run(
            cache: GatorLinearBlockCache,
            output_embeddings: torch.Tensor,
        ) -> torch.Tensor:
            return model.forward_cached(
                cache, output_embeddings=output_embeddings
            )

        with self.assertRaisesRegex(RuntimeError, "context_length"):
            run(full, queries)
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(RuntimeError, "context_length"):
            compiled(full, queries)

    def test_missing_temporal_feature_state_is_rejected(self) -> None:
        model = build_model(
            linear_attention=True, router_occupancy=True
        ).eval()
        cache = model.prefill_cache(torch.randint(4, 16, (2, 8)), block_size=4)
        relays = list(cache.relays)
        relays[0] = relays[0]._replace(attention=None)
        with self.assertRaisesRegex(ValueError, "missing attention"):
            model.forward_cached(cache._replace(relays=tuple(relays)))
        relays = list(cache.relays)
        relays[0] = relays[0]._replace(occupancy_sum=None)
        with self.assertRaisesRegex(ValueError, "missing occupancy"):
            model.forward_cached(cache._replace(relays=tuple(relays)))

    def test_cached_diagnostics_are_labeled_active_only(self) -> None:
        model = build_model().eval()
        cache = model.prefill_cache(torch.randint(4, 16, (2, 8)), block_size=4)
        _, diagnostics = model.forward_cached(cache, return_diagnostics=True)
        self.assertEqual(diagnostics["route_statistics_scope"], "active_block")
        self.assertEqual(model.latest_route_statistics_scope, "active_block")


class CachedEvoTokenTest(unittest.TestCase):
    def test_cached_trajectory_matches_vectorized_reference(self) -> None:
        for mode, feedback_scale in (("relay", 0.5), ("full_pass", 1.0)):
            with self.subTest(mode=mode):
                model = build_model().double().eval()
                refiner = EvoTokenRefiner(
                    model,
                    mask_token_id=3,
                    valid_token_mask=valid_mask(),
                    steps=2,
                    top_k=3,
                    mode=mode,
                    feedback_scale=feedback_scale,
                )
                observed = torch.randint(4, 16, (2, 8))
                initial_ids = torch.full((2, 4), 3, dtype=torch.long)
                initial_pure = torch.zeros_like(initial_ids, dtype=torch.bool)
                alpha = torch.tensor(0.7, dtype=torch.float64)
                reference = refiner(
                    observed,
                    initial_ids,
                    initial_pure,
                    alpha=alpha,
                    inference_promotion_counts=(2, 2),
                )
                cache = model.prefill_cache(observed, block_size=4)
                cached = refiner.forward_cached(
                    cache,
                    initial_ids,
                    initial_pure,
                    alpha=alpha,
                    inference_promotion_counts=(2, 2),
                )
                torch.testing.assert_close(cached[0], reference[0], atol=3e-8, rtol=3e-8)
                torch.testing.assert_close(cached[1], reference[1])
                torch.testing.assert_close(cached[2], reference[2], atol=3e-8, rtol=3e-8)

    def test_block_causal_training_fullgraph_smoke(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        model = build_model(
            linear_attention=True, router_occupancy=True
        ).eval()
        observed = torch.randint(4, 16, (2, 8))
        queries = torch.randn(2, 4, 6)

        def run(observed_ids: torch.Tensor, output_embeddings: torch.Tensor) -> torch.Tensor:
            return model(
                observed_ids,
                output_slots=4,
                output_embeddings=output_embeddings,
            )

        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(observed, queries), run(observed, queries))

    def test_cached_proposal_reuses_graph_at_later_position(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        model = build_model(
            linear_attention=True, router_occupancy=True
        ).eval()
        observed = torch.randint(4, 16, (2, 8))
        accepted = torch.randint(4, 16, (2, 4))
        queries = torch.randn(2, 4, 6)
        first_cache = model.prefill_cache(observed, block_size=4)
        second_cache = model.append_cache(first_cache, accepted)
        graphs: list[torch.fx.GraphModule] = []

        def backend(
            graph: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> object:
            graphs.append(graph)
            return graph.forward

        def run(
            cache: GatorLinearBlockCache,
            output_embeddings: torch.Tensor,
        ) -> torch.Tensor:
            return model.forward_cached(
                cache, output_embeddings=output_embeddings
            )

        compiled = torch.compile(run, backend=backend, fullgraph=True)
        torch.testing.assert_close(
            compiled(first_cache, queries), run(first_cache, queries)
        )
        torch.testing.assert_close(
            compiled(second_cache, queries), run(second_cache, queries)
        )
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
