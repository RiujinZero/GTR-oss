from __future__ import annotations

import inspect
import unittest
from unittest import mock

import torch

from gator_convergent import (
    ConvergentGator,
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
    _padded_grouped_mm_width,
    connection_destinations,
    default_convergent_checkpoint_path,
    graph_reachability,
    main as convergent_main,
)
from gator.models.dense_kernels import folded_cell_node_linear
from gtr_a_experiments import relative_code
from parallel_prefix_benchmark import make_prefix_batch


class ConvergentGatorTests(unittest.TestCase):
    def test_grouped_mm_padding_aligns_bfloat16_widths(self) -> None:
        self.assertEqual(_padded_grouped_mm_width(20, torch.bfloat16), 24)
        self.assertEqual(_padded_grouped_mm_width(24, torch.bfloat16), 24)
        self.assertEqual(_padded_grouped_mm_width(28, torch.bfloat16), 32)

    def test_triton_writer_requires_opaque_packed_memory(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Triton packed writer currently requires",
        ):
            ConvergentGator(packed_memory_writer_backend="triton")

    def test_language_config_serializes_writer_backend(self) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=32,
            router_mode="shared_low_rank",
            ingress_mode="learned",
            route_backend="packed",
            packed_memory_backend="fft_opaque",
            packed_memory_writer_backend="triton",
            active_node_top_k=4,
        )

        self.assertEqual(
            config.to_dict()["packed_memory_writer_backend"],
            "triton",
        )

    def test_execution_backend_is_part_of_default_checkpoint_identity(
        self,
    ) -> None:
        checkpoint = default_convergent_checkpoint_path(
            seed=4,
            route_backend="packed",
            packed_memory_backend="fft_opaque",
            packed_memory_reader_backend="triton",
            packed_memory_writer_backend="triton",
        )
        self.assertEqual(
            checkpoint.name,
            (
                "gator_convergent_packed_fft_opaque_"
                "reader-triton_writer-triton_seed4.pt"
            ),
        )

    def test_standalone_cli_rejects_compiled_raw_fft(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "gator_convergent.py",
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft",
                "--compile",
            ],
        ):
            with self.assertRaises(SystemExit):
                convergent_main()

    def test_split_shared_router_trunk_matches_concatenated_projection(
        self,
    ) -> None:
        torch.manual_seed(123)
        concatenated = ConvergentGator(
            n_nodes=8,
            d_model=5,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=7,
            ingress_mode="learned",
            active_node_top_k=3,
            relays=2,
            lanes=2,
        ).double()
        split = ConvergentGator(
            n_nodes=8,
            d_model=5,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=7,
            ingress_mode="learned",
            active_node_top_k=3,
            relays=2,
            lanes=2,
        ).double()
        split.load_state_dict(concatenated.state_dict(), strict=True)
        hidden_concatenated = torch.randn(
            2,
            6,
            2,
            5,
            dtype=torch.float64,
            requires_grad=True,
        )
        context_concatenated = torch.randn_like(
            hidden_concatenated,
            requires_grad=True,
        )
        hidden_split = (
            hidden_concatenated.detach().clone().requires_grad_(True)
        )
        context_split = (
            context_concatenated.detach().clone().requires_grad_(True)
        )

        expected = torch.nn.functional.gelu(
            torch.nn.functional.linear(
                torch.cat(
                    (hidden_concatenated, context_concatenated),
                    dim=-1,
                ),
                concatenated.router_trunk_weight,
                concatenated.router_trunk_bias,
            )
        )
        actual = split._shared_router_packet_features(
            hidden_split,
            context_split,
        )
        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (
                hidden_concatenated,
                context_concatenated,
                concatenated.router_trunk_weight,
                concatenated.router_trunk_bias,
            ),
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (
                hidden_split,
                context_split,
                split.router_trunk_weight,
                split.router_trunk_bias,
            ),
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=1e-12,
                rtol=1e-12,
            )

    def test_dynamic_router_scores_sources_before_active_node_gather(
        self,
    ) -> None:
        torch.manual_seed(124)
        common = dict(
            n_nodes=8,
            d_model=4,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            active_node_top_k=3,
            relays=2,
            lanes=2,
        )
        indexed = ConvergentGator(**common).double()
        scored = ConvergentGator(**common).double()
        scored.load_state_dict(indexed.state_dict(), strict=True)
        hidden_indexed = torch.randn(
            2,
            3,
            2,
            4,
            dtype=torch.float64,
            requires_grad=True,
        )
        context_indexed = torch.randn_like(
            hidden_indexed,
            requires_grad=True,
        )
        occupancy_indexed = torch.randn(
            2,
            3,
            2,
            3,
            3,
            dtype=torch.float64,
            requires_grad=True,
        )
        address_indexed = torch.randn_like(
            occupancy_indexed,
            requires_grad=True,
        )
        hidden_scored = hidden_indexed.detach().clone().requires_grad_(True)
        context_scored = (
            context_indexed.detach().clone().requires_grad_(True)
        )
        occupancy_scored = (
            occupancy_indexed.detach().clone().requires_grad_(True)
        )
        address_scored = (
            address_indexed.detach().clone().requires_grad_(True)
        )
        source_nodes = torch.randint(
            common["n_nodes"],
            (2, 3, 2, 3),
        )

        indexed_features = torch.nn.functional.gelu(
            torch.nn.functional.linear(
                torch.cat(
                    (hidden_indexed, context_indexed),
                    dim=-1,
                ),
                indexed.router_trunk_weight,
                indexed.router_trunk_bias,
            )
        )
        expected = (
            torch.einsum(
                "btlh,btlkdh->btlkd",
                indexed_features,
                indexed.router_edge_key[source_nodes],
            )
            + torch.einsum(
                "btlkj,dj->btlkd",
                occupancy_indexed,
                indexed.router_occupancy_weight,
            )
            + indexed.router_bias[source_nodes]
            + address_indexed
        )
        actual = scored._shared_low_rank_dynamic_router_logits(
            hidden_scored,
            context_scored,
            occupancy_scored,
            source_nodes,
            address_scored,
        )

        with torch.no_grad():
            packet_features = scored._shared_router_packet_features(
                hidden_scored,
                context_scored,
            )
            all_source_logits = scored._shared_router_all_source_logits(
                packet_features,
            )
        self.assertEqual(
            tuple(all_source_logits.shape),
            (2, 3, 2, common["n_nodes"], common["degree"]),
        )
        dynamic_source = inspect.getsource(
            ConvergentGator._shared_low_rank_dynamic_router_logits
        )
        self.assertIn("_shared_router_all_source_logits", dynamic_source)
        self.assertNotIn("router_edge_key[source_nodes]", dynamic_source)

        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (
                hidden_indexed,
                context_indexed,
                occupancy_indexed,
                address_indexed,
                indexed.router_trunk_weight,
                indexed.router_trunk_bias,
                indexed.router_edge_key,
                indexed.router_occupancy_weight,
                indexed.router_bias,
            ),
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (
                hidden_scored,
                context_scored,
                occupancy_scored,
                address_scored,
                scored.router_trunk_weight,
                scored.router_trunk_bias,
                scored.router_edge_key,
                scored.router_occupancy_weight,
                scored.router_bias,
            ),
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=1e-12,
                rtol=1e-12,
            )

    def test_self_bootstrap_one_relay_skips_receiver_kernel(self) -> None:
        class CountingConvergentGator(ConvergentGator):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self.receiver_kernel_calls = 0

            def _receiver_kernel(self, *args, **kwargs) -> torch.Tensor:
                self.receiver_kernel_calls += 1
                return super()._receiver_kernel(*args, **kwargs)

        common = dict(
            n_nodes=8,
            d_model=4,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            active_node_top_k=3,
            relays=2,
            lanes=2,
            bootstrap_context_mode="self",
        )
        observed = torch.tensor([[1, 2, 3], [4, 5, 6]])
        for route_backend in ("dense", "packed"):
            with self.subTest(route_backend=route_backend):
                model = CountingConvergentGator(
                    **common,
                    route_backend=route_backend,
                )
                with torch.no_grad():
                    model(
                        observed,
                        output_slots=2,
                        relay_limit=1,
                    )
                self.assertEqual(model.receiver_kernel_calls, 0)

                with torch.no_grad():
                    model(
                        observed,
                        output_slots=2,
                        relay_limit=2,
                    )
                self.assertEqual(model.receiver_kernel_calls, 1)

                shared_bootstrap = CountingConvergentGator(
                    **{
                        **common,
                        "bootstrap_context_mode": "shared",
                    },
                    route_backend=route_backend,
                )
                with torch.no_grad():
                    shared_bootstrap(
                        observed,
                        output_slots=2,
                        relay_limit=1,
                    )
                self.assertEqual(
                    shared_bootstrap.receiver_kernel_calls,
                    1,
                )

    def test_packed_route_backend_matches_dense_logits_traces_and_gradients(
        self,
    ) -> None:
        torch.manual_seed(118)
        common = dict(
            n_nodes=8,
            d_model=4,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            active_node_top_k=3,
            relays=2,
            lanes=2,
            bootstrap_context_mode="shared",
            connection_layout="small_world",
            connection_seed=7,
            connection_rewire=0.5,
            terminal_vote_mode="slot",
        )
        dense = ConvergentGator(
            **common,
            route_backend="dense",
        )
        packed = ConvergentGator(
            **common,
            route_backend="packed",
        )
        packed.load_state_dict(dense.state_dict(), strict=True)
        observed = torch.tensor(
            [[1, 7, 3, 9], [2, 5, 4, 0]],
            dtype=torch.long,
        )
        context_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]],
        )
        output_mask = torch.tensor(
            [[True, True, True], [True, True, False]],
        )

        for causal in (False, True):
            with self.subTest(causal=causal):
                dense.zero_grad(set_to_none=True)
                packed.zero_grad(set_to_none=True)
                dense_logits, dense_trace = dense(
                    observed,
                    context_mask,
                    output_slots=3,
                    output_mask=output_mask,
                    causal=causal,
                    temperature=0.7,
                    return_trace=True,
                )
                packed_logits, packed_trace = packed(
                    observed,
                    context_mask,
                    output_slots=3,
                    output_mask=output_mask,
                    causal=causal,
                    temperature=0.7,
                    return_trace=True,
                )
                torch.testing.assert_close(
                    packed_logits,
                    dense_logits,
                    atol=3e-6,
                    rtol=3e-5,
                )
                for key in (
                    "locations",
                    "transition_entropy",
                    "ingress_entropy",
                    "terminal_weights",
                ):
                    torch.testing.assert_close(
                        packed_trace[key],
                        dense_trace[key],
                        atol=3e-6,
                        rtol=3e-5,
                    )
                probe = torch.randn_like(dense_logits)
                (dense_logits * probe).sum().backward()
                (packed_logits * probe).sum().backward()
                dense_parameters = dict(dense.named_parameters())
                for name, parameter in packed.named_parameters():
                    with self.subTest(causal=causal, parameter=name):
                        torch.testing.assert_close(
                            parameter.grad,
                            dense_parameters[name].grad,
                            atol=5e-6,
                            rtol=5e-5,
                        )

    def test_packed_convolution_and_opaque_fft_match_matrix_full_model(
        self,
    ) -> None:
        torch.manual_seed(119)
        common = dict(
            n_nodes=8,
            d_model=5,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=6,
            ingress_mode="learned",
            route_backend="packed",
            active_node_top_k=3,
            relays=2,
            lanes=2,
            bootstrap_context_mode="shared",
            connection_layout="small_world",
            connection_seed=9,
            connection_rewire=0.5,
            terminal_vote_mode="slot",
        )
        matrix = ConvergentGator(
            **common,
            packed_memory_backend="matrix",
        )
        convolution = ConvergentGator(
            **common,
            packed_memory_backend="convolution",
            packed_memory_feature_chunk=2,
        )
        opaque_fft = ConvergentGator(
            **common,
            packed_memory_backend="fft_opaque",
            packed_memory_feature_chunk=2,
        )
        eager_fft = ConvergentGator(
            **common,
            packed_memory_backend="fft",
            packed_memory_feature_chunk=2,
        )
        convolution.load_state_dict(matrix.state_dict(), strict=True)
        opaque_fft.load_state_dict(matrix.state_dict(), strict=True)
        eager_fft.load_state_dict(matrix.state_dict(), strict=True)
        observed = torch.tensor([[1, 7, 3], [2, 5, 4]])
        context_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        output_mask = torch.tensor(
            [[True, True], [True, False]]
        )

        for backend, reference, candidate in (
            ("convolution", matrix, convolution),
            ("fft_opaque", eager_fft, opaque_fft),
        ):
            for causal in (False, True):
                with self.subTest(backend=backend, causal=causal):
                    reference.zero_grad(set_to_none=True)
                    candidate.zero_grad(set_to_none=True)
                    expected, expected_trace = reference(
                        observed,
                        context_mask,
                        output_slots=2,
                        output_mask=output_mask,
                        causal=causal,
                        temperature=0.7,
                        return_trace=True,
                    )
                    actual, actual_trace = candidate(
                        observed,
                        context_mask,
                        output_slots=2,
                        output_mask=output_mask,
                        causal=causal,
                        temperature=0.7,
                        return_trace=True,
                    )
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=3e-6,
                        rtol=3e-5,
                    )
                    for key in (
                        "locations",
                        "transition_entropy",
                        "ingress_entropy",
                        "terminal_weights",
                    ):
                        torch.testing.assert_close(
                            actual_trace[key],
                            expected_trace[key],
                            atol=3e-6,
                            rtol=3e-5,
                        )
                    probe = torch.randn_like(expected)
                    (expected * probe).sum().backward()
                    (actual * probe).sum().backward()
                    expected_parameters = dict(reference.named_parameters())
                    for name, parameter in candidate.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            expected_parameters[name].grad,
                            atol=8e-5,
                            rtol=8e-4,
                        )

    def test_factorized_receiver_memory_matches_dense_definition(self) -> None:
        torch.manual_seed(3)
        model = ConvergentGator(
            n_nodes=8,
            d_model=3,
            degree=3,
            relays=2,
            lanes=2,
        )
        batch, length = 2, 5
        hidden = torch.randn(batch, length, 2, 3)
        locations = torch.softmax(
            torch.randn(batch, length, 2, 8),
            dim=-1,
        )
        write = torch.tensor([1.0, 0.25, 0.5, 0.75, 1.0])

        relation_bias = torch.einsum(
            "ijr,nr->ijn",
            relative_code(length, hidden.device),
            model.relative_node_bias,
        )
        scores = (
            locations.clamp_min(1e-9).log()[:, None]
            + relation_bias[None, :, :, None]
            + write.log()[None, None, :, None, None]
        )
        alpha = scores.flatten(2, 3).softmax(2).view(
            batch,
            length,
            length,
            2,
            8,
        )
        dense = torch.einsum(
            "bijkn,bjkd->bind",
            alpha,
            hidden,
        )

        factorized = model._node_memory(
            hidden,
            locations,
            write,
        )

        torch.testing.assert_close(
            factorized,
            dense,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_hoisted_receiver_kernel_preserves_values_and_gradients(self) -> None:
        torch.manual_seed(31)
        direct = ConvergentGator(
            n_nodes=8,
            d_model=3,
            degree=3,
            relays=2,
            lanes=2,
        )
        hoisted = ConvergentGator(
            n_nodes=8,
            d_model=3,
            degree=3,
            relays=2,
            lanes=2,
        )
        hoisted.load_state_dict(direct.state_dict())
        hidden_direct = torch.randn(2, 5, 2, 3, requires_grad=True)
        hidden_hoisted = hidden_direct.detach().clone().requires_grad_(True)
        location_logits = torch.randn(2, 5, 2, 8)
        locations_direct = torch.softmax(
            location_logits,
            dim=-1,
        ).requires_grad_(True)
        locations_hoisted = (
            locations_direct.detach().clone().requires_grad_(True)
        )
        write = torch.tensor([1.0, 0.25, 0.5, 0.75, 1.0])

        expected = direct._node_memory(
            hidden_direct,
            locations_direct,
            write,
        )
        receiver_kernel = hoisted._receiver_kernel(
            hidden_hoisted.size(1),
            hidden_hoisted.device,
        )
        actual = hoisted._node_memory(
            hidden_hoisted,
            locations_hoisted,
            write,
            receiver_kernel=receiver_kernel,
        )
        expected.square().mean().backward()
        actual.square().mean().backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            hidden_hoisted.grad,
            hidden_direct.grad,
        )
        torch.testing.assert_close(
            locations_hoisted.grad,
            locations_direct.grad,
        )
        torch.testing.assert_close(
            hoisted.relative_node_bias.grad,
            direct.relative_node_bias.grad,
        )

    def test_fused_receiver_mix_matches_two_contractions(self) -> None:
        torch.manual_seed(32)
        model = ConvergentGator(
            n_nodes=8,
            d_model=3,
            degree=3,
            relays=2,
            lanes=2,
        ).double()
        hidden = torch.randn(
            2,
            5,
            2,
            3,
            dtype=torch.float64,
            requires_grad=True,
        )
        location_logits = torch.randn(
            2,
            5,
            2,
            8,
            dtype=torch.float64,
        )
        locations = torch.softmax(location_logits, dim=-1).requires_grad_(True)
        write = torch.tensor(
            [1.0, 1e-9, 0.5, 0.75, 1.0],
            dtype=torch.float64,
        )
        kernel = model._receiver_kernel(5, hidden.device)
        mass = (
            locations.clamp_min(1e-9)
            * write.clamp_min(1e-9)[None, :, None, None]
        )
        node_mass = mass.sum(2)
        node_state = torch.einsum(
            "bjkn,bjkd->bjnd",
            mass,
            hidden,
        )
        numerator = torch.einsum(
            "ijn,bjnd->bind",
            kernel,
            node_state,
        )
        denominator = torch.einsum(
            "ijn,bjn->bin",
            kernel,
            node_mass,
        )
        expected = numerator / denominator.clamp_min(1e-9).unsqueeze(-1)
        actual = model._node_memory(hidden, locations, write)

        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (hidden, locations, model.relative_node_bias),
            retain_graph=True,
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (hidden, locations, model.relative_node_bias),
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=1e-11,
                rtol=1e-11,
            )

    def test_factored_router_matches_expanded_values_and_gradients(self) -> None:
        torch.manual_seed(37)
        batch, length, lanes, sources, degree, width = 2, 5, 3, 4, 3, 6
        semantic_reference = torch.randn(
            batch,
            length,
            lanes,
            width,
            requires_grad=True,
        )
        semantic_factored = (
            semantic_reference.detach().clone().requires_grad_(True)
        )
        occupancy_reference = torch.randn(
            batch,
            lanes,
            sources,
            degree,
            requires_grad=True,
        )
        occupancy_factored = (
            occupancy_reference.detach().clone().requires_grad_(True)
        )
        weight_reference = torch.randn(
            lanes,
            sources,
            degree,
            width + degree,
            requires_grad=True,
        )
        weight_factored = (
            weight_reference.detach().clone().requires_grad_(True)
        )
        bias_reference = torch.randn(
            lanes,
            sources,
            degree,
            requires_grad=True,
        )
        bias_factored = (
            bias_reference.detach().clone().requires_grad_(True)
        )
        address_reference = torch.randn(
            length,
            lanes,
            sources,
            degree,
            requires_grad=True,
        )
        address_factored = (
            address_reference.detach().clone().requires_grad_(True)
        )

        expanded_input = torch.cat(
            (
                semantic_reference.unsqueeze(3).expand(
                    -1,
                    -1,
                    -1,
                    sources,
                    -1,
                ),
                occupancy_reference[:, None].expand(
                    -1,
                    length,
                    -1,
                    -1,
                    -1,
                ),
            ),
            dim=-1,
        )
        expected = (
            torch.einsum(
                "btksi,ksdi->btksd",
                expanded_input,
                weight_reference,
            )
            + bias_reference[None, None]
            + address_reference[None]
        )
        actual = ConvergentGator._factored_router_logits(
            semantic_factored,
            occupancy_factored,
            weight_factored,
            bias_factored,
            address_factored,
        )
        expected.square().mean().backward()
        actual.square().mean().backward()

        torch.testing.assert_close(actual, expected)
        for actual_gradient, expected_gradient in zip(
            (
                semantic_factored.grad,
                occupancy_factored.grad,
                weight_factored.grad,
                bias_factored.grad,
                address_factored.grad,
            ),
            (
                semantic_reference.grad,
                occupancy_reference.grad,
                weight_reference.grad,
                bias_reference.grad,
                address_reference.grad,
            ),
            strict=True,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
            )

    def test_folded_nodes_preserve_arrow_input_output_weight_order(self) -> None:
        torch.manual_seed(41)
        batch, tokens, lanes, nodes, width = 2, 5, 3, 4, 6
        base = torch.randn(
            batch,
            tokens,
            lanes,
            width,
            dtype=torch.float64,
        )
        embedding = torch.randn(
            lanes,
            nodes,
            width,
            dtype=torch.float64,
        )
        # Convergent Arrow stores node weights as [input, output], matching
        # the original ``...d,...de->...e`` relay equation.
        weight = torch.randn(
            lanes,
            nodes,
            width,
            width,
            dtype=torch.float64,
        )
        bias = torch.randn(
            lanes,
            nodes,
            width,
            dtype=torch.float64,
        )
        expected = (
            torch.einsum(
                "btksd,ksde->btkse",
                base.unsqueeze(3) + embedding[None, None],
                weight,
            )
            + bias[None, None]
        )
        actual = folded_cell_node_linear(
            base.permute(0, 2, 1, 3),
            embedding,
            weight.transpose(-1, -2),
            bias,
        ).permute(0, 2, 1, 3, 4)
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_reachable_node_execution_matches_dense_node_execution(self) -> None:
        torch.manual_seed(4)
        sparse = ConvergentGator()
        dense = ConvergentGator(reachable_node_execution=False)
        dense.load_state_dict(sparse.state_dict())
        observed, targets = make_prefix_batch(
            3,
            7,
            torch.device("cpu"),
        )

        sparse_logits = sparse(observed, temperature=0.73)
        dense_logits = dense(observed, temperature=0.73)
        sparse_loss = torch.nn.functional.cross_entropy(
            sparse_logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        dense_loss = torch.nn.functional.cross_entropy(
            dense_logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        sparse_loss.backward()
        dense_loss.backward()

        torch.testing.assert_close(
            sparse_logits,
            dense_logits,
            atol=2e-6,
            rtol=2e-6,
        )
        dense_parameters = dict(dense.named_parameters())
        for name, parameter in sparse.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    parameter.grad,
                    dense_parameters[name].grad,
                    atol=3e-7,
                    rtol=3e-6,
                )

    def test_default_budget_and_lane_seeding(self) -> None:
        model = ConvergentGator()

        self.assertEqual(model.parameter_count, 10_010)
        self.assertEqual(
            model.start_nodes.tolist(),
            [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
        )
        self.assertEqual(
            model.architecture_config,
            {
                "n_nodes": 40,
                "d_model": 8,
                "degree": 3,
                "router_mode": "private_full",
                "router_rank": 16,
                "ingress_mode": "fixed",
                "route_backend": "dense",
                "packed_memory_backend": "matrix",
                "packed_memory_reader_backend": "torch",
                "packed_memory_writer_backend": "torch",
                "packed_memory_feature_chunk": 32,
                "packed_memory_compact_nodes": False,
                "adaptive_route_min_k": None,
                "adaptive_route_mode": "hybrid",
                "adaptive_route_entropy_mix": 0.5,
                "compact_node_dispatch": False,
                "node_compute_mode": "full_private",
                "node_adapter_rank": 8,
                "ingress_groups": None,
                "ingress_top_groups": 2,
                "ingress_learned_groups": True,
                "active_node_top_k": None,
                "active_node_mass_threshold": 0.0,
                "relays": 4,
                "lanes": 10,
                "relay_code_span": 4,
                "write_warmup_relays": 4,
                "bootstrap_context_mode": "shared",
                "connection_layout": "ring",
                "connection_seed": 0,
                "connection_rewire": 1.0,
                "terminal_vote_mode": "global",
                "vocabulary_mode": "private_lane",
                "vocab_size": 16,
                "embedding_dim": 8,
            },
        )

    def test_parameter_formula_covers_fixed_budget_sweep(self) -> None:
        configurations = (
            (40, 9, 6, 9_892),
            (40, 8, 10, 10_010),
            (40, 7, 16, 10_376),
            (30, 9, 10, 9_850),
            (50, 7, 10, 9_810),
        )
        for nodes, width, lanes, expected in configurations:
            with self.subTest(nodes=nodes, width=width, lanes=lanes):
                model = ConvergentGator(
                    n_nodes=nodes,
                    d_model=width,
                    lanes=lanes,
                )
                self.assertEqual(model.parameter_count, expected)
                self.assertEqual(
                    ConvergentGator.parameter_count_for(
                        n_nodes=nodes,
                        d_model=width,
                        lanes=lanes,
                    ),
                    expected,
                )
        relay_counts = {
            ConvergentGator(relays=relays).parameter_count
            for relays in (1, 4, 8)
        }
        self.assertEqual(relay_counts, {10_010})
        degree_five = ConvergentGator(degree=5)
        self.assertEqual(
            degree_five.parameter_count,
            ConvergentGator.parameter_count_for(
                n_nodes=40,
                d_model=8,
                lanes=10,
                degree=5,
            ),
        )

    def test_shared_low_rank_router_reduces_routing_capacity_exactly(
        self,
    ) -> None:
        common = {
            "n_nodes": 66,
            "d_model": 400,
            "lanes": 10,
            "degree": 3,
            "terminal_vote_mode": "slot",
        }
        private_count = ConvergentGator.parameter_count_for(
            router_mode="private_full",
            **common,
        )
        factored_count = ConvergentGator.parameter_count_for(
            router_mode="shared_low_rank",
            router_rank=16,
            ingress_mode="learned",
            **common,
        )
        factored = ConvergentGator(
            router_mode="shared_low_rank",
            router_rank=16,
            ingress_mode="learned",
            active_node_top_k=4,
            **common,
        )

        self.assertEqual(factored.parameter_count, factored_count)
        self.assertEqual(private_count - factored_count, 141_879)
        self.assertIsNone(factored.router_weight)
        self.assertEqual(
            tuple(factored.router_edge_key.shape),
            (66, 3, 16),
        )
        self.assertEqual(
            tuple(factored.ingress_node_key.shape),
            (66, 16),
        )

    def test_learned_ingress_has_global_node_support(self) -> None:
        torch.manual_seed(131)
        model = ConvergentGator(
            n_nodes=12,
            d_model=6,
            lanes=3,
            relays=2,
            router_mode="shared_low_rank",
            router_rank=4,
            ingress_mode="learned",
        )
        observed = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            _, trace = model(
                observed,
                output_slots=2,
                return_trace=True,
            )

        ingress = trace["locations"][:, 0]
        self.assertTrue(bool(ingress.gt(0).all()))
        self.assertTrue(
            torch.allclose(
                ingress.sum(-1),
                torch.ones_like(ingress[..., 0]),
            )
        )
        self.assertTrue(torch.isfinite(trace["ingress_entropy"]))

    def test_active_node_execution_keeps_only_meaningful_mass(self) -> None:
        torch.manual_seed(137)
        model = ConvergentGator(
            n_nodes=20,
            d_model=6,
            lanes=3,
            relays=2,
            router_mode="shared_low_rank",
            router_rank=4,
            ingress_mode="learned",
            active_node_top_k=2,
            active_node_mass_threshold=0.05,
        )
        observed = torch.tensor([[1, 2]])
        logits, trace = model(
            observed,
            output_slots=2,
            return_trace=True,
        )
        logits.square().mean().backward()

        support = trace["locations"].gt(0).sum(-1)
        self.assertLessEqual(int(support.max()), 2)
        self.assertTrue(
            torch.allclose(
                trace["locations"].sum(-1),
                torch.ones_like(trace["locations"][..., 0]),
            )
        )
        node_gradient = model.node_weight.grad.abs().sum(dim=(1, 2))
        self.assertGreater(int(node_gradient.gt(0).sum()), 0)
        self.assertLess(int(node_gradient.gt(0).sum()), model.n_nodes)
        for parameter in (
            model.router_trunk_weight,
            model.router_edge_key,
            model.router_bias,
            model.ingress_node_key,
            model.ingress_node_bias,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_adaptive_packed_routes_report_live_width_and_uncertainty(self) -> None:
        torch.manual_seed(138)
        model = ConvergentGator(
            n_nodes=12,
            d_model=6,
            lanes=3,
            relays=2,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            route_backend="packed",
            active_node_top_k=4,
            adaptive_route_min_k=1,
            adaptive_route_mode="hybrid",
        )
        logits, trace = model(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            output_slots=2,
            return_trace=True,
        )
        support = trace["locations"].gt(0).sum(-1).to(logits.dtype)
        observed_width = support.mean(dim=(0, 2, 3))
        torch.testing.assert_close(trace["active_route_k"], observed_width)
        self.assertEqual(tuple(trace["active_route_k"].shape), (3,))
        self.assertEqual(tuple(trace["route_uncertainty"].shape), (3,))
        self.assertTrue(bool(trace["active_route_k"].ge(1).all()))
        self.assertTrue(bool(trace["active_route_k"].le(4).all()))
        self.assertTrue(bool(trace["route_uncertainty"].ge(0).all()))
        self.assertTrue(bool(trace["route_uncertainty"].le(1).all()))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.router_edge_key.grad).all())

    def test_sparse_grouped_node_update_matches_packet_definition(
        self,
    ) -> None:
        torch.manual_seed(139)
        model = ConvergentGator(
            n_nodes=12,
            d_model=8,
            lanes=3,
            relays=2,
            router_mode="shared_low_rank",
            router_rank=4,
            ingress_mode="learned",
            active_node_top_k=2,
        )
        base = torch.randn(2, 4, 3, 8, requires_grad=True)
        active_nodes = torch.randint(0, 12, (2, 4, 3, 2))
        active_mass = torch.softmax(torch.randn(2, 4, 3, 2), -1)

        actual = model._sparse_node_update(
            base,
            active_nodes,
            active_mass,
        )
        node_input = (
            base.unsqueeze(-2)
            + model.node_embedding.weight[active_nodes]
        )
        expected = (
            active_mass.unsqueeze(-1)
            * torch.tanh(
                torch.einsum(
                    "btkmi,btkmio->btkmo",
                    node_input,
                    model.node_weight[active_nodes],
                )
                + model.node_bias[active_nodes]
            )
        ).sum(-2)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(base.grad).all())
        self.assertTrue(torch.isfinite(model.node_weight.grad).all())
        self.assertEqual(
            int(
                model.node_weight.grad.abs().sum(dim=(1, 2)).gt(0).sum()
            ),
            int(active_nodes.unique().numel()),
        )

    def test_shared_adapter_node_compute_integrates_with_packed_model(self) -> None:
        torch.manual_seed(140)
        common = dict(
            n_nodes=16,
            d_model=8,
            lanes=3,
            relays=2,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            route_backend="packed",
            active_node_top_k=3,
            node_compute_mode="shared_adapter",
            node_adapter_rank=2,
        )
        model = ConvergentGator(**common)
        self.assertEqual(
            model.parameter_count,
            ConvergentGator.parameter_count_for(
                n_nodes=16,
                d_model=8,
                lanes=3,
                degree=3,
                router_mode="shared_low_rank",
                router_rank=5,
                ingress_mode="learned",
                node_compute_mode="shared_adapter",
                node_adapter_rank=2,
            ),
        )
        self.assertIsNone(model.node_weight)
        self.assertIsNotNone(model.shared_node_adapters)
        logits = model(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            output_slots=2,
        )
        logits.square().mean().backward()
        for parameter in model.shared_node_adapters.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_hierarchical_ingress_integrates_and_counts_group_parameters(self) -> None:
        torch.manual_seed(141)
        model = ConvergentGator(
            n_nodes=12,
            d_model=8,
            lanes=3,
            relays=2,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            route_backend="packed",
            active_node_top_k=4,
            ingress_groups=3,
            ingress_top_groups=2,
            ingress_learned_groups=True,
        )
        expected = ConvergentGator.parameter_count_for(
            n_nodes=12,
            d_model=8,
            lanes=3,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            ingress_groups=3,
            ingress_learned_groups=True,
        )
        self.assertEqual(model.parameter_count, expected)
        layout_nodes = model.hierarchical_ingress.group_nodes[
            model.hierarchical_ingress.group_valid
        ]
        self.assertEqual(
            sorted(layout_nodes.tolist()),
            list(range(model.n_nodes)),
        )
        logits, trace = model(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            output_slots=2,
            return_trace=True,
        )
        self.assertLessEqual(
            int(trace["locations"][:, 0].gt(0).sum(-1).max()),
            4,
        )
        logits.square().mean().backward()
        self.assertGreater(
            float(model.hierarchical_ingress.group_keys.grad.abs().sum()),
            0.0,
        )

    def test_sparse_routing_is_independent_across_batch_rows(self) -> None:
        torch.manual_seed(149)
        model = ConvergentGator(
            n_nodes=20,
            d_model=8,
            lanes=3,
            relays=2,
            router_mode="shared_low_rank",
            router_rank=4,
            ingress_mode="learned",
            active_node_top_k=2,
            active_node_mass_threshold=0.01,
        ).eval()
        first = torch.tensor([[1, 2, 3]])
        second = torch.tensor([[9, 8, 7]])
        with torch.no_grad():
            alone = model(first, output_slots=2)
            together = model(
                torch.cat((first, second)),
                output_slots=2,
            )[:1]

        self.assertTrue(torch.allclose(alone, together, atol=2e-6))

    def test_connection_layouts_are_explicit_and_deterministic(self) -> None:
        ring = connection_destinations(
            n_nodes=8,
            degree=3,
            layout="ring",
            seed=0,
        )
        self.assertEqual(
            ring.tolist(),
            [
                [0, 1, 7],
                [1, 2, 0],
                [2, 3, 1],
                [3, 4, 2],
                [4, 5, 3],
                [5, 6, 4],
                [6, 7, 5],
                [7, 0, 6],
            ],
        )
        first = connection_destinations(
            n_nodes=40,
            degree=3,
            layout="small_world",
            seed=19,
            rewire_probability=1.0,
        )
        repeated = connection_destinations(
            n_nodes=40,
            degree=3,
            layout="small_world",
            seed=19,
            rewire_probability=1.0,
        )
        changed = connection_destinations(
            n_nodes=40,
            degree=3,
            layout="small_world",
            seed=20,
            rewire_probability=1.0,
        )
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.equal(first, changed))
        self.assertTrue(
            torch.equal(first[:, 0], torch.arange(40))
        )
        unrewired = connection_destinations(
            n_nodes=40,
            degree=3,
            layout="small_world",
            seed=19,
            rewire_probability=0.0,
        )
        ring_40 = connection_destinations(
            n_nodes=40,
            degree=3,
            layout="ring",
            seed=0,
        )
        torch.testing.assert_close(unrewired, ring_40)
        self.assertTrue(
            torch.equal(first[:, 1], (torch.arange(40) + 1) % 40)
        )
        self.assertTrue(
            all(
                len(set(row)) == 3
                for row in first.tolist()
            )
        )
        with self.assertRaises(ValueError):
            connection_destinations(
                n_nodes=8,
                degree=3,
                layout="dense",
                seed=0,
            )

    def test_connection_generation_does_not_change_weight_seed(self) -> None:
        torch.manual_seed(79)
        ring = ConvergentGator(connection_layout="ring")
        torch.manual_seed(79)
        small_world = ConvergentGator(
            connection_layout="small_world",
            connection_seed=17,
            connection_rewire=0.5,
        )
        for (ring_name, ring_parameter), (
            small_world_name,
            small_world_parameter,
        ) in zip(
            ring.named_parameters(),
            small_world.named_parameters(),
            strict=True,
        ):
            self.assertEqual(ring_name, small_world_name)
            torch.testing.assert_close(
                ring_parameter,
                small_world_parameter,
            )

    def test_graph_reachability_matches_connection_closure(self) -> None:
        destinations = connection_destinations(
            n_nodes=12,
            degree=3,
            layout="small_world",
            seed=7,
        )
        starts = torch.tensor([0, 4, 8])
        nodes, mask, counts = graph_reachability(
            destinations,
            starts,
            relays=3,
        )
        for lane, start in enumerate(starts.tolist()):
            expected = {start}
            for relay in range(4):
                actual = set(
                    nodes[relay, lane][mask[relay, lane]].tolist()
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    int(counts[relay]),
                    int(mask[relay].sum(dim=-1).max()),
                )
                expected = {
                    int(destination)
                    for source in expected
                    for destination in destinations[source].tolist()
                }

    def test_small_world_reachable_execution_matches_dense(self) -> None:
        torch.manual_seed(73)
        sparse = ConvergentGator(
            n_nodes=20,
            lanes=5,
            relays=4,
            connection_layout="small_world",
            connection_seed=11,
        )
        dense = ConvergentGator(
            n_nodes=20,
            lanes=5,
            relays=4,
            connection_layout="small_world",
            connection_seed=11,
            reachable_node_execution=False,
        )
        dense.load_state_dict(sparse.state_dict())
        observed, targets = make_prefix_batch(
            2,
            7,
            torch.device("cpu"),
        )

        sparse_logits = sparse(observed, temperature=0.73)
        dense_logits = dense(observed, temperature=0.73)
        sparse_loss = torch.nn.functional.cross_entropy(
            sparse_logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        dense_loss = torch.nn.functional.cross_entropy(
            dense_logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        sparse_loss.backward()
        dense_loss.backward()

        torch.testing.assert_close(
            sparse_logits,
            dense_logits,
            atol=2e-6,
            rtol=2e-6,
        )
        dense_parameters = dict(dense.named_parameters())
        for name, parameter in sparse.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    parameter.grad,
                    dense_parameters[name].grad,
                    atol=3e-7,
                    rtol=3e-6,
                )

    def test_shared_relay_schedules_make_first_four_hops_identical(self) -> None:
        torch.manual_seed(61)
        four = ConvergentGator(
            relays=4,
            relay_code_span=8,
            write_warmup_relays=4,
        )
        eight = ConvergentGator(
            relays=8,
            relay_code_span=8,
            write_warmup_relays=4,
        )
        eight.load_state_dict(four.state_dict())
        observed = torch.tensor([[1, 2], [3, 4]])
        with torch.no_grad():
            _, four_trace = four(observed, return_trace=True)
            _, eight_trace = eight(observed, return_trace=True)
        torch.testing.assert_close(
            eight_trace["locations"][:, :5],
            four_trace["locations"],
        )
        torch.testing.assert_close(
            eight_trace["transition_entropy"][:4],
            four_trace["transition_entropy"],
        )
        with self.assertRaises(ValueError):
            ConvergentGator(relays=8, relay_code_span=4)
        with self.assertRaises(ValueError):
            ConvergentGator(write_warmup_relays=0)
        with self.assertRaises(ValueError):
            ConvergentGator(bootstrap_context_mode="unknown")
        with self.assertRaises(ValueError):
            ConvergentGator(terminal_vote_mode="unknown")

    def test_self_bootstrap_forces_information_to_follow_routes(self) -> None:
        torch.manual_seed(67)
        observed_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        observed_b = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]])

        one_relay = ConvergentGator(
            relays=1,
            bootstrap_context_mode="self",
        )
        two_relays = ConvergentGator(
            relays=2,
            bootstrap_context_mode="self",
        )
        two_relays.load_state_dict(one_relay.state_dict())

        with torch.no_grad():
            one_a = one_relay(observed_a, output_slots=7)
            one_b = one_relay(observed_b, output_slots=7)
            two_a = two_relays(observed_a, output_slots=7)
            two_b = two_relays(observed_b, output_slots=7)

        torch.testing.assert_close(one_a, one_b)
        self.assertGreater(
            float((two_a - two_b).abs().max()),
            1e-6,
        )

    def test_forward_preserves_distinct_lane_and_location_axes(self) -> None:
        torch.manual_seed(43)
        model = ConvergentGator()
        observed, targets = make_prefix_batch(
            2,
            7,
            torch.device("cpu"),
        )

        logits, trace = model(
            observed,
            output_slots=7,
            temperature=1.25,
            return_trace=True,
        )
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        loss.backward()

        self.assertEqual(tuple(logits.shape), (2, 7, 16))
        self.assertEqual(
            tuple(trace["locations"].shape),
            (2, 5, 15, 10, 40),
        )
        self.assertEqual(
            tuple(trace["lane_logits"].shape),
            (2, 7, 10, 16),
        )
        reconstructed = torch.einsum(
            "k,bokv->bov",
            trace["terminal_weights"],
            trace["lane_logits"],
        )
        torch.testing.assert_close(logits, reconstructed)
        torch.testing.assert_close(
            trace["locations"].sum(-1),
            torch.ones_like(trace["locations"][..., 0]),
        )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_scaled_language_configuration_is_near_130k(self) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=2048,
            d_model=20,
            embedding_dim=32,
            n_nodes=66,
            degree=3,
            relays=2,
            lanes=10,
            blank_token_id=2047,
        )
        model = ConvergentLanguageModel(config)

        self.assertEqual(model.parameter_count, 130_202)
        self.assertEqual(
            model.parameter_count,
            ConvergentGator.shared_language_parameter_count_for(
                vocab_size=2048,
                embedding_dim=32,
                n_nodes=66,
                d_model=20,
                lanes=10,
                degree=3,
            ),
        )

    def test_language_capacity_width_presets_have_exact_counts(self) -> None:
        common = {
            "vocab_size": 2048,
            "embedding_dim": 32,
            "n_nodes": 66,
            "lanes": 10,
            "degree": 3,
        }

        self.assertEqual(
            ConvergentGator.shared_language_parameter_count_for(
                d_model=30,
                **common,
            ),
            186_382,
        )
        self.assertEqual(
            ConvergentGator.shared_language_parameter_count_for(
                d_model=50,
                **common,
            ),
            350_342,
        )
        self.assertEqual(
            ConvergentGator.shared_language_parameter_count_for(
                d_model=54,
                **common,
            ),
            391_390,
        )

    def test_prompt_terminal_vote_starts_as_exact_global_control(self) -> None:
        torch.manual_seed(89)
        global_config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="global",
        )
        prompt_config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="prompt",
        )
        global_model = ConvergentLanguageModel(global_config)
        prompt_model = ConvergentLanguageModel(prompt_config)
        incompatible = prompt_model.load_state_dict(
            global_model.state_dict(),
            strict=False,
        )
        self.assertEqual(
            incompatible.missing_keys,
            ["core.terminal_vote_query"],
        )
        self.assertEqual(incompatible.unexpected_keys, [])
        context = torch.randint(2, 63, (2, 6))
        context_mask = torch.ones_like(context, dtype=torch.bool)

        global_logits = global_model(
            context,
            context_mask,
            output_slots=4,
        )
        prompt_logits, trace = prompt_model(
            context,
            context_mask,
            output_slots=4,
            return_trace=True,
        )

        torch.testing.assert_close(prompt_logits, global_logits)
        self.assertEqual(
            tuple(trace["terminal_weights"].shape),
            (2, 3),
        )
        torch.testing.assert_close(
            trace["terminal_weights"].sum(-1),
            torch.ones(2),
        )
        self.assertEqual(
            prompt_model.parameter_count,
            global_model.parameter_count + 3 * 6,
        )

        prompt_logits.square().mean().backward()
        query_gradient = prompt_model.core.terminal_vote_query.grad
        self.assertIsNotNone(query_gradient)
        self.assertGreater(float(query_gradient.abs().sum()), 0.0)

    def test_slot_terminal_vote_has_direct_address_judges(self) -> None:
        torch.manual_seed(97)
        global_config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="global",
        )
        slot_config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="slot",
        )
        torch.manual_seed(97)
        global_model = ConvergentLanguageModel(global_config)
        torch.manual_seed(97)
        slot_model = ConvergentLanguageModel(slot_config)
        context = torch.randint(2, 63, (2, 6))
        context_mask = torch.ones_like(context, dtype=torch.bool)

        global_logits = global_model(
            context,
            context_mask,
            output_slots=4,
        )
        slot_logits, trace = slot_model(
            context,
            context_mask,
            output_slots=4,
            return_trace=True,
        )

        self.assertFalse(torch.equal(slot_logits, global_logits))
        self.assertEqual(
            tuple(trace["terminal_weights"].shape),
            (2, 4, 3),
        )
        torch.testing.assert_close(
            trace["terminal_weights"].sum(-1),
            torch.ones(2, 4),
        )
        self.assertEqual(
            slot_model.parameter_count,
            global_model.parameter_count + 3 * (6 + 4),
        )
        self.assertGreater(
            float(
                (
                    trace["terminal_weights"][:, 1:]
                    - trace["terminal_weights"][:, :-1]
                )
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )
        self.assertGreater(
            int(torch.unique(trace["terminal_weights"].argmax(-1)).numel()),
            1,
        )

        slot_logits.square().mean().backward()
        query_gradient = slot_model.core.terminal_vote_query.grad
        self.assertIsNotNone(query_gradient)
        self.assertGreater(float(query_gradient.abs().sum()), 0.0)
        address_gradient = slot_model.core.terminal_address_key.grad
        self.assertIsNotNone(address_gradient)
        self.assertGreater(float(address_gradient.abs().sum()), 0.0)

        with torch.no_grad():
            slot_model.core.terminal_vote_query.zero_()
            slot_model.core.terminal_address_key.zero_()
            restored_logits, restored_trace = slot_model(
                context,
                context_mask,
                output_slots=4,
                return_trace=True,
            )
        torch.testing.assert_close(restored_logits, global_logits)
        torch.testing.assert_close(
            restored_trace["terminal_weights"],
            torch.full_like(restored_trace["terminal_weights"], 1.0 / 3),
        )

    def test_legacy_slot_checkpoint_loads_with_zero_address_path(self) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            terminal_vote_mode="slot",
        )
        source = ConvergentLanguageModel(config)
        legacy_state = {
            name: value
            for name, value in source.state_dict().items()
            if name != "core.terminal_address_key"
        }
        restored = ConvergentLanguageModel(config)
        restored.load_state_dict(legacy_state, strict=True)
        torch.testing.assert_close(
            restored.core.terminal_address_key,
            torch.zeros_like(restored.core.terminal_address_key),
        )

    def test_slot_controller_preserves_shared_seed_initialization(self) -> None:
        common = {
            "vocab_size": 64,
            "d_model": 6,
            "embedding_dim": 8,
            "n_nodes": 12,
            "degree": 3,
            "relays": 2,
            "lanes": 3,
            "blank_token_id": 63,
        }
        torch.manual_seed(101)
        global_model = ConvergentLanguageModel(
            ConvergentLanguageConfig(
                terminal_vote_mode="global",
                **common,
            )
        )
        torch.manual_seed(101)
        slot_model = ConvergentLanguageModel(
            ConvergentLanguageConfig(
                terminal_vote_mode="slot",
                **common,
            )
        )

        global_state = global_model.state_dict()
        slot_state = slot_model.state_dict()
        for name, tensor in global_state.items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(tensor, slot_state[name])

    def test_prompt_terminal_vote_parameter_formula_is_exact(self) -> None:
        global_count = ConvergentGator.shared_language_parameter_count_for(
            vocab_size=2048,
            embedding_dim=32,
            n_nodes=90,
            d_model=20,
            lanes=16,
            degree=3,
            terminal_vote_mode="global",
        )
        prompt_count = ConvergentGator.shared_language_parameter_count_for(
            vocab_size=2048,
            embedding_dim=32,
            n_nodes=90,
            d_model=20,
            lanes=16,
            degree=3,
            terminal_vote_mode="prompt",
        )
        slot_count = ConvergentGator.shared_language_parameter_count_for(
            vocab_size=2048,
            embedding_dim=32,
            n_nodes=90,
            d_model=20,
            lanes=16,
            degree=3,
            terminal_vote_mode="slot",
        )
        model = ConvergentLanguageModel(
            ConvergentLanguageConfig(
                vocab_size=2048,
                embedding_dim=32,
                n_nodes=90,
                d_model=20,
                lanes=16,
                degree=3,
                terminal_vote_mode="prompt",
            )
        )

        self.assertEqual(global_count, 158_600)
        self.assertEqual(prompt_count, 158_920)
        self.assertEqual(slot_count, prompt_count + 16 * 4)
        self.assertEqual(model.parameter_count, prompt_count)

    def test_language_forward_masks_padding_and_forbidden_outputs(self) -> None:
        torch.manual_seed(83)
        config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
            forbidden_output_token_ids=(0, 1),
        )
        model = ConvergentLanguageModel(config)
        context = torch.randint(2, 63, (2, 6))
        context_mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, True, False],
            ]
        )
        changed_padding = context.clone()
        changed_padding[~context_mask] = torch.randint(
            2,
            63,
            (int((~context_mask).sum()),),
        )

        logits = model(
            context,
            context_mask,
            output_slots=4,
            temperature=0.7,
        )
        changed_logits = model(
            changed_padding,
            context_mask,
            output_slots=4,
            temperature=0.7,
        )

        self.assertEqual(tuple(logits.shape), (2, 4, 64))
        torch.testing.assert_close(logits, changed_logits)
        self.assertTrue((logits[..., :2] == -1e9).all())
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 64),
            torch.randint(2, 63, (8,)),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_relay_limit_preserves_trained_relay_prefix(self) -> None:
        torch.manual_seed(71)
        model = ConvergentGator(relays=4)
        observed, _ = make_prefix_batch(
            2,
            7,
            torch.device("cpu"),
        )

        with torch.no_grad():
            full_logits, full_trace = model(
                observed,
                output_slots=7,
                return_trace=True,
            )
            limited_full_logits, limited_full_trace = model(
                observed,
                output_slots=7,
                relay_limit=4,
                return_trace=True,
            )
            _, two_trace = model(
                observed,
                output_slots=7,
                relay_limit=2,
                return_trace=True,
            )

        torch.testing.assert_close(full_logits, limited_full_logits)
        torch.testing.assert_close(
            full_trace["locations"],
            limited_full_trace["locations"],
        )
        torch.testing.assert_close(
            two_trace["locations"],
            full_trace["locations"][:, :3],
        )
        torch.testing.assert_close(
            two_trace["transition_entropy"],
            full_trace["transition_entropy"][:2],
        )
        with self.assertRaises(ValueError):
            model(observed, relay_limit=0)
        with self.assertRaises(ValueError):
            model(observed, relay_limit=5)

    def test_relay_depth_expands_each_lane_frontier_by_two_nodes(self) -> None:
        observed = torch.tensor([[1, 2]])
        for relays in (1, 4, 8):
            with self.subTest(relays=relays):
                model = ConvergentGator(relays=relays)
                with torch.no_grad():
                    _, trace = model(observed, return_trace=True)
                locations = trace["locations"]
                support = locations.gt(0).sum(-1)
                expected = torch.tensor(
                    [1 + 2 * relay for relay in range(relays + 1)]
                ).view(1, relays + 1, 1, 1)
                torch.testing.assert_close(
                    support,
                    expected.expand_as(support),
                )

    def test_lane_frontiers_first_collide_after_two_handoffs(self) -> None:
        model = ConvergentGator(relays=4)
        observed = torch.tensor([[1, 2]])
        with torch.no_grad():
            _, trace = model(observed, return_trace=True)
        locations = trace["locations"]
        overlaps = []
        for relay in range(locations.size(1)):
            q = locations[:, relay]
            node_mass = q.sum(dim=2)
            pair_overlap = 0.5 * (
                node_mass.square()
                - q.square().sum(dim=2)
            ).sum(dim=-1).mean()
            overlaps.append(float(pair_overlap))
        self.assertEqual(overlaps[0], 0.0)
        self.assertEqual(overlaps[1], 0.0)
        self.assertGreater(overlaps[2], 0.0)

    def test_optional_route_top_k_limits_support_and_normalizes(self) -> None:
        torch.manual_seed(53)
        model = ConvergentGator(relays=4)
        observed = torch.tensor([[1, 2], [3, 4]])
        with torch.no_grad():
            logits, trace = model(
                observed,
                route_top_k=2,
                return_trace=True,
            )
        routed = trace["locations"][:, 1:]
        self.assertLessEqual(
            int(routed.gt(0).sum(-1).max()),
            2,
        )
        torch.testing.assert_close(
            routed.sum(-1),
            torch.ones_like(routed[..., 0]),
        )
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            model(observed, route_top_k=0)
        with self.assertRaises(ValueError):
            model(
                observed,
                route_top_k=2,
                route_top_k_start=4,
            )
        with torch.no_grad():
            _, scheduled_trace = model(
                observed,
                route_top_k_schedule=(3, 5, 4, 3),
                return_trace=True,
            )
        scheduled_support = scheduled_trace["locations"].gt(0).sum(-1)
        for relay, limit in enumerate((3, 5, 4, 3), start=1):
            self.assertLessEqual(
                int(scheduled_support[:, relay].max()),
                limit,
            )
        with self.assertRaises(ValueError):
            model(
                observed,
                route_top_k=2,
                route_top_k_schedule=(3, 5, 4, 3),
            )
        with self.assertRaises(ValueError):
            model(
                observed,
                route_top_k_schedule=(3, 5),
            )

    def test_broad_lanes_and_graph_sites_all_receive_task_gradient(self) -> None:
        torch.manual_seed(47)
        model = ConvergentGator()
        observed, targets = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        logits = model(
            observed,
            output_slots=7,
            temperature=1.25,
        )
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 16),
            targets.reshape(-1),
        )
        loss.backward()

        lane_gradient = model.token_embedding.grad.abs().sum(dim=(1, 2))
        node_gradient = model.node_weight.grad.abs().sum(dim=(1, 2))
        terminal_gradient = model.terminal_votes.grad.abs()

        self.assertEqual(int(lane_gradient.gt(0).sum()), model.lanes)
        self.assertEqual(int(node_gradient.gt(0).sum()), model.n_nodes)
        self.assertEqual(int(terminal_gradient.gt(0).sum()), model.lanes)


if __name__ == "__main__":
    unittest.main()
