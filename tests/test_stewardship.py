from __future__ import annotations

import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from gator.models.language import GatorLanguageConfig, GatorLanguageModel
from gator.models.stewardship import (
    adaptive_nucleus_stewardship,
    elastic_handoff_beam,
    normalise_stewardship_messages,
    pack_stewardship,
    ring_handoff_destinations,
    transport_packed_stewardship,
    transport_stewardship,
    unpack_stewardship,
)
from gator.tasks import (
    stewardship_route_losses,
    stewardship_route_losses_from_statistics,
)


class StewardshipPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_ring_destinations_include_self_and_balanced_neighbors(self) -> None:
        destinations = ring_handoff_destinations(6, 4)
        self.assertEqual(destinations[0].tolist(), [0, 5, 1, 4, 2])
        self.assertEqual(destinations[5].tolist(), [5, 4, 0, 3, 1])

    def test_forced_handoff_moves_mass_state_and_passport_together(self) -> None:
        ownership = torch.zeros(1, 4, 1)
        ownership[0, 0, 0] = 1.0
        states = torch.zeros(1, 4, 1, 2)
        states[0, 0, 0] = torch.tensor([3.0, 7.0])
        nodes = torch.zeros(1, 4, 1, 2)
        nodes[0, 0, 0, 1] = 1.0
        destinations = ring_handoff_destinations(4, 2)
        handoffs = torch.zeros(1, 4, 1, 3)
        handoffs[..., 2] = 1.0  # +1 action

        next_ownership, next_states, next_nodes = transport_stewardship(
            ownership, states, nodes, handoffs, destinations
        )
        self.assertTrue(torch.equal(next_ownership[0, :, 0], torch.tensor([0.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(next_states[0, 1, 0], torch.tensor([3.0, 7.0])))
        self.assertTrue(torch.equal(next_nodes[0, 1, 0], torch.tensor([0.0, 1.0])))

    def test_colliding_paths_use_mass_weighted_branch_state(self) -> None:
        ownership = torch.tensor([[[0.25], [0.0], [0.75]]])
        states = torch.tensor([[[[2.0]], [[100.0]], [[6.0]]]])
        nodes = torch.tensor([[[[1.0, 0.0]], [[0.5, 0.5]], [[0.0, 1.0]]]])
        handoffs = torch.ones(1, 3, 1, 1)
        destinations = torch.ones(3, 1, dtype=torch.long)

        next_ownership, next_states, next_nodes = transport_stewardship(
            ownership, states, nodes, handoffs, destinations
        )
        self.assertAlmostEqual(float(next_ownership[0, 1, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(next_states[0, 1, 0, 0]), 5.0, places=6)
        self.assertTrue(
            torch.allclose(next_nodes[0, 1, 0], torch.tensor([0.25, 0.75]))
        )

    def test_adaptive_nucleus_narrows_confident_and_widens_uncertain_routes(self) -> None:
        confident = torch.tensor([[[0.96], [0.01], [0.01], [0.01], [0.01]]])
        _, confident_mask = adaptive_nucleus_stewardship(
            confident, keep_mass=0.9, straight_through=False
        )
        self.assertEqual(int(confident_mask.sum()), 1)

        uncertain = torch.full((1, 5, 1), 0.2)
        widened, uncertain_mask = adaptive_nucleus_stewardship(
            uncertain, keep_mass=0.9, straight_through=False
        )
        self.assertEqual(int(uncertain_mask.sum()), 5)
        self.assertAlmostEqual(float(widened.sum()), 1.0, places=6)

    def test_narrow_route_can_reopen_through_uncertain_neighbor_handoffs(self) -> None:
        ownership = torch.zeros(1, 6, 1)
        ownership[0, 0, 0] = 1.0
        states = torch.arange(6.0).view(1, 6, 1, 1)
        nodes = torch.zeros(1, 6, 1, 2)
        nodes[..., 0] = 1.0
        handoffs = torch.full((1, 6, 1, 3), 1.0 / 3.0)
        destinations = ring_handoff_destinations(6, 2)
        transported, _, _ = transport_stewardship(
            ownership, states, nodes, handoffs, destinations
        )
        widened, mask = adaptive_nucleus_stewardship(
            transported, keep_mass=0.99, straight_through=False
        )
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual(
            widened[0, :, 0].gt(0).nonzero().flatten().tolist(), [0, 1, 5]
        )

    def test_confident_route_retains_a_low_mass_scout(self) -> None:
        ownership = torch.tensor([[[0.97], [0.01], [0.01], [0.01]]])
        routed, mask = adaptive_nucleus_stewardship(
            ownership,
            keep_mass=0.9,
            scouts=1,
            exploration_mass=0.02,
            straight_through=False,
        )
        self.assertEqual(int(mask.sum()), 2)
        masses = routed[0, :, 0][mask[0, :, 0]].sort(descending=True).values
        self.assertTrue(torch.allclose(masses, torch.tensor([0.98, 0.02])))

    def test_each_uncertain_steward_can_fan_out_its_own_branch(self) -> None:
        probabilities = torch.tensor(
            [[0.99, 0.0025, 0.0025, 0.0025, 0.0025], [0.2] * 5]
        )
        entropy = torch.tensor([0.0, 1.0])
        routed, mask = elastic_handoff_beam(
            probabilities,
            entropy,
            base_keep_mass=0.9,
            scouts=1,
            exploration_mass=0.01,
            straight_through=False,
        )
        self.assertEqual(mask[0].sum(), 2)
        self.assertEqual(mask[1].sum(), 5)
        self.assertAlmostEqual(float(routed[0].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(routed[1].sum()), 1.0, places=6)
        self.assertTrue(torch.allclose(routed[1], torch.full((5,), 0.2)))

    def test_group_contract_is_counted_inside_max_active(self) -> None:
        ownership = torch.full((1, 4, 1), 0.25)
        with self.assertRaisesRegex(ValueError, "every required cell group"):
            adaptive_nucleus_stewardship(
                ownership,
                keep_mass=0.9,
                max_active=1,
                cell_groups=torch.tensor([0, 1, 0, 1]),
                straight_through=False,
            )

    def test_packed_transport_matches_dense_transport(self) -> None:
        ownership = torch.tensor(
            [[[0.7, 0.0], [0.0, 1.0], [0.3, 0.0], [0.0, 0.0]]]
        )
        states = torch.randn(1, 4, 2, 3)
        nodes = torch.softmax(torch.randn(1, 4, 2, 2), -1)
        destinations = ring_handoff_destinations(4, 2)
        handoffs = torch.softmax(torch.randn(1, 4, 2, 3), -1)
        dense = transport_stewardship(
            ownership, states, nodes, handoffs, destinations
        )
        packed = pack_stewardship(ownership, states, nodes)
        rows = handoffs[packed.batch, packed.cell, packed.token]
        transported = transport_packed_stewardship(
            packed,
            rows,
            destinations,
            batch_size=1,
            cells=4,
            tokens=2,
        )
        expanded = unpack_stewardship(
            transported, batch_size=1, cells=4, tokens=2
        )
        for dense_value, packed_value in zip(dense, expanded):
            self.assertTrue(torch.allclose(dense_value, packed_value, atol=1e-6))

        support = rows.gt(0)
        supported = transport_packed_stewardship(
            packed,
            rows,
            destinations,
            batch_size=1,
            cells=4,
            tokens=2,
            support=support,
        )
        supported_expanded = unpack_stewardship(
            supported, batch_size=1, cells=4, tokens=2
        )
        for expected, actual in zip(expanded, supported_expanded):
            torch.testing.assert_close(actual, expected)

    def test_compiled_bf16_empty_output_group_has_bounded_backward(self) -> None:
        cells = 8
        groups = torch.arange(cells).remainder(2)
        hard_ownership = torch.zeros(1, cells, 1, dtype=torch.bfloat16)
        hard_ownership[:, 0] = 1.0
        hidden = torch.arange(
            1, 1 + cells * 4, dtype=torch.bfloat16
        ).view(1, cells, 1, 4)

        def objective(route_logits: torch.Tensor) -> torch.Tensor:
            probabilities = route_logits.softmax(1)
            ownership = probabilities + (
                hard_ownership - probabilities
            ).detach()
            grouped = []
            for group in range(2):
                selected = groups.eq(group)
                group_ownership = ownership[:, selected]
                group_weights = normalise_stewardship_messages(
                    group_ownership.permute(0, 2, 1),
                    group_ownership.sum(1),
                ).permute(0, 2, 1)
                grouped.append(
                    torch.einsum(
                        "bct,bctd->btd",
                        group_weights,
                        hidden[:, selected],
                    )
                )
            return torch.cat(grouped, -1).float().sum()

        compiled = torch.compile(objective, backend="eager", fullgraph=True)
        route_logits = torch.zeros(
            1, cells, 1, dtype=torch.bfloat16, requires_grad=True
        )
        compiled(route_logits).backward()
        self.assertTrue(torch.isfinite(route_logits.grad).all())
        self.assertLess(float(route_logits.grad.abs().max()), 100.0)

    def test_compiled_bf16_near_empty_output_group_has_bounded_backward(
        self,
    ) -> None:
        cells = 8
        groups = torch.arange(cells).remainder(2)
        hidden = torch.arange(
            1, 1 + cells * 4, dtype=torch.bfloat16
        ).view(1, cells, 1, 4)

        def objective(route_logits: torch.Tensor) -> torch.Tensor:
            probabilities = route_logits.softmax(1)
            hard = torch.zeros_like(probabilities)
            hard[:, 0::2] = probabilities[:, 0::2]
            # Group one is structurally protected by one forced cell even
            # after its probability reaches the BF16 underflow boundary.
            hard[:, 1] = probabilities[:, 1]
            hard = hard / hard.sum(1, keepdim=True)
            ownership = probabilities + (hard - probabilities).detach()
            grouped = []
            for group in range(2):
                selected = groups.eq(group)
                group_ownership = ownership[:, selected]
                group_weights = normalise_stewardship_messages(
                    group_ownership.permute(0, 2, 1),
                    group_ownership.sum(1),
                ).permute(0, 2, 1)
                grouped.append(
                    torch.einsum(
                        "bct,bctd->btd",
                        group_weights,
                        hidden[:, selected],
                    )
                )
            return torch.cat(grouped, -1).float().sum()

        compiled = torch.compile(objective, backend="eager", fullgraph=True)
        route_logits = torch.tensor(
            [10.0, -80.0, 9.0, -82.0, 8.0, -84.0, 7.0, -86.0],
            dtype=torch.bfloat16,
        ).view(1, cells, 1).requires_grad_(True)
        compiled(route_logits).backward()
        self.assertTrue(torch.isfinite(route_logits.grad).all())
        self.assertLess(float(route_logits.grad.abs().max()), 100.0)

    def test_compiled_bf16_empty_routes_have_bounded_backward(self) -> None:
        cells = 8
        hard_ownership = torch.zeros(1, cells, 1, dtype=torch.bfloat16)
        hard_ownership[:, :3] = torch.tensor(
            [0.5, 0.3, 0.2], dtype=torch.bfloat16
        ).view(1, 3, 1)
        states = torch.arange(
            1, cells + 1, dtype=torch.bfloat16
        ).view(1, cells, 1, 1)
        nodes = torch.ones(1, cells, 1, 1, dtype=torch.bfloat16)
        handoffs = torch.ones(1, cells, 1, 1, dtype=torch.bfloat16)
        destinations = torch.arange(cells).view(cells, 1)

        def objective(route_logits: torch.Tensor) -> torch.Tensor:
            probabilities = route_logits.softmax(1)
            ownership = probabilities + (
                hard_ownership - probabilities
            ).detach()
            mass, next_states, next_nodes = transport_stewardship(
                ownership,
                states,
                nodes,
                handoffs,
                destinations,
            )
            return (
                mass.float().sum()
                + next_states.float().sum()
                + next_nodes.float().sum()
            )

        compiled = torch.compile(objective, backend="eager", fullgraph=True)
        route_logits = torch.zeros(
            1, cells, 1, dtype=torch.bfloat16, requires_grad=True
        )
        compiled(route_logits).backward()
        self.assertTrue(torch.isfinite(route_logits.grad).all())
        self.assertLess(float(route_logits.grad.abs().max()), 100.0)

    def test_task_gradient_rewards_the_successful_handoff(self) -> None:
        ownership = torch.zeros(1, 4, 1)
        ownership[0, 0, 0] = 1.0
        states = torch.ones(1, 4, 1, 1)
        nodes = torch.ones(1, 4, 1, 1)
        destinations = ring_handoff_destinations(4, 2)
        route_logits = torch.zeros(1, 4, 1, 3, requires_grad=True)
        handoffs = route_logits.softmax(-1)
        next_ownership, _, _ = transport_stewardship(
            ownership, states, nodes, handoffs, destinations
        )
        # Action two from source zero reaches cell one. Gradient descent on
        # this objective must increase that action relative to staying put.
        loss = -next_ownership[0, 1, 0]
        loss.backward()
        self.assertLess(
            float(route_logits.grad[0, 0, 0, 2]),
            float(route_logits.grad[0, 0, 0, 0]),
        )

    def test_familiar_entropy_pressure_turns_off_for_novel_routes(self) -> None:
        ownership = torch.full((1, 2, 4, 1), 0.25)
        familiar_loss, _ = stewardship_route_losses(
            ownership, torch.zeros(1, 1, 1)
        )
        novel_loss, _ = stewardship_route_losses(
            ownership, torch.ones(1, 1, 1)
        )
        self.assertGreater(float(familiar_loss), 0.99)
        self.assertEqual(float(novel_loss), 0.0)

    def test_route_losses_ignore_invalid_padding(self) -> None:
        ownership = torch.zeros(1, 2, 4, 2)
        ownership[:, :, 0] = 1.0
        # The second token deliberately carries a nonsensical zero-mass route.
        uncertainty = torch.zeros(1, 1, 2)
        valid = torch.tensor([[True, False]])
        entropy, balance = stewardship_route_losses(
            ownership, uncertainty, valid
        )
        self.assertLess(float(entropy), 1e-6)
        self.assertGreater(float(balance), 0.0)

    def test_route_uncertainty_has_finite_empty_route_backward(self) -> None:
        ownership = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]]],
            requires_grad=True,
        )
        handoff_entropy = torch.tensor(
            [[[0.25, 0.5], [0.75, 0.5]]],
            requires_grad=True,
        )
        valid = torch.tensor([[True, False]])

        maximum, mean = GatorLanguageModel._route_uncertainty(
            ownership,
            handoff_entropy,
            valid,
        )
        self.assertEqual(maximum.tolist(), [[0.25, 0.0]])
        gradients = torch.autograd.grad(
            maximum.sum() + mean.sum(),
            (ownership, handoff_entropy),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_packed_route_statistics_match_dense_loss_and_gradients(self) -> None:
        torch.manual_seed(16)
        ownership = torch.rand(
            2, 4, 6, 3, dtype=torch.float64
        )
        ownership = (
            ownership
            / ownership.sum(2, keepdim=True)
        ).requires_grad_(True)
        uncertainty = torch.rand(2, 3, 3, dtype=torch.float64)
        valid = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        expected = stewardship_route_losses(
            ownership,
            uncertainty,
            valid,
        )
        routes = ownership[:, 1:]
        route_entropy = -(
            routes * routes.clamp_min(1e-9).log()
        ).sum(2) / math.log(routes.shape[2])
        valid_mass = valid[:, None, None].to(routes.dtype)
        route_cell_load = (
            (routes * valid_mass).sum(dim=(0, 3))
            / valid.sum().to(routes.dtype)
        )
        actual = stewardship_route_losses_from_statistics(
            route_entropy,
            route_cell_load,
            uncertainty,
            valid,
        )
        torch.testing.assert_close(actual, expected)
        probe = torch.tensor([0.7, -0.2], dtype=torch.float64)
        expected_gradient = torch.autograd.grad(
            expected[0] * probe[0] + expected[1] * probe[1],
            ownership,
            retain_graph=True,
        )[0]
        actual_gradient = torch.autograd.grad(
            actual[0] * probe[0] + actual[1] * probe[1],
            ownership,
        )[0]
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
        )


class StewardshipModelTests(unittest.TestCase):
    @staticmethod
    def _config(mode: str) -> GatorLanguageConfig:
        return GatorLanguageConfig(
            vocab_size=512,
            d_model=8,
            embedding_dim=8,
            n_cells=6,
            n_nodes=4,
            relays=2,
            neighbors=2,
            lexical_mode="communal",
            cell_input_mode="feature_gated",
            cell_output_mode="shared_pool",
            node_topology="symmetric",
            cell_transport_mode=mode,
            stewardship_exploration=0.01,
        )

    def test_default_legacy_state_dict_has_no_stewardship_shell(self) -> None:
        model = GatorLanguageModel(self._config("broadcast_mix"))
        names = set(model.state_dict())
        self.assertNotIn("handoff_weight", names)
        self.assertIn("left_projection.weight", names)
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        logits = model(context, mask, output_slots=3)
        self.assertEqual(tuple(logits.shape), (2, 3, 512))

    def test_model_conserves_routes_and_backpropagates_through_handoffs(self) -> None:
        model = GatorLanguageModel(self._config("stewardship"))
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        targets = torch.randint(0, 512, (2, 3))
        logits, diagnostics = model(
            context,
            mask,
            output_slots=3,
            temperature=1.0,
            stewardship_temperature=1.2,
            return_diagnostics=True,
        )
        ownership = diagnostics["ownership"]
        locations = diagnostics["locations"]
        self.assertLess(float((ownership.sum(2) - 1.0).abs().max().detach()), 2e-6)
        active_node_error = ((locations.sum(-1) - 1.0) * ownership).abs().max()
        self.assertLess(float(active_node_error.detach()), 2e-6)
        loss = F.cross_entropy(logits.reshape(-1, 512), targets.reshape(-1))
        loss.backward()
        self.assertIsNotNone(model.handoff_weight.grad)
        self.assertGreater(float(model.handoff_weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(model.handoff_weight.grad).all())

    def test_reference_stewardship_shell_is_parameter_matched(self) -> None:
        common = dict(
            vocab_size=2048,
            d_model=8,
            embedding_dim=32,
            n_cells=62,
            n_nodes=4,
            relays=4,
            neighbors=4,
            lexical_mode="communal",
            cell_input_mode="feature_gated",
            cell_output_mode="grouped_concat",
            node_topology="hybrid_hemispheric",
        )
        legacy = GatorLanguageModel(
            GatorLanguageConfig(**common, cell_transport_mode="broadcast_mix")
        )
        stewardship = GatorLanguageModel(
            GatorLanguageConfig(**common, cell_transport_mode="stewardship")
        )
        self.assertEqual(stewardship.parameter_count - legacy.parameter_count, 2)

    def test_dense_and_packed_relays_are_numerically_equivalent(self) -> None:
        dense_config = self._config("stewardship")
        packed_config = GatorLanguageConfig(
            **{
                **dense_config.to_dict(),
                "stewardship_execution": "packed_auto",
                "stewardship_packed_threshold": 1.0,
            }
        )
        dense = GatorLanguageModel(dense_config).eval()
        packed = GatorLanguageModel(packed_config).eval()
        packed.load_state_dict(dense.state_dict())
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        with torch.no_grad():
            dense_logits, dense_trace = dense(
                context,
                mask,
                output_slots=3,
                temperature=1.0,
                stewardship_temperature=1.2,
                stewardship_prune=False,
                return_diagnostics=True,
            )
            for packed_from_relay in range(packed.config.relays):
                packed_logits, packed_trace = packed(
                    context,
                    mask,
                    output_slots=3,
                    temperature=1.0,
                    stewardship_temperature=1.2,
                    stewardship_prune=False,
                    stewardship_packed_from_relay=packed_from_relay,
                    return_diagnostics=True,
                )
                self.assertTrue(
                    torch.allclose(dense_logits, packed_logits, atol=2e-6)
                )
                for key in (
                    "ownership",
                    "locations",
                    "handoffs",
                    "route_uncertainty",
                ):
                    self.assertTrue(
                        torch.allclose(
                            dense_trace[key], packed_trace[key], atol=2e-6
                        ),
                        key,
                    )
                self.assertEqual(
                    packed_trace["packed_execution"].tolist(),
                    [
                        relay >= packed_from_relay
                        for relay in range(packed.config.relays)
                    ],
                )

    def test_repack_and_persistent_bucketed_match_on_sparse_pruned_suffix(self) -> None:
        common = {
            **self._config("stewardship").to_dict(),
            "relays": 3,
            "stewardship_execution": "packed_auto",
            "stewardship_packed_threshold": 1.0,
            "stewardship_keep_mass": 0.5,
            "stewardship_handoff_keep_mass": 0.5,
            "stewardship_min_active": 1,
            "stewardship_max_active": 2,
            "stewardship_scouts": 0,
            "stewardship_exploration": 0.0,
        }
        repack = GatorLanguageModel(
            GatorLanguageConfig(
                **{
                    **common,
                    "packed_stewardship_backend": "repack",
                }
            )
        ).eval()
        persistent = GatorLanguageModel(
            GatorLanguageConfig(
                **{
                    **common,
                    "packed_stewardship_backend": "persistent_bucketed",
                }
            )
        ).eval()
        persistent.load_state_dict(repack.state_dict())
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        with torch.no_grad():
            expected_logits, expected = repack(
                context,
                mask,
                output_slots=3,
                stewardship_temperature=0.7,
                stewardship_prune=True,
                stewardship_packed_from_relay=0,
                diagnostics="route_metrics",
            )
            actual_logits, actual = persistent(
                context,
                mask,
                output_slots=3,
                stewardship_temperature=0.7,
                stewardship_prune=True,
                stewardship_packed_from_relay=0,
                diagnostics="route_metrics",
            )
        torch.testing.assert_close(
            actual_logits, expected_logits, rtol=2e-5, atol=2e-6
        )
        for key in (
            "ownership",
            "route_entropy",
            "route_cell_load",
            "route_active_fraction",
            "route_mean_uncertainty",
            "handoffs",
            "route_uncertainty",
            "handoff_support",
        ):
            torch.testing.assert_close(
                actual[key], expected[key], rtol=2e-5, atol=2e-6
            )
        self.assertLessEqual(
            int(actual["ownership"][:, -1].gt(0).sum(1).max()),
            2,
        )

    def test_persistent_suffix_packs_once_and_skips_dense_state_unpack(self) -> None:
        config = GatorLanguageConfig(
            **{
                **self._config("stewardship").to_dict(),
                "relays": 4,
                "packed_stewardship_backend": "persistent_bucketed",
            }
        )
        model = GatorLanguageModel(config).eval()
        context = torch.randint(0, 512, (1, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        with (
            mock.patch(
                "gator.models.language.pack_stewardship",
                wraps=pack_stewardship,
            ) as pack_call,
            mock.patch(
                "gator.models.language.unpack_stewardship",
                wraps=unpack_stewardship,
            ) as unpack_call,
            mock.patch(
                "gator.models.language.packed_ownership",
            ) as ownership_call,
            torch.no_grad(),
        ):
            model(
                context,
                mask,
                output_slots=3,
                stewardship_prune=True,
                stewardship_packed_from_relay=1,
            )
        self.assertEqual(pack_call.call_count, 1)
        self.assertEqual(unpack_call.call_count, 0)
        self.assertEqual(ownership_call.call_count, 0)

    def test_repack_and_persistent_gradients_match_without_pruning(self) -> None:
        common = {
            **self._config("stewardship").to_dict(),
            "stewardship_execution": "packed_auto",
        }
        repack = GatorLanguageModel(
            GatorLanguageConfig(
                **{
                    **common,
                    "packed_stewardship_backend": "repack",
                }
            )
        )
        persistent = GatorLanguageModel(
            GatorLanguageConfig(
                **{
                    **common,
                    "packed_stewardship_backend": "persistent_bucketed",
                }
            )
        )
        persistent.load_state_dict(repack.state_dict())
        context = torch.randint(0, 512, (2, 4))
        mask = torch.ones_like(context, dtype=torch.bool)
        probe = torch.randn(2, 3, 512)
        expected = repack(
            context,
            mask,
            output_slots=3,
            stewardship_prune=False,
            stewardship_packed_from_relay=0,
        )
        actual = persistent(
            context,
            mask,
            output_slots=3,
            stewardship_prune=False,
            stewardship_packed_from_relay=0,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (expected * probe).sum().backward()
        (actual * probe).sum().backward()
        for (expected_name, expected_parameter), (
            actual_name,
            actual_parameter,
        ) in zip(
            repack.named_parameters(),
            persistent.named_parameters(),
            strict=True,
        ):
            self.assertEqual(actual_name, expected_name)
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=1e-4,
                atol=1e-5,
                msg=lambda message, parameter=actual_name: (
                    f"{parameter} gradient mismatch: {message}"
                ),
            )

    def test_stewardship_traffic_preserves_crowding(self) -> None:
        model = GatorLanguageModel(self._config("stewardship"))
        states = torch.randn(1, 6, 2, 8)
        nodes = torch.zeros(1, 6, 2, 4)
        nodes[..., 0] = 1.0
        ownership = torch.zeros(1, 6, 2)
        ownership[:, 0] = 1.0
        write = torch.ones(1, 2)
        relative = torch.zeros(2, 2, 4)
        _, traffic = model._dense_stewardship_context(
            states, nodes, ownership, write, relative
        )
        self.assertAlmostEqual(float(traffic[0, 0, 0].sum()), 6.0, places=5)
        self.assertEqual(float(traffic[0, 1, 0].sum()), 0.0)

    def test_legacy_checkpoint_round_trip_is_exact(self) -> None:
        model = GatorLanguageModel(self._config("broadcast_mix")).eval()
        restored = GatorLanguageModel(self._config("broadcast_mix")).eval()
        restored.load_state_dict(model.state_dict())
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        with torch.no_grad():
            expected = model(context, mask, output_slots=3)
            actual = restored(context, mask, output_slots=3)
        self.assertTrue(torch.equal(expected, actual))

    def test_packed_model_backpropagates_into_handoff_policy(self) -> None:
        config = GatorLanguageConfig(
            **{
                **self._config("stewardship").to_dict(),
                "stewardship_execution": "packed_auto",
                "stewardship_packed_threshold": 1.0,
            }
        )
        model = GatorLanguageModel(config)
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        targets = torch.randint(0, 512, (2, 3))
        logits = model(
            context,
            mask,
            output_slots=3,
            temperature=1.0,
            stewardship_temperature=1.2,
            stewardship_packed_from_relay=0,
        )
        F.cross_entropy(logits.reshape(-1, 512), targets.reshape(-1)).backward()
        self.assertIsNotNone(model.handoff_weight.grad)
        self.assertTrue(torch.isfinite(model.handoff_weight.grad).all())
        self.assertGreater(float(model.handoff_weight.grad.abs().sum()), 0.0)

    def test_route_loss_diagnostics_match_full_without_full_traces(self) -> None:
        model = GatorLanguageModel(self._config("stewardship")).eval()
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        with torch.no_grad():
            full_logits, full = model(
                context,
                mask,
                output_slots=3,
                diagnostics="full",
            )
            route_logits, route = model(
                context,
                mask,
                output_slots=3,
                diagnostics="route_loss",
            )
        self.assertTrue(torch.equal(route_logits, full_logits))
        self.assertEqual(
            set(route),
            {
                "route_entropy",
                "route_cell_load",
                "route_active_fraction",
                "route_mean_uncertainty",
                "route_valid",
            },
        )
        for key in route:
            self.assertTrue(torch.equal(route[key], full[key]), key)
        self.assertIn("locations", full)
        self.assertIn("hidden", full)
        self.assertIn("handoffs", full)

    def test_route_metrics_omit_hidden_and_location_history(self) -> None:
        model = GatorLanguageModel(self._config("stewardship")).eval()
        context = torch.randint(0, 512, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        with torch.no_grad():
            _, diagnostics = model(
                context,
                mask,
                output_slots=3,
                diagnostics="route_metrics",
            )
        self.assertNotIn("locations", diagnostics)
        self.assertNotIn("hidden", diagnostics)
        self.assertIn("ownership", diagnostics)
        self.assertIn("handoffs", diagnostics)
        self.assertIn("packed_execution", diagnostics)

    def test_padded_context_slots_have_no_stewardship_mass(self) -> None:
        model = GatorLanguageModel(self._config("stewardship")).eval()
        context = torch.randint(0, 512, (2, 5))
        mask = torch.tensor(
            [[False, False, True, True, True], [False, True, True, True, True]]
        )
        with torch.no_grad():
            _, diagnostics = model(
                context,
                mask,
                output_slots=3,
                return_diagnostics=True,
            )
        ownership = diagnostics["ownership"]
        invalid = ~diagnostics["route_valid"]
        invalid_mass = ownership.permute(0, 1, 3, 2)[
            invalid[:, None].expand(-1, ownership.shape[1], -1)
        ]
        self.assertEqual(float(invalid_mass.abs().max()), 0.0)

    def test_padding_token_values_cannot_change_logits_or_route_gradients(self) -> None:
        first = GatorLanguageModel(self._config("stewardship"))
        second = GatorLanguageModel(self._config("stewardship"))
        second.load_state_dict(first.state_dict())
        context_a = torch.tensor([[7, 9, 11, 13, 15]])
        context_b = torch.tensor([[101, 203, 11, 13, 15]])
        mask = torch.tensor([[False, False, True, True, True]])
        targets = torch.tensor([[17, 19, 23]])
        logits_a = first(context_a, mask, output_slots=3)
        logits_b = second(context_b, mask, output_slots=3)
        self.assertTrue(torch.equal(logits_a, logits_b))
        F.cross_entropy(logits_a.reshape(-1, 512), targets.reshape(-1)).backward()
        F.cross_entropy(logits_b.reshape(-1, 512), targets.reshape(-1)).backward()
        self.assertTrue(
            torch.equal(first.handoff_weight.grad, second.handoff_weight.grad)
        )


if __name__ == "__main__":
    unittest.main()
