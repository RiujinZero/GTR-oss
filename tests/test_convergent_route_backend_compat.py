from __future__ import annotations

import unittest
from unittest import mock

import torch

from gator.models.packed_frontier import PackedNodeFrontier
from gator_convergent import (
    ConvergentGator,
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)
from language_probe import model_from_checkpoint


class ConvergentRouteBackendCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _packed_model() -> ConvergentGator:
        return ConvergentGator(
            n_nodes=8,
            d_model=4,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            route_backend="packed",
            active_node_top_k=3,
            relays=2,
            lanes=2,
            connection_layout="small_world",
            connection_seed=13,
            connection_rewire=0.5,
            terminal_vote_mode="slot",
        )

    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
        observed = torch.tensor(
            [[1, 7, 3, 9], [2, 5, 4, 0]],
            dtype=torch.long,
        )
        context_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]],
        )
        return observed, context_mask

    def test_legacy_checkpoint_without_route_backend_defaults_to_dense(
        self,
    ) -> None:
        config = ConvergentLanguageConfig(
            vocab_size=64,
            d_model=6,
            embedding_dim=8,
            n_nodes=12,
            degree=3,
            router_mode="shared_low_rank",
            router_rank=5,
            ingress_mode="learned",
            active_node_top_k=3,
            relays=2,
            lanes=3,
            blank_token_id=63,
        )
        reference = ConvergentLanguageModel(config)
        legacy_config = config.to_dict()
        legacy_config.pop("route_backend")

        restored = model_from_checkpoint(
            {
                "model_config": legacy_config,
                "model": reference.state_dict(),
            }
        )

        self.assertIsInstance(restored, ConvergentLanguageModel)
        self.assertEqual(restored.config.route_backend, "dense")
        self.assertEqual(restored.core.route_backend, "dense")
        context = torch.randint(2, 63, (2, 5))
        context_mask = torch.ones_like(context, dtype=torch.bool)
        torch.testing.assert_close(
            restored(context, context_mask, output_slots=4),
            reference(context, context_mask, output_slots=4),
        )

    def test_packed_architecture_config_round_trips_and_loads_strictly(
        self,
    ) -> None:
        torch.manual_seed(71)
        reference = self._packed_model()
        serialized_config = reference.architecture_config

        self.assertEqual(serialized_config["route_backend"], "packed")
        restored = ConvergentGator(**serialized_config)
        restored.load_state_dict(reference.state_dict(), strict=True)

        self.assertEqual(restored.route_backend, "packed")
        self.assertEqual(restored.architecture_config, serialized_config)
        observed, context_mask = self._inputs()
        torch.testing.assert_close(
            restored(observed, context_mask, output_slots=3),
            reference(observed, context_mask, output_slots=3),
        )

    def test_normal_packed_forward_never_densifies_the_frontier(self) -> None:
        torch.manual_seed(73)
        model = self._packed_model()
        observed, context_mask = self._inputs()

        with mock.patch.object(
            PackedNodeFrontier,
            "to_dense",
            side_effect=AssertionError(
                "normal packed execution must not materialize dense routes"
            ),
        ):
            logits = model(
                observed,
                context_mask,
                output_slots=3,
                return_trace=False,
            )

        self.assertEqual(tuple(logits.shape), (2, 3, 16))
        self.assertTrue(torch.isfinite(logits).all())

    def test_packed_trace_densifies_only_for_trace_materialization(self) -> None:
        torch.manual_seed(79)
        model = self._packed_model()
        observed, context_mask = self._inputs()
        original_to_dense = PackedNodeFrontier.to_dense
        calls: list[tuple[int, ...]] = []

        def tracked_to_dense(frontier: PackedNodeFrontier) -> torch.Tensor:
            calls.append(tuple(frontier.nodes.shape))
            return original_to_dense(frontier)

        with mock.patch.object(
            PackedNodeFrontier,
            "to_dense",
            new=tracked_to_dense,
        ):
            logits, trace = model(
                observed,
                context_mask,
                output_slots=3,
                return_trace=True,
            )

        self.assertEqual(tuple(logits.shape), (2, 3, 16))
        self.assertEqual(len(calls), model.relays + 1)
        self.assertEqual(
            tuple(trace["locations"].shape),
            (2, model.relays + 1, 7, model.lanes, model.n_nodes),
        )


if __name__ == "__main__":
    unittest.main()
