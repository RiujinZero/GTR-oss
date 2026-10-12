from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from gator_path.frontier import FrontierState
from gator_path.prefix import (
    GatorPathPrefixConfig,
    GatorPathPrefixModel,
    fixed_address_code,
)
from gator_path_prefix_train import (
    _checkpoint_payload,
    _save_checkpoint,
    evaluate_path_prefix,
)
from parallel_prefix_benchmark import make_prefix_batch


class GatorPathPrefixTests(unittest.TestCase):
    def _model(
        self,
        *,
        cells: int = 1,
        relays: int = 4,
        top_k: int = 2,
        local_policy: bool = False,
        learned_ingress: bool = False,
        frontier_policy: bool = False,
        private_cell_io: bool = False,
        relation_aware: bool = False,
    ) -> GatorPathPrefixModel:
        torch.manual_seed(43)
        return GatorPathPrefixModel(
            GatorPathPrefixConfig(
                num_cells=cells,
                payload_width=8,
                address_width=4,
                relay_steps=relays,
                top_k=top_k,
                local_policy=local_policy,
                learned_ingress=learned_ingress,
                frontier_policy=frontier_policy,
                private_cell_io=private_cell_io,
                relation_aware=relation_aware,
            )
        )

    def test_fixed_addresses_are_continuous_and_parameter_free(self) -> None:
        address = fixed_address_code(
            15,
            4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(address.shape), (15, 4))
        self.assertFalse(address.requires_grad)
        self.assertTrue(torch.isfinite(address).all())
        self.assertGreater(
            float((address[1:] - address[:-1]).abs().sum()),
            0.0,
        )

    def test_initial_state_has_one_packet_per_input_and_output_slot(self) -> None:
        model = self._model()
        observed, _ = make_prefix_batch(3, 7, torch.device("cpu"))

        state, addresses = model.initial_state(
            observed,
            output_slots=7,
        )

        self.assertEqual(tuple(state.payload.shape), (3, 15, 8))
        self.assertEqual(tuple(addresses.shape), (15, 4))
        self.assertEqual(tuple(state.cell_id.shape), (3, 15))
        self.assertNotIn(model.config.num_cells, state.payload.shape)
        torch.testing.assert_close(
            state.payload[:, 8:],
            state.payload[:, 8:9].expand(-1, 7, -1),
        )
        self.assertIs(state.address, addresses)

    def test_learned_ingress_deterministically_selects_first_steward(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            top_k=10,
            local_policy=True,
            learned_ingress=True,
        )
        observed, _ = make_prefix_batch(3, 7, torch.device("cpu"))

        first, _ = model.initial_state(
            observed,
            output_slots=7,
            temperature=1.25,
        )
        second, _ = model.initial_state(
            observed,
            output_slots=7,
            temperature=1.25,
        )

        torch.testing.assert_close(first.cell_id, second.cell_id)
        torch.testing.assert_close(first.node_id, second.node_id)
        torch.testing.assert_close(first.payload, second.payload)
        self.assertGreaterEqual(int(first.cell_id.min()), 0)
        self.assertLess(int(first.cell_id.max()), 10)

    def test_controlled_writing_seeds_context_before_output_packets(self) -> None:
        model = self._model(relays=4)
        schedule = model.write_gate_schedule(
            batch_size=2,
            observed_length=8,
            output_slots=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        torch.testing.assert_close(schedule[:, :, :8], torch.ones(4, 2, 8))
        expected = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        torch.testing.assert_close(
            schedule[:, 0, 8:],
            expected[:, None].expand(-1, 7),
        )

    def test_forward_forms_all_outputs_and_returns_path_diagnostics(self) -> None:
        model = self._model(cells=3)
        observed, targets = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        logits, diagnostics = model(
            observed,
            output_slots=7,
            temperature=0.9,
            return_diagnostics=True,
        )

        self.assertEqual(tuple(logits.shape), (4, 7, 16))
        self.assertEqual(tuple(targets.shape), (4, 7))
        self.assertEqual(diagnostics.relay_steps, 4)
        self.assertEqual(
            tuple(diagnostics.state.payload.shape),
            (4, 15, 8),
        )
        for relay_result in diagnostics.relay_results:
            self.assertEqual(
                tuple(relay_result.route.destination_cell.shape),
                (4, 15, 2),
            )
            self.assertTrue(torch.isfinite(relay_result.state.payload).all())

    def test_one_fresh_batch_update_has_finite_gradients(self) -> None:
        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-3)
        observed, targets = make_prefix_batch(
            16,
            7,
            torch.device("cpu"),
        )

        logits = model(observed, output_slots=7, temperature=1.0)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient_norm))
        self.assertIsNotNone(model.relay.cell_router_weight.grad)
        self.assertIsNotNone(model.relay.node_router_weight.grad)
        self.assertIsNotNone(model.relay.mailbox_address_weight.grad)
        self.assertGreater(
            float(model.relay.cell_router_weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.relay.node_router_weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.relay.mailbox_address_weight.grad.abs().sum()),
            0.0,
        )

    def test_local_policy_carries_one_committed_state_without_route_beam(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            relays=4,
            top_k=10,
            local_policy=True,
            learned_ingress=True,
        )
        observed, targets = make_prefix_batch(
            8,
            7,
            torch.device("cpu"),
        )

        logits, diagnostics = model(
            observed,
            output_slots=7,
            temperature=1.25,
            return_diagnostics=True,
        )
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        loss.backward()

        self.assertEqual(diagnostics.relay_steps, 4)
        for result in diagnostics.relay_results:
            self.assertFalse(result.state.has_topk_support)
        self.assertFalse(diagnostics.state.has_topk_support)
        self.assertIsNotNone(model.ingress_router.weight.grad)
        self.assertGreater(
            float(model.ingress_router.weight.grad.abs().sum()),
            0.0,
        )
        cell_gradient = model.relay.path_transform.weight.grad.abs().sum(
            dim=(1, 2, 3)
        )
        self.assertGreater(
            int(cell_gradient.gt(0).sum()),
            1,
        )

    def test_frontier_policy_carries_only_current_soft_locations(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            relays=4,
            top_k=10,
            learned_ingress=True,
            frontier_policy=True,
        )
        observed, targets = make_prefix_batch(
            8,
            7,
            torch.device("cpu"),
        )

        initial = model.initial_frontier(
            observed,
            output_slots=7,
            temperature=1.25,
        )
        logits, diagnostics = model(
            observed,
            output_slots=7,
            temperature=1.25,
            return_diagnostics=True,
        )
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        loss.backward()

        self.assertEqual(tuple(initial.mass.shape), (8, 15, 10, 4))
        torch.testing.assert_close(
            initial.mass.sum(dim=(2, 3)),
            torch.ones(8, 15),
        )
        self.assertEqual(diagnostics.relay_steps, 4)
        self.assertFalse(hasattr(diagnostics.state, "route"))
        self.assertFalse(hasattr(diagnostics.state, "route_history"))
        torch.testing.assert_close(
            diagnostics.state.mass.sum(dim=(2, 3)),
            torch.ones(8, 15),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        self.assertGreater(
            float(model.ingress_router.weight.grad.abs().sum()),
            0.0,
        )
        transform_gradient = (
            model.frontier.transform_weight.grad.abs().sum(dim=(1, 2, 3))
        )
        self.assertEqual(int(transform_gradient.gt(0).sum()), 10)

    def test_frontier_hard_mode_has_deterministic_first_steward(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            learned_ingress=True,
            frontier_policy=True,
        )
        observed, _ = make_prefix_batch(4, 7, torch.device("cpu"))

        first = model.initial_frontier(
            observed,
            output_slots=7,
            temperature=0.35,
            hard_routes=True,
        )
        second = model.initial_frontier(
            observed,
            output_slots=7,
            temperature=0.35,
            hard_routes=True,
        )

        torch.testing.assert_close(first.mass, second.mass)
        self.assertTrue(
            (first.mass.detach().gt(0).sum(dim=(2, 3)) == 1).all()
        )

    def test_frontier_can_harden_ingress_and_handoffs_independently(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            learned_ingress=True,
            frontier_policy=True,
        )
        observed, _ = make_prefix_batch(4, 7, torch.device("cpu"))

        hard_ingress = model(
            observed,
            output_slots=7,
            temperature=0.35,
            hard_ingress=True,
            hard_handoffs=False,
        )
        hard_handoffs = model(
            observed,
            output_slots=7,
            temperature=0.35,
            hard_ingress=False,
            hard_handoffs=True,
        )

        self.assertEqual(tuple(hard_ingress.shape), (4, 7, 16))
        self.assertEqual(tuple(hard_handoffs.shape), (4, 7, 16))
        self.assertTrue(torch.isfinite(hard_ingress).all())
        self.assertTrue(torch.isfinite(hard_handoffs).all())
        self.assertFalse(torch.equal(hard_ingress, hard_handoffs))

    def test_arrow_capacity_frontier_has_private_tied_cell_io(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            learned_ingress=True,
            frontier_policy=True,
            private_cell_io=True,
        )
        observed, targets = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        logits = model(observed, output_slots=7, temperature=1.25)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        loss.backward()

        self.assertEqual(
            tuple(model.cell_token_embedding.shape),
            (10, 16, 8),
        )
        self.assertEqual(model.parameter_count, 9_748)
        self.assertIsNone(model.token_embedding)
        self.assertEqual(tuple(logits.shape), (4, 7, 16))
        self.assertGreater(
            int(
                model.cell_token_embedding.grad.abs()
                .sum(dim=(1, 2))
                .gt(0)
                .sum()
            ),
            1,
        )

    def test_relation_aware_frontier_reads_are_receiver_specific(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            learned_ingress=True,
            frontier_policy=True,
            relation_aware=True,
        )
        observed, targets = make_prefix_batch(
            4,
            7,
            torch.device("cpu"),
        )

        logits = model(observed, output_slots=7, temperature=1.25)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        loss.backward()

        self.assertEqual(
            tuple(model.frontier.relative_node_bias.shape),
            (10, 4, 4),
        )
        self.assertEqual(model.parameter_count, 8_756)
        self.assertGreater(
            float(model.frontier.relative_node_bias.grad.abs().sum()),
            0.0,
        )

    def test_relative_mailbox_can_select_different_local_writers(
        self,
    ) -> None:
        model = self._model(
            cells=1,
            relays=1,
            learned_ingress=True,
            frontier_policy=True,
            relation_aware=True,
        )
        frontier = model.frontier
        with torch.no_grad():
            frontier.relative_node_bias.zero_()
            # A strong negative absolute-distance bias makes each receiver
            # prefer the writer at its own sequence coordinate.
            frontier.relative_node_bias[0, 0, 1] = -16.0
        mass = torch.zeros(1, 2, 1, 4)
        mass[:, :, 0, 0] = 1.0
        payload = torch.zeros(1, 2, 1, 4, 8)
        payload[:, 1, 0, 0] = 1.0
        state = FrontierState(
            payload=payload,
            mass=mass,
            address=fixed_address_code(
                2,
                4,
                device=torch.device("cpu"),
                dtype=payload.dtype,
            ),
        )

        memory, _ = frontier._mailbox(
            state,
            torch.ones(1, 2),
        )

        self.assertLess(
            float(memory[0, 0, 0, 0].mean().detach()),
            1.0e-5,
        )
        self.assertGreater(
            float(memory[0, 1, 0, 0].mean().detach()),
            1.0 - 1.0e-5,
        )

    def test_restored_arrow_features_are_near_parameter_parity(
        self,
    ) -> None:
        model = self._model(
            cells=10,
            learned_ingress=True,
            frontier_policy=True,
            private_cell_io=True,
            relation_aware=True,
        )

        self.assertEqual(model.parameter_count, 9_908)

    def test_deterministic_fresh_evaluation_metrics_are_bounded(self) -> None:
        model = self._model()

        first = evaluate_path_prefix(
            model,
            length=7,
            device=torch.device("cpu"),
            batch_size=32,
            seed=123,
            temperature=1.0,
        )
        second = evaluate_path_prefix(
            model,
            length=7,
            device=torch.device("cpu"),
            batch_size=32,
            seed=123,
            temperature=1.0,
        )

        self.assertEqual(first, second)
        self.assertGreaterEqual(first["accuracy"], 0.0)
        self.assertLessEqual(first["accuracy"], 1.0)
        self.assertGreaterEqual(first["exact"], 0.0)
        self.assertLessEqual(first["exact"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(first["loss"])))

    def test_checkpoint_payload_round_trips_model_and_optimizer(self) -> None:
        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-3)
        arguments = argparse.Namespace(steps=10, checkpoint_dir=None)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=3,
            best_accuracy=0.25,
            architecture=model.config,
            arguments=arguments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            _save_checkpoint(payload, path)
            restored = torch.load(path, weights_only=False)

        self.assertEqual(
            restored["format"],
            "gator-path-prefix-local-frontier-v3",
        )
        self.assertEqual(restored["step"], 3)
        self.assertEqual(restored["architecture"]["num_cells"], 1)
        clone = self._model()
        clone.load_state_dict(restored["model"])
        for expected, actual in zip(
            model.parameters(),
            clone.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
