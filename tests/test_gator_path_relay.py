from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from gator_path.packet_state import PacketState
from gator_path.relay import GatorPathRelay
from gator_path.topology import fixed_ring_topology


class GatorPathRelayTests(unittest.TestCase):
    def _state(
        self,
        *,
        payload: torch.Tensor,
        cell_id: torch.Tensor,
        node_id: torch.Tensor,
        active: torch.Tensor | None = None,
        write_gate: torch.Tensor | None = None,
        address: torch.Tensor | None = None,
    ) -> PacketState:
        batch, slots, _ = payload.shape
        if active is None:
            active = torch.ones(batch, slots, dtype=torch.bool)
        if write_gate is None:
            write_gate = torch.ones(batch, slots)
        if address is None:
            address = torch.stack(
                (
                    torch.linspace(-1.0, 1.0, slots),
                    torch.linspace(1.0, -1.0, slots),
                ),
                dim=-1,
            )
        return PacketState(
            payload=payload,
            address=address,
            cell_id=cell_id,
            node_id=node_id,
            active=active,
            write_gate=write_gate,
        )

    def _relay(
        self,
        *,
        cells: int = 7,
        width: int = 3,
        top_k: int = 1,
        hard: bool = True,
        local_policy: bool = False,
    ) -> GatorPathRelay:
        torch.manual_seed(17)
        return GatorPathRelay(
            topology=fixed_ring_topology(cells),
            payload_width=width,
            address_width=2,
            top_k=top_k,
            straight_through_hard=hard,
            local_policy=local_policy,
        )

    @staticmethod
    def _choose_action(
        relay: GatorPathRelay,
        *,
        cell_action: int,
        node_action: int,
    ) -> None:
        with torch.no_grad():
            relay.cell_router_weight.zero_()
            relay.cell_router_bias.fill_(-20.0)
            relay.cell_router_bias[:, cell_action] = 20.0
            relay.node_router_weight.zero_()
            relay.node_router_bias.fill_(-20.0)
            relay.node_router_bias[:, :, node_action] = 20.0

    def test_old_snapshot_is_immutable_and_commit_is_functional(self):
        relay = self._relay()
        state = self._state(
            payload=torch.randn(2, 4, 3),
            cell_id=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
            node_id=torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
        )
        before = {
            name: getattr(state, name).clone()
            for name in (
                "payload",
                "cell_id",
                "node_id",
                "active",
                "write_gate",
            )
        }

        result = relay(state)

        self.assertIsNot(result.state, state)
        self.assertIs(result.state.address, state.address)
        for name, expected in before.items():
            torch.testing.assert_close(getattr(state, name), expected)
        self.assertEqual(tuple(result.state.payload.shape), (2, 4, 3))
        self.assertNotIn(relay.topology.num_cells, state.payload.shape)

    def test_moves_are_legal_and_committed_simultaneously(self):
        relay = self._relay(cells=7)
        self._choose_action(relay, cell_action=2, node_action=0)
        source_cell = torch.tensor([[0, 6, 3, 1]])
        source_node = torch.tensor([[0, 1, 2, 3]])
        state = self._state(
            payload=torch.randn(1, 4, 3),
            cell_id=source_cell,
            node_id=source_node,
        )

        result = relay(state)

        torch.testing.assert_close(
            result.state.cell_id,
            (source_cell + 1).remainder(7),
        )
        expected_nodes = torch.tensor(
            [[
                relay.topology.legal_node_destinations[int(node)][0]
                for node in source_node[0]
            ]]
        )
        torch.testing.assert_close(result.state.node_id, expected_nodes)
        for slot in range(state.slot_count):
            source = int(source_node[0, slot])
            destination = int(result.state.node_id[0, slot])
            self.assertIn(
                destination,
                relay.topology.legal_node_destinations[source],
            )

        # Mailbox reads describe old locations, even though all movements
        # have already been committed in result.state.
        torch.testing.assert_close(
            result.mailbox.cell_id, source_cell.flatten().sort().values
        )

    def test_same_location_perturbation_affects_peer_but_remote_one_does_not(
        self,
    ):
        relay = self._relay(cells=5)
        self._choose_action(relay, cell_action=0, node_action=0)
        with torch.no_grad():
            transform = relay.path_transform
            transform.mailbox_weight.copy_(torch.eye(3))
            transform.address_weight.zero_()
            transform.cell_action_embedding.zero_()
            transform.node_edge_embedding.zero_()
            transform.bias.zero_()
            transform.weight.zero_()
            for cell in range(5):
                for node in range(4):
                    transform.weight[cell, node].copy_(torch.eye(3))
            relay.destination_norm_weight.fill_(1.0)
            relay.destination_norm_bias.zero_()

        base_payload = torch.tensor(
            [[[0.7, -0.3, 0.2], [0.1, 0.4, -0.2]]]
        )
        changed_payload = base_payload.clone()
        changed_payload[0, 1] = torch.tensor([3.0, -2.0, 1.0])

        same_base = self._state(
            payload=base_payload,
            cell_id=torch.tensor([[1, 1]]),
            node_id=torch.tensor([[0, 0]]),
        )
        same_changed = self._state(
            payload=changed_payload,
            cell_id=torch.tensor([[1, 1]]),
            node_id=torch.tensor([[0, 0]]),
        )
        remote_base = self._state(
            payload=base_payload,
            cell_id=torch.tensor([[1, 3]]),
            node_id=torch.tensor([[0, 2]]),
        )
        remote_changed = self._state(
            payload=changed_payload,
            cell_id=torch.tensor([[1, 3]]),
            node_id=torch.tensor([[0, 2]]),
        )

        same_a = relay(same_base).state.payload[0, 0]
        same_b = relay(same_changed).state.payload[0, 0]
        remote_a = relay(remote_base).state.payload[0, 0]
        remote_b = relay(remote_changed).state.payload[0, 0]

        self.assertGreater(
            float((same_a - same_b).abs().max().detach()), 1.0e-4
        )
        torch.testing.assert_close(remote_a, remote_b)

    def test_forward_backward_and_router_gradients_are_finite(self):
        relay = self._relay(cells=6, top_k=2, hard=False)
        payload = torch.randn(2, 5, 3, requires_grad=True)
        address = torch.randn(5, 2, requires_grad=True)
        write_gate = torch.rand(2, 5, requires_grad=True)
        state = self._state(
            payload=payload,
            address=address,
            write_gate=write_gate,
            cell_id=torch.tensor(
                [[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]
            ),
            node_id=torch.tensor(
                [[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]]
            ),
        )

        result = relay(state)
        result.route.cell_logits.retain_grad()
        result.route.node_logits.retain_grad()
        loss = (
            result.state.payload.square().mean()
            + 0.01 * result.route.weights.square().mean()
        )
        loss.backward()

        for gradient in (
            payload.grad,
            address.grad,
            write_gate.grad,
            relay.cell_router_weight.grad,
            relay.node_router_weight.grad,
            relay.path_transform.weight.grad,
            result.route.cell_logits.grad,
            result.route.node_logits.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(
            float(relay.cell_router_weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(relay.node_router_weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(result.route.cell_logits.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(result.route.node_logits.grad.abs().sum()), 0.0
        )

    def test_inactive_sentinels_and_payload_are_preserved(self):
        relay = self._relay(cells=5, top_k=2, hard=False)
        payload = torch.randn(1, 4, 3)
        state = self._state(
            payload=payload,
            cell_id=torch.tensor([[0, -1, 2, -1]]),
            node_id=torch.tensor([[0, -1, 3, -1]]),
            active=torch.tensor([[True, False, True, False]]),
            write_gate=torch.tensor([[1.0, 100.0, 1.0, 100.0]]),
        )

        result = relay(state)

        torch.testing.assert_close(
            result.state.payload[~state.active],
            payload[~state.active],
        )
        self.assertEqual(
            result.state.cell_id[~state.active].tolist(), [-1, -1]
        )
        self.assertEqual(
            result.state.node_id[~state.active].tolist(), [-1, -1]
        )
        self.assertEqual(
            result.state.support_cell[~state.active].tolist(),
            [[-1, -1], [-1, -1]],
        )
        self.assertEqual(
            result.state.support_node[~state.active].tolist(),
            [[-1, -1], [-1, -1]],
        )
        torch.testing.assert_close(
            result.state.support_weight[~state.active],
            torch.zeros(2, 2),
        )
        self.assertFalse(bool(result.read.found[~state.active].any()))

    def test_top_two_support_is_compact_and_top_one_is_committed(self):
        relay = self._relay(cells=101, top_k=2, hard=False)
        state = self._state(
            payload=torch.randn(2, 7, 3),
            cell_id=torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6],
                    [100, 99, 98, 97, 96, 95, 94],
                ]
            ),
            node_id=torch.tensor(
                [[0, 1, 2, 3, 0, 1, 2], [3, 2, 1, 0, 3, 2, 1]]
            ),
        )

        result = relay(state)

        self.assertEqual(tuple(result.state.payload.shape), (2, 7, 3))
        self.assertEqual(tuple(result.state.support_cell.shape), (2, 7, 2))
        self.assertEqual(tuple(result.state.support_node.shape), (2, 7, 2))
        self.assertEqual(tuple(result.state.support_weight.shape), (2, 7, 2))
        self.assertNotIn(101, result.state.payload.shape)
        torch.testing.assert_close(
            result.state.cell_id, result.route.destination_cell[..., 0]
        )
        torch.testing.assert_close(
            result.state.node_id, result.route.destination_node[..., 0]
        )
        torch.testing.assert_close(
            result.state.support_weight.sum(dim=-1),
            torch.ones(2, 7),
        )
        for batch in range(state.batch_size):
            for slot in range(state.slot_count):
                source_node = int(state.node_id[batch, slot])
                for branch in range(2):
                    destination_node = int(
                        result.route.destination_node[
                            batch, slot, branch
                        ]
                    )
                    self.assertIn(
                        destination_node,
                        relay.topology.legal_node_destinations[
                            source_node
                        ],
                    )

    def test_local_policy_discards_support_and_backpropagates_broadly(
        self,
    ) -> None:
        relay = self._relay(
            cells=10,
            top_k=10,
            hard=False,
            local_policy=True,
        )
        state = self._state(
            payload=torch.randn(2, 5, 3),
            cell_id=torch.zeros(2, 5, dtype=torch.long),
            node_id=torch.zeros(2, 5, dtype=torch.long),
        )

        result = relay(state, temperature=1.25)
        loss = result.state.payload.square().mean()
        loss.backward()

        self.assertFalse(result.state.has_topk_support)
        torch.testing.assert_close(
            result.state.cell_id,
            result.route.destination_cell[..., 0],
        )
        cell_gradient = relay.path_transform.weight.grad.abs().sum(
            dim=(1, 2, 3)
        )
        self.assertEqual(int(cell_gradient.gt(0).sum()), 1)
        self.assertGreater(float(cell_gradient[0]), 0.0)
        self.assertGreater(
            float(relay.cell_router_weight.grad.abs().sum()),
            0.0,
        )

    def test_local_policy_transform_belongs_to_current_steward(
        self,
    ) -> None:
        relay = self._relay(
            cells=5,
            top_k=1,
            hard=False,
            local_policy=True,
        )
        self._choose_action(relay, cell_action=2, node_action=0)
        state = self._state(
            payload=torch.randn(1, 2, 3),
            cell_id=torch.zeros(1, 2, dtype=torch.long),
            node_id=torch.zeros(1, 2, dtype=torch.long),
        )

        first = relay(state)
        first.state.payload.square().mean().backward()
        first_cell_gradient = relay.path_transform.weight.grad.abs().sum(
            dim=(1, 2, 3)
        )
        self.assertGreater(float(first_cell_gradient[0]), 0.0)
        self.assertEqual(int(first_cell_gradient[1:].gt(0).sum()), 0)

        relay.zero_grad(set_to_none=True)
        second = relay(first.state.snapshot(detach=True))
        second.state.payload.square().mean().backward()
        second_cell_gradient = relay.path_transform.weight.grad.abs().sum(
            dim=(1, 2, 3)
        )
        self.assertGreater(float(second_cell_gradient[1]), 0.0)
        self.assertEqual(int(second_cell_gradient[[0, 2, 3, 4]].gt(0).sum()), 0)

    def test_top_two_selected_deltas_use_compact_weighted_reduction(self):
        relay = self._relay(cells=5, top_k=2, hard=False)
        state = self._state(
            payload=torch.randn(1, 3, 3),
            cell_id=torch.tensor([[0, 2, 4]]),
            node_id=torch.tensor([[0, 1, 3]]),
        )

        result = relay(state)
        route = result.route
        payload = state.payload.unsqueeze(2).expand(-1, -1, 2, -1)
        mailbox_payload = (
            result.read.payload_mean
            + F.linear(
                result.read.address_mean,
                relay.mailbox_address_weight,
            )
        )
        mailbox = mailbox_payload.unsqueeze(2).expand_as(payload)
        address = state.batched_address.unsqueeze(2).expand(
            -1, -1, 2, -1
        )
        source_node = state.node_id.unsqueeze(-1).expand(-1, -1, 2)
        selected_delta = relay.path_transform(
            payload,
            mailbox,
            address,
            route.destination_cell,
            source_node,
            route.destination_node,
            route.cell_action,
            validate_indices=False,
        )
        mixed = (
            selected_delta * route.weights.unsqueeze(-1)
        ).sum(dim=2)
        residual = state.payload + mixed
        mean = residual.mean(dim=-1, keepdim=True)
        variance = residual.var(
            dim=-1, unbiased=False, keepdim=True
        )
        normalized = (
            (residual - mean)
            * torch.rsqrt(variance + relay.norm_epsilon)
        )
        committed_cell = route.destination_cell[..., 0]
        expected = (
            normalized * relay.destination_norm_weight[committed_cell]
            + relay.destination_norm_bias[committed_cell]
        )

        torch.testing.assert_close(result.state.payload, expected)
        self.assertEqual(tuple(selected_delta.shape), (1, 3, 2, 3))
        self.assertEqual(tuple(mixed.shape), (1, 3, 3))

    def test_control_overrides_and_width_validation(self):
        relay = self._relay(top_k=1)
        state = self._state(
            payload=torch.randn(1, 2, 3),
            cell_id=torch.tensor([[0, 1]]),
            node_id=torch.tensor([[0, 1]]),
        )

        result = relay(
            state,
            top_k=2,
            temperature=0.5,
            straight_through_hard=False,
        )
        self.assertEqual(result.route.top_k, 2)

        bad = self._state(
            payload=torch.randn(1, 2, 4),
            cell_id=torch.tensor([[0, 1]]),
            node_id=torch.tensor([[0, 1]]),
        )
        with self.assertRaisesRegex(ValueError, "payload width"):
            relay(bad)

    def test_prior_topk_weights_receive_credit_from_the_next_relay(self):
        relay = self._relay(cells=5, top_k=2, hard=False)
        state = self._state(
            payload=torch.randn(2, 5, 3),
            cell_id=torch.zeros(2, 5, dtype=torch.long),
            node_id=torch.zeros(2, 5, dtype=torch.long),
        )

        first = relay(state, temperature=1.25)
        first.route.weights.retain_grad()
        second = relay(
            first.state,
            temperature=1.25,
            validate_coordinates=False,
        )
        objective = second.state.payload[..., 0].square().mean()
        objective.backward()

        self.assertIsNotNone(first.route.weights.grad)
        self.assertTrue(torch.isfinite(first.route.weights.grad).all())
        self.assertGreater(
            float(first.route.weights.grad.abs().sum()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
