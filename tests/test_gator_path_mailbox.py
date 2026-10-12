from __future__ import annotations

import unittest

import torch

from gator_path.mailbox import (
    build_sparse_mailbox,
    encode_location_keys,
)
from gator_path.packet_state import PacketState


class SparseMailboxTests(unittest.TestCase):
    def _state(
        self,
        *,
        payload: torch.Tensor,
        address: torch.Tensor,
        cell_id: torch.Tensor,
        node_id: torch.Tensor,
        active: torch.Tensor,
        write_gate: torch.Tensor,
    ) -> PacketState:
        return PacketState(
            payload=payload,
            address=address,
            cell_id=cell_id,
            node_id=node_id,
            active=active,
            write_gate=write_gate,
        )

    def test_segmented_aggregation_matches_loop_oracle(self):
        payload = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [9.0, 18.0]],
                [[2.0, 4.0], [4.0, 8.0], [8.0, 16.0], [7.0, 14.0]],
            ]
        )
        address = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [4.0, 1.0]]
        )
        cell_id = torch.tensor([[1, 1, 3, 2], [1, 1, 1, 4]])
        node_id = torch.tensor([[0, 0, 2, 1], [0, 0, 2, 3]])
        active = torch.tensor(
            [[True, True, True, False], [True, True, False, True]]
        )
        gates = torch.tensor([[1.0, 3.0, 0.0, 8.0], [2.0, 1.0, 5.0, 0.5]])
        state = self._state(
            payload=payload,
            address=address,
            cell_id=cell_id,
            node_id=node_id,
            active=active,
            write_gate=gates,
        )

        mailbox = build_sparse_mailbox(
            state, num_cells=5, num_nodes=4
        )

        oracle: dict[int, dict[str, torch.Tensor]] = {}
        for batch in range(state.batch_size):
            for slot in range(state.slot_count):
                if not bool(active[batch, slot]):
                    continue
                key = (
                    batch * 5 * 4
                    + int(cell_id[batch, slot]) * 4
                    + int(node_id[batch, slot])
                )
                group = oracle.setdefault(
                    key,
                    {
                        "mass": torch.tensor(0.0),
                        "payload": torch.zeros(2),
                        "address": torch.zeros(2),
                    },
                )
                gate = gates[batch, slot]
                group["mass"] += gate
                group["payload"] += gate * payload[batch, slot]
                group["address"] += gate * address[slot]

        expected_keys = sorted(oracle)
        self.assertEqual(mailbox.occupied_keys.tolist(), expected_keys)
        for row, key in enumerate(expected_keys):
            expected = oracle[key]
            mass = expected["mass"]
            denominator = mass.clamp_min(1.0e-6)
            torch.testing.assert_close(mailbox.mass[row], mass)
            torch.testing.assert_close(
                mailbox.payload_sum[row],
                expected["payload"],
            )
            torch.testing.assert_close(
                mailbox.address_sum[row],
                expected["address"],
            )
            torch.testing.assert_close(
                mailbox.payload_mean[row],
                expected["payload"] / denominator,
            )
            torch.testing.assert_close(
                mailbox.address_mean[row],
                expected["address"] / denominator,
            )

        decoded = list(
            zip(
                mailbox.batch_id.tolist(),
                mailbox.cell_id.tolist(),
                mailbox.node_id.tolist(),
            )
        )
        self.assertEqual(
            decoded,
            [(0, 1, 0), (0, 3, 2), (1, 1, 0), (1, 4, 3)],
        )

    def test_packet_reads_are_local_and_missing_or_inactive_reads_are_zero(self):
        state = self._state(
            payload=torch.tensor([[[2.0], [6.0], [100.0], [50.0]]]),
            address=torch.tensor([[1.0], [3.0], [7.0], [9.0]]),
            cell_id=torch.tensor([[0, 0, 1, -1]]),
            node_id=torch.tensor([[0, 0, 0, -1]]),
            active=torch.tensor([[True, True, True, False]]),
            write_gate=torch.tensor([[1.0, 1.0, 1.0, 100.0]]),
        )
        mailbox = build_sparse_mailbox(
            state, num_cells=2, num_nodes=1
        )

        read = mailbox.read_packets(state)

        torch.testing.assert_close(
            read.payload_mean,
            torch.tensor([[[4.0], [4.0], [100.0], [0.0]]]),
        )
        torch.testing.assert_close(
            read.address_mean,
            torch.tensor([[[2.0], [2.0], [7.0], [0.0]]]),
        )
        torch.testing.assert_close(
            read.mass, torch.tensor([[2.0, 2.0, 1.0, 0.0]])
        )
        self.assertEqual(read.found.tolist(), [[True, True, True, False]])
        self.assertEqual(read.mailbox_index.tolist(), [[0, 0, 1, -1]])

    def test_zero_write_locations_are_finite_zero_rows(self):
        state = self._state(
            payload=torch.tensor([[[2.0, -1.0], [5.0, 4.0]]]),
            address=torch.tensor([[1.0], [2.0]]),
            cell_id=torch.tensor([[3, 3]]),
            node_id=torch.tensor([[1, 1]]),
            active=torch.ones(1, 2, dtype=torch.bool),
            write_gate=torch.zeros(1, 2),
        )

        mailbox = build_sparse_mailbox(
            state, num_cells=100, num_nodes=4
        )
        read = mailbox.read_packets(state)

        self.assertEqual(mailbox.occupied_count, 1)
        self.assertTrue(torch.isfinite(mailbox.mass).all())
        self.assertTrue(torch.isfinite(mailbox.payload_sum).all())
        self.assertTrue(torch.isfinite(mailbox.address_sum).all())
        self.assertTrue(torch.isfinite(mailbox.payload_mean).all())
        self.assertTrue(torch.isfinite(mailbox.address_mean).all())
        torch.testing.assert_close(mailbox.mass, torch.zeros(1))
        torch.testing.assert_close(mailbox.payload_mean, torch.zeros(1, 2))
        torch.testing.assert_close(mailbox.address_mean, torch.zeros(1, 1))
        self.assertEqual(read.found.tolist(), [[True, True]])

    def test_empty_active_set_ignores_padding_sentinels_and_supports_backward(self):
        payload = torch.randn(2, 3, 4, requires_grad=True)
        gates = torch.randn(2, 3, requires_grad=True)
        state = self._state(
            payload=payload,
            address=torch.randn(3, 2),
            cell_id=torch.full((2, 3), -1, dtype=torch.long),
            node_id=torch.full((2, 3), -1, dtype=torch.long),
            active=torch.zeros(2, 3, dtype=torch.bool),
            write_gate=gates,
        )

        mailbox = build_sparse_mailbox(
            state, num_cells=1_000_000, num_nodes=4
        )
        read = mailbox.read_packets(state)
        loss = (
            mailbox.mass.sum()
            + mailbox.payload_mean.sum()
            + read.mass.sum()
            + read.payload_mean.sum()
        )
        loss.backward()

        self.assertEqual(mailbox.occupied_count, 0)
        self.assertEqual(tuple(mailbox.payload_mean.shape), (0, 4))
        self.assertEqual(tuple(mailbox.address_mean.shape), (0, 2))
        self.assertEqual(read.mailbox_index.tolist(), [[-1] * 3, [-1] * 3])
        self.assertTrue(torch.isfinite(read.payload_mean).all())
        self.assertIsNotNone(payload.grad)
        self.assertIsNotNone(gates.grad)
        torch.testing.assert_close(payload.grad, torch.zeros_like(payload))
        torch.testing.assert_close(gates.grad, torch.zeros_like(gates))

    def test_batched_addresses_are_aggregated_per_batch(self):
        state = self._state(
            payload=torch.ones(2, 2, 1),
            address=torch.tensor(
                [[[1.0], [3.0]], [[10.0], [14.0]]]
            ),
            cell_id=torch.zeros(2, 2, dtype=torch.long),
            node_id=torch.zeros(2, 2, dtype=torch.long),
            active=torch.ones(2, 2, dtype=torch.bool),
            write_gate=torch.ones(2, 2),
        )

        mailbox = build_sparse_mailbox(
            state, num_cells=1, num_nodes=1
        )

        torch.testing.assert_close(
            mailbox.address_mean, torch.tensor([[2.0], [12.0]])
        )
        self.assertEqual(mailbox.batch_id.tolist(), [0, 1])

    def test_backward_is_finite_for_payloads_and_write_gates(self):
        payload = torch.tensor(
            [[[1.0, 2.0], [4.0, -1.0], [8.0, 3.0]]],
            requires_grad=True,
        )
        gates = torch.tensor(
            [[0.25, 0.75, 0.5]], requires_grad=True
        )
        address = torch.tensor(
            [[1.0], [2.0], [3.0]], requires_grad=True
        )
        state = self._state(
            payload=payload,
            address=address,
            cell_id=torch.tensor([[2, 2, 1]]),
            node_id=torch.tensor([[0, 0, 1]]),
            active=torch.ones(1, 3, dtype=torch.bool),
            write_gate=gates,
        )

        mailbox = build_sparse_mailbox(
            state, num_cells=4, num_nodes=2
        )
        read = mailbox.read_packets(state)
        loss = (
            read.payload_mean.square().sum()
            + 0.2 * read.address_mean.square().sum()
            + 0.1 * mailbox.mass.square().sum()
        )
        loss.backward()

        for gradient in (payload.grad, gates.grad, address.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_storage_tracks_occupied_keys_not_total_cells(self):
        state = self._state(
            payload=torch.randn(1, 4, 3),
            address=torch.randn(4, 2),
            cell_id=torch.tensor([[0, 0, 999_999, 999_999]]),
            node_id=torch.tensor([[0, 0, 3, 3]]),
            active=torch.ones(1, 4, dtype=torch.bool),
            write_gate=torch.ones(1, 4),
        )

        mailbox = build_sparse_mailbox(
            state, num_cells=1_000_000, num_nodes=4
        )

        self.assertEqual(mailbox.occupied_count, 2)
        self.assertEqual(tuple(mailbox.payload_mean.shape), (2, 3))
        self.assertEqual(tuple(mailbox.address_mean.shape), (2, 2))
        tensor_elements = sum(
            tensor.numel()
            for tensor in (
                mailbox.occupied_keys,
                mailbox.batch_id,
                mailbox.cell_id,
                mailbox.node_id,
                mailbox.mass,
                mailbox.payload_sum,
                mailbox.address_sum,
                mailbox.payload_mean,
                mailbox.address_mean,
            )
        )
        self.assertEqual(tensor_elements, 2 * (5 + 2 * 3 + 2 * 2))

    def test_invalid_active_locations_are_rejected(self):
        base = dict(
            payload=torch.randn(1, 2, 2),
            address=torch.randn(2, 1),
            active=torch.tensor([[True, False]]),
            write_gate=torch.ones(1, 2),
        )
        invalid_cell = self._state(
            **base,
            cell_id=torch.tensor([[3, -1]]),
            node_id=torch.tensor([[0, -1]]),
        )
        with self.assertRaisesRegex(ValueError, "active cell_id"):
            build_sparse_mailbox(
                invalid_cell, num_cells=3, num_nodes=2
            )

        invalid_node = self._state(
            **base,
            cell_id=torch.tensor([[0, -1]]),
            node_id=torch.tensor([[2, -1]]),
        )
        with self.assertRaisesRegex(ValueError, "active node_id"):
            build_sparse_mailbox(
                invalid_node, num_cells=3, num_nodes=2
            )

    def test_key_lookup_returns_minus_one_for_absent_locations(self):
        state = self._state(
            payload=torch.ones(1, 1, 1),
            address=torch.ones(1, 1),
            cell_id=torch.tensor([[2]]),
            node_id=torch.tensor([[1]]),
            active=torch.ones(1, 1, dtype=torch.bool),
            write_gate=torch.ones(1, 1),
        )
        mailbox = build_sparse_mailbox(
            state, num_cells=4, num_nodes=2
        )
        keys = encode_location_keys(
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 2, 3]),
            torch.tensor([0, 1, 1]),
            num_cells=4,
            num_nodes=2,
        )

        self.assertEqual(mailbox.lookup(keys).tolist(), [-1, 0, -1])

    def test_topk_support_writes_and_reads_persistent_sparse_branches(self):
        support_weight = torch.tensor(
            [[[0.25, 0.75], [0.5, 0.5]]],
            requires_grad=True,
        )
        state = PacketState(
            payload=torch.tensor([[[2.0], [6.0]]]),
            address=torch.tensor([[1.0], [3.0]]),
            cell_id=torch.zeros(1, 2, dtype=torch.long),
            node_id=torch.zeros(1, 2, dtype=torch.long),
            active=torch.ones(1, 2, dtype=torch.bool),
            write_gate=torch.ones(1, 2),
            support_cell=torch.tensor([[[0, 1], [0, 1]]]),
            support_node=torch.zeros(1, 2, 2, dtype=torch.long),
            support_weight=support_weight,
        )

        mailbox = build_sparse_mailbox(
            state,
            num_cells=2,
            num_nodes=1,
        )
        support_read = mailbox.read_support(state)

        self.assertEqual(mailbox.occupied_count, 2)
        torch.testing.assert_close(
            mailbox.mass,
            torch.tensor([0.75, 1.25]),
        )
        torch.testing.assert_close(
            mailbox.payload_mean,
            torch.tensor([[14.0 / 3.0], [3.6]]),
        )
        self.assertEqual(tuple(support_read.mass.shape), (1, 2, 2))
        self.assertEqual(
            tuple(support_read.payload_mean.shape),
            (1, 2, 2, 1),
        )
        loss = (
            mailbox.mass.square().sum()
            + mailbox.payload_mean.square().sum()
            + support_read.payload_mean.square().sum()
        )
        loss.backward()
        self.assertIsNotNone(support_weight.grad)
        self.assertTrue(torch.isfinite(support_weight.grad).all())
        self.assertGreater(float(support_weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
