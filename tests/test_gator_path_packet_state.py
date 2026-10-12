from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import torch

from gator_path.packet_state import PacketState


class PacketStateTests(unittest.TestCase):
    def _state(self, *, top_k: int | None = None) -> PacketState:
        batch_size, slots, width = 2, 5, 7
        kwargs = {}
        if top_k is not None:
            kwargs = {
                "support_cell": torch.zeros(
                    batch_size, slots, top_k, dtype=torch.long
                ),
                "support_node": torch.ones(
                    batch_size, slots, top_k, dtype=torch.long
                ),
                "support_weight": torch.full(
                    (batch_size, slots, top_k),
                    1.0 / top_k,
                ),
            }
        return PacketState(
            payload=torch.randn(batch_size, slots, width),
            address=torch.randn(slots, 3),
            cell_id=torch.zeros(
                batch_size, slots, dtype=torch.long
            ),
            node_id=torch.ones(
                batch_size, slots, dtype=torch.long
            ),
            active=torch.ones(
                batch_size, slots, dtype=torch.bool
            ),
            write_gate=torch.rand(batch_size, slots),
            trace=torch.randn(batch_size, slots, 4),
            **kwargs,
        )

    def test_top1_state_has_one_payload_per_packet(self):
        state = self._state()

        self.assertEqual(tuple(state.payload.shape), (2, 5, 7))
        self.assertEqual(tuple(state.cell_id.shape), (2, 5))
        self.assertEqual(tuple(state.node_id.shape), (2, 5))
        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.slot_count, 5)
        self.assertEqual(state.payload_width, 7)
        self.assertEqual(state.address_width, 3)
        self.assertEqual(state.trace_width, 4)
        self.assertEqual(state.route_k, 1)
        self.assertFalse(state.has_topk_support)

    def test_shared_address_expands_as_a_nonallocating_batch_view(self):
        state = self._state()

        batched = state.batched_address

        self.assertEqual(tuple(batched.shape), (2, 5, 3))
        self.assertEqual(batched.stride(0), 0)
        self.assertEqual(batched.data_ptr(), state.address.data_ptr())
        torch.testing.assert_close(batched[0], state.address)
        torch.testing.assert_close(batched[1], state.address)

    def test_validated_batch_specific_address_is_returned_directly(self):
        state = self._state()
        address = torch.randn(2, 5, 3)

        batched_state = PacketState(
            payload=state.payload,
            address=address,
            cell_id=state.cell_id,
            node_id=state.node_id,
            active=state.active,
            write_gate=state.write_gate,
        )

        self.assertIs(batched_state.batched_address, address)
        with self.assertRaisesRegex(ValueError, "batched address"):
            PacketState(
                payload=state.payload,
                address=torch.randn(3, 5, 3),
                cell_id=state.cell_id,
                node_id=state.node_id,
                active=state.active,
                write_gate=state.write_gate,
            )

    def test_rejects_any_payload_replication_over_cells(self):
        state = self._state()

        with self.assertRaisesRegex(
            ValueError, "no cell axis"
        ):
            PacketState(
                payload=torch.randn(2, 11, 5, 7),
                address=state.address,
                cell_id=state.cell_id,
                node_id=state.node_id,
                active=state.active,
                write_gate=state.write_gate,
            )

    def test_sparse_topk_support_is_compact_and_all_or_none(self):
        state = self._state(top_k=2)

        self.assertTrue(state.has_topk_support)
        self.assertEqual(state.route_k, 2)
        self.assertEqual(tuple(state.support_cell.shape), (2, 5, 2))
        self.assertEqual(tuple(state.support_node.shape), (2, 5, 2))
        self.assertEqual(tuple(state.support_weight.shape), (2, 5, 2))

        with self.assertRaisesRegex(ValueError, "provided together"):
            state.replace(support_weight=None)
        with self.assertRaisesRegex(ValueError, r"\[B, L, K\]"):
            state.replace(
                support_node=torch.zeros(2, 5, 3, dtype=torch.long)
            )

    def test_validates_packet_shapes_and_semantic_dtypes(self):
        state = self._state()

        with self.assertRaisesRegex(ValueError, r"cell_id.*\[B, L\]"):
            state.replace(cell_id=torch.zeros(2, 5, 1, dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "torch.long"):
            state.replace(cell_id=torch.zeros(2, 5))
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            state.replace(active=torch.ones(2, 5))
        with self.assertRaisesRegex(ValueError, r"trace.*\[B, L, D_t\]"):
            state.replace(trace=torch.zeros(2, 4, 4))

    def test_snapshot_copies_mutable_state_and_shares_address(self):
        state = self._state(top_k=2)
        snapshot = state.snapshot()

        self.assertIs(snapshot.address, state.address)
        for name in (
            "payload",
            "cell_id",
            "node_id",
            "active",
            "write_gate",
            "trace",
            "support_cell",
            "support_node",
            "support_weight",
        ):
            original = getattr(state, name)
            copied = getattr(snapshot, name)
            self.assertIsNot(copied, original)
            torch.testing.assert_close(copied, original)

        state.payload[0, 0, 0] = 100.0
        self.assertNotEqual(
            float(snapshot.payload[0, 0, 0]),
            float(state.payload[0, 0, 0]),
        )

    def test_snapshot_can_detach_autograd_history(self):
        state = self._state()
        payload = state.payload.detach().requires_grad_(True)
        state = state.replace(payload=payload)

        connected = state.snapshot()
        detached = state.snapshot(detach=True)

        self.assertTrue(connected.payload.requires_grad)
        self.assertIsNotNone(connected.payload.grad_fn)
        self.assertFalse(detached.payload.requires_grad)
        self.assertIsNone(detached.payload.grad_fn)

    def test_replace_is_validated_and_preserves_immutable_address(self):
        state = self._state()
        payload = torch.randn_like(state.payload)

        updated = state.replace(payload=payload, trace=None)

        self.assertIs(updated.payload, payload)
        self.assertIs(updated.address, state.address)
        self.assertIsNone(updated.trace)
        with self.assertRaisesRegex(ValueError, "address is immutable"):
            state.replace(address=state.address.clone())
        with self.assertRaisesRegex(TypeError, "unknown"):
            state.replace(not_a_field=payload)

    def test_state_fields_cannot_be_reassigned(self):
        state = self._state()

        with self.assertRaises(FrozenInstanceError):
            state.payload = torch.zeros_like(state.payload)


if __name__ == "__main__":
    unittest.main()
