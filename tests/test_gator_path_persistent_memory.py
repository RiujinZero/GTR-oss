from __future__ import annotations

import unittest

import torch

from gator_path.persistent_memory import (
    build_path_relative_memory,
    build_relative_memory,
    relative_code,
    update_path_relative_memory,
    update_relative_memory,
)


def _brute_memory(
    payload: torch.Tensor,
    cell_id: torch.Tensor,
    node_id: torch.Tensor,
    write_gate: torch.Tensor,
    relation_bias: torch.Tensor,
) -> torch.Tensor:
    batch, length, width = payload.shape
    cells, nodes, _ = relation_bias.shape
    relations = relative_code(
        length,
        device=payload.device,
        dtype=payload.dtype,
    )
    result = torch.zeros(batch, length, cells, nodes, width)
    for cell in range(cells):
        for node in range(nodes):
            present = (cell_id == cell) & (node_id == node)
            score = torch.einsum(
                "ijr,r->ij",
                relations,
                relation_bias[cell, node],
            )[None].expand(batch, -1, -1)
            weight = (
                score.exp()
                * write_gate[:, None]
                * present[:, None]
            )
            denominator = weight.sum(dim=2, keepdim=True)
            normalized = weight / denominator.clamp_min(1.0e-8)
            memory = torch.einsum(
                "bij,bjd->bid",
                normalized,
                payload,
            )
            result[:, :, cell, node] = torch.where(
                denominator > 0,
                memory,
                torch.zeros_like(memory),
            )
    return result


class PersistentRelativeMemoryTests(unittest.TestCase):
    def _packets(self):
        torch.manual_seed(19)
        payload = torch.randn(2, 5, 4)
        cell_id = torch.tensor(
            [[0, 1, 2, 0, 1], [2, 2, 1, 0, 0]],
            dtype=torch.long,
        )
        node_id = torch.tensor(
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
            dtype=torch.long,
        )
        write_gate = torch.tensor(
            [[1.0, 1.0, 0.5, 0.25, 1.0], [1.0, 0.5, 1.0, 1.0, 0.5]]
        )
        relation_bias = torch.randn(3, 2, 4) * 0.3
        return payload, cell_id, node_id, write_gate, relation_bias

    def test_initial_carried_memory_matches_dense_rebuild(self) -> None:
        payload, cell_id, node_id, write_gate, relation_bias = (
            self._packets()
        )

        state = build_relative_memory(
            payload=payload,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_gate,
            relation_bias=relation_bias,
        )
        expected = _brute_memory(
            payload,
            cell_id,
            node_id,
            write_gate,
            relation_bias,
        )

        torch.testing.assert_close(
            state.value,
            expected,
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    def test_departure_arrival_update_matches_final_rebuild(self) -> None:
        payload, cell_id, node_id, write_gate, relation_bias = (
            self._packets()
        )
        state = build_relative_memory(
            payload=payload,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_gate,
            relation_bias=relation_bias,
        )
        new_payload = payload + torch.randn_like(payload) * 0.1
        new_cell = (cell_id + 1).remainder(3)
        new_node = (node_id + (cell_id == 0)).remainder(2)
        new_write = write_gate.roll(1, dims=1)

        updated = update_relative_memory(
            state,
            old_payload=payload,
            old_cell_id=cell_id,
            old_node_id=node_id,
            old_write_gate=write_gate,
            new_payload=new_payload,
            new_cell_id=new_cell,
            new_node_id=new_node,
            new_write_gate=new_write,
            relation_bias=relation_bias,
        )
        rebuilt = build_relative_memory(
            payload=new_payload,
            cell_id=new_cell,
            node_id=new_node,
            write_gate=new_write,
            relation_bias=relation_bias,
        )

        torch.testing.assert_close(
            updated.numerator,
            rebuilt.numerator,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        torch.testing.assert_close(
            updated.denominator,
            rebuilt.denominator,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        torch.testing.assert_close(
            updated.value,
            rebuilt.value,
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    def test_inactive_packets_do_not_change_memory(self) -> None:
        payload, cell_id, node_id, write_gate, relation_bias = (
            self._packets()
        )
        state = build_relative_memory(
            payload=payload,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_gate,
            relation_bias=relation_bias,
        )
        inactive = torch.zeros_like(cell_id, dtype=torch.bool)

        updated = update_relative_memory(
            state,
            old_payload=payload,
            old_cell_id=cell_id,
            old_node_id=node_id,
            old_write_gate=write_gate,
            new_payload=payload + 100.0,
            new_cell_id=(cell_id + 1).remainder(3),
            new_node_id=(node_id + 1).remainder(2),
            new_write_gate=write_gate,
            relation_bias=relation_bias,
            active=inactive,
        )

        torch.testing.assert_close(updated.numerator, state.numerator)
        torch.testing.assert_close(updated.denominator, state.denominator)

    def test_carried_update_has_finite_gradients(self) -> None:
        payload, cell_id, node_id, write_gate, relation_bias = (
            self._packets()
        )
        payload.requires_grad_()
        relation_bias.requires_grad_()
        state = build_relative_memory(
            payload=payload,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_gate,
            relation_bias=relation_bias,
        )
        new_payload = payload * 0.9 + 0.1
        updated = update_relative_memory(
            state,
            old_payload=payload,
            old_cell_id=cell_id,
            old_node_id=node_id,
            old_write_gate=write_gate,
            new_payload=new_payload,
            new_cell_id=(cell_id + 1).remainder(3),
            new_node_id=node_id,
            new_write_gate=write_gate,
            relation_bias=relation_bias,
        )

        updated.value.square().mean().backward()

        self.assertTrue(torch.isfinite(payload.grad).all())
        self.assertTrue(torch.isfinite(relation_bias.grad).all())
        self.assertGreater(float(payload.grad.abs().sum()), 0.0)
        self.assertGreater(float(relation_bias.grad.abs().sum()), 0.0)

    def test_single_trace_path_memory_matches_packet_memory(self) -> None:
        payload, cell_id, node_id, write_gate, relation_bias = (
            self._packets()
        )
        position_id = torch.arange(payload.shape[1])[None].expand(
            payload.shape[0],
            -1,
        )

        packet_state = build_relative_memory(
            payload=payload,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_gate,
            relation_bias=relation_bias,
        )
        path_state = build_path_relative_memory(
            payload=payload,
            position_id=position_id,
            cell_id=cell_id,
            node_id=node_id,
            write_gate=write_gate,
            relation_bias=relation_bias,
            receiver_length=payload.shape[1],
        )

        torch.testing.assert_close(
            path_state.numerator,
            packet_state.numerator,
        )
        torch.testing.assert_close(
            path_state.denominator,
            packet_state.denominator,
        )

    def test_cooperative_trace_update_matches_rebuild(self) -> None:
        payload, cell_id, node_id, write_gate, relation_bias = (
            self._packets()
        )
        batch, length, width = payload.shape
        paths = 2
        path_payload = (
            payload[:, :, None]
            + torch.tensor([0.0, 0.2])[None, None, :, None]
        ).reshape(batch, length * paths, width)
        path_cell = torch.stack(
            (cell_id, (cell_id + 1).remainder(3)),
            dim=2,
        ).reshape(batch, length * paths)
        path_node = torch.stack(
            (node_id, (node_id + 1).remainder(2)),
            dim=2,
        ).reshape(batch, length * paths)
        path_write = write_gate[:, :, None].expand(
            -1,
            -1,
            paths,
        ).reshape(batch, length * paths)
        position_id = torch.arange(length).repeat_interleave(paths)[
            None
        ].expand(batch, -1)
        state = build_path_relative_memory(
            payload=path_payload,
            position_id=position_id,
            cell_id=path_cell,
            node_id=path_node,
            write_gate=path_write,
            relation_bias=relation_bias,
            receiver_length=length,
        )
        new_payload = path_payload * 0.95 + 0.05
        new_cell = (path_cell + 1).remainder(3)
        new_node = (path_node + 1).remainder(2)

        updated = update_path_relative_memory(
            state,
            position_id=position_id,
            old_payload=path_payload,
            old_cell_id=path_cell,
            old_node_id=path_node,
            old_write_gate=path_write,
            new_payload=new_payload,
            new_cell_id=new_cell,
            new_node_id=new_node,
            new_write_gate=path_write,
            relation_bias=relation_bias,
            receiver_length=length,
        )
        rebuilt = build_path_relative_memory(
            payload=new_payload,
            position_id=position_id,
            cell_id=new_cell,
            node_id=new_node,
            write_gate=path_write,
            relation_bias=relation_bias,
            receiver_length=length,
        )

        torch.testing.assert_close(
            updated.numerator,
            rebuilt.numerator,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        torch.testing.assert_close(
            updated.denominator,
            rebuilt.denominator,
            atol=2.0e-6,
            rtol=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
