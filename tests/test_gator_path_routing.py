from __future__ import annotations

import unittest

import torch

from gator_path.routing import route_local_actions
from gator_path.topology import fixed_ring_topology


class LocalRoutingTests(unittest.TestCase):
    def _zeros(
        self,
        *,
        batch: int = 1,
        slots: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch, slots, 5),
            torch.zeros(batch, slots, 3),
        )

    def test_routes_wrap_cells_and_decode_source_local_node_actions(
        self,
    ) -> None:
        topology = fixed_ring_topology(10)
        source_cell = torch.tensor([[0, 9]], dtype=torch.long)
        source_node = torch.tensor([[0, 1]], dtype=torch.long)
        cell_logits = torch.full((1, 2, 5), -20.0)
        node_logits = torch.full((1, 2, 3), -20.0)
        cell_logits[0, 0, 1] = 20.0  # offset -1
        cell_logits[0, 1, 4] = 20.0  # offset +2
        node_logits[0, 0, 1] = 20.0  # source 0: destination 1
        node_logits[0, 1, 1] = 20.0  # source 1: destination 2

        result = route_local_actions(
            topology=topology,
            source_cell=source_cell,
            source_node=source_node,
            cell_logits=cell_logits,
            node_logits=node_logits,
        )

        torch.testing.assert_close(
            result.destination_cell,
            torch.tensor([[[9], [1]]]),
        )
        torch.testing.assert_close(
            result.destination_node,
            torch.tensor([[[1], [2]]]),
        )
        torch.testing.assert_close(
            result.cell_action,
            torch.tensor([[[1], [4]]]),
        )
        torch.testing.assert_close(
            result.node_action,
            torch.tensor([[[1], [1]]]),
        )

    def test_destination_table_can_replace_ring_geometry(self) -> None:
        topology = fixed_ring_topology(10)
        source_cell = torch.tensor([[8, 3]], dtype=torch.long)
        source_node = torch.tensor([[0, 0]], dtype=torch.long)
        cell_logits = torch.full((1, 2, 5), -20.0)
        node_logits = torch.full((1, 2, 3), -20.0)
        cell_logits[..., 4] = 20.0
        node_logits[..., 0] = 20.0
        destination_table = torch.tensor(
            topology.cell_destination_table,
            dtype=torch.long,
        )
        destination_table[8, 4] = 7
        destination_table[3, 4] = 9

        result = route_local_actions(
            topology=topology,
            source_cell=source_cell,
            source_node=source_node,
            cell_logits=cell_logits,
            node_logits=node_logits,
            cell_destination_table=destination_table,
        )

        torch.testing.assert_close(
            result.destination_cell,
            torch.tensor([[[7], [9]]]),
        )
        torch.testing.assert_close(
            result.cell_action,
            torch.tensor([[[4], [4]]]),
        )

    def test_different_out_degrees_are_masked_before_selection(self) -> None:
        topology = fixed_ring_topology(7)
        cell_logits, node_logits = self._zeros(slots=2)
        source_cell = torch.tensor([[1, 1]])
        source_node = torch.tensor([[0, 1]])
        node_logits[0, 0, 2] = 1000.0
        node_logits[0, 1, 2] = 1000.0

        result = route_local_actions(
            topology=topology,
            source_cell=source_cell,
            source_node=source_node,
            cell_logits=cell_logits,
            node_logits=node_logits,
        )

        self.assertFalse(bool(result.node_legal_mask[0, 0, 2]))
        self.assertTrue(bool(result.node_legal_mask[0, 1, 2]))
        self.assertEqual(float(result.node_probabilities[0, 0, 2]), 0.0)
        self.assertEqual(int(result.destination_node[0, 1, 0]), 3)
        self.assertIn(
            int(result.destination_node[0, 0, 0]),
            topology.legal_node_destinations[0],
        )

    def test_top_two_ties_use_stable_cell_major_action_order(self) -> None:
        topology = fixed_ring_topology(10)
        cell_logits, node_logits = self._zeros()

        result = route_local_actions(
            topology=topology,
            source_cell=torch.tensor([[4]]),
            source_node=torch.tensor([[0]]),
            cell_logits=cell_logits,
            node_logits=node_logits,
            top_k=2,
        )

        # Source node zero has local destinations [0, 1].  Equal joint
        # probabilities therefore choose (stay, 0), then (stay, 1).
        torch.testing.assert_close(
            result.cell_action,
            torch.tensor([[[0, 0]]]),
        )
        torch.testing.assert_close(
            result.node_action,
            torch.tensor([[[0, 1]]]),
        )
        torch.testing.assert_close(
            result.joint_action,
            torch.tensor([[[0, 1]]]),
        )
        torch.testing.assert_close(
            result.weights,
            torch.tensor([[[0.5, 0.5]]]),
        )

    def test_joint_probabilities_have_factorized_semantics(self) -> None:
        topology = fixed_ring_topology(9)
        source_cell = torch.tensor([[3]])
        source_node = torch.tensor([[0]])
        cell_logits = torch.tensor([[[0.0, 1.0, -1.0, 2.0, -2.0]]])
        node_logits = torch.tensor([[[2.0, -1.0, 100.0]]])

        result = route_local_actions(
            topology=topology,
            source_cell=source_cell,
            source_node=source_node,
            cell_logits=cell_logits,
            node_logits=node_logits,
            top_k=2,
            temperature=0.5,
        )

        expected_cell = torch.softmax(cell_logits / 0.5, dim=-1)
        expected_node = torch.softmax(
            torch.tensor([[[4.0, -2.0, -torch.inf]]]),
            dim=-1,
        )
        expected_joint = (
            expected_cell.unsqueeze(-1) * expected_node.unsqueeze(-2)
        )
        torch.testing.assert_close(
            result.cell_probabilities,
            expected_cell,
        )
        torch.testing.assert_close(
            result.node_probabilities,
            expected_node,
        )
        torch.testing.assert_close(
            result.joint_probabilities,
            expected_joint,
        )
        torch.testing.assert_close(
            result.joint_probabilities.sum(dim=(-2, -1)),
            torch.ones(1, 1),
        )

    def test_hard_top_one_is_one_forward_with_soft_head_gradients(
        self,
    ) -> None:
        torch.manual_seed(7)
        topology = fixed_ring_topology(11)
        cell_logits = torch.randn(2, 3, 5, requires_grad=True)
        node_logits = torch.randn(2, 3, 3, requires_grad=True)

        result = route_local_actions(
            topology=topology,
            source_cell=torch.tensor([[0, 5, 10], [2, 4, 6]]),
            source_node=torch.tensor([[0, 1, 2], [3, 0, 1]]),
            cell_logits=cell_logits,
            node_logits=node_logits,
            straight_through_hard=True,
        )

        torch.testing.assert_close(
            result.weights,
            torch.ones_like(result.weights),
        )
        result.weights.sum().backward()
        self.assertIsNotNone(cell_logits.grad)
        self.assertIsNotNone(node_logits.grad)
        self.assertGreater(float(cell_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(node_logits.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(cell_logits.grad).all())
        self.assertTrue(torch.isfinite(node_logits.grad).all())

    def test_action_tensor_sizes_do_not_depend_on_global_cell_count(
        self,
    ) -> None:
        cell_logits, node_logits = self._zeros(batch=2, slots=4)
        source_node = torch.zeros(2, 4, dtype=torch.long)
        shapes = []
        for num_cells in (7, 101):
            result = route_local_actions(
                topology=fixed_ring_topology(num_cells),
                source_cell=torch.full(
                    (2, 4),
                    num_cells - 1,
                    dtype=torch.long,
                ),
                source_node=source_node,
                cell_logits=cell_logits,
                node_logits=node_logits,
                top_k=2,
            )
            shapes.append(
                (
                    tuple(result.cell_probabilities.shape),
                    tuple(result.node_probabilities.shape),
                    tuple(result.joint_probabilities.shape),
                    tuple(result.destination_cell.shape),
                )
            )

        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(
            shapes[0],
            ((2, 4, 5), (2, 4, 3), (2, 4, 5, 3), (2, 4, 2)),
        )

    def test_validation_rejects_bad_shapes_ranges_and_controls(self) -> None:
        topology = fixed_ring_topology(5)
        cell_logits, node_logits = self._zeros()
        kwargs = {
            "topology": topology,
            "source_cell": torch.tensor([[0]]),
            "source_node": torch.tensor([[0]]),
            "cell_logits": cell_logits,
            "node_logits": node_logits,
        }

        with self.assertRaisesRegex(ValueError, "cell_logits"):
            route_local_actions(
                **{**kwargs, "cell_logits": torch.zeros(1, 1, 6)}
            )
        with self.assertRaisesRegex(ValueError, "source_cell values"):
            route_local_actions(
                **{**kwargs, "source_cell": torch.tensor([[5]])}
            )
        with self.assertRaisesRegex(ValueError, "source_node values"):
            route_local_actions(
                **{**kwargs, "source_node": torch.tensor([[4]])}
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            route_local_actions(**kwargs, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "top_k exceeds"):
            route_local_actions(**kwargs, top_k=11)
        with self.assertRaisesRegex(TypeError, "cell_destination_table"):
            route_local_actions(
                **kwargs,
                cell_destination_table=torch.zeros(5, 5),
            )
        with self.assertRaisesRegex(ValueError, "invalid cell"):
            invalid_table = torch.tensor(
                topology.cell_destination_table,
                dtype=torch.long,
            )
            invalid_table[0, 0] = 5
            route_local_actions(
                **kwargs,
                cell_destination_table=invalid_table,
            )


if __name__ == "__main__":
    unittest.main()
