from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from gator_path.transforms import SelectedPathTransform


class SelectedPathTransformTests(unittest.TestCase):
    def _module(
        self,
        *,
        cells: int = 5,
        nodes: int = 4,
        width: int = 3,
        address_width: int = 2,
    ) -> SelectedPathTransform:
        torch.manual_seed(11)
        return SelectedPathTransform(
            num_cells=cells,
            num_nodes=nodes,
            payload_width=width,
            address_width=address_width,
            num_cell_actions=5,
        )

    def test_selected_vectorization_matches_scalar_oracle(self) -> None:
        module = self._module()
        payload = torch.randn(2, 3, 3)
        mailbox = torch.randn_like(payload)
        address = torch.randn(2, 3, 2)
        destination_cell = torch.tensor([[0, 2, 4], [1, 3, 0]])
        source_node = torch.tensor([[0, 1, 2], [3, 0, 1]])
        destination_node = torch.tensor([[1, 3, 0], [0, 1, 2]])
        cell_action = torch.tensor([[0, 2, 4], [1, 3, 0]])

        actual = module(
            payload,
            mailbox,
            address,
            destination_cell,
            source_node,
            destination_node,
            cell_action,
        )

        expected = torch.empty_like(payload)
        for batch in range(2):
            for slot in range(3):
                cell = int(destination_cell[batch, slot])
                source = int(source_node[batch, slot])
                node = int(destination_node[batch, slot])
                action = int(cell_action[batch, slot])
                base = (
                    payload[batch, slot]
                    + F.linear(
                        mailbox[batch, slot], module.mailbox_weight
                    )
                    + F.linear(
                        address[batch, slot], module.address_weight
                    )
                    + module.cell_action_embedding[action]
                    + module.node_edge_embedding[source, node]
                )
                expected[batch, slot] = torch.tanh(
                    F.linear(
                        base,
                        module.weight[cell, node],
                        module.bias[cell, node],
                    )
                )

        torch.testing.assert_close(actual, expected)

    def test_topk_prefix_remains_compact_and_cell_count_independent(self) -> None:
        payload = torch.randn(2, 7, 2, 3)
        mailbox = torch.randn_like(payload)
        address = torch.randn(2, 7, 2, 2)
        source_node = torch.zeros(2, 7, 2, dtype=torch.long)
        destination_node = torch.ones(2, 7, 2, dtype=torch.long)
        cell_action = torch.zeros(2, 7, 2, dtype=torch.long)

        small = self._module(cells=5)
        large = self._module(cells=101)
        small_output = small(
            payload,
            mailbox,
            address,
            torch.zeros(2, 7, 2, dtype=torch.long),
            source_node,
            destination_node,
            cell_action,
        )
        large_output = large(
            payload,
            mailbox,
            address,
            torch.full((2, 7, 2), 100, dtype=torch.long),
            source_node,
            destination_node,
            cell_action,
        )

        self.assertEqual(tuple(small_output.shape), (2, 7, 2, 3))
        self.assertEqual(tuple(large_output.shape), (2, 7, 2, 3))
        self.assertNotIn(101, large_output.shape)

    def test_only_selected_private_parameters_receive_gradients(self) -> None:
        module = self._module(cells=4)
        payload = torch.randn(1, 2, 3, requires_grad=True)
        mailbox = torch.randn_like(payload)
        address = torch.randn(1, 2, 2)
        destination_cell = torch.tensor([[0, 2]])
        source_node = torch.tensor([[0, 1]])
        destination_node = torch.tensor([[1, 3]])
        cell_action = torch.tensor([[0, 2]])

        output = module(
            payload,
            mailbox,
            address,
            destination_cell,
            source_node,
            destination_node,
            cell_action,
        )
        output.square().sum().backward()

        self.assertGreater(
            float(module.weight.grad[0, 1].abs().sum()), 0.0
        )
        self.assertGreater(
            float(module.weight.grad[2, 3].abs().sum()), 0.0
        )
        unselected = module.weight.grad.clone()
        unselected[0, 1] = 0
        unselected[2, 3] = 0
        self.assertEqual(float(unselected.abs().sum()), 0.0)
        self.assertGreater(
            float(module.bias.grad[0, 1].abs().sum()), 0.0
        )
        self.assertGreater(
            float(module.bias.grad[2, 3].abs().sum()), 0.0
        )
        unselected_bias = module.bias.grad.clone()
        unselected_bias[0, 1] = 0
        unselected_bias[2, 3] = 0
        self.assertEqual(float(unselected_bias.abs().sum()), 0.0)
        for selected_action in (0, 2):
            self.assertGreater(
                float(
                    module.cell_action_embedding.grad[
                        selected_action
                    ].abs().sum()
                ),
                0.0,
            )
        action_grad = module.cell_action_embedding.grad.clone()
        action_grad[0] = 0
        action_grad[2] = 0
        self.assertEqual(float(action_grad.abs().sum()), 0.0)
        for source, destination in ((0, 1), (1, 3)):
            self.assertGreater(
                float(
                    module.node_edge_embedding.grad[
                        source, destination
                    ].abs().sum()
                ),
                0.0,
            )
        edge_grad = module.node_edge_embedding.grad.clone()
        edge_grad[0, 1] = 0
        edge_grad[1, 3] = 0
        self.assertEqual(float(edge_grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(payload.grad).all())

    def test_topk_values_and_input_gradients_match_branch_oracle(self) -> None:
        module = self._module(cells=6)
        payload = torch.randn(1, 2, 2, 3, requires_grad=True)
        mailbox = torch.randn(
            1, 2, 2, 3, requires_grad=True
        )
        address = torch.randn(1, 2, 2, 2, requires_grad=True)
        destination_cell = torch.tensor([[[0, 1], [4, 5]]])
        source_node = torch.tensor([[[0, 1], [2, 3]]])
        destination_node = torch.tensor([[[1, 3], [0, 1]]])
        cell_action = torch.tensor([[[0, 1], [3, 4]]])

        actual = module(
            payload,
            mailbox,
            address,
            destination_cell,
            source_node,
            destination_node,
            cell_action,
        )
        expected = torch.empty_like(actual)
        for slot in range(2):
            for branch in range(2):
                cell = int(destination_cell[0, slot, branch])
                source = int(source_node[0, slot, branch])
                node = int(destination_node[0, slot, branch])
                action = int(cell_action[0, slot, branch])
                base = (
                    payload[0, slot, branch]
                    + F.linear(
                        mailbox[0, slot, branch],
                        module.mailbox_weight,
                    )
                    + F.linear(
                        address[0, slot, branch],
                        module.address_weight,
                    )
                    + module.cell_action_embedding[action]
                    + module.node_edge_embedding[source, node]
                )
                expected[0, slot, branch] = torch.tanh(
                    F.linear(
                        base,
                        module.weight[cell, node],
                        module.bias[cell, node],
                    )
                )

        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        for value in (payload, mailbox, address):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.0)

    def test_invalid_selection_or_shape_is_rejected(self) -> None:
        module = self._module()
        payload = torch.randn(1, 2, 3)
        mailbox = torch.randn_like(payload)
        address = torch.randn(1, 2, 2)
        valid = torch.zeros(1, 2, dtype=torch.long)

        with self.assertRaisesRegex(AssertionError, "destination_cell"):
            module(
                payload,
                mailbox,
                address,
                torch.tensor([[0, -1]]),
                valid,
                valid,
                valid,
            )
        with self.assertRaisesRegex(ValueError, "mailbox"):
            module(
                payload,
                mailbox[:, :1],
                address,
                valid,
                valid,
                valid,
                valid,
            )
        with self.assertRaisesRegex(TypeError, "torch.long"):
            module(
                payload,
                mailbox,
                address,
                valid.float(),
                valid,
                valid,
                valid,
            )


if __name__ == "__main__":
    unittest.main()
