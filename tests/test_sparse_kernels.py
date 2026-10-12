from __future__ import annotations

import unittest

import torch

from gator.models.sparse_kernels import (
    bucketed_cell_node_linear,
    bucketed_group_linear,
    build_packed_group_plan,
)
from gator.models.stewardship import (
    PackedStewardship,
    adaptive_nucleus_stewardship,
    bucketed_packed_receiver_context,
    pack_stewardship,
    packed_ownership,
    packed_readout_hidden,
    packed_route_uncertainty,
    padded_packed_receiver_context,
    prune_packed_stewardship,
    transport_packed_stewardship,
)


class PackedGroupPlanTests(unittest.TestCase):
    def test_bucket_plan_covers_each_row_once_at_boundaries(self):
        lengths = [1, 2, 3, 4, 5, 8, 9]
        keys = torch.cat(
            [
                torch.full((length,), key, dtype=torch.long)
                for key, length in enumerate(lengths)
            ]
        )
        permutation = torch.randperm(keys.numel())
        keys = keys[permutation]
        plan = build_packed_group_plan(
            keys,
            maximum_group_size=9,
        )

        seen = torch.zeros_like(keys)
        key_capacity = {}
        for bucket in plan.buckets:
            seen = seen.index_add(
                0,
                bucket.row_indices.reshape(-1),
                bucket.valid.to(torch.long).reshape(-1),
            )
            for group_key, count in zip(
                bucket.group_keys.tolist(),
                bucket.counts.tolist(),
            ):
                key_capacity[group_key] = (count, bucket.capacity)
        self.assertTrue(torch.equal(seen, torch.ones_like(seen)))
        self.assertEqual(
            [key_capacity[index][0] for index in range(len(lengths))],
            lengths,
        )
        self.assertEqual(
            [key_capacity[index][1] for index in range(len(lengths))],
            [1, 2, 4, 4, 8, 8, 9],
        )

    def test_bucketed_group_linear_matches_row_gather_and_gradients(self):
        torch.manual_seed(4)
        keys = torch.tensor([3, 1, 3, 0, 1, 3, 2], dtype=torch.long)
        plan = build_packed_group_plan(
            keys,
            maximum_group_size=7,
        )
        base = (
            torch.randn(7, 5, dtype=torch.float64),
            torch.randn(5, 6, 5, dtype=torch.float64),
            torch.randn(5, 6, dtype=torch.float64),
        )
        reference_operands = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        bucketed_operands = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        reference = (
            torch.einsum(
                "mi,moi->mo",
                reference_operands[0],
                reference_operands[1][keys],
            )
            + reference_operands[2][keys]
        )
        actual = bucketed_group_linear(
            *bucketed_operands,
            plan,
        )
        torch.testing.assert_close(actual, reference)
        probe = torch.randn_like(reference)
        expected_gradients = torch.autograd.grad(
            (reference * probe).sum(), reference_operands
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(), bucketed_operands
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient
            )
        self.assertEqual(
            float(actual_gradients[1][4].abs().sum()), 0.0
        )
        self.assertGreater(
            float(actual_gradients[1][:4].abs().sum()), 0.0
        )

    def test_bucket_plan_rejects_invalid_bounds_and_keys(self):
        with self.assertRaisesRegex(
            RuntimeError, "maximum_group_size"
        ):
            build_packed_group_plan(
                torch.tensor([0, 0, 0], dtype=torch.long),
                maximum_group_size=2,
            )
        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            build_packed_group_plan(
                torch.tensor([0, -1], dtype=torch.long),
                maximum_group_size=2,
            )

    def test_bucketed_cell_nodes_skip_inactive_parameter_slices(self):
        torch.manual_seed(41)
        keys = torch.tensor([1, 3, 1, 3, 1], dtype=torch.long)
        plan = build_packed_group_plan(
            keys,
            maximum_group_size=5,
        )
        base = (
            torch.randn(5, 4, dtype=torch.float64),
            torch.randn(5, 3, 4, dtype=torch.float64),
            torch.randn(5, 3, 4, 4, dtype=torch.float64),
            torch.randn(5, 3, 4, dtype=torch.float64),
        )
        reference_operands = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        bucketed_operands = tuple(
            value.clone().requires_grad_(True) for value in base
        )
        node_input = (
            reference_operands[0][:, None]
            + reference_operands[1][keys]
        )
        reference = (
            torch.einsum(
                "mni,mnoi->mno",
                node_input,
                reference_operands[2][keys],
            )
            + reference_operands[3][keys]
        )
        actual = bucketed_cell_node_linear(
            *bucketed_operands,
            plan,
        )
        torch.testing.assert_close(actual, reference)
        probe = torch.randn_like(reference)
        expected_gradients = torch.autograd.grad(
            (reference * probe).sum(), reference_operands
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(), bucketed_operands
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
            )
        for parameter_gradient in actual_gradients[1:]:
            self.assertEqual(
                float(parameter_gradient[[0, 2, 4]].abs().sum()),
                0.0,
            )


class BucketedPackedPrimitiveTests(unittest.TestCase):
    def _irregular_packed(self, *, dtype=torch.float64):
        torch.manual_seed(8)
        batch_size, cells, tokens, hidden, nodes = 2, 4, 6, 3, 3
        ownership = torch.zeros(
            batch_size, cells, tokens, dtype=dtype
        )
        ownership[0, 0, :6] = 1.0
        ownership[0, 1, :3] = 1.0
        ownership[0, 2, :1] = 1.0
        ownership[1, 0, :5] = 1.0
        ownership[1, 2, :2] = 1.0
        ownership[1, 3, :4] = 1.0
        states = torch.randn(
            batch_size, cells, tokens, hidden, dtype=dtype
        )
        nodes_value = torch.softmax(
            torch.randn(
                batch_size, cells, tokens, nodes, dtype=dtype
            ),
            -1,
        )
        return (
            pack_stewardship(ownership, states, nodes_value),
            batch_size,
            cells,
            tokens,
            hidden,
            nodes,
        )

    def test_packed_record_and_transport_reject_invalid_structure(self):
        with self.assertRaisesRegex(ValueError, "leading row count"):
            PackedStewardship(
                batch=torch.tensor([0]),
                cell=torch.tensor([0]),
                token=torch.tensor([0]),
                mass=torch.ones(1),
                state=torch.zeros(2, 3),
                node=torch.ones(1, 2),
            )
        invalid = PackedStewardship(
            batch=torch.tensor([0]),
            cell=torch.tensor([0]),
            token=torch.tensor([2]),
            mass=torch.ones(1),
            state=torch.zeros(1, 3),
            node=torch.ones(1, 2),
        )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            transport_packed_stewardship(
                invalid,
                torch.ones(1, 1),
                torch.tensor([[0]]),
                batch_size=1,
                cells=1,
                tokens=2,
            )

    def test_bucketed_receiver_matches_global_padding_and_gradients(self):
        (
            base_packed,
            _,
            cells,
            tokens,
            hidden,
            nodes,
        ) = self._irregular_packed()
        write_base = torch.tensor(
            [
                [1.0, 1.0, 0.5, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        relative = torch.randn(
            tokens, tokens, 4, dtype=torch.float64
        )
        bias_base = torch.randn(
            cells, nodes, 4, dtype=torch.float64
        )

        def operands():
            state = base_packed.state.detach().clone().requires_grad_(True)
            node = base_packed.node.detach().clone().requires_grad_(True)
            mass = base_packed.mass.detach().clone().requires_grad_(True)
            write = write_base.clone().requires_grad_(True)
            bias = bias_base.clone().requires_grad_(True)
            return state, node, mass, write, bias

        legacy_values = operands()
        bucketed_values = operands()

        def make_packed(values):
            return PackedStewardship(
                batch=base_packed.batch,
                cell=base_packed.cell,
                token=base_packed.token,
                mass=values[2],
                state=values[0],
                node=values[1],
            )

        expected = padded_packed_receiver_context(
            make_packed(legacy_values),
            write=legacy_values[3],
            relative_code=relative,
            relative_node_bias=legacy_values[4],
            cells=cells,
        )
        actual = bucketed_packed_receiver_context(
            make_packed(bucketed_values),
            write=bucketed_values[3],
            relative_code=relative,
            relative_node_bias=bucketed_values[4],
            cells=cells,
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertTrue(torch.isfinite(actual[0]).all())
        probe_context = torch.randn(
            base_packed.size, hidden, dtype=torch.float64
        )
        probe_traffic = torch.randn(
            base_packed.size, nodes, dtype=torch.float64
        )
        expected_gradients = torch.autograd.grad(
            (expected[0] * probe_context).sum()
            + (expected[1] * probe_traffic).sum(),
            legacy_values,
        )
        actual_gradients = torch.autograd.grad(
            (actual[0] * probe_context).sum()
            + (actual[1] * probe_traffic).sum(),
            bucketed_values,
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1e-10,
                atol=1e-11,
            )

    def test_packed_nucleus_matches_dense_nucleus(self):
        torch.manual_seed(12)
        batch_size, cells, tokens = 2, 6, 3
        ownership = torch.rand(
            batch_size, cells, tokens, dtype=torch.float64
        )
        support = torch.rand(batch_size, cells, tokens).gt(0.35)
        support[:, 0] = True
        ownership = ownership * support
        ownership = ownership / ownership.sum(1, keepdim=True)
        states = torch.randn(batch_size, cells, tokens, 2)
        nodes = torch.softmax(
            torch.randn(batch_size, cells, tokens, 2), -1
        )
        packed = pack_stewardship(ownership, states, nodes)
        keep_mass = torch.tensor(
            [[0.75, 0.9, 0.6], [0.8, 0.7, 0.95]],
            dtype=torch.float64,
        )
        groups = torch.tensor([0, 1, 0, 1, 0, 1])
        dense_input = ownership.clone().requires_grad_(True)
        packed_input = PackedStewardship(
            batch=packed.batch,
            cell=packed.cell,
            token=packed.token,
            mass=packed.mass.clone().requires_grad_(True),
            state=packed.state,
            node=packed.node,
        )
        expected, _ = adaptive_nucleus_stewardship(
            dense_input,
            keep_mass=keep_mass,
            min_active=1,
            max_active=4,
            cell_groups=groups,
            cell_group_count=2,
            scouts=1,
            exploration_mass=0.02,
            straight_through=False,
        )
        routed, _ = prune_packed_stewardship(
            packed_input,
            keep_mass=keep_mass,
            batch_size=batch_size,
            cells=cells,
            tokens=tokens,
            min_active=1,
            max_active=4,
            cell_groups=groups,
            cell_group_count=2,
            scouts=1,
            exploration_mass=0.02,
            straight_through=False,
        )
        actual = packed_ownership(
            routed,
            batch_size=batch_size,
            cells=cells,
            tokens=tokens,
        )
        torch.testing.assert_close(actual, expected)
        probe = torch.randn_like(expected)
        expected_gradient = torch.autograd.grad(
            (expected * probe).sum(), dense_input
        )[0][packed.batch, packed.cell, packed.token]
        actual_gradient = torch.autograd.grad(
            (actual * probe).sum(), packed_input.mass
        )[0]
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-10,
            atol=1e-11,
        )

    def test_packed_pruning_intentionally_drops_dead_route_gradients(self):
        ownership = torch.tensor(
            [[[0.4], [0.3], [0.2], [0.1]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        states = torch.zeros(1, 4, 1, 1, dtype=torch.float64)
        nodes = torch.ones(1, 4, 1, 1, dtype=torch.float64)
        packed_base = pack_stewardship(ownership, states, nodes)
        packed_mass = packed_base.mass.detach().clone().requires_grad_(True)
        packed = PackedStewardship(
            batch=packed_base.batch,
            cell=packed_base.cell,
            token=packed_base.token,
            mass=packed_mass,
            state=packed_base.state,
            node=packed_base.node,
        )
        dense, _ = adaptive_nucleus_stewardship(
            ownership,
            keep_mass=0.6,
            min_active=1,
            max_active=2,
            scouts=0,
            exploration_mass=0.0,
            straight_through=True,
        )
        routed, keep = prune_packed_stewardship(
            packed,
            keep_mass=0.6,
            batch_size=1,
            cells=4,
            tokens=1,
            min_active=1,
            max_active=2,
            scouts=0,
            exploration_mass=0.0,
            straight_through=True,
        )
        sparse = packed_ownership(
            routed,
            batch_size=1,
            cells=4,
            tokens=1,
        )
        torch.testing.assert_close(sparse, dense)
        probe = torch.tensor(
            [[[1.0], [2.0], [3.0], [4.0]]],
            dtype=torch.float64,
        )
        dense_gradient = torch.autograd.grad(
            (dense * probe).sum(),
            ownership,
        )[0]
        sparse_gradient = torch.autograd.grad(
            (sparse * probe).sum(),
            packed_mass,
        )[0]
        self.assertEqual(keep.tolist(), [True, True, False, False])
        torch.testing.assert_close(
            sparse_gradient[:2],
            dense_gradient[0, :2, 0],
        )
        torch.testing.assert_close(
            sparse_gradient[2:],
            torch.zeros(2, dtype=torch.float64),
        )
        self.assertGreater(
            float(dense_gradient[0, 2:, 0].abs().sum()),
            0.0,
        )

    def test_packed_uncertainty_and_readout_match_dense(self):
        torch.manual_seed(21)
        batch_size, cells, tokens, hidden = 2, 4, 5, 3
        ownership = torch.rand(batch_size, cells, tokens)
        ownership[:, :, 0] = 0.0
        ownership[:, 0, 1:] += 1.0
        ownership[:, :, 1:] = ownership[:, :, 1:] / ownership[
            :, :, 1:
        ].sum(1, keepdim=True)
        states = torch.randn(batch_size, cells, tokens, hidden)
        nodes = torch.softmax(
            torch.randn(batch_size, cells, tokens, 2), -1
        )
        packed = pack_stewardship(ownership, states, nodes)
        entropy_dense = torch.rand(batch_size, cells, tokens)
        entropy_rows = entropy_dense[
            packed.batch, packed.cell, packed.token
        ]
        valid = torch.ones(batch_size, tokens, dtype=torch.bool)
        valid[:, 0] = False
        maximum, mean = packed_route_uncertainty(
            packed,
            entropy_rows,
            batch_size=batch_size,
            tokens=tokens,
            route_valid=valid,
        )
        expected_maximum = entropy_dense.masked_fill(
            ~ownership.gt(0), -torch.inf
        ).amax(1)
        expected_maximum = torch.where(
            valid, expected_maximum, torch.zeros_like(expected_maximum)
        )
        expected_mean = torch.where(
            valid,
            (ownership * entropy_dense).sum(1),
            torch.zeros_like(mean),
        )
        torch.testing.assert_close(maximum, expected_maximum)
        torch.testing.assert_close(mean, expected_mean)

        context_tokens, output_tokens = 2, 3
        address = torch.randn(cells, tokens, hidden)
        target_state = states[:, :, context_tokens:] + address[
            None, :, context_tokens:
        ]
        expected_shared = (
            ownership[:, :, context_tokens:, None] * target_state
        ).sum(1)
        actual_shared = packed_readout_hidden(
            packed,
            address,
            batch_size=batch_size,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
        )
        torch.testing.assert_close(actual_shared, expected_shared)

        cell_groups = torch.tensor([0, 1, 0, 1])
        expected_groups = []
        for group in range(2):
            selected = cell_groups.eq(group)
            mass = ownership[:, selected, context_tokens:]
            weights = mass / mass.sum(1, keepdim=True).clamp_min(1e-9)
            expected_groups.append(
                (
                    weights[..., None]
                    * target_state[:, selected]
                ).sum(1)
            )
        expected_grouped = torch.cat(expected_groups, -1)
        actual_grouped = packed_readout_hidden(
            packed,
            address,
            batch_size=batch_size,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            cell_groups=cell_groups,
            cell_group_count=2,
        )
        torch.testing.assert_close(actual_grouped, expected_grouped)


if __name__ == "__main__":
    unittest.main()
