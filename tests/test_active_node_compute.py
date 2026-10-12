from __future__ import annotations

import copy
import unittest

import torch

from gator.models.active_node_compute import (
    SharedTrunkNodeAdapters,
    full_private_parameter_count,
    full_private_sparse_macs,
    shared_trunk_adapter_macs,
    shared_trunk_adapter_parameter_count,
)


def packed_to_dense_mass(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    *,
    n_nodes: int,
) -> torch.Tensor:
    dense = mass.new_zeros(mass.shape[:-1] + (n_nodes,))
    return dense.scatter_add(
        -1,
        nodes,
        torch.where(valid, mass, torch.zeros_like(mass)),
    )


class SharedTrunkNodeAdaptersTests(unittest.TestCase):
    def test_parameter_and_compute_formulas(self) -> None:
        module = SharedTrunkNodeAdapters(
            n_nodes=17,
            width=12,
            adapter_rank=3,
        )
        actual = sum(parameter.numel() for parameter in module.parameters())
        expected = 12 * 12 + 12 + 17 * (2 * 12 + 2 * 12 * 3)
        self.assertEqual(module.exact_parameter_count, expected)
        self.assertEqual(actual, expected)
        self.assertEqual(
            shared_trunk_adapter_parameter_count(
                n_nodes=17,
                width=12,
                adapter_rank=3,
            ),
            expected,
        )
        self.assertEqual(
            shared_trunk_adapter_macs(
                packet_count=40,
                active_nodes_per_packet=2,
                width=12,
                adapter_rank=3,
            ),
            40 * 12 * 12 + 40 * 2 * 2 * 12 * 3,
        )
        self.assertEqual(
            full_private_parameter_count(n_nodes=17, width=12),
            17 * (12 * 12 + 12),
        )
        self.assertEqual(
            full_private_sparse_macs(
                packet_count=40,
                active_nodes_per_packet=2,
                width=12,
            ),
            40 * 2 * 12 * 12,
        )

    def test_grouped_execution_dtype_contract_is_explicit(self) -> None:
        self.assertEqual(
            SharedTrunkNodeAdapters._grouped_execution_dtype(torch.float32),
            torch.bfloat16,
        )
        self.assertEqual(
            SharedTrunkNodeAdapters._grouped_execution_dtype(torch.bfloat16),
            torch.bfloat16,
        )
        self.assertEqual(
            SharedTrunkNodeAdapters._grouped_execution_dtype(torch.float16),
            torch.float16,
        )

    def test_packed_matches_dense_values_and_all_gradients(self) -> None:
        torch.manual_seed(4)
        packed_module = SharedTrunkNodeAdapters(
            n_nodes=7,
            width=5,
            adapter_rank=2,
            adapter_scale=0.7,
        ).double()
        dense_module = copy.deepcopy(packed_module)

        packed_base = torch.randn(2, 3, 2, 5, dtype=torch.double)
        packed_base.requires_grad_(True)
        dense_base = packed_base.detach().clone().requires_grad_(True)
        # Entries are unique per packet so dense gradients with respect to
        # node mass map one-to-one back to the packed representation.
        nodes = torch.tensor(
            [
                [
                    [[0, 3, 5], [1, 2, 6]],
                    [[1, 4, 6], [0, 2, 5]],
                    [[0, 2, 4], [1, 3, 6]],
                ],
                [
                    [[1, 3, 5], [0, 2, 4]],
                    [[0, 4, 6], [1, 2, 5]],
                    [[2, 3, 6], [0, 1, 5]],
                ],
            ],
            dtype=torch.long,
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, 1, 1, 2] = False
        valid[1, 2, 0, 1] = False
        packed_mass = torch.rand(nodes.shape, dtype=torch.double)
        packed_mass.requires_grad_(True)
        dense_mass = packed_to_dense_mass(
            nodes,
            packed_mass.detach(),
            valid,
            n_nodes=7,
        ).requires_grad_(True)

        packed = packed_module(packed_base, nodes, packed_mass, valid)
        dense = dense_module.dense_reference(dense_base, dense_mass)
        torch.testing.assert_close(packed, dense, rtol=1e-11, atol=1e-11)

        weight = torch.randn_like(packed)
        (packed * weight).sum().backward()
        (dense * weight).sum().backward()
        torch.testing.assert_close(
            packed_base.grad,
            dense_base.grad,
            rtol=1e-10,
            atol=1e-10,
        )
        dense_mass_grad = dense_mass.grad.gather(-1, nodes)
        dense_mass_grad = torch.where(
            valid,
            dense_mass_grad,
            torch.zeros_like(dense_mass_grad),
        )
        torch.testing.assert_close(
            packed_mass.grad,
            dense_mass_grad,
            rtol=1e-10,
            atol=1e-10,
        )
        for (packed_name, packed_parameter), (
            dense_name,
            dense_parameter,
        ) in zip(
            packed_module.named_parameters(),
            dense_module.named_parameters(),
            strict=True,
        ):
            self.assertEqual(packed_name, dense_name)
            torch.testing.assert_close(
                packed_parameter.grad,
                dense_parameter.grad,
                rtol=1e-10,
                atol=1e-10,
                msg=lambda message, name=packed_name: f"{name}: {message}",
            )

    def test_only_selected_nodes_receive_private_gradients(self) -> None:
        torch.manual_seed(5)
        module = SharedTrunkNodeAdapters(
            n_nodes=8,
            width=6,
            adapter_rank=2,
        )
        base = torch.randn(2, 2, 3, 6, requires_grad=True)
        nodes = torch.tensor([1, 5], dtype=torch.long).view(1, 1, 1, 2)
        nodes = nodes.expand(2, 2, 3, -1)
        mass = torch.rand(2, 2, 3, 2)
        module(base, nodes, mass).square().mean().backward()

        selected = {1, 5}
        for name in (
            "node_embedding",
            "adapter_down",
            "adapter_up",
            "node_bias",
        ):
            gradient = getattr(module, name).grad
            self.assertIsNotNone(gradient)
            for node in range(module.n_nodes):
                if node in selected:
                    self.assertGreater(float(gradient[node].abs().sum()), 0.0)
                else:
                    self.assertEqual(float(gradient[node].abs().sum()), 0.0)

    def test_invalid_entries_contribute_zero(self) -> None:
        torch.manual_seed(6)
        module = SharedTrunkNodeAdapters(
            n_nodes=5,
            width=4,
            adapter_rank=2,
        )
        base = torch.randn(1, 2, 1, 4)
        nodes = torch.tensor([[[[1, 3]], [[2, 4]]]])
        mass = torch.tensor([[[[0.25, 99.0]], [[0.8, 101.0]]]])
        valid = torch.tensor([[[[True, False]], [[True, False]]]])
        result = module(base, nodes, mass, valid)
        expected = module(base, nodes[..., :1], mass[..., :1])
        torch.testing.assert_close(result, expected)

    def test_duplicate_routes_preserve_weighted_node_semantics(self) -> None:
        torch.manual_seed(61)
        packed_module = SharedTrunkNodeAdapters(
            n_nodes=6,
            width=5,
            adapter_rank=3,
        ).double()
        dense_module = copy.deepcopy(packed_module)
        packed_base = torch.randn(
            1,
            3,
            2,
            5,
            dtype=torch.double,
            requires_grad=True,
        )
        dense_base = packed_base.detach().clone().requires_grad_(True)
        nodes = torch.tensor(
            [
                [
                    [[1, 1, 4], [2, 5, 2]],
                    [[3, 0, 3], [4, 4, 1]],
                    [[5, 2, 5], [0, 3, 0]],
                ]
            ]
        )
        packed_mass = torch.rand(
            nodes.shape,
            dtype=torch.double,
            requires_grad=True,
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        dense_mass = packed_to_dense_mass(
            nodes,
            packed_mass.detach(),
            valid,
            n_nodes=packed_module.n_nodes,
        ).requires_grad_(True)

        packed = packed_module(packed_base, nodes, packed_mass, valid)
        dense = dense_module.dense_reference(dense_base, dense_mass)
        torch.testing.assert_close(packed, dense, rtol=1e-11, atol=1e-11)

        probe = torch.randn_like(packed)
        (packed * probe).sum().backward()
        (dense * probe).sum().backward()
        torch.testing.assert_close(
            packed_base.grad,
            dense_base.grad,
            rtol=1e-10,
            atol=1e-10,
        )
        expected_mass_gradient = dense_mass.grad.gather(-1, nodes)
        torch.testing.assert_close(
            packed_mass.grad,
            expected_mass_gradient,
            rtol=1e-10,
            atol=1e-10,
        )
        dense_parameters = dict(dense_module.named_parameters())
        for name, parameter in packed_module.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                dense_parameters[name].grad,
                rtol=1e-10,
                atol=1e-10,
                msg=lambda message, parameter_name=name: (
                    f"{parameter_name}: {message}"
                ),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_grouped_dispatch_matches_dense_values_and_gradients(
        self,
    ) -> None:
        if (
            not hasattr(torch, "_grouped_mm")
            or torch.cuda.get_device_capability() < (8, 0)
        ):
            self.skipTest("CUDA grouped MM requires an Ampere-or-newer GPU")
        torch.manual_seed(62)
        packed_module = SharedTrunkNodeAdapters(
            n_nodes=9,
            width=10,
            adapter_rank=3,
            adapter_scale=0.4,
        ).cuda()
        self.assertTrue(packed_module._can_use_grouped_mm(packed_module.node_bias))
        dense_module = copy.deepcopy(packed_module)
        packed_base = torch.randn(
            2,
            3,
            2,
            10,
            device="cuda",
            requires_grad=True,
        )
        dense_base = packed_base.detach().clone().requires_grad_(True)
        nodes = torch.randint(
            0,
            9,
            (2, 3, 2, 4),
            device="cuda",
        )
        valid = torch.ones_like(nodes, dtype=torch.bool)
        valid[0, 1, 1, -1] = False
        packed_mass = torch.rand(
            nodes.shape,
            device="cuda",
            requires_grad=True,
        )
        dense_mass = packed_to_dense_mass(
            nodes,
            packed_mass.detach(),
            valid,
            n_nodes=packed_module.n_nodes,
        ).requires_grad_(True)

        packed = packed_module(packed_base, nodes, packed_mass, valid)
        dense = dense_module.dense_reference(dense_base, dense_mass)
        # FP32 is the master/residual dtype, but the two grouped adapter GEMMs
        # deliberately use the frozen Blackwell BF16 execution boundary.
        torch.testing.assert_close(packed, dense, rtol=2e-2, atol=2e-2)
        probe = torch.randn_like(packed)
        (packed * probe).sum().backward()
        (dense * probe).sum().backward()
        torch.testing.assert_close(
            packed_base.grad,
            dense_base.grad,
            rtol=4e-2,
            atol=4e-2,
        )
        expected_mass_gradient = dense_mass.grad.gather(-1, nodes)
        expected_mass_gradient = torch.where(
            valid,
            expected_mass_gradient,
            torch.zeros_like(expected_mass_gradient),
        )
        torch.testing.assert_close(
            packed_mass.grad,
            expected_mass_gradient,
            rtol=4e-2,
            atol=4e-2,
        )
        dense_parameters = dict(dense_module.named_parameters())
        for name, parameter in packed_module.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                dense_parameters[name].grad,
                rtol=5e-2,
                atol=5e-2,
                msg=lambda message, parameter_name=name: (
                    f"{parameter_name}: {message}"
                ),
            )

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile unavailable")
    def test_fullgraph_compile_smoke(self) -> None:
        torch.manual_seed(7)
        module = SharedTrunkNodeAdapters(
            n_nodes=9,
            width=8,
            adapter_rank=2,
        )
        base = torch.randn(1, 3, 2, 8)
        nodes = torch.randint(0, 9, (1, 3, 2, 3))
        mass = torch.softmax(torch.randn(1, 3, 2, 3), dim=-1)
        valid = torch.ones_like(nodes, dtype=torch.bool)
        expected = module(base, nodes, mass, valid)
        compiled = torch.compile(module, backend="eager", fullgraph=True)
        actual = compiled(base, nodes, mass, valid)
        torch.testing.assert_close(actual, expected)

    def test_validation_rejects_invalid_formula_inputs(self) -> None:
        with self.assertRaises(ValueError):
            shared_trunk_adapter_parameter_count(
                n_nodes=0,
                width=8,
                adapter_rank=2,
            )
        with self.assertRaises(ValueError):
            shared_trunk_adapter_macs(
                packet_count=1,
                active_nodes_per_packet=0,
                width=8,
                adapter_rank=2,
            )


if __name__ == "__main__":
    unittest.main()
