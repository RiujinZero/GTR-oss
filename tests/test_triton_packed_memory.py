from __future__ import annotations

import unittest
from unittest import mock

import torch

import gator.models.triton_packed_memory as packed_reader_module
from gator.models.triton_packed_memory import (
    packed_reader_context,
    torch_packed_reader_context,
    triton_packed_reader_available,
)


def _inputs(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(913)
    batch, tokens, active_nodes, features = 2, 4, 5, 7
    lanes, routes = 3, 3
    numerator = torch.randn(
        batch,
        tokens,
        active_nodes,
        features,
        device=device,
        dtype=dtype,
    )
    denominator = torch.rand(
        batch,
        tokens,
        active_nodes,
        device=device,
        dtype=dtype,
    )
    denominator[0, 1, 2] = 1e-12
    denominator[1, 2, 3] = 1e-6
    reader_nodes = torch.randint(
        0,
        active_nodes,
        (batch, tokens, lanes, routes),
        device=device,
    )
    # Force collisions so the Triton backward must accumulate gradients.
    reader_nodes[..., 1] = reader_nodes[..., 0]
    reader_mass = torch.rand(
        batch,
        tokens,
        lanes,
        routes,
        device=device,
        dtype=dtype,
    )
    reader_mass[0, -1, 1, -1] = 0
    return numerator, denominator, reader_nodes, reader_mass


class TorchPackedReaderTests(unittest.TestCase):
    def test_torch_oracle_matches_direct_index_equation(self) -> None:
        numerator, denominator, nodes, mass = _inputs()
        epsilon = 1e-6
        actual = torch_packed_reader_context(
            numerator,
            denominator,
            nodes,
            mass,
            epsilon=epsilon,
        )
        expected = torch.zeros(
            numerator.shape[0],
            numerator.shape[1],
            nodes.shape[2],
            numerator.shape[-1],
            dtype=numerator.dtype,
        )
        for batch in range(numerator.shape[0]):
            for token in range(numerator.shape[1]):
                for lane in range(nodes.shape[2]):
                    for route in range(nodes.shape[3]):
                        node = int(nodes[batch, token, lane, route])
                        expected[batch, token, lane] += (
                            mass[batch, token, lane, route]
                            * numerator[batch, token, node]
                            / denominator[batch, token, node].clamp_min(epsilon)
                        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_torch_dispatch_preserves_all_gradients(self) -> None:
        numerator, denominator, nodes, mass = _inputs()
        numerator.requires_grad_(True)
        denominator.requires_grad_(True)
        mass.requires_grad_(True)
        direct = packed_reader_context(
            numerator,
            denominator,
            nodes,
            mass,
            epsilon=1e-6,
            backend="torch",
        )
        probe = torch.randn_like(direct)
        gradients = torch.autograd.grad(
            (direct * probe).sum(),
            (numerator, denominator, mass),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
        # Duplicate node IDs must accumulate into the same numerator row.
        duplicate_node = int(nodes[0, 0, 0, 0])
        self.assertGreater(
            float(gradients[0][0, 0, duplicate_node].abs().sum()),
            0.0,
        )

    def test_explicit_torch_adjoint_matches_autograd_oracle(self) -> None:
        numerator, denominator, nodes, mass = _inputs()
        for value in (numerator, denominator, mass):
            value.requires_grad_(True)
        output = torch_packed_reader_context(
            numerator,
            denominator,
            nodes,
            mass,
            epsilon=1e-6,
        )
        probe = torch.randn_like(output)
        expected = torch.autograd.grad(
            (output * probe).sum(),
            (numerator, denominator, mass),
        )
        actual = packed_reader_module._packed_reader_backward_unchecked(
            probe,
            numerator.detach(),
            denominator.detach(),
            nodes,
            mass.detach(),
            epsilon=1e-6,
            backend="torch",
        )
        for candidate, reference in zip(actual, expected):
            torch.testing.assert_close(
                candidate,
                reference,
                atol=1e-10,
                rtol=1e-10,
            )

    def test_trusted_internal_dispatch_skips_repeated_tensor_scans(self) -> None:
        inputs = _inputs()
        expected = torch_packed_reader_context(*inputs, epsilon=1e-6)
        with mock.patch.object(
            packed_reader_module,
            "_validate_inputs",
            side_effect=AssertionError("unexpected repeated validation"),
        ):
            actual = packed_reader_module._packed_reader_context_unchecked(
                *inputs,
                epsilon=1e-6,
                backend="torch",
            )
        torch.testing.assert_close(actual, expected)

    def test_invalid_backend_is_rejected(self) -> None:
        inputs = _inputs()
        with self.assertRaisesRegex(ValueError, "must be 'torch' or 'triton'"):
            packed_reader_context(*inputs, backend="cuda")  # type: ignore[arg-type]

    def test_explicit_triton_backend_never_falls_back_on_cpu(self) -> None:
        inputs = _inputs(dtype=torch.float32)
        with mock.patch.object(
            packed_reader_module,
            "torch_packed_reader_context",
            side_effect=AssertionError("silent fallback"),
        ):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                packed_reader_context(*inputs, backend="triton")

    def test_input_contract_rejects_unsafe_node_ids(self) -> None:
        numerator, denominator, nodes, mass = _inputs()
        nodes[0, 0, 0, 0] = numerator.shape[2]
        with self.assertRaises((AssertionError, RuntimeError)):
            packed_reader_context(
                numerator,
                denominator,
                nodes,
                mass,
                backend="torch",
            )

    def test_availability_probe_is_boolean(self) -> None:
        self.assertIsInstance(triton_packed_reader_available(), bool)


@unittest.skipUnless(
    torch.cuda.is_available() and triton_packed_reader_available(),
    "CUDA Triton is unavailable",
)
class TritonPackedReaderTests(unittest.TestCase):
    def test_custom_op_registration_with_opcheck(self) -> None:
        if not hasattr(torch.library, "opcheck"):
            self.skipTest("torch.library.opcheck is unavailable")
        numerator, denominator, nodes, mass = _inputs(
            device="cuda",
            dtype=torch.float32,
        )
        numerator.requires_grad_(True)
        denominator.requires_grad_(True)
        mass.requires_grad_(True)
        result = torch.library.opcheck(
            torch.ops.gator.packed_reader_forward.default,
            (numerator, denominator, nodes, mass, 1e-6),
        )
        self.assertTrue(all(value == "SUCCESS" for value in result.values()))

    def test_forward_and_backward_match_torch_oracle(self) -> None:
        torch_inputs = _inputs(device="cuda", dtype=torch.float32)
        triton_inputs = tuple(
            value.detach().clone() if value.is_floating_point() else value.clone()
            for value in torch_inputs
        )
        torch_numerator, torch_denominator, nodes, torch_mass = torch_inputs
        triton_numerator, triton_denominator, triton_nodes, triton_mass = (
            triton_inputs
        )
        for value in (
            torch_numerator,
            torch_denominator,
            torch_mass,
            triton_numerator,
            triton_denominator,
            triton_mass,
        ):
            value.requires_grad_(True)

        expected = packed_reader_context(
            torch_numerator,
            torch_denominator,
            nodes,
            torch_mass,
            epsilon=1e-6,
            backend="torch",
        )
        actual = packed_reader_context(
            triton_numerator,
            triton_denominator,
            triton_nodes,
            triton_mass,
            epsilon=1e-6,
            backend="triton",
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (torch_numerator, torch_denominator, torch_mass),
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (triton_numerator, triton_denominator, triton_mass),
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=4e-5,
                rtol=4e-5,
            )

    def test_compiled_training_boundary(self) -> None:
        numerator, denominator, nodes, mass = _inputs(
            device="cuda", dtype=torch.float32
        )
        numerator.requires_grad_(True)
        denominator.requires_grad_(True)
        mass.requires_grad_(True)

        @torch.compile(fullgraph=True)
        def compiled(
            numerator: torch.Tensor,
            denominator: torch.Tensor,
            nodes: torch.Tensor,
            mass: torch.Tensor,
        ) -> torch.Tensor:
            return packed_reader_context(
                numerator,
                denominator,
                nodes,
                mass,
                epsilon=1e-6,
                backend="triton",
            )

        output = compiled(numerator, denominator, nodes, mass)
        output.square().mean().backward()
        for gradient in (numerator.grad, denominator.grad, mass.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_noncontiguous_multi_block_inputs_match_oracle(self) -> None:
        torch.manual_seed(929)
        batch, tokens, active_nodes, features = 1, 3, 4, 137
        lanes, routes = 2, 2
        # Receiver integration supplies these token-major views by permuting
        # node-major convolution output. Exercise those real strides, plus a
        # width greater than one Triton block so backward partial-dot atomics
        # are covered.
        numerator_source = torch.randn(
            batch,
            active_nodes,
            tokens,
            features,
            device="cuda",
        )
        denominator_source = torch.rand(
            batch,
            active_nodes,
            tokens,
            device="cuda",
        )
        numerator = numerator_source.permute(0, 2, 1, 3).requires_grad_(True)
        denominator = denominator_source.permute(0, 2, 1).requires_grad_(True)
        nodes = torch.randint(
            0,
            active_nodes,
            (batch, tokens, lanes, routes),
            device="cuda",
        )
        mass = torch.rand(
            batch,
            tokens,
            lanes,
            routes,
            device="cuda",
            requires_grad=True,
        )
        oracle_inputs = (
            numerator.detach().clone().requires_grad_(True),
            denominator.detach().clone().requires_grad_(True),
            nodes.clone(),
            mass.detach().clone().requires_grad_(True),
        )
        expected = packed_reader_context(*oracle_inputs, backend="torch")
        actual = packed_reader_context(
            numerator,
            denominator,
            nodes,
            mass,
            backend="triton",
        )
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        probe = torch.randn_like(actual)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (oracle_inputs[0], oracle_inputs[1], oracle_inputs[3]),
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (numerator, denominator, mass),
        )
        for candidate, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(
                candidate,
                reference,
                atol=8e-5,
                rtol=8e-5,
            )


if __name__ == "__main__":
    unittest.main()
