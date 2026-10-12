from __future__ import annotations

import unittest
from unittest import mock

import torch

import gator.models.triton_packed_writer as writer_module
from gator.models.triton_packed_writer import (
    packed_writer_adjoint,
    packed_writer_features,
    torch_packed_writer_adjoint,
    torch_packed_writer_features,
    triton_packed_writer_available,
)


def _inputs(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    unique: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7401 + int(unique))
    batch, tokens, lanes, width, routes, active = 2, 3, 2, 7, 3, 7
    hidden = torch.randn(batch, tokens, lanes, width, generator=generator)
    write = torch.rand(batch, tokens, generator=generator) + 0.1
    if unique:
        base = torch.arange(lanes * routes).reshape(lanes, routes)
        ids = torch.stack(
            [
                torch.stack(
                    [(base + batch_index + token) % active for token in range(tokens)]
                )
                for batch_index in range(batch)
            ]
        )
    else:
        ids = torch.randint(
            0,
            active,
            (batch, tokens, lanes, routes),
            generator=generator,
        )
        # Force both within-lane and cross-lane collisions.
        ids[..., 1] = ids[..., 0]
        ids[:, :, 1, 2] = ids[:, :, 0, 0]
    route_delta = torch.rand(
        batch,
        tokens,
        lanes,
        routes,
        generator=generator,
    )
    # Zero-delta entries model invalid/padded routes while retaining a safe ID.
    route_delta[:, 0, 0, 2] = 0
    route_delta[0, 2, 1, 0] = 0
    return (
        hidden.to(device=device, dtype=dtype),
        write.to(device=device, dtype=dtype),
        ids.to(device=device, dtype=torch.long),
        route_delta.to(device=device, dtype=dtype),
    )


def _manual_writer(
    hidden: torch.Tensor,
    write: torch.Tensor,
    ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
    include_mass: bool,
) -> torch.Tensor:
    batch, tokens, lanes, _ = hidden.shape
    features = stop - start
    output = hidden.new_empty(
        batch,
        features + int(include_mass),
        active_count,
        tokens,
    )
    for batch_index in range(batch):
        for active in range(active_count):
            for token in range(tokens):
                state = (
                    epsilon
                    * write[batch_index, token]
                    * hidden[batch_index, token, :, start:stop].sum(dim=0)
                )
                mass = lanes * epsilon * write[batch_index, token]
                for lane in range(lanes):
                    for route in range(ids.shape[-1]):
                        if int(ids[batch_index, token, lane, route]) != active:
                            continue
                        scale = (
                            write[batch_index, token]
                            * route_delta[batch_index, token, lane, route]
                        )
                        state = state + (
                            scale
                            * hidden[batch_index, token, lane, start:stop]
                        )
                        mass = mass + scale
                if include_mass:
                    output[batch_index, 0, active, token] = mass
                    output[batch_index, 1:, active, token] = state
                else:
                    output[batch_index, :, active, token] = state
    return output


class TorchPackedWriterTests(unittest.TestCase):
    def test_public_model_package_exports_writer_api(self) -> None:
        import gator.models as models

        self.assertIs(models.packed_writer_features, packed_writer_features)
        self.assertIs(models.packed_writer_adjoint, packed_writer_adjoint)
        self.assertIs(
            models.triton_packed_writer_available,
            triton_packed_writer_available,
        )

    def test_availability_probe_is_boolean(self) -> None:
        self.assertIsInstance(triton_packed_writer_available(), bool)

    def test_forward_matches_loop_for_unique_duplicate_and_tail_chunks(self) -> None:
        for unique in (False, True):
            hidden, write, ids, route_delta = _inputs(unique=unique)
            for start, stop, include_mass in (
                (0, 7, False),
                (0, 4, True),
                (4, 7, False),
                (6, 7, True),
            ):
                expected = _manual_writer(
                    hidden,
                    write,
                    ids,
                    route_delta,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                )
                actual = torch_packed_writer_features(
                    hidden,
                    write,
                    ids,
                    route_delta,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                )
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.stride(-1), 1)

    def test_explicit_adjoint_matches_autograd_for_every_input_gradient(self) -> None:
        for unique in (False, True):
            for start, stop, include_mass in (
                (0, 4, True),
                (3, 7, False),
                (6, 7, True),
            ):
                hidden, write, ids, route_delta = _inputs(unique=unique)
                hidden.requires_grad_(True)
                write.requires_grad_(True)
                route_delta.requires_grad_(True)
                output = torch_packed_writer_features(
                    hidden,
                    write,
                    ids,
                    route_delta,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                )
                probe = torch.linspace(
                    -0.7,
                    1.1,
                    output.numel(),
                    dtype=output.dtype,
                    device=output.device,
                ).reshape_as(output)
                expected_hidden, expected_write, expected_delta = torch.autograd.grad(
                    (output * probe).sum(),
                    (hidden, write, route_delta),
                )
                actual_hidden, actual_write, actual_delta = (
                    torch_packed_writer_adjoint(
                        hidden.detach(),
                        write.detach(),
                        ids,
                        route_delta.detach(),
                        probe,
                        start=start,
                        stop=stop,
                        active_count=7,
                        epsilon=1e-5,
                        include_mass=include_mass,
                    )
                )
                torch.testing.assert_close(
                    actual_hidden,
                    expected_hidden[..., start:stop],
                    atol=1e-12,
                    rtol=1e-12,
                )
                torch.testing.assert_close(
                    actual_write, expected_write, atol=1e-12, rtol=1e-12
                )
                torch.testing.assert_close(
                    actual_delta, expected_delta, atol=1e-12, rtol=1e-12
                )

    def test_zero_delta_has_exact_zero_delta_scaled_state_effect(self) -> None:
        hidden, write, ids, route_delta = _inputs(unique=False)
        route_delta.zero_()
        actual = packed_writer_features(
            hidden,
            write,
            ids,
            route_delta,
            start=2,
            stop=5,
            active_count=7,
            epsilon=1e-4,
            include_mass=True,
            backend="torch",
        )
        state = (
            1e-4 * write[..., None] * hidden[..., 2:5].sum(dim=2)
        )
        expected_state = state.permute(0, 2, 1)[:, :, None].expand(
            -1, -1, 7, -1
        )
        expected_mass = (
            2e-4 * write[:, None].expand(-1, 7, -1)
        )
        torch.testing.assert_close(actual[:, 0], expected_mass)
        torch.testing.assert_close(actual[:, 1:], expected_state)

    def test_explicit_triton_backend_never_falls_back_on_cpu(self) -> None:
        hidden, write, ids, route_delta = _inputs(unique=True)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            packed_writer_features(
                hidden,
                write,
                ids,
                route_delta,
                start=0,
                stop=3,
                active_count=7,
                backend="triton",
            )
        grad_features = hidden.new_zeros(2, 3, 7, 3)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            packed_writer_adjoint(
                hidden,
                write,
                ids,
                route_delta,
                grad_features,
                start=0,
                stop=3,
                active_count=7,
                backend="triton",
            )

    def test_invalid_backend_and_unsafe_ids_are_rejected(self) -> None:
        hidden, write, ids, route_delta = _inputs(unique=True)
        with self.assertRaisesRegex(ValueError, "backend"):
            packed_writer_features(
                hidden,
                write,
                ids,
                route_delta,
                start=0,
                stop=3,
                active_count=7,
                backend="invalid",  # type: ignore[arg-type]
            )
        unsafe = ids.clone()
        unsafe[0, 0, 0, 0] = -1
        route_delta = route_delta.clone()
        route_delta[0, 0, 0, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "outside"):
            torch_packed_writer_features(
                hidden,
                write,
                unsafe,
                route_delta,
                start=0,
                stop=3,
                active_count=7,
            )


@unittest.skipUnless(
    torch.cuda.is_available() and triton_packed_writer_available(),
    "CUDA Triton packed writer is unavailable",
)
class TritonPackedWriterTests(unittest.TestCase):
    def test_registered_autograd_matches_torch_oracle(self) -> None:
        source = _inputs(device="cuda", dtype=torch.float32, unique=False)
        torch_hidden = source[0].detach().clone().requires_grad_(True)
        torch_write = source[1].detach().clone().requires_grad_(True)
        torch_delta = source[3].detach().clone().requires_grad_(True)
        triton_hidden = source[0].detach().clone().requires_grad_(True)
        triton_write = source[1].detach().clone().requires_grad_(True)
        triton_delta = source[3].detach().clone().requires_grad_(True)
        expected = packed_writer_features(
            torch_hidden,
            torch_write,
            source[2],
            torch_delta,
            start=2,
            stop=7,
            active_count=7,
            epsilon=1e-5,
            include_mass=True,
            backend="torch",
        )
        actual = packed_writer_features(
            triton_hidden,
            triton_write,
            source[2],
            triton_delta,
            start=2,
            stop=7,
            active_count=7,
            epsilon=1e-5,
            include_mass=True,
            backend="triton",
        )
        probe = torch.randn_like(expected)
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (torch_hidden, torch_write, torch_delta),
        )
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (triton_hidden, triton_write, triton_delta),
        )
        for candidate, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(
                candidate,
                reference,
                atol=5e-5,
                rtol=5e-5,
            )

    def test_production_width_128_forward_and_adjoint_match_oracle(self) -> None:
        torch.manual_seed(9902)
        batch, tokens, lanes, width, routes, active_count = 1, 17, 3, 128, 4, 23
        hidden = torch.randn(batch, tokens, lanes, width, device="cuda")
        write = torch.rand(batch, tokens, device="cuda")
        ids = torch.randint(
            0,
            active_count,
            (batch, tokens, lanes, routes),
            device="cuda",
        )
        ids[..., 1] = ids[..., 0]
        route_delta = torch.rand_like(ids, dtype=torch.float32)
        inputs = (hidden, write, ids, route_delta)
        expected = packed_writer_features(
            *inputs,
            start=0,
            stop=128,
            active_count=active_count,
            epsilon=1e-6,
            include_mass=True,
            backend="torch",
        )
        actual = packed_writer_features(
            *inputs,
            start=0,
            stop=128,
            active_count=active_count,
            epsilon=1e-6,
            include_mass=True,
            backend="triton",
        )
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        probe = torch.randn_like(actual)
        expected_gradients = packed_writer_adjoint(
            *inputs,
            probe,
            start=0,
            stop=128,
            active_count=active_count,
            epsilon=1e-6,
            include_mass=True,
            backend="torch",
        )
        actual_gradients = packed_writer_adjoint(
            *inputs,
            probe,
            start=0,
            stop=128,
            active_count=active_count,
            epsilon=1e-6,
            include_mass=True,
            backend="triton",
        )
        for candidate, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(
                candidate,
                reference,
                atol=5e-5,
                rtol=5e-5,
            )

    def test_forward_and_explicit_adjoint_match_torch_oracle(self) -> None:
        for unique in (False, True):
            for start, stop, include_mass in (
                (0, 4, True),
                (3, 7, False),
                (6, 7, True),
            ):
                inputs = _inputs(device="cuda", dtype=torch.float32, unique=unique)
                hidden, write, ids, route_delta = inputs
                expected = packed_writer_features(
                    *inputs,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                    backend="torch",
                )
                actual = packed_writer_features(
                    *inputs,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                    backend="triton",
                )
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                generator = torch.Generator(device="cuda").manual_seed(9901)
                grad_features = torch.randn(
                    actual.shape,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                expected_gradients = packed_writer_adjoint(
                    *inputs,
                    grad_features,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                    backend="torch",
                )
                actual_gradients = packed_writer_adjoint(
                    *inputs,
                    grad_features,
                    start=start,
                    stop=stop,
                    active_count=7,
                    epsilon=1e-5,
                    include_mass=include_mass,
                    backend="triton",
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients
                ):
                    torch.testing.assert_close(
                        actual_gradient,
                        expected_gradient,
                        atol=3e-5,
                        rtol=3e-5,
                    )

    def test_dispatch_proves_triton_forward_and_adjoint_are_invoked(self) -> None:
        inputs = _inputs(device="cuda", dtype=torch.float32, unique=False)
        native_forward = writer_module._triton_packed_writer_features
        native_adjoint = writer_module._triton_packed_writer_adjoint
        self.assertIsNotNone(native_forward)
        self.assertIsNotNone(native_adjoint)
        with mock.patch.object(
            writer_module,
            "_triton_packed_writer_features",
            wraps=native_forward,
        ) as wrapped_forward:
            output = packed_writer_features(
                *inputs,
                start=1,
                stop=7,
                active_count=7,
                include_mass=True,
                backend="triton",
            )
        wrapped_forward.assert_called_once()
        with mock.patch.object(
            writer_module,
            "_triton_packed_writer_adjoint",
            wraps=native_adjoint,
        ) as wrapped_adjoint:
            packed_writer_adjoint(
                *inputs,
                torch.ones_like(output),
                start=1,
                stop=7,
                active_count=7,
                include_mass=True,
                backend="triton",
            )
        wrapped_adjoint.assert_called_once()

    def test_strict_dtype_and_chunk_width_contracts_do_not_fallback(self) -> None:
        hidden, write, ids, route_delta = _inputs(
            device="cuda", dtype=torch.float64, unique=True
        )
        with self.assertRaisesRegex(TypeError, "float32 only"):
            packed_writer_features(
                hidden,
                write,
                ids,
                route_delta,
                start=0,
                stop=7,
                active_count=7,
                backend="triton",
            )

        wide_hidden = torch.randn(1, 1, 1, 129, device="cuda")
        wide_write = torch.ones(1, 1, device="cuda")
        wide_ids = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.long)
        wide_delta = torch.ones(1, 1, 1, 1, device="cuda")
        with self.assertRaisesRegex(ValueError, "up to 128"):
            packed_writer_features(
                wide_hidden,
                wide_write,
                wide_ids,
                wide_delta,
                start=0,
                stop=129,
                active_count=1,
                backend="triton",
            )


if __name__ == "__main__":
    unittest.main()
