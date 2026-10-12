from __future__ import annotations

import unittest

import torch

from gator.models.language import GatorLanguageConfig, GatorLanguageModel
from gator.models.memory import (
    attention_receiver_context,
    convolution_receiver_context,
    linear_scan_receiver_context,
)


class ReceiverMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    @staticmethod
    def _inputs(*, dtype: torch.dtype = torch.float64):
        batch, cells, tokens, hidden, nodes = 2, 3, 6, 4, 4
        states = torch.randn(batch, cells, tokens, hidden, dtype=dtype)
        node_locations = torch.randn(
            batch, cells, tokens, nodes, dtype=dtype
        ).softmax(-1)
        writer_mass = torch.rand(batch, cells, tokens, dtype=dtype)
        writer_mass[0, 1, :2] = 0
        relative_bias = torch.randn(cells, nodes, 4, dtype=dtype) * 0.2
        return states, node_locations, writer_mass, relative_bias

    def test_convolution_matches_quadratic_attention(self) -> None:
        inputs = self._inputs()
        expected = attention_receiver_context(*inputs)
        actual = convolution_receiver_context(*inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_convolution_matches_reference_gradients(self) -> None:
        base = self._inputs()
        attention_inputs = tuple(
            value.detach().clone().requires_grad_(True) for value in base
        )
        convolution_inputs = tuple(
            value.detach().clone().requires_grad_(True) for value in base
        )
        probe = torch.randn_like(base[0])

        attention_output = attention_receiver_context(*attention_inputs)
        attention_loss = (attention_output * probe).sum()
        attention_gradients = torch.autograd.grad(
            attention_loss, attention_inputs
        )

        convolution_output = convolution_receiver_context(*convolution_inputs)
        convolution_loss = (convolution_output * probe).sum()
        convolution_gradients = torch.autograd.grad(
            convolution_loss, convolution_inputs
        )

        for actual, expected in zip(
            convolution_gradients, attention_gradients
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-9)

    def test_no_live_writer_returns_finite_zero_context(self) -> None:
        states, nodes, mass, bias = self._inputs(dtype=torch.float32)
        mass.zero_()
        inputs = tuple(
            value.detach().clone().requires_grad_(True)
            for value in (states, nodes, mass, bias)
        )
        context = convolution_receiver_context(*inputs)
        self.assertTrue(torch.isfinite(context).all())
        self.assertEqual(float(context.detach().abs().sum()), 0.0)
        gradients = torch.autograd.grad(context.sum(), inputs)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(float(gradient.abs().sum()), 0.0)

    def test_linear_scan_returns_finite_output_and_gradients(self) -> None:
        # linear_scan is a genuinely different computation from the other
        # two backends (bidirectional decayed_scan, not a relative-position
        # kernel), so unlike test_convolution_matches_quadratic_attention
        # there is no reference to numerically match -- this checks it
        # produces a well-formed, differentiable output on the same input
        # contract instead.
        inputs = self._inputs(dtype=torch.float32)
        inputs = tuple(value.requires_grad_(True) for value in inputs)
        context = linear_scan_receiver_context(*inputs)
        self.assertEqual(context.shape, inputs[0].shape)
        self.assertTrue(torch.isfinite(context).all())
        gradients = torch.autograd.grad(context.sum(), inputs)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_linear_scan_ignores_bias_channels_two_and_three(self) -> None:
        # relative_node_bias's four channels are reinterpreted for this
        # backend as (decay, forward/backward mix, unused, unused); the
        # unused channels must get exactly zero gradient, not just "small".
        states, node_locations, writer_mass, bias = self._inputs(
            dtype=torch.float32
        )
        bias = bias.requires_grad_(True)
        context = linear_scan_receiver_context(
            states, node_locations, writer_mass, bias
        )
        (gradient,) = torch.autograd.grad(context.sum(), (bias,))
        self.assertEqual(float(gradient[..., 2].abs().sum()), 0.0)
        self.assertEqual(float(gradient[..., 3].abs().sum()), 0.0)
        self.assertGreater(float(gradient[..., 0].abs().sum()), 0.0)
        self.assertGreater(float(gradient[..., 1].abs().sum()), 0.0)

    def test_linear_scan_no_live_writer_returns_finite_zero_context(
        self,
    ) -> None:
        states, nodes, mass, bias = self._inputs(dtype=torch.float32)
        mass.zero_()
        inputs = tuple(
            value.detach().clone().requires_grad_(True)
            for value in (states, nodes, mass, bias)
        )
        context = linear_scan_receiver_context(*inputs)
        self.assertTrue(torch.isfinite(context).all())
        self.assertEqual(float(context.detach().abs().sum()), 0.0)
        gradients = torch.autograd.grad(context.sum(), inputs)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_tiny_live_writer_has_bounded_bf16_backward(self) -> None:
        for backend in (
            convolution_receiver_context,
            linear_scan_receiver_context,
        ):
            with self.subTest(backend=backend.__name__):
                states = torch.tensor(
                    [[[[1.0, 2.0], [3.0, 4.0]]]],
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                nodes = torch.ones(
                    1, 1, 2, 1, dtype=torch.bfloat16, requires_grad=True
                )
                mass = torch.full(
                    (1, 1, 2),
                    5.5e-40,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                bias = torch.zeros(
                    1, 1, 4, dtype=torch.bfloat16, requires_grad=True
                )
                kwargs = (
                    {"block_size": 1}
                    if backend is linear_scan_receiver_context
                    else {}
                )
                context = backend(states, nodes, mass, bias, **kwargs)
                self.assertTrue(torch.isfinite(context).all())
                gradients = torch.autograd.grad(
                    context.float().sum(), (states, nodes, mass, bias)
                )
                for gradient in gradients:
                    self.assertTrue(torch.isfinite(gradient).all())

    def test_linear_scan_model_forward_and_backward(self) -> None:
        model = GatorLanguageModel(
            GatorLanguageConfig(
                vocab_size=32,
                d_model=4,
                embedding_dim=4,
                n_cells=4,
                n_nodes=4,
                relays=2,
                neighbors=2,
                mask_token_id=3,
                lexical_mode="communal",
                node_topology="asymmetric",
                receiver_memory_backend="linear_scan",
            )
        ).train()
        context = torch.randint(4, 32, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)
        logits = model(context, mask, output_slots=3)
        loss = logits.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNotNone(model.start_logits.grad)
        self.assertTrue(torch.isfinite(model.start_logits.grad).all())

    def test_model_backends_share_parameters_and_outputs(self) -> None:
        common = dict(
            vocab_size=32,
            d_model=4,
            embedding_dim=4,
            n_cells=4,
            n_nodes=4,
            relays=2,
            neighbors=2,
            mask_token_id=3,
            lexical_mode="communal",
            node_topology="asymmetric",
        )
        attention = GatorLanguageModel(
            GatorLanguageConfig(
                **common, receiver_memory_backend="attention"
            )
        ).eval()
        convolution = GatorLanguageModel(
            GatorLanguageConfig(
                **common, receiver_memory_backend="convolution"
            )
        ).eval()
        convolution.load_state_dict(attention.state_dict())
        context = torch.randint(4, 32, (2, 5))
        mask = torch.tensor(
            [[False, True, True, True, True], [True, True, True, True, True]]
        )
        with torch.no_grad():
            expected = attention(context, mask, output_slots=3)
            actual = convolution(context, mask, output_slots=3)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_model_forward_supports_bfloat16_autocast(self) -> None:
        model = GatorLanguageModel(
            GatorLanguageConfig(
                vocab_size=32,
                d_model=4,
                embedding_dim=4,
                n_cells=4,
                n_nodes=4,
                relays=2,
                neighbors=2,
                mask_token_id=3,
                lexical_mode="communal",
                node_topology="asymmetric",
                receiver_memory_backend="convolution",
            )
        ).train()
        context = torch.randint(4, 32, (2, 5))
        mask = torch.ones_like(context, dtype=torch.bool)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(context, mask, output_slots=3)
            loss = logits.float().square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNotNone(model.start_logits.grad)
        self.assertTrue(torch.isfinite(model.start_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
