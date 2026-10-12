"""End-to-end integration check: dense_node_backend="monarch" wired
into the real GatorLanguageModel forward/backward, not just the
standalone primitive tests in tests/test_monarch_node_linear.py.

Unlike hgrn_scan/gated_deltanet (which replace a kernel/mechanism
while representing the same or a superset of the original math),
Monarch is a structurally different, much smaller parameterization --
a Monarch-initialized model has no dense-equivalent reference to match
output against, so these tests check finiteness/gradient-flow/param-
identity properties rather than numerical agreement with another
backend.

Requires CUDA.
"""

import unittest

import torch

from gator.models.language import GatorLanguageConfig, GatorLanguageModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class MonarchNodeLinearModelIntegrationTests(unittest.TestCase):
    @staticmethod
    def _config(**overrides) -> GatorLanguageConfig:
        base = dict(
            vocab_size=32,
            d_model=8,
            embedding_dim=16,
            lexical_embedding_dim=8,
            n_cells=4,
            n_nodes=2,
            relays=2,
            neighbors=2,
            mask_token_id=3,
            lexical_mode="private_hash",
            cell_input_mode="shared_projection",
            cell_output_mode="shared_pool",
            cell_transport_mode="broadcast_mix",
            stewardship_execution="dense",
            receiver_memory_backend="convolution",
            dense_node_backend="monarch",
            monarch_block_size=2,  # d_model=8 -> p=2, q=4
            dense_kernel_padding=8,
            node_topology="symmetric",
        )
        base.update(overrides)
        return GatorLanguageConfig(**base)

    @staticmethod
    def _batch(vocab_size: int):
        torch.manual_seed(0)
        context = torch.randint(4, vocab_size, (2, 4), device="cuda")
        mask = torch.ones_like(context, dtype=torch.bool)
        return context, mask

    def test_allocates_monarch_params_not_dense_node_weight(self):
        config = self._config()
        model = GatorLanguageModel(config).cuda()
        self.assertIsNone(model.node_weight)
        self.assertIsNotNone(model.monarch_left)
        self.assertIsNotNone(model.monarch_right)
        # p=2, q=4 -> monarch_left:[c,n,q,p,p]=[4,2,4,2,2], monarch_right:[c,n,p,q,q]=[4,2,2,4,4]
        self.assertEqual(tuple(model.monarch_left.shape), (4, 2, 4, 2, 2))
        self.assertEqual(tuple(model.monarch_right.shape), (4, 2, 2, 4, 4))

    def test_folded_bmm_model_has_dense_node_weight_not_monarch_params(self):
        config = self._config(dense_node_backend="folded_bmm", monarch_block_size=2)
        model = GatorLanguageModel(config).cuda()
        self.assertIsNotNone(model.node_weight)
        self.assertIsNone(model.monarch_left)
        self.assertIsNone(model.monarch_right)

    def test_forward_and_backward_finite(self):
        config = self._config()
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        logits = model(context, mask, output_slots=4)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"non-finite gradient for {name}",
                )

    def test_gradients_flow_to_monarch_factors(self):
        config = self._config()
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        model(context, mask, output_slots=4).sum().backward()
        self.assertIsNotNone(model.monarch_left.grad)
        self.assertIsNotNone(model.monarch_right.grad)
        self.assertIsNotNone(model.node_embedding.grad)
        self.assertIsNotNone(model.node_bias.grad)
        self.assertTrue(torch.isfinite(model.monarch_left.grad).all())
        self.assertTrue(torch.isfinite(model.monarch_right.grad).all())
        # Not all-zero -- a real, non-degenerate gradient signal reached
        # the factors, not just a finite-but-trivial one.
        self.assertGreater(model.monarch_left.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.monarch_right.grad.abs().sum().item(), 0.0)

    def test_multi_relay_forward_and_backward_finite(self):
        config = self._config(relays=3)
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        logits = model(context, mask, output_slots=4)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertTrue(torch.isfinite(model.monarch_left.grad).all())
        self.assertTrue(torch.isfinite(model.monarch_right.grad).all())

    def test_activation_checkpointing_matches_no_checkpointing(self):
        # The Monarch args are threaded through _dense_relay_transform's
        # signature as flat positional tensors specifically so
        # torch.utils.checkpoint.checkpoint can track them for grad --
        # verify checkpointing on vs off produce the same gradients.
        torch.manual_seed(3)
        config_no_ckpt = self._config(activation_checkpointing=False)
        torch.manual_seed(3)
        config_ckpt = self._config(activation_checkpointing=True)
        model_no_ckpt = GatorLanguageModel(config_no_ckpt).cuda().train()
        model_ckpt = GatorLanguageModel(config_ckpt).cuda().train()
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())
        context, mask = self._batch(config_no_ckpt.vocab_size)

        model_no_ckpt(context, mask, output_slots=4).sum().backward()
        model_ckpt(context, mask, output_slots=4).sum().backward()

        self.assertTrue(
            torch.allclose(
                model_no_ckpt.monarch_left.grad,
                model_ckpt.monarch_left.grad,
                rtol=1e-3,
                atol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                model_no_ckpt.monarch_right.grad,
                model_ckpt.monarch_right.grad,
                rtol=1e-3,
                atol=1e-4,
            )
        )

    def test_rejects_monarch_block_size_not_dividing_d_model(self):
        with self.assertRaises(ValueError):
            self._config(monarch_block_size=3)  # d_model=8, 8%3 != 0

    def test_rejects_monarch_with_non_bf16_precision(self):
        with self.assertRaises(ValueError):
            self._config(node_compute_precision="mxfp8")

    def test_rejects_monarch_with_packed_stewardship(self):
        with self.assertRaises(ValueError):
            self._config(stewardship_execution="packed_auto")


if __name__ == "__main__":
    unittest.main()
