"""End-to-end integration check: receiver_memory_hgrn_scan=True wired
into the real GatorLanguageModel forward/backward, not just the
standalone module tests in tests/test_hgrn_receiver_scan.py. Requires
CUDA and flash-linear-attention, same as that file.
"""

import unittest

import torch

from gator.models.language import GatorLanguageConfig, GatorLanguageModel

try:
    import fla  # noqa: F401

    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention is not installed")
class HgrnReceiverScanModelIntegrationTests(unittest.TestCase):
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
            receiver_memory_backend="linear_scan",
            dense_node_backend="folded_bmm",
            dense_kernel_padding=8,
            node_topology="symmetric",
            block_causal=True,
            block_size=4,
            position_reference_length=8,
        )
        base.update(overrides)
        return GatorLanguageConfig(**base)

    @staticmethod
    def _batch(vocab_size: int):
        torch.manual_seed(0)
        context = torch.randint(4, vocab_size, (2, 4), device="cuda")
        mask = torch.ones_like(context, dtype=torch.bool)
        return context, mask

    def test_matches_ungraphed_output_closely(self):
        torch.manual_seed(1)
        config_eager = self._config(receiver_memory_hgrn_scan=False)
        torch.manual_seed(1)
        config_hgrn = self._config(receiver_memory_hgrn_scan=True)
        model_eager = GatorLanguageModel(config_eager).cuda().eval()
        model_hgrn = GatorLanguageModel(config_hgrn).cuda().eval()
        model_hgrn.load_state_dict(model_eager.state_dict())
        context, mask = self._batch(config_eager.vocab_size)
        with torch.no_grad():
            logits_eager = model_eager(context, mask, output_slots=4)
            logits_hgrn = model_hgrn(context, mask, output_slots=4)
        self.assertTrue(
            torch.allclose(logits_eager, logits_hgrn, rtol=1e-2, atol=1e-2),
            "hgrn_scan model diverged from the decayed_scan reference",
        )

    def test_forward_and_backward_finite(self):
        config = self._config(receiver_memory_hgrn_scan=True)
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

    def test_multi_relay_gradients_match_eager_reference(self):
        torch.manual_seed(2)
        config_eager = self._config(receiver_memory_hgrn_scan=False, relays=3)
        torch.manual_seed(2)
        config_hgrn = self._config(receiver_memory_hgrn_scan=True, relays=3)
        model_eager = GatorLanguageModel(config_eager).cuda().train()
        model_hgrn = GatorLanguageModel(config_hgrn).cuda().train()
        model_hgrn.load_state_dict(model_eager.state_dict())
        context, mask = self._batch(config_eager.vocab_size)

        model_eager(context, mask, output_slots=4).sum().backward()
        model_hgrn(context, mask, output_slots=4).sum().backward()

        grads_eager = {
            name: p.grad.clone()
            for name, p in model_eager.named_parameters()
            if p.grad is not None
        }
        grads_hgrn = {
            name: p.grad.clone()
            for name, p in model_hgrn.named_parameters()
            if p.grad is not None
        }
        self.assertEqual(set(grads_eager), set(grads_hgrn))
        for name in grads_eager:
            self.assertTrue(
                torch.allclose(grads_eager[name], grads_hgrn[name], rtol=5e-2, atol=5e-2),
                f"gradient for {name} diverged between eager and hgrn_scan at relays=3",
            )

    def test_rejects_hgrn_scan_without_block_causal(self):
        with self.assertRaises(ValueError):
            self._config(
                receiver_memory_hgrn_scan=True,
                block_causal=False,
                block_size=None,
                position_reference_length=None,
            )

    def test_rejects_hgrn_scan_combined_with_cuda_graph(self):
        with self.assertRaises(ValueError):
            self._config(
                receiver_memory_hgrn_scan=True,
                receiver_memory_cuda_graph=True,
            )


if __name__ == "__main__":
    unittest.main()
