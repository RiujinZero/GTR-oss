"""End-to-end integration check: gator/mx_precision.py wired into the
real GatorLanguageModel forward/backward, not just the standalone
module tests in tests/test_mx_precision.py. Requires CUDA + torchao,
same as that file.
"""

import unittest

import torch

try:
    import torchao  # noqa: F401

    _TORCHAO_AVAILABLE = True
except ImportError:
    _TORCHAO_AVAILABLE = False

try:
    import transformer_engine  # noqa: F401

    _TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    _TRANSFORMER_ENGINE_AVAILABLE = False

from gator.models.language import GatorLanguageConfig, GatorLanguageModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TORCHAO_AVAILABLE, "torchao is not installed")
class MXPrecisionModelIntegrationTests(unittest.TestCase):
    @staticmethod
    def _config(**overrides) -> GatorLanguageConfig:
        base = dict(
            vocab_size=32,
            d_model=8,
            embedding_dim=16,
            lexical_embedding_dim=8,
            n_cells=4,
            n_nodes=2,
            relays=1,
            neighbors=2,
            mask_token_id=3,
            lexical_mode="private_hash",
            cell_input_mode="shared_projection",
            cell_output_mode="shared_pool",
            cell_transport_mode="broadcast_mix",
            stewardship_execution="dense",
            receiver_memory_backend="convolution",
            dense_node_backend="folded_bmm",
            dense_kernel_padding=8,
            node_topology="symmetric",
        )
        base.update(overrides)
        return GatorLanguageConfig(**base)

    @staticmethod
    def _batch(vocab_size: int):
        torch.manual_seed(0)
        context = torch.randint(4, vocab_size, (2, 6), device="cuda")
        mask = torch.ones_like(context, dtype=torch.bool)
        return context, mask

    def test_bf16_default_is_unaffected_by_new_config_fields(self):
        # Defaults (node_compute_precision="bf16", readout_precision="bf16")
        # must be byte-identical to a config that doesn't set them at all --
        # confirms the new dispatch branches don't change default behavior.
        torch.manual_seed(1)
        model_a = self._config()
        torch.manual_seed(1)
        model_b = self._config(node_compute_precision="bf16", readout_precision="bf16")
        model_a_instance = GatorLanguageModel(model_a).cuda().eval()
        model_b_instance = GatorLanguageModel(model_b).cuda().eval()
        model_b_instance.load_state_dict(model_a_instance.state_dict())
        context, mask = self._batch(model_a.vocab_size)
        with torch.no_grad():
            logits_a = model_a_instance(context, mask, output_slots=2)
            logits_b = model_b_instance(context, mask, output_slots=2)
        self.assertTrue(torch.equal(logits_a, logits_b))

    def test_mxfp8_node_compute_forward_backward_finite(self):
        config = self._config(node_compute_precision="mxfp8")
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        logits = model(context, mask, output_slots=2)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"non-finite gradient for {name}",
                )

    def test_nvfp4_node_compute_forward_backward_finite(self):
        config = self._config(node_compute_precision="nvfp4")
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        logits = model(context, mask, output_slots=2)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"non-finite gradient for {name}",
                )

    def test_mxfp8_readout_forward_backward_finite(self):
        config = self._config(readout_precision="mxfp8")
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        logits = model(context, mask, output_slots=2)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"non-finite gradient for {name}",
                )

    def test_mxfp4_readout_forward_backward_finite(self):
        config = self._config(readout_precision="mxfp4")
        model = GatorLanguageModel(config).cuda().train()
        context, mask = self._batch(config.vocab_size)
        logits = model(context, mask, output_slots=2)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"non-finite gradient for {name}",
                )

    def test_node_compute_precision_rejects_einsum_backend(self):
        with self.assertRaises(ValueError):
            self._config(
                dense_node_backend="einsum", node_compute_precision="mxfp8"
            )

    def test_invalid_precision_choice_rejected(self):
        with self.assertRaises(ValueError):
            self._config(node_compute_precision="fp99")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(_TRANSFORMER_ENGINE_AVAILABLE, "transformer_engine is not installed")
class FP8TECudaGraphMultiRelayIntegrationTests(unittest.TestCase):
    """node_compute_precision='fp8_te' at relays>1 -- the real production
    pattern (node compute runs once per relay before one combined
    backward). Separate from MXPrecisionModelIntegrationTests above,
    which uses relays=1 and torchao gating (fp8_te needs
    transformer_engine, not torchao) and therefore never exercised this.

    Regression test for a real bug found during the receiver-memory
    CUDA-graph investigation: te_fp8_cudagraph_node_linear's graphed
    instance, shared across relay calls via one cache keyed only by
    shape, silently broke gradients (or crashed with
    AttributeError: 'NoneType' object has no attribute 'clone') when
    called more than once before the shared backward() ran -- fixed by
    keying the cache additionally by graph_slot (the relay index), so
    each relay gets its own dedicated graphed instance. See
    Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for the full
    investigation.
    """

    @staticmethod
    def _config(**overrides) -> GatorLanguageConfig:
        # d_model must be a multiple of 16 and batch*tokens a multiple of
        # 8 -- TE's real FP8 hardware GEMM requires this (checked via
        # assert_dim_for_fp8_exec); the tiny d_model=8 shape
        # MXPrecisionModelIntegrationTests uses above is fine for the
        # fake-quant precisions (no such constraint) but too small here.
        base = dict(
            vocab_size=32,
            d_model=64,
            embedding_dim=64,
            lexical_embedding_dim=64,
            n_cells=8,
            n_nodes=4,
            relays=3,
            neighbors=2,
            mask_token_id=3,
            lexical_mode="private_hash",
            cell_input_mode="shared_projection",
            cell_output_mode="shared_pool",
            cell_transport_mode="broadcast_mix",
            stewardship_execution="dense",
            receiver_memory_backend="convolution",
            dense_node_backend="folded_bmm",
            dense_kernel_padding=8,
            node_topology="symmetric",
        )
        base.update(overrides)
        return GatorLanguageConfig(**base)

    @staticmethod
    def _batch(vocab_size: int):
        # context(14) + output_slots(2) = 16 total; batch(2)*16 = 32,
        # divisible by 8 -- required by TE's real FP8 GEMM shape check.
        torch.manual_seed(0)
        context = torch.randint(4, vocab_size, (2, 14), device="cuda")
        mask = torch.ones_like(context, dtype=torch.bool)
        return context, mask

    def test_relay_gradients_match_bf16_reference_at_relays_three(self):
        torch.manual_seed(3)
        config_bf16 = self._config(node_compute_precision="bf16")
        torch.manual_seed(3)
        config_fp8 = self._config(node_compute_precision="fp8_te")
        model_bf16 = GatorLanguageModel(config_bf16).cuda().train()
        model_fp8 = GatorLanguageModel(config_fp8).cuda().train()
        model_fp8.load_state_dict(model_bf16.state_dict())
        context, mask = self._batch(config_bf16.vocab_size)

        model_bf16(context, mask, output_slots=2).sum().backward()
        model_fp8(context, mask, output_slots=2).sum().backward()

        grads_bf16 = {
            name: p.grad.clone()
            for name, p in model_bf16.named_parameters()
            if p.grad is not None
        }
        grads_fp8 = {
            name: p.grad.clone()
            for name, p in model_fp8.named_parameters()
            if p.grad is not None
        }
        # Intersection, not exact set equality: node_bias/node_embedding
        # only get real gradients via the einsum node-compute path (not
        # exercised here, dense_node_backend='folded_bmm' is required for
        # fp8_te) and their presence/absence differs between the two
        # models for reasons unrelated to what this test checks (shared
        # cross-relay parameters like node_weight/node_offset/
        # router_weight not getting corrupted) -- not a regression.
        shared_names = set(grads_bf16) & set(grads_fp8)
        self.assertGreater(len(shared_names), 0, "no shared gradient-bearing parameters to compare")
        for name in shared_names:
            self.assertTrue(torch.isfinite(grads_fp8[name]).all(), f"{name}.grad not finite")
            # Loose tolerance -- fp8_te is real lower-precision compute, not
            # bit-identical to bf16, but should track the same direction/
            # magnitude, not silently corrupted by a shared-graph bug.
            self.assertTrue(
                torch.allclose(grads_bf16[name], grads_fp8[name], rtol=0.3, atol=0.3),
                f"gradient for {name} diverged sharply between bf16 and "
                "fp8_te at relays=3 -- possible cross-relay graph corruption",
            )


if __name__ == "__main__":
    unittest.main()
