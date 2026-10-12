import unittest

import torch

from gator.models.language import (
    GatorLanguageConfig,
    GatorLanguageModel,
    _fixed_code,
)
from gator.models.memory import linear_scan_receiver_context


class LinearScanBlockCacheTests(unittest.TestCase):
    def test_vectorized_blocks_equal_sequential_cached_blocks(self) -> None:
        torch.manual_seed(12)
        states = torch.randn(1, 2, 4, 3)
        locations = torch.randn(1, 2, 4, 2).softmax(-1)
        writer_mass = torch.rand(1, 2, 4)
        bias = torch.randn(2, 2, 4)

        full, full_cache = linear_scan_receiver_context(
            states,
            locations,
            writer_mass,
            bias,
            block_size=2,
            return_cache=True,
        )
        first, cache = linear_scan_receiver_context(
            states[:, :, :2],
            locations[:, :, :2],
            writer_mass[:, :, :2],
            bias,
            block_size=2,
            return_cache=True,
        )
        second, sequential_cache = linear_scan_receiver_context(
            states[:, :, 2:],
            locations[:, :, 2:],
            writer_mass[:, :, 2:],
            bias,
            block_size=2,
            cache=cache,
            return_cache=True,
        )

        torch.testing.assert_close(full, torch.cat((first, second), dim=2))
        torch.testing.assert_close(
            full_cache.forward_state, sequential_cache.forward_state
        )

    def test_later_block_cannot_change_an_earlier_block(self) -> None:
        torch.manual_seed(13)
        states = torch.randn(1, 2, 4, 3)
        locations = torch.randn(1, 2, 4, 2).softmax(-1)
        writer_mass = torch.rand(1, 2, 4)
        bias = torch.randn(2, 2, 4)
        changed = states.clone()
        changed[:, :, 2:] += 100.0

        original = linear_scan_receiver_context(
            states, locations, writer_mass, bias, block_size=2
        )
        modified = linear_scan_receiver_context(
            changed, locations, writer_mass, bias, block_size=2
        )

        torch.testing.assert_close(original[:, :, :2], modified[:, :, :2])


class GatorARWBlockCacheTests(unittest.TestCase):
    @staticmethod
    def _model(
        cell_transport_mode: str = "stewardship",
        *,
        stewardship_execution: str = "dense",
    ) -> GatorLanguageModel:
        return GatorLanguageModel(
            GatorLanguageConfig(
                vocab_size=19,
                d_model=4,
                embedding_dim=4,
                n_cells=4,
                n_nodes=2,
                relays=2,
                neighbors=2,
                mask_token_id=3,
                node_topology="symmetric",
                cell_transport_mode=cell_transport_mode,
                stewardship_execution=stewardship_execution,
                receiver_memory_backend="linear_scan",
                block_causal=True,
                block_size=2,
                position_reference_length=16,
            )
        ).eval()

    def test_fixed_addresses_do_not_rescale_after_append(self) -> None:
        whole = _fixed_code(6, torch.device("cpu"), reference_length=16)
        suffix = _fixed_code(
            2, torch.device("cpu"), offset=4, reference_length=16
        )
        torch.testing.assert_close(whole[4:], suffix)

    def test_cached_proposal_matches_vectorized_block_causal_forward(self) -> None:
        torch.manual_seed(21)
        model = self._model()
        context = torch.tensor([[4, 5, 6, 7]])
        mask = torch.ones_like(context, dtype=torch.bool)

        with torch.no_grad():
            vectorized = model(context, mask, output_slots=2)
            cache = model.prefill_cache(context, mask)
            cached = model.forward_cached(cache, output_slots=2)

        torch.testing.assert_close(vectorized, cached, rtol=1e-5, atol=1e-6)
        self.assertEqual(int(cache.next_position), 4)
        self.assertEqual(len(cache.relays), model.config.relays)

    def test_naked_broadcast_cached_proposal_matches_vectorized_forward(self) -> None:
        torch.manual_seed(25)
        model = self._model(
            "broadcast_mix",
            stewardship_execution="packed_auto",
        )
        self.assertEqual(model.config.cell_input_mode, "shared_projection")
        self.assertEqual(model.config.cell_output_mode, "shared_pool")
        context = torch.tensor([[4, 5, 6, 7]])
        mask = torch.ones_like(context, dtype=torch.bool)

        with torch.no_grad():
            vectorized = model(context, mask, output_slots=2)
            cache = model.prefill_cache(context, mask)
            cached = model.forward_cached(cache, output_slots=2)

        torch.testing.assert_close(vectorized, cached, rtol=1e-5, atol=1e-6)
        self.assertEqual(int(cache.next_position), 4)
        self.assertEqual(len(cache.relays), model.config.relays)

    def test_append_is_immutable_and_matches_full_prefill(self) -> None:
        torch.manual_seed(22)
        model = self._model()
        context = torch.tensor([[4, 5, 6, 7]])
        accepted = torch.tensor([[8, 9]])

        first = model.prefill_cache(context)
        first_snapshot = tuple(
            relay.receiver.forward_state.clone() for relay in first.relays
        )
        appended = model.append_cache(first, accepted)
        complete = model.prefill_cache(torch.cat((context, accepted), dim=1))

        self.assertEqual(int(first.next_position), 4)
        self.assertEqual(int(appended.next_position), 6)
        for old, snapshot in zip(first.relays, first_snapshot, strict=True):
            torch.testing.assert_close(old.receiver.forward_state, snapshot)
        for actual, expected in zip(appended.relays, complete.relays, strict=True):
            torch.testing.assert_close(
                actual.receiver.forward_state,
                expected.receiver.forward_state,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                actual.traffic_numerator,
                expected.traffic_numerator,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(actual.write_sum, expected.write_sum)

    def test_masked_context_slot_does_not_write_into_cache(self) -> None:
        torch.manual_seed(23)
        model = self._model()
        mask = torch.tensor([[True, False]])
        first = model.prefill_cache(torch.tensor([[4, 5]]), mask)
        changed = model.prefill_cache(torch.tensor([[4, 18]]), mask)

        for actual, expected in zip(first.relays, changed.relays, strict=True):
            torch.testing.assert_close(
                actual.receiver.forward_state,
                expected.receiver.forward_state,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                actual.traffic_numerator,
                expected.traffic_numerator,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(actual.write_sum, expected.write_sum)

    def test_bf16_block_training_backward_is_finite(self) -> None:
        torch.manual_seed(24)
        model = self._model().train()
        context = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])
        mask = torch.ones_like(context, dtype=torch.bool)
        target = torch.tensor([[8, 9], [9, 8]])

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(context, mask, output_slots=2)
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), target
            )
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_broadcast_masked_context_slot_does_not_write(self) -> None:
        torch.manual_seed(27)
        model = self._model("broadcast_mix")
        mask = torch.tensor([[True, False]])
        first = model.prefill_cache(torch.tensor([[4, 5]]), mask)
        changed = model.prefill_cache(torch.tensor([[4, 18]]), mask)

        for actual, expected in zip(first.relays, changed.relays, strict=True):
            torch.testing.assert_close(
                actual.receiver.forward_state,
                expected.receiver.forward_state,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                actual.traffic_numerator,
                expected.traffic_numerator,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(actual.write_sum, expected.write_sum)

    def test_broadcast_bf16_training_backward_is_finite(self) -> None:
        torch.manual_seed(28)
        model = self._model("broadcast_mix").train()
        context = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])
        mask = torch.ones_like(context, dtype=torch.bool)
        target = torch.tensor([[8, 9], [9, 8]])

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(context, mask, output_slots=2)
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), target
            )
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_broadcast_append_matches_full_prefill(self) -> None:
        torch.manual_seed(26)
        model = self._model("broadcast_mix")
        context = torch.tensor([[4, 5, 6, 7]])
        accepted = torch.tensor([[8, 9]])

        appended = model.append_cache(model.prefill_cache(context), accepted)
        complete = model.prefill_cache(torch.cat((context, accepted), dim=1))

        for actual, expected in zip(appended.relays, complete.relays, strict=True):
            torch.testing.assert_close(
                actual.receiver.forward_state,
                expected.receiver.forward_state,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                actual.traffic_numerator,
                expected.traffic_numerator,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(actual.write_sum, expected.write_sum)


if __name__ == "__main__":
    unittest.main()
