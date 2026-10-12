from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from gator_linear import (
    EvoTokenRefiner,
    GatorLinearConfig,
    GatorLinearPrefixConfig,
    GatorLinearPrefixModel,
    balanced_promotion_counts,
    masked_trajectory_cross_entropy,
    position_gradient_statistics,
    sample_masked_denoising_inputs,
    sample_training_schedule,
    soft_token_embeddings,
    topk_token_distribution,
)


def build_model(
    *,
    relays: int = 2,
    tie_token_embeddings: bool = False,
) -> GatorLinearPrefixModel:
    torch.manual_seed(7)
    core = GatorLinearConfig(
        vocab_size=16,
        streams=3,
        nodes=8,
        width=6,
        relays=relays,
        degree=3,
        buses=2,
        mailbox_channels=2,
        mailbox_width=4,
        basis_rank=3,
        address_width=4,
        router_rank=4,
        conv_kernel=3,
        context_length=12,
        causal=False,
        controlled_writing=True,
        tie_token_embeddings=tie_token_embeddings,
    )
    return GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core))


def valid_mask() -> torch.Tensor:
    mask = torch.zeros(16, dtype=torch.bool)
    mask[4:] = True
    return mask


class SoftOutputInterfaceTest(unittest.TestCase):
    def test_token_embedding_tying_is_optional_and_counted_once(self) -> None:
        untied = build_model(tie_token_embeddings=False)
        tied = build_model(tie_token_embeddings=True)

        self.assertIsNot(
            untied.token_embedding.weight,
            untied.vocabulary.weight,
        )
        self.assertIs(
            tied.token_embedding.weight,
            tied.vocabulary.weight,
        )
        expected_saving = (
            tied.config.core.vocab_size * tied.config.core.width
        )
        self.assertEqual(
            untied.parameter_count - tied.parameter_count,
            expected_saving,
        )
        parameter_names = dict(tied.named_parameters())
        self.assertIn("token_embedding.weight", parameter_names)
        self.assertNotIn("vocabulary.weight", parameter_names)

        tied.to(dtype=torch.float64)
        self.assertIs(
            tied.token_embedding.weight,
            tied.vocabulary.weight,
        )

    def test_tied_embedding_survives_checkpoint_and_receives_gradient(self) -> None:
        model = build_model(tie_token_embeddings=True)
        observed = torch.randint(4, 16, (2, 5))
        model(observed, output_slots=4).square().mean().backward()
        gradient = model.token_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

        restored = build_model(tie_token_embeddings=True)
        restored.load_state_dict(model.state_dict())
        self.assertIs(
            restored.token_embedding.weight,
            restored.vocabulary.weight,
        )
        torch.testing.assert_close(
            restored.token_embedding.weight,
            model.token_embedding.weight,
        )

    def test_output_logits_only_adapts_requested_slots(self) -> None:
        model = build_model().eval()
        hidden = torch.randn(2, 9, 3, 6, requires_grad=True)
        output_slots = 4

        expected_adapted = torch.einsum(
            "blsd,sde->blse", hidden, model.stream_adapter
        )
        expected_weights = torch.softmax(
            model.fusion_score(hidden).squeeze(-1), dim=-1
        )
        expected_fused = torch.einsum(
            "blsd,bls->bld", expected_adapted, expected_weights
        )
        expected_logits = model.vocabulary(expected_fused)[
            :, -output_slots:
        ]
        expected_grad = torch.autograd.grad(
            expected_logits.square().mean(),
            hidden,
            retain_graph=True,
        )[0]

        original_einsum = torch.einsum
        with mock.patch(
            "gator_linear.prefix.torch.einsum",
            wraps=original_einsum,
        ) as einsum:
            actual_logits, actual_weights = model.output_logits(
                hidden,
                output_slots=output_slots,
            )
        actual_grad = torch.autograd.grad(
            actual_logits.square().mean(),
            hidden,
        )[0]

        torch.testing.assert_close(actual_logits, expected_logits)
        torch.testing.assert_close(actual_weights, expected_weights)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(
            tuple(einsum.call_args_list[0].args[1].shape),
            (2, output_slots, 3, 6),
        )
        self.assertEqual(
            tuple(einsum.call_args_list[1].args[2].shape),
            (2, output_slots, 3),
        )

    def test_explicit_query_embeddings_preserve_the_legacy_forward(self) -> None:
        model = build_model().eval()
        observed = torch.randint(4, 16, (3, 5))
        explicit = model.output_query.view(1, 1, -1).expand(3, 4, -1)
        with torch.no_grad():
            legacy = model(observed, output_slots=4)
            supplied = model(
                observed,
                output_slots=4,
                output_embeddings=explicit,
            )
        torch.testing.assert_close(legacy, supplied, rtol=0, atol=0)

    def test_output_embedding_gradient_is_live(self) -> None:
        model = build_model()
        observed = torch.randint(4, 16, (2, 5))
        output = torch.randn(2, 4, 6, requires_grad=True)
        model(
            observed,
            output_slots=4,
            output_embeddings=output,
        ).square().mean().backward()
        self.assertIsNotNone(output.grad)
        self.assertGreater(float(output.grad.abs().sum()), 0.0)

    def test_bad_output_embedding_shape_is_rejected(self) -> None:
        model = build_model()
        with self.assertRaisesRegex(ValueError, "output_embeddings"):
            model(
                torch.randint(4, 16, (2, 5)),
                output_slots=4,
                output_embeddings=torch.randn(2, 3, 6),
            )

    def test_output_position_ids_reassign_logical_addresses(self) -> None:
        model = build_model().eval()
        observed = torch.randint(4, 16, (2, 5))
        identity = torch.arange(4)
        permutation = torch.tensor([2, 0, 3, 1])
        legacy_hidden, legacy_addresses = model.initial_hidden(
            observed, output_slots=4
        )
        identity_hidden, identity_addresses = model.initial_hidden(
            observed,
            output_slots=4,
            output_position_ids=identity,
        )
        permuted_hidden, permuted_addresses = model.initial_hidden(
            observed,
            output_slots=4,
            output_position_ids=permutation,
        )
        torch.testing.assert_close(
            identity_hidden, legacy_hidden, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            identity_addresses, legacy_addresses, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            permuted_addresses[:5], legacy_addresses[:5]
        )
        torch.testing.assert_close(
            permuted_addresses[5:],
            legacy_addresses[5:].index_select(0, permutation),
        )
        self.assertFalse(torch.equal(permuted_hidden, legacy_hidden))

    def test_output_position_ids_must_be_a_permutation(self) -> None:
        model = build_model()
        observed = torch.randint(4, 16, (2, 5))
        with self.assertRaisesRegex(ValueError, "permutation"):
            model.initial_hidden(
                observed,
                output_slots=4,
                output_position_ids=torch.tensor([0, 0, 2, 3]),
            )
        with self.assertRaisesRegex(TypeError, "long"):
            model.initial_hidden(
                observed,
                output_slots=4,
                output_position_ids=torch.arange(4, dtype=torch.int32),
            )


class EvoTokenPrimitiveTest(unittest.TestCase):
    def test_position_gradient_statistics_detects_alignment_and_conflict(self) -> None:
        active = torch.tensor([2.0], requires_grad=True)
        unused = torch.tensor([3.0], requires_grad=True)
        losses = torch.stack((active[0], active[0], -active[0]))
        norms, cos_prefix = position_gradient_statistics(
            losses,
            ((active,), (unused,)),
        )
        torch.testing.assert_close(norms[0], torch.ones(3))
        torch.testing.assert_close(norms[1], torch.zeros(3))
        self.assertTrue(bool(torch.isnan(cos_prefix[:, 0]).all()))
        torch.testing.assert_close(cos_prefix[0, 1], torch.tensor(1.0))
        torch.testing.assert_close(cos_prefix[0, 2], torch.tensor(-1.0))
        self.assertTrue(bool(torch.isnan(cos_prefix[1]).all()))
        self.assertIsNone(active.grad)
        self.assertIsNone(unused.grad)

        losses.sum().backward()
        torch.testing.assert_close(active.grad, torch.tensor([1.0]))
        self.assertIsNone(unused.grad)

    def test_masked_trajectory_loss_ignores_initially_visible_tokens(self) -> None:
        logits = torch.zeros(2, 1, 2, 3, requires_grad=True)
        targets = torch.tensor([[0, 1]])
        visible = torch.tensor([[True, False]])
        loss, token_loss = masked_trajectory_cross_entropy(
            logits,
            targets,
            visible,
            visible_weight=0.0,
        )
        expected = token_loss[:, :, 1].mean()
        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertEqual(float(logits.grad[:, :, 0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[:, :, 1].abs().sum()), 0.0)

    def test_unit_visible_weight_matches_full_trajectory_mean(self) -> None:
        logits = torch.randn(2, 3, 4, 7)
        targets = torch.randint(0, 7, (3, 4))
        visible = torch.rand(3, 4) < 0.5
        loss, token_loss = masked_trajectory_cross_entropy(
            logits,
            targets,
            visible,
            visible_weight=1.0,
        )
        torch.testing.assert_close(loss, token_loss.mean())

    def test_default_objective_matches_the_legacy_weighted_mean(self) -> None:
        logits = torch.randn(2, 3, 4, 7)
        targets = torch.randint(0, 7, (3, 4))
        visible = torch.rand(3, 4) < 0.5
        visible_weight = 0.25
        loss, token_loss = masked_trajectory_cross_entropy(
            logits,
            targets,
            visible,
            visible_weight=visible_weight,
        )
        expanded_visible = visible.unsqueeze(0).expand_as(token_loss)
        weights = torch.where(
            expanded_visible,
            torch.full_like(token_loss, visible_weight),
            torch.ones_like(token_loss),
        )
        expected = (token_loss * weights).sum() / weights.sum()
        torch.testing.assert_close(loss, expected)

    def test_masked_trajectory_loss_rejects_invalid_visible_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "visible_weight"):
            masked_trajectory_cross_entropy(
                torch.randn(2, 1, 2, 3),
                torch.tensor([[0, 1]]),
                torch.tensor([[True, False]]),
                visible_weight=1.1,
            )

    def test_position_weights_balance_and_mask_direct_credit(self) -> None:
        logits = torch.randn(2, 3, 4, 7, requires_grad=True)
        targets = torch.randint(0, 7, (3, 4))
        visible = torch.zeros_like(targets, dtype=torch.bool)
        weights = torch.tensor([0.0, 1.0, 2.0, 3.0])
        loss, token_loss = masked_trajectory_cross_entropy(
            logits,
            targets,
            visible,
            position_weights=weights,
        )
        expanded = weights.view(1, 1, -1).expand_as(token_loss)
        expected = (token_loss * expanded).sum() / expanded.sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[:, :, 0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[:, :, 3].abs().sum()), 0.0)

    def test_worst_position_objective_is_scaled_and_bounded(self) -> None:
        logits = torch.zeros(1, 1, 3, 3)
        targets = torch.tensor([[0, 1, 2]])
        logits[0, 0, 0, 0] = 5.0
        logits[0, 0, 2, 2] = -5.0
        visible = torch.zeros_like(targets, dtype=torch.bool)
        mean_loss, token_loss = masked_trajectory_cross_entropy(
            logits, targets, visible
        )
        worst_loss, _ = masked_trajectory_cross_entropy(
            logits,
            targets,
            visible,
            worst_position_weight=1.0,
            worst_position_temperature=0.1,
        )
        self.assertGreater(float(worst_loss), float(mean_loss))
        self.assertLessEqual(float(worst_loss), float(token_loss.max()))

        tied_logits = torch.zeros(1, 1, 3, 3)
        tied_mean, _ = masked_trajectory_cross_entropy(
            tied_logits, targets, visible
        )
        tied_worst, _ = masked_trajectory_cross_entropy(
            tied_logits,
            targets,
            visible,
            worst_position_weight=1.0,
            worst_position_temperature=0.1,
        )
        torch.testing.assert_close(tied_worst, tied_mean)

    def test_worst_position_objective_respects_position_weights(self) -> None:
        logits = torch.zeros(1, 1, 3, 3)
        targets = torch.tensor([[0, 1, 2]])
        logits[0, 0, 0, 0] = 3.0
        logits[0, 0, 2, 2] = -3.0
        visible = torch.zeros_like(targets, dtype=torch.bool)
        weights = torch.tensor([0.0, 1.0, 3.0])
        temperature = 0.5
        actual, token_loss = masked_trajectory_cross_entropy(
            logits,
            targets,
            visible,
            position_weights=weights,
            worst_position_weight=1.0,
            worst_position_temperature=temperature,
        )
        position_loss = token_loss.mean(dim=(0, 1))
        active = weights > 0
        expected = temperature * (
            torch.logsumexp(
                position_loss[active] / temperature + weights[active].log(),
                dim=0,
            )
            - weights[active].sum().log()
        )
        torch.testing.assert_close(actual, expected)

    def test_topk_filters_illegal_tokens_and_renormalizes(self) -> None:
        logits = torch.randn(2, 4, 16, requires_grad=True)
        logits.data[..., 0] = 100.0
        token_ids, probabilities, confidence, retained = (
            topk_token_distribution(
                logits,
                top_k=3,
                valid_token_mask=valid_mask(),
            )
        )
        self.assertTrue(bool((token_ids >= 4).all()))
        torch.testing.assert_close(
            probabilities.sum(-1),
            torch.ones_like(probabilities[..., 0]),
        )
        self.assertFalse(probabilities.requires_grad)
        self.assertFalse(confidence.requires_grad)
        self.assertTrue(bool(((retained > 0) & (retained <= 1)).all()))

    def test_sparse_embedding_matches_manual_weighting(self) -> None:
        weight = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
        token_ids = torch.tensor([[[4, 8, 12]]])
        probabilities = torch.tensor([[[0.2, 0.3, 0.5]]])
        actual = soft_token_embeddings(token_ids, probabilities, weight)
        expected = (
            0.2 * weight[4] + 0.3 * weight[8] + 0.5 * weight[12]
        ).view(1, 1, 6)
        torch.testing.assert_close(actual, expected)

    def test_balanced_promotions_scale_across_slot_counts(self) -> None:
        self.assertEqual(balanced_promotion_counts(8, 4), (2, 2, 2, 2))
        self.assertEqual(balanced_promotion_counts(6, 4), (2, 1, 2, 1))
        self.assertEqual(balanced_promotion_counts(4, 4), (1, 1, 1, 1))
        self.assertEqual(balanced_promotion_counts(2, 4), (1, 0, 1, 0))
        self.assertEqual(balanced_promotion_counts(10, 4), (3, 2, 3, 2))
        self.assertEqual(balanced_promotion_counts(1, 4), (1, 0, 0, 0))

    def test_balanced_promotions_reject_invalid_sizes(self) -> None:
        for output_slots, steps in ((0, 4), (4, 0), (True, 4), (4, False)):
            with self.subTest(output_slots=output_slots, steps=steps):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    balanced_promotion_counts(output_slots, steps)

    def test_two_slots_can_use_four_refinement_stages(self) -> None:
        targets = torch.randint(4, 16, (5, 2))
        initial_ids, initial_pure, promotions, counts = (
            sample_training_schedule(targets, mask_token_id=3, steps=4)
        )
        self.assertEqual(counts, (1, 0, 1, 0))
        self.assertFalse(bool(initial_pure.any()))
        self.assertTrue(bool(initial_ids.eq(3).all()))
        self.assertTrue(bool(promotions.sum(dim=0).eq(1).all()))
        torch.testing.assert_close(
            promotions.sum(dim=(1, 2)),
            torch.tensor(counts) * targets.shape[0],
        )

    def test_training_schedule_is_seeded_disjoint_and_complete(self) -> None:
        targets = torch.randint(4, 16, (5, 4))
        first = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            transition_choices=(1, 2),
            generator=torch.Generator().manual_seed(19),
        )
        second = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            transition_choices=(1, 2),
            generator=torch.Generator().manual_seed(19),
        )
        for left, right in zip(first[:3], second[:3]):
            torch.testing.assert_close(left, right)
        self.assertEqual(first[3], second[3])

        initial_ids, initial_pure, promotions, counts = first
        self.assertTrue(bool(initial_ids[~initial_pure].eq(3).all()))
        self.assertTrue(bool(initial_ids[initial_pure].eq(targets[initial_pure]).all()))
        self.assertFalse(bool((promotions[0] & promotions[1]).any()))
        self.assertFalse(bool((promotions & initial_pure.unsqueeze(0)).any()))
        torch.testing.assert_close(
            promotions.sum(dim=(1, 2)),
            torch.tensor(counts) * targets.shape[0],
        )

    def test_training_schedule_accepts_explicit_slot_orders(self) -> None:
        targets = torch.randint(4, 16, (2, 4))
        order = torch.tensor([[3, 2, 1, 0], [1, 3, 0, 2]])
        _, initial_pure, promotions, counts = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            promotion_counts=(2, 2),
            slot_order=order,
        )
        self.assertEqual(counts, (2, 2))
        self.assertFalse(bool(initial_pure.any()))
        for row in range(order.shape[0]):
            torch.testing.assert_close(
                promotions[0, row].nonzero(as_tuple=False).squeeze(-1),
                order[row, :2].sort().values,
            )
        with self.assertRaisesRegex(ValueError, "permutation"):
            sample_training_schedule(
                targets,
                mask_token_id=3,
                steps=2,
                promotion_counts=(2, 2),
                slot_order=torch.tensor(
                    [[0, 0, 2, 3], [0, 1, 2, 3]]
                ),
            )

    def test_masked_bootstrap_always_masks_each_row(self) -> None:
        targets = torch.randint(4, 16, (32, 4))
        inputs, masked = sample_masked_denoising_inputs(
            targets,
            mask_token_id=3,
            minimum_mask_rate=0.0,
            maximum_mask_rate=0.0,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertTrue(bool(masked.any(dim=1).all()))
        self.assertTrue(bool(inputs[masked].eq(3).all()))


class RelayRefinementTest(unittest.TestCase):
    def test_full_pass_accepts_output_position_permutation(self) -> None:
        model = build_model(relays=2).eval()
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=2,
            top_k=3,
            mode="full_pass",
            feedback_scale=1.0,
        )
        observed = torch.randint(4, 16, (2, 5))
        targets = torch.randint(4, 16, (2, 4))
        initial_ids, initial_pure, promotions, _ = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            promotion_counts=(2, 2),
        )
        result = refiner(
            observed,
            initial_ids,
            initial_pure,
            alpha=torch.full((2, 2), 0.7),
            promotion_schedule=promotions,
            output_position_ids=torch.tensor([2, 3, 0, 1]),
        )
        self.assertEqual(result[0].shape, (2, 2, 4, 16))
        self.assertEqual(result[1].shape, (2, 4))

    def test_position_starvation_controls_train_together(self) -> None:
        model = build_model(relays=2)
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=2,
            top_k=3,
            mode="relay",
            feedback_scale=0.5,
        )
        observed = torch.randint(4, 16, (2, 5))
        logical_targets = torch.randint(4, 16, (2, 4))
        physical_to_logical = torch.tensor([2, 3, 0, 1])
        targets = logical_targets.index_select(1, physical_to_logical)
        order = torch.tensor([[3, 2, 1, 0], [1, 3, 0, 2]])
        initial_ids, initial_pure, promotions, _ = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            promotion_counts=(2, 2),
            slot_order=order,
        )
        stage_logits, _, _ = refiner(
            observed,
            initial_ids,
            initial_pure,
            alpha=torch.full((2, 2), 0.7),
            promotion_schedule=promotions,
            output_position_ids=physical_to_logical,
        )
        loss, token_loss = masked_trajectory_cross_entropy(
            stage_logits,
            targets,
            initial_pure,
            position_weights=torch.tensor([0.0, 0.0, 1.0, 3.0]),
            worst_position_weight=0.25,
            worst_position_temperature=0.5,
        )
        position_losses = token_loss.mean(dim=(0, 1))
        relay_ids = {
            id(parameter)
            for relay in model.core.relays
            for parameter in relay.parameters()
        }
        shared = tuple(
            parameter
            for parameter in model.parameters()
            if id(parameter) not in relay_ids
        )
        groups = (shared,) + tuple(
            tuple(relay.parameters()) for relay in model.core.relays
        )
        norms, cos_prefix = position_gradient_statistics(
            position_losses, groups
        )
        self.assertEqual(norms.shape, (3, 4))
        self.assertEqual(cos_prefix.shape, (3, 4))
        self.assertTrue(bool(torch.isfinite(norms).all()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_output_position_permutations_reuse_one_compiled_graph(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        model = build_model(relays=2).eval()
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=2,
            top_k=3,
            mode="relay",
            feedback_scale=0.5,
        )
        observed = torch.randint(4, 16, (2, 5))
        targets = torch.randint(4, 16, (2, 4))
        initial_ids, initial_pure, promotions, _ = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            promotion_counts=(2, 2),
        )
        alpha = torch.full((2, 2), 0.7)
        graphs: list[torch.fx.GraphModule] = []

        def backend(
            graph: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> object:
            graphs.append(graph)
            return graph.forward

        def run(position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return refiner(
                observed,
                initial_ids,
                initial_pure,
                alpha=alpha,
                promotion_schedule=promotions,
                output_position_ids=position_ids,
            )

        compiled = torch.compile(run, backend=backend, fullgraph=True)
        for position_ids in (
            torch.arange(4),
            torch.tensor([2, 3, 0, 1]),
        ):
            expected = run(position_ids)
            actual = compiled(position_ids)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right)
        self.assertEqual(len(graphs), 1)

    def test_compiled_bfloat16_position_objective_backward(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        model = build_model(relays=2)
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=2,
            top_k=3,
            mode="relay",
            feedback_scale=0.5,
        )
        observed = torch.randint(4, 16, (2, 5))
        logical_targets = torch.randint(4, 16, (2, 4))
        position_ids = torch.tensor([2, 3, 0, 1])
        targets = logical_targets.index_select(1, position_ids)
        initial_ids, initial_pure, promotions, _ = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            promotion_counts=(2, 2),
        )
        alpha = torch.full((2, 2), 0.7)
        position_weights = torch.tensor([0.0, 1.0, 2.0, 3.0])

        def objective(mapping: torch.Tensor) -> torch.Tensor:
            with torch.autocast(
                device_type="cpu",
                dtype=torch.bfloat16,
            ):
                stage_logits, _, _ = refiner(
                    observed,
                    initial_ids,
                    initial_pure,
                    alpha=alpha,
                    promotion_schedule=promotions,
                    output_position_ids=mapping,
                )
                loss, _ = masked_trajectory_cross_entropy(
                    stage_logits,
                    targets,
                    initial_pure,
                    position_weights=position_weights,
                    worst_position_weight=0.25,
                    worst_position_temperature=0.5,
                )
            return loss

        compiled = torch.compile(
            objective,
            backend="eager",
            fullgraph=True,
        )
        loss = compiled(position_ids)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_zero_feedback_matches_the_uninterrupted_core(self) -> None:
        model = build_model().eval()
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=2,
            top_k=3,
            mode="relay",
            feedback_scale=0.0,
        ).eval()
        observed = torch.randint(4, 16, (2, 5))
        initial_ids = torch.full((2, 4), 3, dtype=torch.long)
        initial_pure = torch.zeros_like(initial_ids, dtype=torch.bool)
        promotions = torch.zeros(2, 2, 4, dtype=torch.bool)
        promotions[0, :, :1] = True
        promotions[1, :, 1:2] = True
        alpha = torch.full((2, 2), 0.7)
        initial_embeddings = F.embedding(
            initial_ids, model.token_embedding.weight
        )
        with torch.no_grad():
            expected = model(
                observed,
                output_slots=4,
                output_embeddings=initial_embeddings,
            )
            stages, _, _ = refiner(
                observed,
                initial_ids,
                initial_pure,
                alpha=alpha,
                promotion_schedule=promotions,
            )
        torch.testing.assert_close(stages[-1], expected, rtol=0, atol=0)

    def test_feedback_starts_after_stage_zero_and_changes_later_logits(self) -> None:
        model = build_model().eval()
        common = {
            "mask_token_id": 3,
            "valid_token_mask": valid_mask(),
            "steps": 2,
            "top_k": 3,
            "mode": "relay",
        }
        off = EvoTokenRefiner(model, feedback_scale=0.0, **common).eval()
        on = EvoTokenRefiner(model, feedback_scale=0.5, **common).eval()
        observed = torch.randint(4, 16, (2, 5))
        initial_ids = torch.full((2, 4), 3, dtype=torch.long)
        pure = torch.zeros_like(initial_ids, dtype=torch.bool)
        promotions = torch.zeros(2, 2, 4, dtype=torch.bool)
        promotions[0, :, :2] = True
        promotions[1, :, 2:] = True
        alpha = torch.full((2, 2), 0.7)
        with torch.no_grad():
            stages_off, _, _ = off(
                observed,
                initial_ids,
                pure,
                alpha=alpha,
                promotion_schedule=promotions,
            )
            stages_on, _, _ = on(
                observed,
                initial_ids,
                pure,
                alpha=alpha,
                promotion_schedule=promotions,
            )
        torch.testing.assert_close(stages_off[0], stages_on[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(stages_off[1], stages_on[1]))

    def test_relay_trajectory_loss_reaches_every_relay(self) -> None:
        model = build_model()
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=2,
            mode="relay",
            feedback_scale=0.5,
        )
        observed = torch.randint(4, 16, (3, 5))
        targets = torch.randint(4, 16, (3, 4))
        initial_ids, pure, promotions, _ = sample_training_schedule(
            targets,
            mask_token_id=3,
            steps=2,
            transition_choices=(1, 2),
            generator=torch.Generator().manual_seed(5),
        )
        stages, _, retained = refiner(
            observed,
            initial_ids,
            pure,
            alpha=torch.full((2, 3), 0.7),
            promotion_schedule=promotions,
        )
        self.assertEqual(tuple(stages.shape), (2, 3, 4, 16))
        self.assertEqual(tuple(retained.shape), (2, 3, 4))
        expanded_targets = targets.unsqueeze(0).expand(2, -1, -1)
        F.cross_entropy(
            stages.reshape(-1, 16), expanded_targets.reshape(-1)
        ).backward()
        for index, relay in enumerate(model.core.relays):
            total = sum(
                float(parameter.grad.abs().sum())
                for parameter in relay.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(total, 0.0, f"relay {index} received no gradient")

    def test_inference_promotes_every_slot_and_returns_legal_ids(self) -> None:
        for mode, scale in (("relay", 0.5), ("full_pass", 1.0)):
            model = build_model().eval()
            refiner = EvoTokenRefiner(
                model,
                mask_token_id=3,
                valid_token_mask=valid_mask(),
                steps=2,
                mode=mode,
                feedback_scale=scale,
            ).eval()
            observed = torch.randint(4, 16, (2, 5))
            initial_ids = torch.full((2, 4), 3, dtype=torch.long)
            pure = torch.zeros_like(initial_ids, dtype=torch.bool)
            with torch.no_grad():
                stages, decoded, _ = refiner(
                    observed,
                    initial_ids,
                    pure,
                    alpha=torch.tensor(0.7),
                    inference_transition_tokens=2,
                )
            self.assertEqual(tuple(stages.shape), (2, 2, 4, 16))
            self.assertTrue(bool((decoded >= 4).all()))


    def test_inference_scales_to_fewer_slots_than_stages(self) -> None:
        counts = balanced_promotion_counts(2, 4)
        for mode, scale in (("relay", 0.5), ("full_pass", 1.0)):
            with self.subTest(mode=mode):
                model = build_model(relays=4).eval()
                refiner = EvoTokenRefiner(
                    model,
                    mask_token_id=3,
                    valid_token_mask=valid_mask(),
                    steps=4,
                    mode=mode,
                    feedback_scale=scale,
                ).eval()
                observed = torch.randint(4, 16, (2, 5))
                initial_ids = torch.full((2, 2), 3, dtype=torch.long)
                pure = torch.zeros_like(initial_ids, dtype=torch.bool)
                with torch.no_grad():
                    stages, decoded, _ = refiner(
                        observed,
                        initial_ids,
                        pure,
                        alpha=torch.tensor(0.7),
                        inference_promotion_counts=counts,
                    )
                self.assertEqual(tuple(stages.shape), (4, 2, 2, 16))
                self.assertEqual(tuple(decoded.shape), (2, 2))
                self.assertTrue(bool((decoded >= 4).all()))

    def test_inference_rejects_promotion_counts_with_wrong_total(self) -> None:
        model = build_model(relays=4).eval()
        refiner = EvoTokenRefiner(
            model,
            mask_token_id=3,
            valid_token_mask=valid_mask(),
            steps=4,
            mode="relay",
            feedback_scale=0.5,
        ).eval()
        initial_ids = torch.full((2, 2), 3, dtype=torch.long)
        pure = torch.zeros_like(initial_ids, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "sum to output_slots"):
            refiner(
                torch.randint(4, 16, (2, 5)),
                initial_ids,
                pure,
                alpha=torch.tensor(0.7),
                inference_promotion_counts=(1, 0, 0, 0),
            )


if __name__ == "__main__":
    unittest.main()
