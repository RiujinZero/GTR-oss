from __future__ import annotations

import unittest

import torch
from torch import nn

import validate_convergent_fft_backend as validator


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 17, width: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.output = nn.Linear(width, vocab_size)
        self.unused = nn.Parameter(torch.randn(width))

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        output_slots: int,
        output_token_ids: torch.Tensor,
        output_mask: torch.Tensor,
        causal: bool,
        temperature: float | torch.Tensor,
    ) -> torch.Tensor:
        del causal
        context = self.embedding(context_ids)
        weights = context_mask.to(context.dtype).unsqueeze(-1)
        pooled = (context * weights).sum(1) / weights.sum(1).clamp_min(1)
        output = self.embedding(output_token_ids[:, :output_slots])
        output = output * output_mask[:, :output_slots, None]
        return self.output((output + pooled[:, None]) / temperature)


class FullModelValidatorContractTests(unittest.TestCase):
    def test_parser_accepts_release_runner_flags_and_aliases(self) -> None:
        args = validator.parse_args(
            [
                "--device",
                "cuda:1",
                "--batch-size",
                "2",
                "--context-tokens",
                "13",
                "--output-slots",
                "7",
                "--nodes",
                "12",
                "--width",
                "16",
                "--embedding-dim",
                "10",
                "--lanes",
                "3",
                "--route-width",
                "4",
                "--degree",
                "5",
                "--relays",
                "3",
                "--router-rank",
                "8",
                "--vocab-size",
                "256",
                "--feature-chunk",
                "8",
                "--causal",
                "--compile",
                "--fullgraph",
            ]
        )
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(args.d_model, 16)
        self.assertEqual(args.active_node_top_k, 4)
        self.assertEqual(args.packed_memory_feature_chunk, 8)
        self.assertTrue(args.causal)
        self.assertTrue(args.compile)
        self.assertTrue(args.fullgraph)
        self.assertEqual(args.packed_memory_writer_backend, "triton")

    def test_parser_rejects_noncompiled_or_non_cuda_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --compile"):
            validator.parse_args(["--no-compile"])
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            validator.parse_args(["--device", "cpu"])

    def test_synthetic_language_batch_is_seeded_shifted_and_masked(self) -> None:
        shape = validator.ModelShape(
            batch_size=3,
            context_tokens=5,
            output_slots=4,
            vocab_size=31,
            embedding_dim=6,
            d_model=6,
            nodes=7,
            lanes=2,
            routes=3,
            degree=3,
            relays=2,
            router_rank=4,
            feature_chunk=3,
        )
        first = validator.build_synthetic_batch(shape, seed=9)
        second = validator.build_synthetic_batch(shape, seed=9)
        self.assertTrue(torch.equal(first.context_ids, second.context_ids))
        self.assertTrue(torch.equal(first.targets, second.targets))
        self.assertTrue(torch.equal(first.output_token_ids[:, 0], torch.ones(3)))
        self.assertTrue(
            torch.equal(first.output_token_ids[:, 1:], first.targets[:, :-1])
        )
        self.assertFalse(bool(first.context_mask[0, -1]))
        self.assertEqual(int(first.context_ids[0, -1]), 0)
        self.assertGreaterEqual(int(first.targets.min()), 4)

    def test_backend_configs_change_execution_only_and_load_strictly(self) -> None:
        args = validator.parse_args([])
        shape = validator.shape_from_args(args)
        reference_config = validator.build_model_config(
            shape, seed=3, candidate=False
        )
        candidate_config = validator.build_model_config(
            shape, seed=3, candidate=True
        )
        self.assertEqual(reference_config.route_backend, "packed")
        self.assertEqual(reference_config.packed_memory_backend, "fft")
        self.assertEqual(
            reference_config.packed_memory_reader_backend, "torch"
        )
        self.assertEqual(
            reference_config.packed_memory_writer_backend, "torch"
        )
        self.assertEqual(candidate_config.packed_memory_backend, "fft_opaque")
        self.assertEqual(candidate_config.packed_memory_reader_backend, "triton")
        self.assertEqual(candidate_config.packed_memory_writer_backend, "triton")

        torch.manual_seed(3)
        reference = validator.ConvergentLanguageModel(reference_config)
        initial = validator.state_dict_to_cpu(reference)
        candidate = validator.ConvergentLanguageModel(candidate_config)
        candidate.load_state_dict(initial, strict=True)
        validator.assert_state_dict_exact(candidate.state_dict(), initial)
        self.assertEqual(reference.parameter_count, candidate.parameter_count)

    def test_cpu_snapshot_contract_checks_all_gradients_and_adamw_deltas(self) -> None:
        torch.manual_seed(41)
        reference = TinyLanguageModel()
        initial = validator.state_dict_to_cpu(reference)
        candidate = TinyLanguageModel()
        candidate.load_state_dict(initial, strict=True)
        shape = validator.ModelShape(
            batch_size=2,
            context_tokens=4,
            output_slots=3,
            vocab_size=17,
            embedding_dim=6,
            d_model=6,
            nodes=4,
            lanes=2,
            routes=2,
            degree=2,
            relays=1,
            router_rank=3,
            feature_chunk=3,
        )
        batch = validator.build_synthetic_batch(shape, seed=12)
        kwargs = {
            "output_slots": shape.output_slots,
            "vocab_size": shape.vocab_size,
            "causal": True,
            "temperature": torch.tensor(0.7),
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        }
        expected = validator.capture_training_snapshot(
            reference, reference, batch, initial, **kwargs
        )
        actual = validator.capture_training_snapshot(
            candidate, candidate, batch, initial, **kwargs
        )
        summaries = validator.compare_snapshots(
            actual,
            expected,
            atol=0.0,
            rtol=0.0,
            delta_atol=0.0,
            delta_rtol=0.0,
        )
        self.assertEqual(summaries["outputs"].tensors_checked, 2)
        self.assertEqual(
            summaries["gradients"].tensors_checked,
            len(dict(reference.named_parameters())) - 1,
        )
        self.assertEqual(summaries["gradients"].tensors_both_none, 1)
        self.assertEqual(
            summaries["parameter_deltas"].tensors_checked,
            len(dict(reference.named_parameters())),
        )

    def test_comparison_fails_closed_on_one_named_gradient_drift(self) -> None:
        expected = validator.TrainingSnapshot(
            logits=torch.zeros(1, 1, 2),
            loss=torch.tensor(0.0),
            gradients={"weight": torch.zeros(2)},
            parameter_deltas={"weight": torch.zeros(2)},
        )
        actual = validator.TrainingSnapshot(
            logits=expected.logits.clone(),
            loss=expected.loss.clone(),
            gradients={"weight": torch.tensor([0.0, 0.01])},
            parameter_deltas={"weight": torch.zeros(2)},
        )
        with self.assertRaises(AssertionError):
            validator.compare_snapshots(
                actual,
                expected,
                atol=1e-5,
                rtol=1e-5,
                delta_atol=1e-5,
                delta_rtol=1e-5,
            )

    def test_nonfinite_tensor_fails_closed(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            validator.require_finite("probe", torch.tensor([float("nan")]))

    def test_compile_evidence_requires_at_least_one_graph(self) -> None:
        self.assertEqual(
            validator._compile_evidence(
                {
                    "stats": {"unique_graphs": 2},
                    "graph_break": {"one": 3},
                }
            ),
            (2, 3),
        )
        with self.assertRaisesRegex(RuntimeError, "no recorded Dynamo graph"):
            validator._compile_evidence(
                {"stats": {}, "graph_break": {}}
            )


if __name__ == "__main__":
    unittest.main()
