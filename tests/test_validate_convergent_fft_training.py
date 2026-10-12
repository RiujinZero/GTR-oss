from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import validate_convergent_fft_training as validator


class ConvergentFFTTrainingCliTests(unittest.TestCase):
    def test_defaults_are_small_and_pin_the_candidate_backend(self) -> None:
        args = validator.parse_args([])
        shape = validator.shape_from_args(args)
        config = validator.build_model_config(shape, seed=args.seed)

        self.assertEqual(shape.batch_size, 1)
        self.assertEqual(shape.context_tokens, 7)
        self.assertEqual(shape.output_slots, 5)
        self.assertEqual(shape.d_model, 8)
        self.assertEqual(shape.nodes, 9)
        self.assertEqual(shape.lanes, 2)
        self.assertEqual(shape.routes, 3)
        self.assertEqual(config.route_backend, "packed")
        self.assertEqual(config.packed_memory_backend, "fft_opaque")
        self.assertEqual(config.packed_memory_reader_backend, "triton")
        self.assertEqual(config.packed_memory_writer_backend, "triton")
        self.assertEqual(config.router_mode, "shared_low_rank")
        self.assertEqual(config.ingress_mode, "learned")
        self.assertEqual(config.terminal_vote_mode, "slot")
        self.assertTrue(args.compile)
        self.assertFalse(args.fullgraph)
        self.assertEqual(args.packed_memory_writer_backend, "triton")

    def test_target_shape_knobs_and_aliases_are_exposed(self) -> None:
        args = validator.parse_args(
            [
                "--steps",
                "7",
                "--batch-size",
                "6",
                "--context-tokens",
                "512",
                "--output-slots",
                "128",
                "--vocab-size",
                "32768",
                "--embedding-dim",
                "768",
                "--width",
                "768",
                "--nodes",
                "154",
                "--lanes",
                "10",
                "--routes",
                "32",
                "--degree",
                "3",
                "--relays",
                "2",
                "--router-rank",
                "96",
                "--feature-chunk",
                "32",
                "--causal",
                "--fullgraph",
                "--checkpoint-path",
                "candidate.pt",
            ]
        )
        shape = validator.shape_from_args(args)
        self.assertEqual(shape.batch_size, 6)
        self.assertEqual(shape.context_tokens, 512)
        self.assertEqual(shape.output_slots, 128)
        self.assertEqual(shape.d_model, 768)
        self.assertEqual(shape.nodes, 154)
        self.assertEqual(shape.routes, 32)
        self.assertEqual(shape.router_rank, 96)
        self.assertTrue(args.causal)
        self.assertTrue(args.fullgraph)
        self.assertEqual(args.checkpoint_path, Path("candidate.pt"))

    def test_fail_closed_cli_rejects_cpu_uncompiled_and_unsafe_shapes(self) -> None:
        cases = (
            (["--device", "cpu"], "requires CUDA"),
            (["--no-compile"], "requires --compile"),
            (["--nodes", "2", "--lanes", "3"], "lanes cannot exceed"),
            (
                ["--nodes", "2", "--active-node-top-k", "3"],
                "active_node_top_k cannot exceed",
            ),
            (
                [
                    "--nodes",
                    "2",
                    "--active-node-top-k",
                    "2",
                    "--degree",
                    "3",
                ],
                "degree cannot exceed",
            ),
            (["--steps", "0"], "must be positive"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, message):
                    validator.parse_args(argv)

    def test_cuda_preflight_fails_without_cuda_instead_of_falling_back(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                validator.require_cuda_triton("cuda:0")

    def test_dynamo_metrics_require_and_report_real_graph_capture(self) -> None:
        counters = {
            "stats": {"unique_graphs": 2, "calls_captured": 17},
            "frames": {"total": 3, "ok": 2},
            "graph_break": {"reason one": 2, "reason two": 1},
        }
        with mock.patch.object(torch._dynamo.utils, "counters", counters):
            self.assertEqual(
                validator.dynamo_metrics(),
                {
                    "unique_graphs": 2,
                    "calls_captured": 17,
                    "frames_total": 3,
                    "frames_ok": 2,
                    "graph_breaks": 3,
                },
            )
        counters["stats"]["unique_graphs"] = 0
        with mock.patch.object(torch._dynamo.utils, "counters", counters):
            with self.assertRaisesRegex(AssertionError, "captured no Dynamo graph"):
                validator.dynamo_metrics()

    def test_compile_defaults_to_production_graph_semantics(self) -> None:
        args = validator.parse_args([])
        model = mock.Mock(spec=torch.nn.Module)
        compiled = object()
        with (
            mock.patch("torch.compile", return_value=compiled) as compile_mock,
            mock.patch("torch._dynamo.reset"),
            mock.patch.object(torch._dynamo.utils, "counters", mock.Mock()),
        ):
            self.assertIs(validator.compile_candidate(model, args), compiled)
        compile_mock.assert_called_once_with(
            model,
            fullgraph=False,
            mode="default",
        )

        fullgraph_args = validator.parse_args(["--fullgraph"])
        with (
            mock.patch("torch.compile", return_value=compiled) as compile_mock,
            mock.patch("torch._dynamo.reset"),
            mock.patch.object(torch._dynamo.utils, "counters", mock.Mock()),
        ):
            validator.compile_candidate(model, fullgraph_args)
        self.assertTrue(compile_mock.call_args.kwargs["fullgraph"])


class DeterministicBatchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = validator.parse_args(["--batch-size", "2"])
        self.shape = validator.shape_from_args(self.args)

    def test_batches_are_index_addressable_and_round_trip_exactly(self) -> None:
        first = validator.build_synthetic_batch(
            self.shape,
            seed=11,
            batch_index=4,
        )
        repeated = validator.build_synthetic_batch(
            self.shape,
            seed=11,
            batch_index=4,
        )
        next_batch = validator.build_synthetic_batch(
            self.shape,
            seed=11,
            batch_index=5,
        )
        restored = validator.SyntheticLanguageBatch.from_state_dict(
            first.state_dict()
        )
        for name in first.state_dict():
            torch.testing.assert_close(getattr(first, name), getattr(repeated, name))
            torch.testing.assert_close(getattr(first, name), getattr(restored, name))
        self.assertFalse(torch.equal(first.targets, next_batch.targets))
        self.assertTrue(torch.equal(first.output_token_ids[:, 1:], first.targets[:, :-1]))
        self.assertFalse(first.context_mask[1, -1])

    def test_invalid_teacher_prefix_is_rejected(self) -> None:
        batch = validator.build_synthetic_batch(
            self.shape,
            seed=0,
            batch_index=0,
        )
        broken = validator.SyntheticLanguageBatch.from_state_dict(batch.state_dict())
        broken.output_token_ids[:, 1] = 2
        with self.assertRaisesRegex(ValueError, "teacher prefix"):
            validator.validate_batch(broken, self.shape)


class OptimizerAndCheckpointContractTests(unittest.TestCase):
    @staticmethod
    def _initialized_optimizer() -> tuple[torch.nn.Module, torch.optim.AdamW]:
        torch.manual_seed(21)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            foreach=False,
            fused=False,
        )
        loss = sum(parameter.square().mean() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return model, optimizer

    def test_serialized_optimizer_ids_map_back_to_stable_parameter_names(self) -> None:
        model, optimizer = self._initialized_optimizer()
        names = validator.optimizer_parameter_names(model, optimizer)
        serialized = validator._tree_to_cpu(optimizer.state_dict())
        expected = validator.named_optimizer_state_to_cpu(model, optimizer)
        actual = validator.serialized_named_optimizer_state(serialized, names)
        self.assertEqual(actual.keys(), expected.keys())
        self.assertTrue(any(name.endswith("::exp_avg") for name in actual))
        self.assertTrue(any(name.endswith("::exp_avg_sq") for name in actual))
        self.assertTrue(any(name.endswith("::step") for name in actual))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def _candidate_payload(self) -> tuple[dict, validator.ModelShape, object, object]:
        args = validator.parse_args(["--steps", "2"])
        shape = validator.shape_from_args(args)
        config = validator.build_model_config(shape, seed=args.seed)
        model = validator.ConvergentLanguageModel(config)
        optimizer = validator.make_optimizer(model, args)
        # Populate AdamW state for every parameter without invoking the CUDA-
        # only candidate receiver.
        loss = sum(parameter.square().mean() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        names = validator.optimizer_parameter_names(model, optimizer)
        optimizer_state = validator._tree_to_cpu(optimizer.state_dict())
        next_batch = validator.build_synthetic_batch(
            shape,
            seed=args.seed,
            batch_index=args.steps,
        )
        payload = {
            "format": validator.CHECKPOINT_FORMAT,
            "version": validator.CHECKPOINT_VERSION,
            "shape": validator.asdict(shape),
            "model_config": config.to_dict(),
            "training_spec": validator.training_spec(args),
            "completed_steps": args.steps,
            "next_batch_index": args.steps,
            "model_state": validator.state_dict_to_cpu(model),
            "optimizer_state": optimizer_state,
            "optimizer_parameter_names": names,
            "next_batch": next_batch.state_dict(),
            "cpu_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state": torch.zeros(16, dtype=torch.uint8),
        }
        return payload, shape, args, config

    def test_checkpoint_round_trip_preserves_complete_resume_contract(self) -> None:
        payload, shape, args, config = self._candidate_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = validator.save_checkpoint(Path(directory) / "candidate.pt", payload)
            restored = validator.load_checkpoint(path)
        next_batch = validator.validate_checkpoint_payload(
            restored,
            shape=shape,
            config=config,
            args=args,
        )
        self.assertEqual(restored["completed_steps"], 2)
        self.assertEqual(restored["next_batch_index"], 2)
        validator.validate_batch(next_batch, shape)

    def test_checkpoint_validation_rejects_stale_next_batch_state(self) -> None:
        payload, shape, args, config = self._candidate_payload()
        payload["next_batch_index"] = args.steps + 1
        with self.assertRaisesRegex(ValueError, "next_batch_index"):
            validator.validate_checkpoint_payload(
                payload,
                shape=shape,
                config=config,
                args=args,
            )


class JsonResultContractTests(unittest.TestCase):
    def test_main_emits_machine_readable_pass_record(self) -> None:
        report = {
            "event": "validation",
            "status": "PASS",
            "scope": validator.VALIDATION_SCOPE,
        }
        stream = io.StringIO()
        with mock.patch.object(validator, "run_validation", return_value=report):
            with redirect_stdout(stream):
                validator.main([])
        parsed = json.loads(stream.getvalue().strip().splitlines()[-1])
        self.assertEqual(parsed, report)

    def test_main_emits_failure_json_and_reraises(self) -> None:
        stream = io.StringIO()
        with mock.patch.object(
            validator,
            "run_validation",
            side_effect=RuntimeError("synthetic failure"),
        ):
            with redirect_stdout(stream):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    validator.main([])
        parsed = json.loads(stream.getvalue().strip().splitlines()[-1])
        self.assertEqual(parsed["status"], "FAIL")
        self.assertEqual(parsed["scope"], validator.VALIDATION_SCOPE)
        self.assertEqual(parsed["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
