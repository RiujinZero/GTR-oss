from __future__ import annotations

import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest import mock

import benchmark_convergent_routing
import run_blackwell_release_gate as gate
import validate_convergent_fft_backend
import validate_convergent_fft_training
import validate_fused_fft_backend


class BlackwellReleaseGateContractTests(unittest.TestCase):
    def test_evidence_identity_uses_environment_and_cli_overrides(self) -> None:
        environment_revision = "A" * 40
        environment_bundle = "B" * 64
        with mock.patch.dict(
            "os.environ",
            {
                "GTR_SOURCE_REVISION": environment_revision,
                "GTR_BUNDLE_SHA256": environment_bundle,
            },
        ):
            environment_args = gate.parse_args([])
            cli_args = gate.parse_args(
                [
                    "--source-revision",
                    "C" * 40,
                    "--bundle-sha256",
                    "D" * 64,
                ]
            )
        self.assertEqual(environment_args.source_revision, "a" * 40)
        self.assertEqual(environment_args.bundle_sha256, "b" * 64)
        self.assertEqual(cli_args.source_revision, "c" * 40)
        self.assertEqual(cli_args.bundle_sha256, "d" * 64)

    def test_malformed_evidence_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "40-character"):
            gate.parse_args(["--source-revision", "abc123"])
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            gate.parse_args(["--bundle-sha256", "z" * 64])

    def test_runtime_environment_records_build_versions(self) -> None:
        properties = mock.Mock(total_memory=96 * 1024**3)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            mock.patch("torch.cuda.get_device_name", return_value="Blackwell"),
            mock.patch("torch.cuda.get_device_properties", return_value=properties),
            mock.patch.object(
                gate,
                "triton_packed_reader_available",
                return_value=True,
            ),
            mock.patch.object(
                gate,
                "triton_packed_writer_available",
                return_value=True,
            ),
            mock.patch.object(gate, "package_version", return_value="3.3.1"),
        ):
            environment = gate.require_target_environment(
                "cuda:0",
                require_blackwell=True,
            )
        self.assertEqual(environment["python_version"], platform.python_version())
        self.assertEqual(environment["triton_version"], "3.3.1")
        self.assertTrue(environment["triton_writer_available"])
        self.assertIn("torch_git_version", environment)
        self.assertEqual(environment["cuda_capability"], [12, 0])

    def test_full_model_equivalence_is_required_by_default(self) -> None:
        args = gate.parse_args([])
        self.assertEqual(args.full_model_equivalence, "required")
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(
                gate,
                "full_model_equivalence_available",
                return_value=True,
            ):
                stages = gate.build_stages(args, output_dir=Path(root))
        self.assertEqual(len(stages), 15)
        self.assertEqual(
            [
                stage.name
                for stage in stages
                if stage.name.startswith("full_model_equivalence")
            ],
            [
                "full_model_equivalence_seed0_noncausal",
                "full_model_equivalence_seed0_causal",
                "full_model_equivalence_seed1_noncausal",
                "full_model_equivalence_seed1_causal",
            ],
        )

    def test_default_gate_covers_seed_and_causality_matrix(self) -> None:
        args = gate.parse_args(
            [
                "--full-model-equivalence",
                "skip",
                "--training-resume",
                "skip",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            stages = gate.build_stages(args, output_dir=Path(root))
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "cuda_unit_tests",
                "quick_validator_seed0_noncausal",
                "quick_validator_seed0_causal",
                "quick_validator_seed1_noncausal",
                "quick_validator_seed1_causal",
                "production_validator_seed0_noncausal",
                "production_validator_seed0_causal",
                "production_validator_seed1_noncausal",
                "production_validator_seed1_causal",
            ],
        )
        self.assertTrue(all(stage.command[0] == sys.executable for stage in stages))
        self.assertEqual(
            Path(stages[0].command[2]),
            gate.CUDA_UNIT_GATE_SCRIPT,
        )
        self.assertIn("tests.test_triton_packed_writer", stages[0].command)
        self.assertTrue(Path(stages[1].command[2]).is_absolute())

    def test_only_production_causal_receiver_gets_noise_envelope(self) -> None:
        args = gate.parse_args(
            [
                "--full-model-equivalence",
                "skip",
                "--training-resume",
                "skip",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            stages = gate.build_stages(args, output_dir=Path(root))

        validator_stages = stages[1:]
        for stage in validator_stages:
            parsed = validate_fused_fft_backend.parse_args(
                list(stage.command[3:])
            )
            is_production_causal = (
                stage.name.startswith("production_validator_")
                and stage.name.rsplit("_", 1)[-1] == "causal"
            )
            if is_production_causal:
                self.assertEqual(parsed.grad_mass_atol, 1e-3)
                self.assertEqual(parsed.grad_mass_rtol, 1.5e-3)
                self.assertEqual(parsed.grad_mass_max_relative_l2, 2e-5)
                self.assertEqual(parsed.grad_mass_max_absolute, 2.5e-2)
                self.assertEqual(parsed.grad_mass_max_outliers, 4)
                self.assertEqual(parsed.grad_mass_max_outlier_fraction, 2e-5)
            else:
                self.assertIsNone(parsed.grad_mass_atol)
                self.assertIsNone(parsed.grad_mass_rtol)
                self.assertIsNone(parsed.grad_mass_max_relative_l2)
                self.assertIsNone(parsed.grad_mass_max_absolute)
                self.assertIsNone(parsed.grad_mass_max_outliers)
                self.assertIsNone(parsed.grad_mass_max_outlier_fraction)

    def test_noise_envelope_must_be_finite_and_nonnegative(self) -> None:
        for arguments in (
            ["--production-causal-grad-mass-atol", "nan"],
            ["--production-causal-grad-mass-rtol", "inf"],
            ["--production-causal-grad-mass-max-relative-l2", "-1"],
            ["--production-causal-grad-mass-max-absolute", "-1"],
            ["--production-causal-grad-mass-max-outliers", "-1"],
            ["--production-causal-grad-mass-max-outlier-fraction", "nan"],
            ["--production-causal-grad-mass-max-outlier-fraction", "2"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "finite|nonnegative"):
                    gate.parse_args(arguments)

    def test_performance_ab_is_optional_last_and_artifact_isolated(self) -> None:
        args = gate.parse_args(
            [
                "--performance-ab",
                "--full-model-equivalence",
                "skip",
                "--training-resume",
                "skip",
                "--benchmark-batch-size",
                "2",
                "--benchmark-warmup",
                "1",
                "--benchmark-iterations",
                "3",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            stages = gate.build_stages(args, output_dir=root_path)
            performance = stages[-4:]
            self.assertEqual(
                [stage.name for stage in performance],
                [
                    "benchmark_dense_matrix_torch",
                    "benchmark_packed_fft_opaque_torch",
                    "benchmark_packed_fft_opaque_triton_reader_torch_writer",
                    "benchmark_packed_fft_opaque_triton_reader_triton_writer",
                ],
            )
            parsed = [
                benchmark_convergent_routing.parse_args(
                    list(stage.command[3:])
                )
                for stage in performance
            ]
        self.assertEqual(
            [
                (
                    item.route_backend,
                    item.packed_memory_backend,
                    item.packed_memory_reader_backend,
                    item.packed_memory_writer_backend,
                )
                for item in parsed
            ],
            [
                ("dense", "matrix", "torch", "torch"),
                ("packed", "fft_opaque", "torch", "torch"),
                ("packed", "fft_opaque", "triton", "torch"),
                ("packed", "fft_opaque", "triton", "triton"),
            ],
        )
        self.assertTrue(all(item.compile for item in parsed))
        self.assertTrue(all(item.phases == ("training_step",) for item in parsed))
        self.assertTrue(
            all(item.decode_workload == "teacher_forced" for item in parsed)
        )
        self.assertTrue(all(item.gradient_clip == 1.0 for item in parsed))
        frozen_evidence_knobs = {
            (
                item.batch_size,
                item.context_tokens,
                item.output_slots,
                item.vocab_size,
                item.embedding_dim,
                item.d_model,
                item.nodes,
                item.lanes,
                item.relays,
                item.degree,
                item.router_rank,
                item.active_node_top_k,
                item.packed_memory_feature_chunk,
                item.warmup_steps,
                item.measured_steps,
                item.seed,
                item.phases,
            )
            for item in parsed
        }
        self.assertEqual(len(frozen_evidence_knobs), 1)
        only_knobs = next(iter(frozen_evidence_knobs))
        self.assertEqual(only_knobs[0], 2)
        self.assertEqual(only_knobs[14], 3)
        self.assertEqual(len({item.json_output for item in parsed}), 4)
        self.assertTrue(all(stage.continue_on_failure for stage in performance))

    def test_validator_stage_commands_are_executable_by_the_parser(self) -> None:
        args = gate.parse_args(
            [
                "--quick-tokens",
                "11",
                "--production-tokens",
                "96",
                "--production-width",
                "32",
                "--causal",
                "--full-model-equivalence",
                "skip",
                "--training-resume",
                "skip",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            stages = gate.build_stages(args, output_dir=Path(root))
        for stage in stages[1:]:
            parsed = validate_fused_fft_backend.parse_args(
                list(stage.command[3:])
            )
            self.assertTrue(parsed.compile)
            self.assertTrue(parsed.causal)
        self.assertEqual(
            validate_fused_fft_backend.parse_args(
                list(stages[1].command[3:])
            ).tokens,
            11,
        )

    def test_seed_and_mode_overrides_reduce_correctness_matrix(self) -> None:
        args = gate.parse_args(
            [
                "--validation-seed",
                "7",
                "--causal-mode",
                "noncausal",
                "--full-model-equivalence",
                "skip",
                "--training-resume",
                "skip",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            stages = gate.build_stages(args, output_dir=Path(root))
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "cuda_unit_tests",
                "quick_validator_seed7_noncausal",
                "production_validator_seed7_noncausal",
            ],
        )

    def test_full_model_equivalence_auto_and_required_contract(self) -> None:
        auto = gate.parse_args(["--full-model-equivalence", "auto"])
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(
                gate,
                "full_model_equivalence_available",
                return_value=True,
            ):
                stages = gate.build_stages(auto, output_dir=Path(root))
        equivalence = [
            stage for stage in stages if stage.name.startswith("full_model_")
        ]
        self.assertEqual(
            [stage.name for stage in equivalence],
            [
                "full_model_equivalence_seed0_noncausal",
                "full_model_equivalence_seed0_causal",
                "full_model_equivalence_seed1_noncausal",
                "full_model_equivalence_seed1_causal",
            ],
        )
        command = " ".join(equivalence[0].command)
        self.assertIn("validate_convergent_fft_backend.py", command)
        self.assertIn("--seed 0", command)
        self.assertIn("--no-causal", command)
        self.assertIn("--compile", command)
        for stage in equivalence:
            parsed = validate_convergent_fft_backend.parse_args(
                list(stage.command[3:])
            )
            self.assertTrue(parsed.compile)

        required = gate.parse_args(
            [
                "--full-model-equivalence",
                "required",
                "--training-resume",
                "skip",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(
                gate,
                "full_model_equivalence_available",
                return_value=False,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "is missing"):
                    gate.build_stages(required, output_dir=Path(root))

    def test_training_resume_is_required_and_parser_compatible(self) -> None:
        args = gate.parse_args(
            [
                "--validation-seed",
                "7",
                "--full-model-equivalence",
                "skip",
                "--training-steps",
                "11",
                "--training-seed",
                "3",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            with mock.patch.object(
                gate,
                "training_resume_available",
                return_value=True,
            ):
                stages = gate.build_stages(args, output_dir=root_path)
            training = [
                stage
                for stage in stages
                if stage.name.startswith("candidate_training_resume")
            ]
            self.assertEqual(
                [stage.name for stage in training],
                [
                    "candidate_training_resume_seed3_noncausal",
                    "candidate_training_resume_seed3_causal",
                ],
            )
            parsed = [
                validate_convergent_fft_training.parse_args(
                    list(stage.command[3:])
                )
                for stage in training
            ]
        self.assertTrue(all(item.compile for item in parsed))
        self.assertEqual([item.causal for item in parsed], [False, True])
        self.assertTrue(all(item.steps == 11 for item in parsed))
        self.assertTrue(all(item.seed == 3 for item in parsed))
        self.assertEqual(
            len({item.checkpoint_path for item in parsed}),
            2,
        )

        required = gate.parse_args(
            [
                "--full-model-equivalence",
                "skip",
                "--training-resume",
                "required",
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(
                gate,
                "training_resume_available",
                return_value=False,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "is missing"):
                    gate.build_stages(required, output_dir=Path(root))

    def test_execute_stages_stops_on_first_failure(self) -> None:
        specs = (
            gate.StageSpec("one", ("one",)),
            gate.StageSpec("two", ("two",)),
            gate.StageSpec("three", ("three",)),
        )
        calls = []

        def runner(spec, output_dir, combined):
            del output_dir, combined
            calls.append(spec.name)
            return gate.StageResult(
                name=spec.name,
                command=spec.command,
                log_path=f"{spec.name}.log",
                started_utc="start",
                completed_utc="end",
                duration_seconds=0.1,
                return_code=4 if spec.name == "two" else 0,
                status="failed" if spec.name == "two" else "passed",
            )

        results = gate.execute_stages(
            specs,
            Path("unused"),
            io.StringIO(),
            runner=runner,
        )
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual([result.name for result in results], ["one", "two"])

    def test_performance_stages_continue_for_complete_failed_evidence(self) -> None:
        specs = (
            gate.StageSpec("a", ("a",), continue_on_failure=True),
            gate.StageSpec("b", ("b",), continue_on_failure=True),
        )
        calls = []

        def runner(spec, output_dir, combined):
            del output_dir, combined
            calls.append(spec.name)
            return gate.StageResult(
                name=spec.name,
                command=spec.command,
                log_path=f"{spec.name}.log",
                started_utc="start",
                completed_utc="end",
                duration_seconds=0.1,
                return_code=1 if spec.name == "a" else 0,
                status="failed" if spec.name == "a" else "passed",
            )

        results = gate.execute_stages(
            specs,
            Path("unused"),
            io.StringIO(),
            runner=runner,
        )
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual([result.status for result in results], ["failed", "passed"])

    def test_run_stage_streams_and_records_subprocess_output(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            combined = io.StringIO()
            spec = gate.StageSpec(
                "cpu_contract",
                (
                    sys.executable,
                    "-c",
                    "print('release-gate-stream-test', flush=True)",
                ),
            )
            result = gate.run_stage(spec, root_path, combined)
            stage_text = (root_path / "cpu_contract.log").read_text(
                encoding="utf-8"
            )
        self.assertEqual(result.status, "passed")
        self.assertIn("release-gate-stream-test", stage_text)
        self.assertIn("release-gate-stream-test", combined.getvalue())

    def test_atomic_summary_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "summary.json"
            gate.write_summary(path, {"status": "failed", "stages": []})
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_suffix(".json.tmp").exists())
        self.assertEqual(parsed["status"], "failed")

    def test_preflight_rejects_cpu_without_claiming_a_pass(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                gate.require_target_environment(
                    "cuda:0",
                    require_blackwell=True,
                )

    def test_main_persists_failed_preflight_summary(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "failed_gate"
            with mock.patch("torch.cuda.is_available", return_value=False):
                return_code = gate.main(
                    [
                        "--output-dir",
                        str(output),
                        "--allow-missing-identity",
                    ]
                )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            log = (output / "release_gate.log").read_text(encoding="utf-8")
        self.assertEqual(return_code, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stages"], [])
        self.assertIn("requires CUDA", summary["preflight_error"])
        self.assertIn("release_gate_error", log)

    def test_main_rejects_missing_release_identity(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "missing_identity"
            return_code = gate.main(["--output-dir", str(output)])
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(return_code, 1)
        self.assertIn("requires --source-revision", summary["preflight_error"])

    def test_main_records_evidence_identity_in_environment(self) -> None:
        revision = "1" * 40
        bundle = "2" * 64
        runtime = {
            "python_version": "3.test",
            "python_implementation": "CPython",
            "torch_version": "2.test",
            "torch_git_version": "3" * 40,
            "torch_cuda_version": "13.test",
            "triton_version": "4.test",
            "device": "cuda:0",
            "gpu": "Blackwell",
            "cuda_capability": [12, 0],
            "total_vram_bytes": 1,
            "triton_reader_available": True,
            "triton_writer_available": True,
        }
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "evidence_gate"
            with (
                mock.patch.object(
                    gate,
                    "require_target_environment",
                    return_value=runtime,
                ),
                mock.patch.object(gate, "build_stages", return_value=()),
            ):
                return_code = gate.main(
                    [
                        "--output-dir",
                        str(output),
                        "--source-revision",
                        revision,
                        "--bundle-sha256",
                        bundle,
                    ]
                )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(return_code, 0)
        self.assertEqual(summary["evidence_identity"]["source_revision"], revision)
        self.assertEqual(summary["environment"]["source_revision"], revision)
        self.assertEqual(summary["environment"]["bundle_sha256"], bundle)


if __name__ == "__main__":
    unittest.main()
