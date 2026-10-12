from pathlib import Path
import tempfile
import unittest

from run_teacher_forced_language_colab import (
    EXPECTED_PARAMETERS,
    build_command,
    preflight_parameter_count,
    validate_checkpoint_dir,
    write_run_spec,
)


class TeacherForcedColabRunnerTests(unittest.TestCase):
    def test_frozen_parameter_counts(self) -> None:
        for width, expected in EXPECTED_PARAMETERS.items():
            with self.subTest(width=width):
                self.assertEqual(preflight_parameter_count(width), expected)

    def test_d20_command_preserves_capacity_ab_controls(self) -> None:
        command = build_command(
            width=20,
            steps=30_000,
            log_every=1_000,
            checkpoint_dir=Path("/tmp/control"),
            compile_model=True,
            decode_mode="teacher_forced",
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--decode-mode teacher_forced", joined)
        self.assertIn("--formation-objective slot_ce", joined)
        self.assertIn("--d-model 20", joined)
        self.assertIn("--context-tokens 512", joined)
        self.assertIn("--response-tokens 128", joined)
        self.assertIn("--batch-size 8", joined)
        self.assertIn("--eval-examples 0", joined)
        self.assertIn("--streaming-exhaustion restart", joined)
        self.assertIn("--terminal-vote-mode slot", joined)
        self.assertIn(
            "--tinystories-data "
            "/content/drive/MyDrive/GTR/training_data/"
            "TinyStories-train.txt",
            joined,
        )
        self.assertIn(
            "--tinystories-validation-data "
            "/content/drive/MyDrive/GTR/training_data/"
            "TinyStories-valid.txt",
            joined,
        )
        self.assertIn("--tinystories-mix-fraction 0.40", joined)
        self.assertIn("--compile", command)

    def test_handoff_command_exposes_requested_accuracy_gate(self) -> None:
        command = build_command(
            width=20,
            steps=100_000,
            log_every=1_000,
            checkpoint_dir=Path("/tmp/handoff"),
            compile_model=True,
            decode_mode="handoff",
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--decode-mode handoff", joined)
        self.assertIn(
            "--handoff-start-content-accuracy 0.60",
            joined,
        )
        self.assertIn(
            "--handoff-end-content-accuracy 0.90",
            joined,
        )
        self.assertIn("--handoff-accuracy-ema-decay 0.50", joined)

    def test_graduated_handoff_command_exposes_cadence_and_horizon(
        self,
    ) -> None:
        command = build_command(
            width=400,
            steps=800_000,
            log_every=10_000,
            checkpoint_dir=Path("/tmp/graduated"),
            compile_model=True,
            decode_mode="graduated_handoff",
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--decode-mode graduated_handoff", joined)
        self.assertIn("--graduated-baseline-fraction 0.05", joined)
        self.assertIn("--graduated-accuracy-step 0.05", joined)
        self.assertIn("--graduated-release-accuracy 0.90", joined)
        self.assertIn("--handoff-accuracy-ema-decay 0.50", joined)
        self.assertIn("--scheduler-total-steps 800000", joined)

    def test_prefix_denoise_command_preserves_mix_and_support_curriculum(
        self,
    ) -> None:
        command = build_command(
            width=20,
            steps=1_000_000,
            log_every=5_000,
            eval_examples=256,
            checkpoint_dir=Path("/tmp/prefix-denoise"),
            compile_model=True,
            decode_mode="prefix_denoise",
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--decode-mode prefix_denoise", joined)
        self.assertIn("--streaming-exhaustion restart", joined)
        self.assertIn("--prefix-retention-start 0.95", joined)
        self.assertIn("--prefix-retention-floor 0.50", joined)
        self.assertIn("--prefix-denoise-warmup-steps 100000", joined)
        self.assertIn("--prefix-zero-row-fraction 0.05", joined)
        self.assertIn("--prefix-short-row-fraction 0.10", joined)
        self.assertIn("--prefix-span-fraction 0.50", joined)
        self.assertIn("--prefix-context-utility-start 0.02", joined)
        self.assertIn("--prefix-context-utility-end 0.10", joined)

    def test_token_handoff_command_removes_one_teacher_token_at_a_time(
        self,
    ) -> None:
        command = build_command(
            width=20,
            steps=2_000_000,
            log_every=5_000,
            batch_size=32,
            eval_batch_size=32,
            eval_examples=64,
            checkpoint_dir=Path("/tmp/token-handoff"),
            compile_model=True,
            decode_mode="token_handoff",
            resume_checkpoint=Path("/tmp/teacher-best.pt"),
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--decode-mode token_handoff", joined)
        self.assertIn("--token-handoff-start-removed 1", joined)
        self.assertIn("--token-handoff-max-removed 0", joined)
        self.assertIn("--token-handoff-advance-ratio 0.95", joined)
        self.assertIn("--token-handoff-retreat-ratio 0.75", joined)
        self.assertIn("--token-handoff-full-probe-fraction 0.02", joined)
        self.assertIn(
            f"--resume-checkpoint {Path('/tmp/teacher-best.pt')}",
            joined,
        )
        self.assertIn("--resume-optimizer", joined)

    def test_chunk_parallel_command_forms_eight_tokens_from_gold_chunks(
        self,
    ) -> None:
        command = build_command(
            width=20,
            steps=200_000,
            log_every=5_000,
            eval_examples=64,
            checkpoint_dir=Path("/tmp/chunk8"),
            compile_model=True,
            decode_mode="chunk_parallel",
            chunk_size=8,
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--decode-mode chunk_parallel", joined)
        self.assertIn("--chunk-size 8", joined)
        # References stay long; only the selected next chunk is formed.
        self.assertIn("--response-tokens 128", joined)
        self.assertIn("--context-tokens 512", joined)
        self.assertIn("--d-model 20", joined)
        self.assertIn("--terminal-vote-mode slot", joined)

    def test_eval_subset_is_forwarded_to_trainer(self) -> None:
        command = build_command(
            width=20,
            steps=1_000,
            log_every=100,
            eval_examples=64,
            checkpoint_dir=Path("/tmp/eval-subset"),
            compile_model=True,
            decode_mode="handoff",
            python_executable="python",
        )
        self.assertIn("--eval-examples 64", " ".join(command))

    def test_batch_sizes_are_exposed_without_changing_defaults(self) -> None:
        command = build_command(
            width=20,
            steps=1_000,
            log_every=100,
            batch_size=128,
            eval_batch_size=64,
            checkpoint_dir=Path("/tmp/batch-128"),
            compile_model=True,
            decode_mode="teacher_forced",
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--batch-size 128", joined)
        self.assertIn("--eval-batch-size 64", joined)

    def test_run_spec_and_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "run"
            self.assertEqual(
                validate_checkpoint_dir(checkpoint_dir),
                checkpoint_dir,
            )
            spec = write_run_spec(
                checkpoint_dir,
                width=20,
                steps=30_000,
                parameter_count=130_442,
                command=("python", "trainer.py"),
                decode_mode="teacher_forced",
            )
            text = spec.read_text(encoding="utf-8")
            self.assertIn('"d_model": 20', text)
            self.assertIn('"parameters": 130442', text)
            self.assertIn('"decode_mode": "teacher_forced"', text)
            self.assertIn('"roneneldan/TinyStories": 0.4', text)
            with self.assertRaises(FileExistsError):
                validate_checkpoint_dir(checkpoint_dir)


if __name__ == "__main__":
    unittest.main()
