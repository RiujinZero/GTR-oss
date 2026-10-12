from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_gator_140m_colab import (
    DriveMirror,
    EXPECTED_PARAMETERS,
    build_command,
    default_run_name,
    parse_args,
    preflight_parameter_count,
    write_run_spec,
)


class Gator140MRunnerTests(unittest.TestCase):
    def test_frozen_parameter_count(self) -> None:
        self.assertEqual(preflight_parameter_count(), EXPECTED_PARAMETERS)
        self.assertEqual(EXPECTED_PARAMETERS, 140_193_699)

    def test_command_freezes_shared_routing_and_local_checkpoint(self) -> None:
        checkpoint = Path("/content/local/run")
        command = build_command(
            steps=100,
            batch_size=2,
            eval_examples=8,
            log_every=10,
            checkpoint_dir=checkpoint,
            compile_model=True,
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--router-mode shared_low_rank", joined)
        self.assertIn("--router-rank 96", joined)
        self.assertIn("--ingress-mode learned", joined)
        self.assertIn("--active-node-top-k 32", joined)
        self.assertIn("--route-backend dense", joined)
        self.assertIn("--packed-memory-backend matrix", joined)
        self.assertIn("--packed-memory-reader-backend torch", joined)
        self.assertIn("--packed-memory-writer-backend torch", joined)
        self.assertIn("--terminal-vote-mode slot", joined)
        self.assertIn("--batch-size 2", joined)
        self.assertIn(f"--checkpoint-dir {checkpoint}", joined)
        self.assertTrue(command[-1] == "--compile")

    def test_command_exposes_packed_receiver_backend(self) -> None:
        command = build_command(
            steps=10,
            batch_size=6,
            eval_examples=8,
            log_every=5,
            checkpoint_dir=Path("/content/local/packed"),
            compile_model=False,
            route_backend="packed",
            packed_memory_backend="fft_opaque",
            packed_memory_reader_backend="triton",
            packed_memory_writer_backend="triton",
            packed_memory_feature_chunk=64,
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--route-backend packed", joined)
        self.assertIn("--packed-memory-backend fft_opaque", joined)
        self.assertIn("--packed-memory-reader-backend triton", joined)
        self.assertIn("--packed-memory-writer-backend triton", joined)
        self.assertIn("--packed-memory-feature-chunk 64", joined)
        self.assertNotIn("--compile", command)

    def test_compiled_fft_is_rejected_before_launch(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "run_gator_140m_colab.py",
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft",
                "--compile",
            ],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_eager_fft_remains_available(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "run_gator_140m_colab.py",
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft",
                "--no-compile",
            ],
        ):
            args = parse_args()
        self.assertFalse(args.compile)
        self.assertEqual(args.packed_memory_backend, "fft")

    def test_compiled_opaque_fft_with_triton_reader_is_available(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "run_gator_140m_colab.py",
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft_opaque",
                "--packed-memory-reader-backend",
                "triton",
                "--packed-memory-writer-backend",
                "triton",
                "--compile",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.compile)
        self.assertEqual(args.packed_memory_backend, "fft_opaque")
        self.assertEqual(args.packed_memory_reader_backend, "triton")
        self.assertEqual(args.packed_memory_writer_backend, "triton")

    def test_default_identity_and_run_spec_include_memory_backends(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "run_gator_140m_colab.py",
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft_opaque",
                "--packed-memory-reader-backend",
                "triton",
                "--packed-memory-writer-backend",
                "triton",
            ],
        ):
            args = parse_args()
        run_name = default_run_name(args, "20260721_120000")
        self.assertIn(
            "packed_fft_opaque_reader-triton_writer-triton",
            run_name,
        )

        with tempfile.TemporaryDirectory() as root:
            spec = Path(root) / "RUN_SPEC.json"
            write_run_spec(
                spec,
                run_name=run_name,
                command=["python", "gator_language_train.py"],
                args=args,
            )
            payload = json.loads(spec.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["architecture"]["packed_memory_reader_backend"],
            "triton",
        )
        self.assertEqual(
            payload["architecture"]["packed_memory_writer_backend"],
            "triton",
        )

    def test_log_batches_by_step_reports(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            local = root_path / "local"
            drive = root_path / "drive"
            local.mkdir()
            now = [0.0]
            mirror = DriveMirror(
                local_dir=local,
                drive_dir=drive,
                sync_seconds=900,
                log_updates_per_sync=2,
                clock=lambda: now[0],
            )
            mirror.observe("step=100 loss=1\n")
            mirror.observe("context='x'\n")
            self.assertFalse((drive / "train.log").exists())
            mirror.observe("step=200 loss=0.9\n")
            self.assertEqual(
                (drive / "train.log").read_text(encoding="utf-8"),
                "step=100 loss=1\ncontext='x'\nstep=200 loss=0.9\n",
            )

    def test_time_sync_and_final_checkpoint_copy(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            local = root_path / "local"
            drive = root_path / "drive"
            local.mkdir()
            now = [0.0]
            mirror = DriveMirror(
                local_dir=local,
                drive_dir=drive,
                sync_seconds=10,
                log_updates_per_sync=20,
                clock=lambda: now[0],
            )
            mirror.observe("hello\n")
            now[0] = 11.0
            mirror.observe("world\n")
            self.assertEqual(
                (drive / "train.log").read_text(encoding="utf-8"),
                "hello\nworld\n",
            )
            (local / "latest.pt").write_bytes(b"latest")
            (local / "best.pt").write_bytes(b"best")
            mirror.finalize()
            self.assertEqual((drive / "latest.pt").read_bytes(), b"latest")
            self.assertEqual((drive / "best.pt").read_bytes(), b"best")


if __name__ == "__main__":
    unittest.main()
