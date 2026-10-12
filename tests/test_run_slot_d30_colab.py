import tempfile
import unittest
from pathlib import Path

from run_slot_d30_colab import (
    EXPECTED_MODE,
    EXPECTED_PARAMETERS,
    build_command,
    preflight_parameter_count,
    validate_checkpoint_dir,
)


class SlotD30ColabRunnerTests(unittest.TestCase):
    def test_exact_parameter_preflight(self) -> None:
        self.assertEqual(preflight_parameter_count(), EXPECTED_PARAMETERS)

    def test_command_has_slot_mode_and_distinct_checkpoint(self) -> None:
        checkpoint = Path("/tmp/formation_slotjury_v1")
        command = build_command(
            checkpoint_dir=checkpoint,
            steps=300_000,
            log_every=1_000,
            compile_model=True,
            python_executable="python",
        )
        mode_index = command.index("--terminal-vote-mode")
        checkpoint_index = command.index("--checkpoint-dir")
        self.assertEqual(command[mode_index + 1], EXPECTED_MODE)
        self.assertEqual(command[checkpoint_index + 1], str(checkpoint))
        self.assertIn("--compile", command)
        precision_index = command.index("--float32-matmul-precision")
        self.assertEqual(command[precision_index + 1], "high")
        self.assertIn("--cudnn-tf32", command)
        self.assertEqual(command[command.index("--steps") + 1], "300000")
        self.assertEqual(
            command[command.index("--scheduler-total-steps") + 1],
            "300000",
        )

    def test_refuses_ambiguous_checkpoint_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "without 'slot'"):
            validate_checkpoint_dir(Path("/tmp/global_control"))

    def test_refuses_nonempty_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "formation_slot"
            checkpoint.mkdir()
            (checkpoint / "best.pt").write_bytes(b"occupied")
            with self.assertRaisesRegex(FileExistsError, "refusing"):
                validate_checkpoint_dir(checkpoint)


if __name__ == "__main__":
    unittest.main()
