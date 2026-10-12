from pathlib import Path
import tempfile
import unittest

import torch

from archive_teacher_forced_colab_run import archive_run, file_sha256


class ArchiveTeacherForcedRunTests(unittest.TestCase):
    def test_archive_copies_bundle_and_records_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            checkpoint = {
                "format": "gator-language-v2",
                "step": 123,
                "model": {"weight": torch.ones(2, 3)},
                "model_config": {"d_model": 20},
                "task_config": {"context_tokens": 512},
                "data_config": {"decode_mode": "teacher_forced"},
                "validation": {"loss": 2.0},
                "training_evaluation": None,
            }
            torch.save(checkpoint, run_dir / "best.pt")
            bundle = root / "bundle.zip"
            bundle.write_bytes(b"frozen source")

            manifest = archive_run(run_dir, bundle)

            self.assertTrue(manifest.is_file())
            frozen_bundle = run_dir / "SOURCE_BUNDLE.zip"
            self.assertEqual(file_sha256(frozen_bundle), file_sha256(bundle))
            text = manifest.read_text(encoding="utf-8")
            self.assertIn('"step": 123', text)
            self.assertIn('"parameter_tensors_numel": 6', text)
            self.assertIn('"decode_mode": "teacher_forced"', text)


if __name__ == "__main__":
    unittest.main()
