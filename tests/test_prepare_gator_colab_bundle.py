from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from prepare_gator_colab_bundle import build_bundle


class PrepareGatorColabBundleTests(unittest.TestCase):
    def test_bundle_contains_explicit_tests_package_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "gator_colab_bundle.zip"
            build_bundle(output)
            with ZipFile(output) as archive:
                names = set(archive.namelist())

        self.assertIn("tests/__init__.py", names)
        self.assertIn("tests/test_triton_packed_memory.py", names)
        self.assertIn("tests/test_triton_packed_writer.py", names)
        self.assertIn("tests/test_opaque_fft_memory.py", names)
        self.assertIn("gator/models/triton_packed_writer.py", names)

    def test_extracted_bundle_wins_over_conflicting_tests_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output = temporary_root / "gator_colab_bundle.zip"
            source_root = temporary_root / "source"
            fake_packages = temporary_root / "fake_packages"
            fake_tests = fake_packages / "tests"
            fake_tests.mkdir(parents=True)
            (fake_tests / "__init__.py").write_text(
                'raise RuntimeError("shadow package imported")\n',
                encoding="utf-8",
            )
            build_bundle(output)
            with ZipFile(output) as archive:
                archive.extractall(source_root)

            environment = os.environ.copy()
            existing_python_path = environment.get("PYTHONPATH")
            python_path_parts = [str(fake_packages)]
            if existing_python_path:
                python_path_parts.append(existing_python_path)
            environment["PYTHONPATH"] = os.pathsep.join(python_path_parts)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import tests; "
                        "print(Path(tests.__file__).resolve())"
                    ),
                ],
                cwd=source_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            Path(result.stdout.strip()),
            (source_root / "tests" / "__init__.py").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
