from __future__ import annotations

import io
import importlib
from pathlib import Path
import unittest
from unittest import mock

import run_cuda_unit_gate


class _PassingTest(unittest.TestCase):
    def runTest(self) -> None:
        self.assertTrue(True)


class _SkippedTest(unittest.TestCase):
    @unittest.skip("target path unavailable")
    def runTest(self) -> None:
        self.fail("skip decorator should prevent execution")


class CudaUnitGateTests(unittest.TestCase):
    def test_bundled_tests_are_an_explicit_importable_package(self) -> None:
        package = importlib.import_module("tests")
        package_file = Path(package.__file__).resolve()
        self.assertEqual(package_file.name, "__init__.py")
        self.assertEqual(package_file.parent, Path(__file__).resolve().parent)

    def _run(self, case: unittest.TestCase) -> tuple[int, str]:
        stream = io.StringIO()
        with mock.patch.object(
            unittest.defaultTestLoader,
            "loadTestsFromNames",
            return_value=unittest.TestSuite([case]),
        ):
            return_code = run_cuda_unit_gate.run_modules(
                ["synthetic"],
                stream=stream,
            )
        return return_code, stream.getvalue()

    def test_pass_requires_at_least_one_unskipped_success(self) -> None:
        return_code, output = self._run(_PassingTest())
        self.assertEqual(return_code, 0)
        self.assertIn('"skipped": 0', output)
        self.assertIn('"tests_run": 1', output)

    def test_skip_is_a_gate_failure(self) -> None:
        return_code, output = self._run(_SkippedTest())
        self.assertEqual(return_code, 1)
        self.assertIn('"skipped": 1', output)


if __name__ == "__main__":
    unittest.main()
