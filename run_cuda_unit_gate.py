"""Run named CUDA unit-test modules and fail when any test is skipped.

Ordinary ``python -m unittest`` exits zero when a test is skipped.  That is
useful for portable development suites but unsafe for a target-GPU release
gate: an unavailable CUDA/Triton path must not masquerade as a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO
import unittest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="+", help="unittest module names")
    return parser


def run_modules(modules: Sequence[str], *, stream: TextIO) -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(list(modules))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "event": "cuda_unit_gate",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
    }
    stream.write(json.dumps(report, sort_keys=True) + "\n")
    stream.flush()
    unacceptable = (
        report["failures"]
        + report["errors"]
        + report["skipped"]
        + report["expected_failures"]
        + report["unexpected_successes"]
    )
    return 0 if report["tests_run"] > 0 and unacceptable == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_modules(args.modules, stream=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
