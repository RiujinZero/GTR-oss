from __future__ import annotations

import argparse
import unittest

from gator.cli import (
    add_data_pipeline_arguments,
    data_pipeline_config_from_args,
    preparation_config_from_args,
)


class DataPipelineCliTests(unittest.TestCase):
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_data_pipeline_arguments(parser)
        return parser

    def test_defaults_enable_bounded_prefetch_but_not_process_workers(self):
        args = self.parser().parse_args([])

        batch = data_pipeline_config_from_args(
            args,
            batch_size=32,
            device="cpu",
        )
        preparation = preparation_config_from_args(args)

        self.assertEqual(batch.mode, "prefetch")
        self.assertEqual(batch.prefetch_batches, 2)
        self.assertFalse(batch.pin_memory)
        self.assertEqual(preparation.backend, "serial")
        self.assertEqual(preparation.chunk_size, 256)

    def test_explicit_parallel_and_transfer_policy_round_trips(self):
        args = self.parser().parse_args(
            [
                "--batch-pipeline",
                "serial",
                "--prefetch-batches",
                "4",
                "--pin-memory",
                "--no-non-blocking-transfer",
                "--no-dedicated-copy-stream",
                "--resume-data-state",
                "exact",
                "--tokenizer-workers",
                "3",
                "--tokenizer-worker-backend",
                "threads",
                "--tokenizer-chunk-size",
                "17",
                "--tokenizer-pending-chunks",
                "5",
            ]
        )

        batch = data_pipeline_config_from_args(
            args,
            batch_size=8,
            device="cuda:0",
        )
        preparation = preparation_config_from_args(args)

        self.assertEqual(batch.mode, "serial")
        self.assertEqual(batch.prefetch_batches, 4)
        self.assertTrue(batch.pin_memory)
        self.assertFalse(batch.non_blocking)
        self.assertFalse(batch.dedicated_cuda_stream)
        self.assertEqual(args.resume_data_state, "exact")
        self.assertEqual(preparation.backend, "threads")
        self.assertEqual(preparation.workers, 3)
        self.assertEqual(preparation.chunk_size, 17)
        self.assertEqual(preparation.max_pending_chunks, 5)

    def test_negative_workers_fail_validation(self):
        args = self.parser().parse_args(["--tokenizer-workers", "-1"])
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            preparation_config_from_args(args)

    def test_portable_exact_resume_policy_is_explicit(self):
        args = self.parser().parse_args(
            ["--resume-data-state", "portable_exact"]
        )
        self.assertEqual(args.resume_data_state, "portable_exact")


if __name__ == "__main__":
    unittest.main()
