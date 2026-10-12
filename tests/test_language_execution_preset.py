from __future__ import annotations

import argparse
import unittest

from gator_language_train import (
    apply_execution_preset,
    explicit_cli_destinations,
)


class LanguageExecutionPresetTests(unittest.TestCase):
    def test_manual_preserves_individual_options(self) -> None:
        args = argparse.Namespace(
            execution_preset="manual",
            compile=False,
            stewardship_execution="dense",
            receiver_memory_backend="attention",
        )

        self.assertIs(apply_execution_preset(args), args)
        self.assertFalse(args.compile)
        self.assertEqual(args.stewardship_execution, "dense")
        self.assertEqual(args.receiver_memory_backend, "attention")

    def test_dense_gpu_selects_exact_compiled_dense_kernels(self) -> None:
        args = argparse.Namespace(execution_preset="dense_gpu")

        apply_execution_preset(args)

        self.assertTrue(args.compile)
        self.assertEqual(args.float32_matmul_precision, "high")
        self.assertTrue(args.cudnn_tf32)
        self.assertEqual(args.receiver_memory_backend, "convolution")
        self.assertEqual(args.dense_router_backend, "einsum")
        self.assertEqual(args.dense_node_backend, "folded_bmm")
        self.assertEqual(args.stewardship_execution, "dense")
        self.assertEqual(args.batch_pipeline, "prefetch")

    def test_optimized_gpu_adds_persistent_sparse_execution(self) -> None:
        args = argparse.Namespace(execution_preset="optimized_gpu")

        apply_execution_preset(args)

        self.assertTrue(args.compile)
        self.assertEqual(args.stewardship_execution, "packed_auto")
        self.assertEqual(args.stewardship_packed_threshold, 0.5)
        self.assertEqual(
            args.packed_stewardship_backend,
            "persistent_bucketed",
        )
        self.assertEqual(args.stewardship_execution_check_every, 100)
        self.assertEqual(args.prefetch_batches, 2)
        self.assertIsNone(args.pin_memory)
        self.assertTrue(args.non_blocking_transfer)
        self.assertTrue(args.dedicated_copy_stream)

    def test_safe_preset_keeps_structural_optimizations_in_strict_fp32(self) -> None:
        args = argparse.Namespace(execution_preset="optimized_gpu_safe")

        apply_execution_preset(args)

        self.assertFalse(args.compile)
        self.assertEqual(args.float32_matmul_precision, "highest")
        self.assertFalse(args.cudnn_tf32)
        self.assertEqual(args.receiver_memory_backend, "convolution")
        self.assertEqual(args.dense_node_backend, "folded_bmm")
        self.assertEqual(args.stewardship_execution, "packed_auto")
        self.assertEqual(
            args.packed_stewardship_backend,
            "persistent_bucketed",
        )

    def test_explicit_low_level_options_override_preset(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--execution-preset")
        parser.add_argument("--compile", action="store_true")
        parser.add_argument(
            "--cudnn-tf32",
            action=argparse.BooleanOptionalAction,
        )
        command = [
            "--execution-preset",
            "optimized_gpu_safe",
            "--compile",
            "--cudnn-tf32",
        ]
        args = parser.parse_args(command)

        apply_execution_preset(
            args,
            explicit_destinations=explicit_cli_destinations(parser, command),
        )

        self.assertTrue(args.compile)
        self.assertTrue(args.cudnn_tf32)
        self.assertEqual(args.stewardship_execution, "packed_auto")

    def test_unknown_preset_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported execution preset"):
            apply_execution_preset(
                argparse.Namespace(execution_preset="imaginary")
            )


if __name__ == "__main__":
    unittest.main()
