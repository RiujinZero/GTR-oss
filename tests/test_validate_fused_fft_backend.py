from __future__ import annotations

import unittest
from unittest import mock

import torch

import validate_fused_fft_backend as validator


class FusedFFTValidatorContractTests(unittest.TestCase):
    def test_default_parser_is_a_quick_irregular_shape(self) -> None:
        args = validator.parse_args([])
        self.assertEqual(
            (
                args.batch,
                args.tokens,
                args.lanes,
                args.routes,
                args.nodes,
                args.width,
                args.feature_chunk,
                args.warmup,
                args.iterations,
            ),
            (1, 17, 2, 3, 9, 8, 8, 3, 10),
        )
        self.assertEqual(args.packed_memory_writer_backend, "triton")

    def test_parser_exposes_every_shape_and_timing_axis(self) -> None:
        args = validator.parse_args(
            [
                "--batch",
                "2",
                "--tokens",
                "96",
                "--lanes",
                "10",
                "--top-k",
                "32",
                "--nodes",
                "154",
                "--d-model",
                "64",
                "--chunk",
                "16",
                "--warmup",
                "1",
                "--iterations",
                "4",
                "--causal",
            ]
        )
        self.assertEqual(
            (
                args.batch,
                args.tokens,
                args.lanes,
                args.routes,
                args.nodes,
                args.width,
                args.feature_chunk,
                args.warmup,
                args.iterations,
                args.causal,
            ),
            (2, 96, 10, 32, 154, 64, 16, 1, 4, True),
        )

    def test_parser_exposes_component_diagnostics(self) -> None:
        args = validator.parse_args(["--diagnose-components"])
        self.assertTrue(args.diagnose_components)

    def test_parser_exposes_grad_mass_tolerance_override(self) -> None:
        args = validator.parse_args(
            [
                "--grad-mass-atol",
                "0.001",
                "--grad-mass-rtol",
                "0.002",
                "--grad-mass-max-relative-l2",
                "0.00001",
                "--grad-mass-max-absolute",
                "0.008",
                "--grad-mass-max-outliers",
                "4",
                "--grad-mass-max-outlier-fraction",
                "0.00002",
            ]
        )
        self.assertEqual(args.grad_mass_atol, 0.001)
        self.assertEqual(args.grad_mass_rtol, 0.002)
        self.assertEqual(args.grad_mass_max_relative_l2, 0.00001)
        self.assertEqual(args.grad_mass_max_absolute, 0.008)
        self.assertEqual(args.grad_mass_max_outliers, 4)
        self.assertEqual(args.grad_mass_max_outlier_fraction, 0.00002)

    def test_readme_command_aliases_parse_verbatim(self) -> None:
        args = validator.parse_args(
            [
                "--batch-size",
                "1",
                "--tokens",
                "96",
                "--nodes",
                "154",
                "--lanes",
                "10",
                "--route-width",
                "32",
                "--width",
                "32",
                "--feature-chunk",
                "32",
                "--warmup-steps",
                "10",
                "--measured-steps",
                "50",
                "--compile",
            ]
        )
        self.assertEqual(
            (
                args.batch,
                args.tokens,
                args.nodes,
                args.lanes,
                args.routes,
                args.width,
                args.feature_chunk,
                args.warmup,
                args.iterations,
                args.compile,
            ),
            (1, 96, 154, 10, 32, 32, 32, 10, 50, True),
        )

    def test_repo_style_aliases_parse(self) -> None:
        args = validator.parse_args(
            [
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft_opaque",
                "--packed-memory-reader-backend",
                "triton",
                "--packed-memory-writer-backend",
                "triton",
                "--active-node-top-k",
                "8",
                "--d-model",
                "64",
                "--packed-memory-feature-chunk",
                "16",
            ]
        )
        self.assertEqual(
            (
                args.route_backend,
                args.packed_memory_backend,
                args.packed_memory_reader_backend,
                args.packed_memory_writer_backend,
                args.routes,
                args.width,
                args.feature_chunk,
            ),
            ("packed", "fft_opaque", "triton", "triton", 8, 64, 16),
        )

    def test_no_compile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --compile"):
            validator.parse_args(["--no-compile"])

    def test_nonfinite_numerical_contract_is_rejected(self) -> None:
        for arguments in (
            ["--epsilon", "nan"],
            ["--atol", "inf"],
            ["--rtol", "nan"],
            ["--grad-mass-atol", "inf"],
            ["--grad-mass-rtol", "nan"],
            ["--grad-mass-max-relative-l2", "inf"],
            ["--grad-mass-max-absolute", "nan"],
            ["--grad-mass-max-outlier-fraction", "nan"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "finite"):
                    validator.parse_args(arguments)

    def test_sparse_outlier_contract_requires_both_limits_and_guards(self) -> None:
        for arguments in (
            ["--grad-mass-max-outliers", "4"],
            ["--grad-mass-max-outlier-fraction", "0.1"],
            [
                "--grad-mass-max-outliers",
                "4",
                "--grad-mass-max-outlier-fraction",
                "0.1",
            ],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "together|guards"):
                    validator.parse_args(arguments)

    def test_invalid_unique_route_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "routes cannot exceed nodes"):
            validator.parse_args(["--routes", "10", "--nodes", "9"])

    def test_cuda_preflight_fails_clearly_on_cpu(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                validator.require_cuda_triton()

    def test_cuda_preflight_fails_clearly_without_triton(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(
                validator,
                "triton_packed_reader_available",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires .*Triton"):
                validator.require_cuda_triton()

    def test_cuda_preflight_fails_clearly_without_triton_writer(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(
                validator,
                "triton_packed_reader_available",
                return_value=True,
            ),
            mock.patch.object(
                validator,
                "triton_packed_writer_available",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "writer"):
                validator.require_cuda_triton()

    def test_input_builder_produces_unique_normalized_fp32_routes(self) -> None:
        shape = validator.Shape(
            batch=2,
            tokens=5,
            lanes=3,
            routes=4,
            nodes=7,
            width=6,
            feature_chunk=3,
        )
        hidden, nodes, mass, valid, write, bias = (
            validator.build_unique_inputs(shape, device="cpu", seed=3)
        )
        self.assertEqual(tuple(hidden.shape), (2, 5, 3, 6))
        self.assertEqual(tuple(nodes.shape), (2, 5, 3, 4))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertTrue(valid.all())
        sorted_nodes = nodes.sort(dim=-1).values
        self.assertTrue((sorted_nodes[..., 1:] != sorted_nodes[..., :-1]).all())
        torch.testing.assert_close(
            mass.sum(dim=-1),
            torch.ones(2, 5, 3),
        )
        for tensor in (hidden, mass, write, bias):
            self.assertTrue(tensor.requires_grad)

    def test_drift_reports_zero_for_identical_tensors(self) -> None:
        value = torch.tensor([0.0, 1.0, -3.0])
        result = validator.drift(value, value.clone())
        self.assertEqual(result.maximum_absolute, 0.0)
        self.assertEqual(result.maximum_relative, 0.0)
        self.assertEqual(result.relative_l2, 0.0)
        self.assertEqual(result.mismatched, 0)
        self.assertEqual(result.nonfinite, 0)
        self.assertEqual(result.total, 3)

    def test_drift_reports_tolerance_normalized_worst_element(self) -> None:
        expected = torch.tensor([1.0, 1.0, 2.0])
        actual = torch.tensor([1.0, 1.002, 2.0001])
        result = validator.drift(
            actual,
            expected,
            atol=5e-4,
            rtol=5e-4,
        )
        self.assertEqual(result.mismatched, 1)
        self.assertEqual(result.nonfinite, 0)
        self.assertEqual(result.total, 3)
        self.assertEqual(result.worst_index, (1,))
        self.assertAlmostEqual(result.worst_actual, 1.002, places=5)
        self.assertAlmostEqual(result.worst_expected, 1.0, places=5)
        self.assertAlmostEqual(result.worst_allowed, 0.001, places=6)
        self.assertGreater(result.worst_tolerance_ratio, 1.9)

    def test_compare_values_collects_every_tensor_failure(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("first", "second"),
            (torch.tensor([1.0]), torch.tensor([2.0])),
            (torch.tensor([0.0]), torch.tensor([0.0])),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(len(failures), 2)
        self.assertIn("unit/first", failures[0])
        self.assertIn("unit/second", failures[1])

    def test_compare_values_rejects_truncated_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            validator.compare_values(
                "unit",
                ("first", "second"),
                (torch.tensor([1.0]),),
                (torch.tensor([1.0]), torch.tensor([2.0])),
                atol=0.0,
                rtol=0.0,
            )

    def test_compare_values_collects_shape_mismatch(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("shape",),
            (torch.zeros(2),),
            (torch.zeros(3),),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("shape mismatch", failures[0])

    def test_sparse_policy_rejects_dtype_metadata_mismatch(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([1.0], dtype=torch.float16),),
            (torch.tensor([1.0], dtype=torch.float32),),
            atol=1e-3,
            rtol=1.5e-3,
            relative_l2_limits={"grad_mass": 1e-5},
            maximum_absolute_limits={"grad_mass": 8e-3},
            sparse_outlier_limits={"grad_mass": (4, 2e-5)},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("metadata mismatch", failures[0])
        self.assertIn("dtype", failures[0])

    def test_drift_counts_nonfinite_values_as_mismatches(self) -> None:
        result = validator.drift(
            torch.tensor([float("nan"), 1.0]),
            torch.tensor([0.0, 1.0]),
            atol=1.0,
            rtol=1.0,
        )
        self.assertEqual(result.mismatched, 1)
        self.assertEqual(result.nonfinite, 1)
        self.assertEqual(result.worst_tolerance_ratio, float("inf"))
        self.assertEqual(result.relative_l2, float("inf"))

    def test_compare_values_applies_only_named_tolerance_override(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass", "other"),
            (torch.tensor([1.0015]), torch.tensor([1.0015])),
            (torch.tensor([1.0]), torch.tensor([1.0])),
            atol=5e-4,
            rtol=5e-4,
            tolerance_overrides={"grad_mass": (1e-3, 2e-3)},
            relative_l2_limits={"grad_mass": 1e-2},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("unit/other", failures[0])

    def test_compare_values_enforces_relative_l2_guard(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([1.1]),),
            (torch.tensor([1.0]),),
            atol=1.0,
            rtol=1.0,
            relative_l2_limits={"grad_mass": 1e-2},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("relative_l2", failures[0])

    def test_compare_values_enforces_maximum_absolute_guard(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([1.1]),),
            (torch.tensor([1.0]),),
            atol=1.0,
            rtol=1.0,
            maximum_absolute_limits={"grad_mass": 0.05},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("max_abs", failures[0])

    def test_compare_values_accepts_bounded_sparse_outliers(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([0.0719, 10.0, 20.0]),),
            (torch.tensor([0.0739, 10.0, 20.0]),),
            atol=1e-3,
            rtol=1.5e-3,
            relative_l2_limits={"grad_mass": 1e-3},
            maximum_absolute_limits={"grad_mass": 8e-3},
            sparse_outlier_limits={"grad_mass": (1, 0.5)},
        )
        self.assertEqual(failures, [])

    def test_production_sparse_policy_accepts_observed_single_outlier(self) -> None:
        expected = torch.ones(204_800)
        actual = expected.clone()
        expected[0] = 0.0739154816
        actual[0] = 0.0718832016
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (actual,),
            (expected,),
            atol=1e-3,
            rtol=1.5e-3,
            relative_l2_limits={"grad_mass": 1e-5},
            maximum_absolute_limits={"grad_mass": 8e-3},
            sparse_outlier_limits={"grad_mass": (4, 2e-5)},
        )
        self.assertEqual(failures, [])

    def test_compare_values_rejects_excess_sparse_outliers(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([0.0719, 0.0719, 1000.0]),),
            (torch.tensor([0.0739, 0.0739, 1000.0]),),
            atol=1e-3,
            rtol=1.5e-3,
            relative_l2_limits={"grad_mass": 1e-3},
            maximum_absolute_limits={"grad_mass": 8e-3},
            sparse_outlier_limits={"grad_mass": (1, 0.5)},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("sparse elementwise outliers", failures[0])

    def test_compare_values_sparse_policy_requires_aggregate_guards(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires relative-L2"):
            validator.compare_values(
                "unit",
                ("grad_mass",),
                (torch.tensor([0.0719]),),
                (torch.tensor([0.0739]),),
                atol=1e-3,
                rtol=1.5e-3,
                sparse_outlier_limits={"grad_mass": (1, 1.0)},
            )

    def test_sparse_acceptance_does_not_bypass_maximum_absolute_guard(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([0.05, 1000.0]),),
            (torch.tensor([0.0, 1000.0]),),
            atol=1e-3,
            rtol=1.5e-3,
            relative_l2_limits={"grad_mass": 1e-3},
            maximum_absolute_limits={"grad_mass": 8e-3},
            sparse_outlier_limits={"grad_mass": (1, 1.0)},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("max_abs", failures[0])

    def test_compare_values_explicitly_rejects_matching_nonfinite_values(self) -> None:
        failures = validator.compare_values(
            "unit",
            ("grad_mass",),
            (torch.tensor([float("inf")]),),
            (torch.tensor([float("inf")]),),
            atol=1.0,
            rtol=1.0,
        )
        self.assertTrue(any("nonfinite" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
