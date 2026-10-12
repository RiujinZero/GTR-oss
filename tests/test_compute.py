import argparse
import unittest
from unittest import mock

import torch
from torch import nn

from gator.cli import add_compute_arguments, compute_config_from_args
from gator.compute import (
    CompileConfig,
    ComputeConfig,
    Float32RuntimeConfig,
    compile_module,
    configure_float32_runtime,
    mark_compiled_iteration_begin,
    resolve_float32_runtime,
)


class ComputeConfigTests(unittest.TestCase):
    def test_defaults_round_trip_to_dictionary(self):
        config = ComputeConfig()
        self.assertEqual(config.float32.matmul_precision, "highest")
        self.assertIsNone(config.float32.cudnn_tf32)
        self.assertFalse(config.compile.enabled)
        self.assertEqual(
            config.to_dict(),
            {
                "float32": {
                    "matmul_precision": "highest",
                    "cudnn_tf32": None,
                },
                "compile": {
                    "enabled": False,
                    "mode": "default",
                    "shape_padding": None,
                },
            },
        )

    def test_compile_mode_and_custom_options_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            CompileConfig(
                enabled=True,
                mode="max-autotune",
                shape_padding=True,
            )

    def test_cli_parses_explicit_tensor_core_policy(self):
        parser = argparse.ArgumentParser()
        add_compute_arguments(parser)
        args = parser.parse_args(
            [
                "--compile",
                "--compile-shape-padding",
                "--float32-matmul-precision",
                "high",
                "--cudnn-tf32",
            ]
        )
        config = compute_config_from_args(args)
        self.assertTrue(config.compile.enabled)
        self.assertTrue(config.compile.shape_padding)
        self.assertEqual(config.float32.matmul_precision, "high")
        self.assertTrue(config.float32.cudnn_tf32)


class RuntimePolicyTests(unittest.TestCase):
    def test_configure_applies_and_restores_global_cpu_policy(self):
        previous_precision = torch.get_float32_matmul_precision()
        previous_cudnn = torch.backends.cudnn.allow_tf32
        try:
            resolved = configure_float32_runtime(
                Float32RuntimeConfig(
                    matmul_precision="highest",
                    cudnn_tf32=False,
                ),
                "cpu",
            )
            self.assertEqual(
                torch.get_float32_matmul_precision(), "highest"
            )
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(resolved.tf32_supported)
            self.assertEqual(resolved.device_type, "cpu")
        finally:
            torch.set_float32_matmul_precision(previous_precision)
            torch.backends.cudnn.allow_tf32 = previous_cudnn

    def test_resolver_reports_blackwell_and_t4_capability(self):
        config = Float32RuntimeConfig(matmul_precision="high")
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            with (
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch(
                    "torch.cuda.get_device_capability",
                    return_value=(10, 0),
                ),
                mock.patch(
                    "torch.cuda.get_device_name",
                    return_value="Blackwell",
                ),
            ):
                blackwell = resolve_float32_runtime(config, "cuda")
            self.assertTrue(blackwell.tf32_supported)
            self.assertTrue(
                blackwell.matmul_tensor_core_policy_effective
            )

            with (
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch(
                    "torch.cuda.get_device_capability",
                    return_value=(7, 5),
                ),
                mock.patch(
                    "torch.cuda.get_device_name",
                    return_value="T4",
                ),
            ):
                t4 = resolve_float32_runtime(config, "cuda")
            self.assertFalse(t4.tf32_supported)
            self.assertFalse(t4.matmul_tensor_core_policy_effective)
        finally:
            torch.set_float32_matmul_precision(previous_precision)


class CompilePolicyTests(unittest.TestCase):
    def test_disabled_compile_is_identity(self):
        module = nn.Linear(2, 3)
        self.assertIs(compile_module(module, CompileConfig()), module)

    def test_shape_padding_is_passed_as_backend_option(self):
        module = nn.Linear(2, 3)
        compiled = object()
        with mock.patch("torch.compile", return_value=compiled) as call:
            result = compile_module(
                module,
                CompileConfig(enabled=True, shape_padding=True),
            )
        self.assertIs(result, compiled)
        call.assert_called_once_with(
            module,
            options={"shape_padding": True},
        )

    def test_named_mode_is_passed_without_custom_options(self):
        module = nn.Linear(2, 3)
        with mock.patch("torch.compile", return_value=module) as call:
            compile_module(
                module,
                CompileConfig(enabled=True, mode="reduce-overhead"),
            )
        call.assert_called_once_with(module, mode="reduce-overhead")

    def test_compile_configures_module_cuda_graph_accumulation(self):
        module = mock.Mock()
        module.configure_cudagraph_accumulation = mock.Mock()
        with mock.patch("torch.compile", return_value=module):
            compile_module(
                module,
                CompileConfig(enabled=True, mode="max-autotune"),
            )
            module.configure_cudagraph_accumulation.assert_called_once_with(
                True
            )
            module.configure_cudagraph_accumulation.reset_mock()
            compile_module(
                module,
                CompileConfig(
                    enabled=True,
                    mode="max-autotune-no-cudagraphs",
                ),
            )
        module.configure_cudagraph_accumulation.assert_called_once_with(False)

    def test_compile_ignores_noncallable_cuda_graph_hook(self):
        module = nn.Linear(2, 3)
        module.configure_cudagraph_accumulation = object()
        with mock.patch("torch.compile", return_value=module) as call:
            result = compile_module(
                module,
                CompileConfig(enabled=True, mode="max-autotune"),
            )
        self.assertIs(result, module)
        call.assert_called_once_with(module, mode="max-autotune")

    def test_cuda_graph_modes_mark_each_compiled_iteration(self):
        for mode in ("reduce-overhead", "max-autotune"):
            with self.subTest(mode=mode), mock.patch(
                "torch.compiler.cudagraph_mark_step_begin"
            ) as marker:
                marked = mark_compiled_iteration_begin(
                    CompileConfig(enabled=True, mode=mode),
                    "cuda",
                )
            self.assertTrue(marked)
            marker.assert_called_once_with()

    def test_non_cuda_and_no_cudagraph_modes_do_not_mark(self):
        with mock.patch(
            "torch.compiler.cudagraph_mark_step_begin"
        ) as marker:
            self.assertFalse(
                mark_compiled_iteration_begin(
                    CompileConfig(enabled=True, mode="max-autotune"),
                    "cpu",
                )
            )
            self.assertFalse(
                mark_compiled_iteration_begin(
                    CompileConfig(
                        enabled=True,
                        mode="max-autotune-no-cudagraphs",
                    ),
                    "cuda",
                )
            )
            self.assertFalse(
                mark_compiled_iteration_begin(CompileConfig(), "cuda")
            )
            self.assertFalse(
                mark_compiled_iteration_begin(
                    CompileConfig(enabled=True, mode="default"),
                    "cuda",
                )
            )
        marker.assert_not_called()

    def test_missing_cuda_graph_marker_fails_with_safe_fallback(self):
        with mock.patch.object(
            torch.compiler,
            "cudagraph_mark_step_begin",
            None,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "max-autotune-no-cudagraphs",
            ):
                mark_compiled_iteration_begin(
                    CompileConfig(enabled=True, mode="max-autotune"),
                    "cuda",
                )


if __name__ == "__main__":
    unittest.main()
