from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import benchmark_convergent_routing as benchmark


class ConvergentRoutingBenchmarkTests(unittest.TestCase):
    def test_parser_accepts_complete_route_shape(self) -> None:
        args = benchmark.parse_args(
            [
                "--route-backend",
                "both",
                "--nodes",
                "308",
                "--d-model",
                "768",
                "--embedding-dim",
                "512",
                "--lanes",
                "10",
                "--relays",
                "2",
                "--degree",
                "3",
                "--active-node-top-k",
                "32",
                "--router-rank",
                "96",
                "--batch-size",
                "6",
                "--context-tokens",
                "512",
                "--output-slots",
                "128",
                "--warmup-steps",
                "20",
                "--measured-steps",
                "100",
                "--device",
                "cpu",
                "--dtype",
                "bfloat16",
                "--compile",
            ]
        )

        self.assertEqual(args.route_backend, "both")
        self.assertEqual(args.nodes, 308)
        self.assertEqual(args.d_model, 768)
        self.assertEqual(args.embedding_dim, 512)
        self.assertEqual(args.active_node_top_k, 32)
        self.assertEqual(args.router_rank, 96)
        self.assertEqual(args.float32_matmul_precision, "high")
        self.assertTrue(args.cudnn_tf32)
        self.assertTrue(args.compile)
        self.assertEqual(args.phases, benchmark.PHASE_NAMES)
        self.assertEqual(args.decode_workload, "parallel")
        self.assertIsNone(args.gradient_clip)
        self.assertEqual(args.packed_memory_writer_backend, "torch")

    def test_parser_accepts_triton_fft_writer(self) -> None:
        args = benchmark.parse_args(
            [
                "--route-backend",
                "packed",
                "--packed-memory-backend",
                "fft_opaque",
                "--packed-memory-writer-backend",
                "triton",
            ]
        )

        self.assertEqual(args.packed_memory_writer_backend, "triton")

    def test_teacher_forced_synthetic_batch_is_shifted(self) -> None:
        args = benchmark.parse_args(
            [
                "--batch-size",
                "2",
                "--context-tokens",
                "3",
                "--output-slots",
                "4",
                "--vocab-size",
                "32",
                "--decode-workload",
                "teacher_forced",
                "--gradient-clip",
                "1.0",
            ]
        )
        _, _, targets, output_token_ids, output_mask = (
            benchmark._synthetic_batch(args, torch.device("cpu"))
        )
        self.assertTrue(output_token_ids[:, 0].eq(1).all())
        torch.testing.assert_close(output_token_ids[:, 1:], targets[:, :-1])
        self.assertTrue(output_mask.all())

    def test_phase_selector_is_repeatable_and_deduplicated(self) -> None:
        args = benchmark.parse_args(
            [
                "--phase",
                "training_step",
                "--phases",
                "training_step",
                "--phase",
                "forward",
            ]
        )
        self.assertEqual(args.phases, ("training_step", "forward"))

    def test_parser_defaults_embedding_width_and_rejects_bad_frontier(self) -> None:
        args = benchmark.parse_args(
            ["--nodes", "8", "--lanes", "2", "--d-model", "12"]
        )
        self.assertEqual(args.embedding_dim, 12)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmark.parse_args(
                    [
                        "--nodes",
                        "8",
                        "--lanes",
                        "2",
                        "--active-node-top-k",
                        "9",
                    ]
                )

    def test_route_shape_estimates_are_exact(self) -> None:
        estimates = benchmark.route_shape_estimates(
            batch_size=2,
            total_tokens=5,
            nodes=12,
            width=8,
            lanes=3,
            relays=2,
            top_k=2,
            degree=3,
            value_element_size=4,
        )

        self.assertEqual(estimates["packet_rows"], 30)
        self.assertEqual(estimates["dense_frontier_values"], 360)
        self.assertEqual(estimates["dense_frontier_bytes"], 1_440)
        self.assertEqual(estimates["dense_route_history_bytes"], 4_320)
        self.assertEqual(estimates["packed_frontier_entries"], 60)
        self.assertEqual(
            estimates["packed_frontier_bytes_int64_ids"],
            60 * (8 + 4),
        )
        self.assertEqual(
            estimates["dense_edge_candidates_per_relay"],
            1_080,
        )
        self.assertEqual(
            estimates["packed_edge_candidates_per_relay"],
            180,
        )
        self.assertEqual(
            estimates["dense_node_memory_values_per_relay"],
            2 * 5 * 12 * 8,
        )
        self.assertEqual(
            estimates["packed_requested_memory_values_per_relay"],
            60 * 8,
        )

    def test_cpu_smoke_times_all_four_scopes_for_both_backends(self) -> None:
        if not benchmark.route_backend_is_supported():
            self.skipTest("packed route backend has not landed")
        args = benchmark.parse_args(
            [
                "--route-backend",
                "both",
                "--nodes",
                "4",
                "--d-model",
                "4",
                "--embedding-dim",
                "4",
                "--lanes",
                "2",
                "--relays",
                "1",
                "--degree",
                "3",
                "--active-node-top-k",
                "2",
                "--router-rank",
                "2",
                "--batch-size",
                "1",
                "--context-tokens",
                "2",
                "--output-slots",
                "2",
                "--vocab-size",
                "16",
                "--warmup-steps",
                "0",
                "--measured-steps",
                "1",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )

        result = benchmark.run_benchmark(args)

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["hardware"]["device"], "cpu")
        self.assertEqual(
            result["shape"]["packed_memory_writer_backend"],
            "torch",
        )
        self.assertEqual(
            [item["route_backend"] for item in result["backends"]],
            ["dense", "packed"],
        )
        for backend_result in result["backends"]:
            self.assertGreater(backend_result["parameters"], 0)
            self.assertEqual(
                [phase["name"] for phase in backend_result["phases"]],
                [
                    "forward",
                    "forward_backward",
                    "optimizer",
                    "training_step",
                ],
            )
            for phase in backend_result["phases"]:
                self.assertGreater(phase["milliseconds_per_step"], 0.0)
                self.assertGreater(phase["steps_per_second"], 0.0)
                self.assertIsNone(phase["peak_allocated_bytes"])
                self.assertIsNone(phase["peak_reserved_bytes"])

    def test_cpu_training_step_selector_skips_mutating_preludes(self) -> None:
        args = benchmark.parse_args(
            [
                "--route-backend",
                "dense",
                "--nodes",
                "4",
                "--d-model",
                "4",
                "--embedding-dim",
                "4",
                "--lanes",
                "2",
                "--relays",
                "1",
                "--degree",
                "3",
                "--active-node-top-k",
                "2",
                "--router-rank",
                "2",
                "--batch-size",
                "1",
                "--context-tokens",
                "2",
                "--output-slots",
                "2",
                "--vocab-size",
                "16",
                "--warmup-steps",
                "0",
                "--measured-steps",
                "1",
                "--device",
                "cpu",
                "--phase",
                "training_step",
                "--decode-workload",
                "teacher_forced",
                "--gradient-clip",
                "1.0",
            ]
        )
        result = benchmark.run_benchmark(args)
        self.assertEqual(result["hardware"]["phases"], ["training_step"])
        self.assertEqual(result["hardware"]["decode_workload"], "teacher_forced")
        self.assertTrue(result["hardware"]["causal"])
        self.assertEqual(result["hardware"]["gradient_clip"], 1.0)
        self.assertEqual(
            [phase["name"] for phase in result["backends"][0]["phases"]],
            ["training_step"],
        )
        self.assertTrue(result["backends"][0]["causal"])

    def test_json_and_console_outputs_are_machine_readable(self) -> None:
        result = {
            "schema_version": 1,
            "hardware": {
                "device": "cpu",
                "dtype": "float32",
                "compile": False,
            },
            "shape": {
                "batch_size": 1,
                "context_tokens": 2,
                "output_slots": 2,
                "d_model": 4,
                "nodes": 4,
                "lanes": 2,
                "relays": 1,
                "degree": 2,
                "router_rank": 2,
                "active_node_top_k": 2,
            },
            "route_shape_estimates": {
                "dense_frontier_bytes": 64,
                "packed_frontier_bytes_int64_ids": 96,
                "dense_edge_candidates_per_relay": 32,
                "packed_edge_candidates_per_relay": 16,
            },
            "backends": [
                {
                    "route_backend": "dense",
                    "parameters": 123,
                    "phases": [
                        {
                            "name": "forward",
                            "milliseconds_per_step": 1.0,
                            "steps_per_second": 1000.0,
                            "peak_allocated_bytes": None,
                            "peak_reserved_bytes": None,
                            "peak_allocated_delta_bytes": None,
                            "peak_reserved_delta_bytes": None,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            payload = benchmark.emit_json(
                result,
                output=output,
                print_payload=False,
            )
            self.assertEqual(json.loads(payload), result)
            self.assertEqual(json.loads(output.read_text("utf-8")), result)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            benchmark.print_console(result)
        text = stream.getvalue()
        self.assertIn("backend=dense", text)
        self.assertIn("phase=forward", text)


if __name__ == "__main__":
    unittest.main()
