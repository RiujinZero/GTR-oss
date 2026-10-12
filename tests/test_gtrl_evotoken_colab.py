from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import gtrl_evotoken_colab as runner


def source_text(relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    module_location = str(Path(__file__))
    marker = ".zip"
    marker_index = module_location.lower().find(marker)
    if marker_index < 0:
        raise FileNotFoundError(path)
    archive_path = Path(module_location[: marker_index + len(marker)])
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(relative_path).decode("utf-8")


class RunConfigTest(unittest.TestCase):
    def test_defaults_preserve_the_current_four_slot_run(self) -> None:
        config = runner.resolve_config(runner.RunConfig())
        self.assertEqual(config.output_slots, 4)
        self.assertEqual(config.context_length, 132)
        self.assertEqual(config.promotion_counts, (1, 1, 1, 1))
        self.assertEqual(config.experiment_tag, "cts_masked_s4_v5")
        self.assertEqual(
            config.experiment_version,
            "gtrl-130k-evotoken-relay-cts-masked-s4-v5",
        )
        self.assertTrue(config.compile)
        self.assertEqual(config.amp_dtype, "bf16")
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.checkpoint_every, 30_000)
        self.assertEqual(config.scan_chunk, 64)
        self.assertEqual(config.graph_layout, "small_world")
        self.assertEqual(config.graph_rewire, 0.5)
        self.assertTrue(config.private_stream_io)
        self.assertEqual(config.private_stream_rank, 4)
        self.assertFalse(config.heterogeneous_streams)
        self.assertTrue(config.readout_bias)
        self.assertFalse(config.tie_token_embeddings)
        self.assertFalse(config.block_causal)
        self.assertFalse(config.router_occupancy)
        self.assertEqual(config.linear_attention_key_dim, 8)
        self.assertEqual(config.optimizer_beta1, 0.9)
        self.assertEqual(config.optimizer_beta2, 0.999)
        self.assertEqual(config.optimizer_epsilon, 1e-8)
        self.assertEqual(config.slot_permutation_mode, "none")
        self.assertEqual(config.promotion_schedule_mode, "random")
        self.assertEqual(config.inference_promotion_mode, "confidence")
        self.assertEqual(config.position_objective_coordinates, "logical")
        self.assertEqual(config.tail_loss_slots, 0)
        self.assertEqual(config.position_balance_gamma, 0.0)
        self.assertEqual(config.worst_position_weight, 0.0)
        self.assertEqual(config.position_gradient_every_evals, 0)

    def test_slot_counts_scale_without_other_edits(self) -> None:
        expected = {
            2: (1, 0, 1, 0),
            4: (1, 1, 1, 1),
            6: (2, 1, 2, 1),
            8: (2, 2, 2, 2),
            10: (3, 2, 3, 2),
        }
        for slots, promotions in expected.items():
            with self.subTest(slots=slots):
                config = runner.resolve_config(
                    runner.RunConfig(output_slots=slots)
                )
                self.assertEqual(config.promotion_counts, promotions)
                self.assertIn(f"s{slots}_v5", config.local_log)
                self.assertIn(f"s{slots}_v5", config.drive_checkpoint)

    def test_tied_embeddings_get_an_isolated_default_namespace(self) -> None:
        config, _ = runner.parse_config(["--tie-token-embeddings"])
        self.assertTrue(config.tie_token_embeddings)
        self.assertEqual(config.experiment_tag, "cts_masked_s4_tied_v5")
        self.assertIn("s4_tied_v5", config.local_checkpoint)
        self.assertIn("s4_tied_v5", config.drive_checkpoint)

        untied, _ = runner.parse_config(["--no-tie-token-embeddings"])
        self.assertFalse(untied.tie_token_embeddings)
        self.assertEqual(untied.experiment_tag, "cts_masked_s4_v5")

    def test_block_causal_gets_an_isolated_cli_namespace(self) -> None:
        config, _ = runner.parse_config(["--block-causal"])
        self.assertTrue(config.block_causal)
        self.assertEqual(config.experiment_tag, "cts_masked_s4_block_v5")
        self.assertIn("s4_block_v5", config.local_checkpoint)
        self.assertIn("s4_block_v5", config.drive_checkpoint)

        tied, _ = runner.parse_config(
            ["--tie-token-embeddings", "--block-causal"]
        )
        self.assertEqual(
            tied.experiment_tag, "cts_masked_s4_tied_block_v5"
        )
        legacy, _ = runner.parse_config(["--no-block-causal"])
        self.assertEqual(legacy.experiment_tag, "cts_masked_s4_v5")

    def test_causal_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.resolve_config(
                runner.RunConfig(causal=True, block_causal=True)
            )

    def test_block_causal_requires_complete_observed_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by output_slots"):
            runner.resolve_config(
                runner.RunConfig(
                    observed=127,
                    output_slots=4,
                    block_causal=True,
                )
            )

    def test_extra_cache_capacity_is_separate_and_namespaced(self) -> None:
        config, _ = runner.parse_config(
            ["--block-causal", "--context-length", "256"]
        )
        self.assertEqual(config.observed + config.output_slots, 132)
        self.assertEqual(config.context_length, 256)
        self.assertEqual(
            config.experiment_tag, "cts_masked_s4_block_ctx256_v5"
        )
        with self.assertRaisesRegex(ValueError, "cannot be shorter"):
            runner.resolve_config(runner.RunConfig(context_length=131))

    def test_position_ablation_flags_get_an_isolated_namespace(self) -> None:
        config, _ = runner.parse_config(
            [
                "--output-slots",
                "6",
                "--slot-permutation-mode",
                "rotate",
                "--promotion-schedule-mode",
                "rotating",
                "--inference-promotion-mode",
                "tail_first",
                "--position-objective-coordinates",
                "physical",
                "--tail-loss-slots",
                "2",
                "--position-balance-gamma",
                "0.5",
                "--worst-position-weight",
                "0.25",
                "--position-gradient-every-evals",
                "10",
                "--position-gradient-max-slots",
                "6",
            ]
        )
        self.assertEqual(config.output_slots, 6)
        self.assertEqual(config.tail_loss_slots, 2)
        self.assertEqual(config.position_gradient_every_evals, 10)
        self.assertRegex(
            config.experiment_tag,
            r"^cts_masked_s6_permrot_promrot_infertailfirst_"
            r"pbal_worst25_tail2_phys_a[0-9a-f]{8}_v5$",
        )

    def test_numeric_position_ablations_cannot_share_artifact_paths(self) -> None:
        first = runner.resolve_config(
            runner.RunConfig(position_balance_gamma=0.5)
        )
        second = runner.resolve_config(
            runner.RunConfig(position_balance_gamma=0.75)
        )
        third = runner.resolve_config(
            runner.RunConfig(
                position_balance_gamma=0.5,
                position_balance_decay=0.95,
            )
        )
        self.assertNotEqual(first.experiment_tag, second.experiment_tag)
        self.assertNotEqual(first.experiment_tag, third.experiment_tag)
        self.assertNotEqual(first.drive_checkpoint, second.drive_checkpoint)
        diagnostics_only = runner.resolve_config(
            runner.RunConfig(position_gradient_every_evals=10)
        )
        self.assertEqual(diagnostics_only.experiment_tag, "cts_masked_s4_v5")

    def test_initial_model_checkpoint_is_exposed_for_tail_probes(self) -> None:
        config, _ = runner.parse_config(
            ["--initial-model-checkpoint", "/drive/baseline.pt"]
        )
        self.assertEqual(
            config.initial_model_checkpoint,
            "/drive/baseline.pt",
        )

    def test_invalid_position_ablation_settings_are_rejected(self) -> None:
        invalid = (
            (
                {"slot_permutation_mode": "mystery"},
                "slot_permutation_mode",
            ),
            ({"promotion_schedule_mode": "mystery"}, "promotion_schedule_mode"),
            ({"inference_promotion_mode": "mystery"}, "inference_promotion_mode"),
            ({"tail_loss_slots": 5}, "tail_loss_slots"),
            ({"position_balance_gamma": -0.1}, "position_balance_gamma"),
            ({"position_balance_decay": 1.0}, "position_balance_decay"),
            ({"position_balance_max_ratio": 0.9}, "max_ratio"),
            ({"worst_position_weight": 1.1}, "worst_position_weight"),
            ({"position_gradient_every_evals": -1}, "gradient_every"),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    runner.resolve_config(runner.RunConfig(**overrides))
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            runner.resolve_config(
                runner.RunConfig(
                    causal=True,
                    conv_kernel=8,
                    slot_permutation_mode="rotate",
                )
            )

    def test_every_config_field_has_a_cli_flag(self) -> None:
        parser = runner.build_parser()
        destinations = {
            action.dest
            for action in parser._actions
        }
        expected = {
            field.name for field in runner.fields(runner.RunConfig)
        }
        self.assertTrue(expected.issubset(destinations))

    def test_cli_overrides_paths_shape_and_runtime(self) -> None:
        config, _ = runner.parse_config(
            [
                "--drive-root",
                "/drive/custom",
                "--bundle",
                "/drive/bundle.zip",
                "--code-dir",
                "/content/custom_code",
                "--output-slots",
                "2",
                "--nodes",
                "64",
                "--batch-size",
                "8",
                "--learning-rate",
                "0.0003",
                "--no-compile",
                "--no-monitor",
                "--promotion-counts",
                "1,0,1,0",
            ]
        )
        self.assertEqual(config.drive_root, "/drive/custom")
        self.assertEqual(Path(config.bundle), Path("/drive/bundle.zip"))
        self.assertEqual(config.code_dir, "/content/custom_code")
        self.assertEqual(config.output_slots, 2)
        self.assertEqual(config.nodes, 64)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertFalse(config.compile)
        self.assertFalse(config.monitor)
        self.assertEqual(config.promotion_counts, (1, 0, 1, 0))

    def test_json_config_then_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "output_slots": 8,
                        "batch_size": 4,
                        "drive_root": "/drive/from_json",
                    }
                ),
                encoding="utf-8",
            )
            config, _ = runner.parse_config(
                [
                    "--config",
                    str(path),
                    "--output-slots",
                    "2",
                    "--batch-size",
                    "12",
                ]
            )
        self.assertEqual(config.output_slots, 2)
        self.assertEqual(config.batch_size, 12)
        self.assertEqual(config.drive_root, "/drive/from_json")
        self.assertEqual(config.promotion_counts, (1, 0, 1, 0))

    def test_config_round_trip_is_lossless(self) -> None:
        original = runner.resolve_config(
            runner.RunConfig(
                output_slots=10,
                observed=120,
                experiment_tag="custom_s10",
                feedback_scale=0.25,
                block_causal=True,
            )
        )
        restored = runner.config_from_dict(runner.config_dict(original))
        self.assertEqual(runner.config_dict(restored), runner.config_dict(original))

    def test_invalid_promotion_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to output_slots"):
            runner.resolve_config(
                runner.RunConfig(
                    output_slots=4,
                    promotion_counts=(1, 0, 1, 0),
                )
            )

    def test_broad_replacement_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "too broad"):
            runner.resolve_config(runner.RunConfig(code_dir="/content"))

    def test_relay_steps_must_match_relays(self) -> None:
        with self.assertRaisesRegex(ValueError, "relays == trajectory_steps"):
            runner.resolve_config(
                runner.RunConfig(relays=3, trajectory_steps=4)
            )


class LauncherTest(unittest.TestCase):
    def test_worker_source_is_syntax_valid_and_gracefully_checkpointed(self) -> None:
        compile(runner.WORKER_SOURCE, "<worker>", "exec")
        self.assertIn("signal.SIGTERM", runner.WORKER_SOURCE)
        self.assertIn("CHECKPOINT_ON_EXIT", runner.WORKER_SOURCE)
        self.assertIn(
            "tie_token_embeddings=TIE_TOKEN_EMBEDDINGS",
            runner.WORKER_SOURCE,
        )
        self.assertIn(
            "checkpoint tie_token_embeddings mismatch",
            runner.WORKER_SOURCE,
        )
        self.assertIn("block_causal=BLOCK_CAUSAL", runner.WORKER_SOURCE)
        self.assertIn(
            "checkpoint block_causal mismatch",
            runner.WORKER_SOURCE,
        )
        self.assertIn(
            "checkpoint position-ablation signature mismatch",
            runner.WORKER_SOURCE,
        )
        self.assertIn("map_physical_to_logical", runner.WORKER_SOURCE)
        self.assertIn("pos_grad step=", runner.WORKER_SOURCE)
        self.assertIn("train_promote_logical=", runner.WORKER_SOURCE)
        self.assertIn("train_promote_physical=", runner.WORKER_SOURCE)
        self.assertIn("infer_promote_stage=", runner.WORKER_SOURCE)
        self.assertIn("checkpoint=warmstart_loaded", runner.WORKER_SOURCE)
        self.assertIn(
            "warm-start checkpoint model architecture mismatch",
            runner.WORKER_SOURCE,
        )
        for field in (
            "graph_layout",
            "graph_seed",
            "graph_rewire",
            "tie_token_embeddings",
            "causal",
            "block_causal",
        ):
            self.assertIn(f'"{field}"', runner.WORKER_SOURCE)
        self.assertNotIn("__OUTPUT_SLOTS__", runner.WORKER_SOURCE)

    def test_worker_command_calls_the_same_module(self) -> None:
        config = runner.resolve_config(runner.RunConfig())
        command = runner.worker_command(config)
        self.assertEqual(command[1:4], ["-u", "-m", "gtrl_evotoken_colab"])
        self.assertEqual(command[4], "--worker-config")
        self.assertEqual(command[5], config.local_config)

    def test_drive_log_can_omit_or_retain_samples(self) -> None:
        text = "step=1\nsample='hello'\ncheckpoint=ok\n"
        self.assertNotIn(
            "sample=",
            runner.drive_safe_text(text, mirror_samples=False),
        )
        self.assertEqual(
            runner.drive_safe_text(text, mirror_samples=True),
            text,
        )

    def test_monitor_digest_keeps_position_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "noise\n"
                "step=300 loss=4.0\n"
                "position=ema_logical=[0.2 0.1]\n"
                "pos_grad step=300 group=shared norm=[1,0.1]\n",
                encoding="utf-8",
            )
            digest = runner.digest(path, lines=10)
        self.assertIn("step=300", digest)
        self.assertIn("position=ema_logical", digest)
        self.assertIn("pos_grad step=300", digest)
        self.assertNotIn("noise", digest)

    def test_interrupting_the_monitor_stops_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = runner.resolve_config(
                runner.RunConfig(
                    bundle=str(Path(directory) / "bundle.zip"),
                    drive_root=str(Path(directory) / "drive"),
                    local_root=str(Path(directory) / "local"),
                    local_log=str(Path(directory) / "train.log"),
                    local_checkpoint=str(Path(directory) / "ckpt.pt"),
                    progress_archive=str(Path(directory) / "progress.log"),
                    interval_seconds=0.01,
                    stop_wait_seconds=1.0,
                )
            )

            class FakeProcess:
                def __init__(self) -> None:
                    self.pid = 4321
                    self.exited = False

                def poll(self) -> int | None:
                    return 0 if self.exited else None

            process = FakeProcess()
            killed: list[tuple[int, int]] = []

            def fake_kill(pid: int, sig: int) -> None:
                killed.append((pid, sig))
                process.exited = True

            calls = {"count": 0}

            def fake_sleep(seconds: float) -> None:
                calls["count"] += 1
                if calls["count"] == 1:
                    raise KeyboardInterrupt

            with mock.patch.object(runner.os, "kill", fake_kill), \
                    mock.patch.object(runner.time, "sleep", fake_sleep):
                runner.monitor_worker(config, process)  # type: ignore[arg-type]

        self.assertEqual(killed, [(process.pid, runner.signal.SIGTERM)])
        self.assertIsNotNone(process.poll())

    def test_thin_colab_cell_only_invokes_the_module(self) -> None:
        path = Path(
            "Runbooks/colab_cells/"
            "gtrl_130k_evotoken_relay_drive_cell.py"
        )
        text = source_text(path.as_posix())
        self.assertLess(len(text.splitlines()), 60)
        self.assertIn("from gtrl_evotoken_colab import main", text)
        self.assertIn("main(RUN_ARGS)", text)
        self.assertNotIn("TRAIN_SCRIPT", text)
        compile(text, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
