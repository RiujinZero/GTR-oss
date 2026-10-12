from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from gator.diagnostics import AdamWWeightTracer
from gator_linear_prefix_train import build_parser


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(inputs)))

    def procoim_parameter_groups(
        self,
    ) -> dict[str, list[nn.Parameter]]:
        return {
            "node_compute": list(self.first.parameters()),
            "readout": list(self.second.parameters()),
        }


class AdamWWeightTracerTests(unittest.TestCase):
    def test_trace_matches_exact_weights_gradients_updates_and_moments(
        self,
    ) -> None:
        torch.manual_seed(7)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.01,
            betas=(0.8, 0.9),
            weight_decay=0.05,
        )
        names = dict(model.named_parameters())
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in names.items()
        }
        gradient_samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in names
        }
        update_samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in names
        }

        with tempfile.TemporaryDirectory() as directory:
            tracer = AdamWWeightTracer(
                model,
                optimizer,
                directory,
                interval=2,
                step_statistics=True,
            )
            initial = tracer.maybe_snapshot(
                0,
                force=True,
                context={"phase": "initial"},
            )
            self.assertIsNotNone(initial)
            assert initial is not None
            self.assertTrue(initial.artifact_path.is_file())

            for step in (1, 2):
                inputs = torch.tensor(
                    [
                        [0.2 * step, -0.3, 0.5],
                        [-0.4, 0.1 * step, 0.7],
                    ]
                )
                targets = torch.tensor([[0.3, -0.2], [-0.1, 0.4]])
                optimizer.zero_grad(set_to_none=True)
                loss = (model(inputs) - targets).square().mean()
                loss.backward()
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=0.01
                    )
                )
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in names.items()
                }
                for name, parameter in names.items():
                    assert parameter.grad is not None
                    gradient_samples[name].append(
                        parameter.grad.detach().clone()
                    )
                tracer.before_optimizer_step(
                    step=step,
                    loss=float(loss.detach()),
                    preclip_gradient_norm=preclip,
                    gradient_clip=0.01,
                )
                optimizer.step()
                tracer.after_optimizer_step()
                for name, parameter in names.items():
                    update_samples[name].append(
                        parameter.detach() - before[name]
                    )
                snapshot = tracer.maybe_snapshot(
                    step,
                    context={"phase": "train"},
                )

            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.interval_steps, 2)
            self.assertIn("node_compute", snapshot.summary["groups"])
            self.assertIn("readout", snapshot.summary["groups"])
            self.assertIn("weight_trace step=2", snapshot.log_line())
            self.assertLessEqual(
                snapshot.summary["global"]["path_efficiency"],
                1.0 + 1e-6,
            )
            self.assertGreater(snapshot.summary["global"]["delta_norm"], 0.0)
            self.assertGreater(snapshot.summary["clip_rate"], 0.0)

            payload = torch.load(
                snapshot.artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                payload["format"], "gator-adamw-weight-trace"
            )
            for name, parameter in names.items():
                expected_delta = (
                    parameter.detach().cpu() - initial_parameters[name]
                )
                torch.testing.assert_close(
                    payload["parameters"][name],
                    parameter.detach().cpu(),
                )
                torch.testing.assert_close(
                    payload["interval_delta"][name],
                    expected_delta,
                )
                torch.testing.assert_close(
                    payload["mean_gradient"][name],
                    torch.stack(gradient_samples[name]).mean(dim=0),
                )
                torch.testing.assert_close(
                    payload["rms_update"][name],
                    torch.stack(update_samples[name])
                    .square()
                    .mean(dim=0)
                    .sqrt(),
                )
                state = optimizer.state[parameter]
                torch.testing.assert_close(
                    payload["exp_avg"][name],
                    state["exp_avg"].detach().cpu(),
                )
                torch.testing.assert_close(
                    payload["exp_avg_sq"][name],
                    state["exp_avg_sq"].detach().cpu(),
                )

            summary_lines = (
                Path(directory) / "summary.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(summary_lines), 2)
            self.assertEqual(json.loads(summary_lines[-1])["step"], 2)

    def test_boundary_mode_records_exact_endpoint_without_step_walk(
        self,
    ) -> None:
        torch.manual_seed(11)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        endpoint_gradients: dict[str, torch.Tensor] = {}
        with tempfile.TemporaryDirectory() as directory:
            tracer = AdamWWeightTracer(
                model,
                optimizer,
                directory,
                interval=2,
            )
            tracer.maybe_snapshot(0, force=True)
            self.assertFalse(tracer.step_statistics)
            self.assertEqual(tracer._gradient_sum, {})

            for step in (1, 2):
                optimizer.zero_grad(set_to_none=True)
                inputs = torch.full((3, 3), 0.1 * step)
                loss = model(inputs).square().mean()
                loss.backward()
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                )
                tracer.before_optimizer_step(
                    step=step,
                    loss=float(loss.detach()),
                    preclip_gradient_norm=preclip,
                    gradient_clip=1.0,
                )
                if step == 2:
                    endpoint_gradients = {
                        name: parameter.grad.detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                optimizer.step()
                tracer.after_optimizer_step()
                snapshot = tracer.maybe_snapshot(step)

            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertFalse(snapshot.summary["step_statistics"])
            self.assertEqual(
                snapshot.summary["gradient_statistic"],
                "endpoint_clipped_gradient",
            )
            self.assertIsNone(
                snapshot.summary["global"]["path_efficiency"]
            )
            self.assertIn("path_efficiency=nan", snapshot.log_line())
            assert snapshot.artifact_path is not None
            payload = torch.load(
                snapshot.artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    payload["interval_delta"][name],
                    parameter.detach().cpu() - initial[name],
                )
                torch.testing.assert_close(
                    payload["endpoint_gradient"][name],
                    endpoint_gradients[name],
                )

    def test_summary_only_trace_and_parser_flags(self) -> None:
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            tracer = AdamWWeightTracer(
                model,
                optimizer,
                directory,
                interval=500,
                save_tensors=False,
            )
            initial = tracer.maybe_snapshot(0, force=True)
            self.assertIsNotNone(initial)
            assert initial is not None
            self.assertIsNone(initial.artifact_path)
            self.assertFalse(list(Path(directory).glob("*.pt")))

        parser = build_parser()
        args = parser.parse_args(
            [
                "--adamw-weight-trace-every",
                "500",
                "--adamw-weight-trace-dir",
                "trace",
                "--no-adamw-weight-trace-save-tensors",
                "--adamw-weight-trace-step-statistics",
            ]
        )
        self.assertEqual(args.adamw_weight_trace_every, 500)
        self.assertEqual(args.adamw_weight_trace_dir, Path("trace"))
        self.assertFalse(args.adamw_weight_trace_save_tensors)
        self.assertTrue(args.adamw_weight_trace_step_statistics)


if __name__ == "__main__":
    unittest.main()
