"""Resumable block-coordinate 100M Gator curriculum for a browser Colab cell.

Each invocation runs one evaluation segment.  The companion launch cell calls
this program repeatedly, so phase transitions and rewinds occur without manual
editing while the browser-owned cell remains active.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time

from run_gator_100m_curriculum_colab import (
    ACTIVE_NODE_TOP_K,
    BASE_LEARNING_RATE,
    DEGREE,
    D_MODEL,
    EMBEDDING_DIM,
    EXPECTED_PARAMETERS,
    LANES,
    N_NODES,
    RELAYS,
    ROUTER_RANK,
    TARGET_STEPS,
    VOCAB_SIZE,
    _atomic_copy,
    _atomic_json,
    _common_command,
    preflight_parameter_count,
)


PHASE_A = "A_coadapt_teacher"
PHASE_B = "B_route_jury_teacher"
PHASE_C = "C_private_nodes_handoff"
COMPLETE = "complete"
STATE_NAME = "STAGED_CURRICULUM_STATE.json"

DEFAULT_CONFIG = Path("/content/GTR/GATOR100M_COLAB_CONFIG.json")
COMPONENT_LR = {
    "lexical": 0.50,
    "lane_core": 1.00,
    "private_node": 0.625,
    "route_gate": 1.00,
}
DECODE_MODES = (
    "parallel",
    "serial",
    "chunk_parallel",
    "teacher_forced",
    "handoff",
    "graduated_handoff",
    "token_handoff",
    "prefix_denoise",
)


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_paths = (
        "drive_run_root",
        "local_work_root",
        "smoltalk_everyday",
        "tinystories_train",
        "tinystories_validation",
        "lmsys_chat",
        "tulu3_sft",
        "smoltalk_preferences",
    )
    missing = [name for name in required_paths if not payload.get(name)]
    if missing:
        raise ValueError(f"config is missing paths: {', '.join(missing)}")
    return payload


def fresh_state(config: dict) -> dict:
    return {
        "schema_version": 1,
        "run_name": config["run_name"],
        "status": "active",
        "phase": PHASE_A,
        "retained_step": 0,
        "attempted_steps": 0,
        "segment_index": 0,
        "phase_segments": 0,
        "phase_entry_step": 0,
        "phase_best_teacher_ppl": None,
        "phase_best_step": None,
        "fixed_route_temperature": None,
        "history": [],
        "contract": {
            "parameters": EXPECTED_PARAMETERS,
            "architecture": {
                "vocab_size": VOCAB_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "d_model": D_MODEL,
                "nodes": N_NODES,
                "lanes": LANES,
                "degree": DEGREE,
                "relays": RELAYS,
                "router_rank": ROUTER_RANK,
                "active_node_top_k": ACTIVE_NODE_TOP_K,
            },
            "all_material_interleaved_in": [PHASE_A, PHASE_B],
            "phase_a_train_components": [
                "lexical", "lane_core", "private_node", "route_gate"
            ],
            "phase_b_train_components": ["route_gate"],
            "phase_c_train_components": ["private_node"],
            "handoff_start_removed": 0,
            "handoff_min_teacher_accuracy": config[
                "handoff_min_teacher_accuracy"
            ],
            "handoff_min_frontier_accuracy": config[
                "handoff_min_frontier_accuracy"
            ],
            "handoff_advance_ratio": config["handoff_advance_ratio"],
            "component_lr_multipliers": COMPONENT_LR,
        },
    }


def load_checkpoint_metrics(path: Path) -> dict:
    import torch

    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except (RuntimeError, TypeError, ValueError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    def compact(value: dict | None) -> dict | None:
        if not value:
            return None
        return {
            key: float(value[key])
            for key in (
                "content_loss",
                "content_perplexity",
                "content_accuracy",
                "content_top5_accuracy",
                "formation_coverage",
            )
            if value.get(key) is not None
        }

    try:
        data_config = checkpoint.get("data_config") or {}
        handoff = data_config.get("token_handoff_state") or {}
        decode_mode = str(data_config.get("decode_mode", "unknown"))
        partial = compact(checkpoint.get("validation"))
        teacher = compact(checkpoint.get("auxiliary_validation"))
        free = compact(checkpoint.get("support_validation"))
        if decode_mode == "teacher_forced":
            teacher = partial
        return {
            "step": int(checkpoint["step"]),
            "decode_mode": decode_mode,
            "removed_tokens": handoff.get("removed_tokens"),
            "teacher": teacher,
            "partial": partial,
            "free": free,
        }
    finally:
        del checkpoint
        gc.collect()


def replace_flag(command: list[str], flag: str, value: object) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def replace_repeated_flag(
    command: list[str], flag: str, values: list[object]
) -> None:
    indices = [index for index, item in enumerate(command) if item == flag]
    if len(indices) != len(values):
        raise RuntimeError(
            f"expected {len(values)} occurrences of {flag}, found {len(indices)}"
        )
    for index, value in zip(indices, values, strict=True):
        command[index + 1] = str(value)


def build_segment_command(
    *,
    config: dict,
    phase: str,
    steps: int,
    checkpoint_dir: Path,
    resume_checkpoint: Path | None,
    route_temperature: float | None,
) -> list[str]:
    command = _common_command(
        steps=steps,
        batch_size=int(config["batch_size"]),
        eval_examples=int(config["eval_examples"]),
        log_every=int(config["log_every_steps"]),
        checkpoint_dir=checkpoint_dir,
        route_backend=str(config["route_backend"]),
        packed_memory_backend=str(config["packed_memory_backend"]),
        packed_memory_reader_backend=str(
            config.get("packed_memory_reader_backend", "torch")
        ),
        packed_memory_writer_backend=str(
            config.get("packed_memory_writer_backend", "torch")
        ),
        packed_memory_feature_chunk=int(
            config.get("packed_memory_feature_chunk", 32)
        ),
        compile_model=bool(config["compile"]),
        max_runtime_seconds=int(config["child_runtime_seconds"]),
        checkpoint_every_seconds=float(config["checkpoint_sync_seconds"]),
        python_executable=sys.executable,
    )
    replace_flag(command, "--data", config["smoltalk_everyday"])
    replace_flag(command, "--formation-objective", config["formation_objective"])
    replace_flag(command, "--tinystories-data", config["tinystories_train"])
    replace_flag(
        command,
        "--tinystories-validation-data",
        config["tinystories_validation"],
    )
    replace_flag(command, "--lmsys-data", config["lmsys_chat"])
    replace_repeated_flag(
        command,
        "--chat-data",
        [config["tulu3_sft"], config["smoltalk_preferences"]],
    )
    command.extend(
        ["--slot-isolation-slots", str(config["slot_isolation_slots"])]
    )
    replace_flag(command, "--optimizer-lexical-lr-multiplier", COMPONENT_LR["lexical"])
    replace_flag(command, "--optimizer-lane-core-lr-multiplier", COMPONENT_LR["lane_core"])
    replace_flag(command, "--optimizer-route-gate-lr-multiplier", COMPONENT_LR["route_gate"])
    command.extend(
        [
            "--optimizer-private-node-lr-multiplier",
            str(COMPONENT_LR["private_node"]),
            "--optimizer-retain-frozen-state",
        ]
    )

    if phase == PHASE_A:
        train_components = ("lexical", "lane_core", "private_node", "route_gate")
        decode_mode = config["phase_a_decode_mode"]
        command.extend(["--decode-mode", decode_mode])
        command.extend(["--route-temperature-schedule", "staged_reheat"])
        command.extend(
            [
                "--route-temperature-start", "1.25",
                "--route-temperature-middle", "1.25",
                "--route-temperature-end", "1.25",
                "--route-temperature-step-offset", "0",
                "--route-temperature-total-steps", "1",
            ]
        )
    elif phase == PHASE_B:
        train_components = ("route_gate",)
        decode_mode = config["phase_b_decode_mode"]
        command.extend(["--decode-mode", decode_mode])
        command.extend(["--route-temperature-schedule", "staged_reheat"])
        command.extend(
            [
                "--route-temperature-start", "1.25",
                "--route-temperature-middle", "1.25",
                "--route-temperature-end", "1.25",
                "--route-temperature-step-offset", "0",
                "--route-temperature-total-steps", "1",
            ]
        )
    elif phase == PHASE_C:
        train_components = ("private_node",)
        fixed = 1.25 if route_temperature is None else float(route_temperature)
        decode_mode = config["phase_c_decode_mode"]
        command.extend(["--decode-mode", decode_mode])
        if decode_mode == "token_handoff":
            command.extend(
                [
                "--token-handoff-start-removed", "0",
                "--token-handoff-max-removed", "0",
                "--token-handoff-advance-ratio",
                str(config["handoff_advance_ratio"]),
                "--token-handoff-retreat-ratio",
                str(config["handoff_retreat_ratio"]),
                "--token-handoff-min-teacher-accuracy",
                str(config["handoff_min_teacher_accuracy"]),
                "--token-handoff-min-frontier-accuracy",
                str(config["handoff_min_frontier_accuracy"]),
                "--token-handoff-max-teacher-ppl-ratio", "0",
                "--token-handoff-retreat-teacher-ppl-ratio", "0",
                "--token-handoff-advance-patience",
                str(config["handoff_advance_patience"]),
                "--token-handoff-retreat-patience",
                str(config["handoff_retreat_patience"]),
                "--token-handoff-full-probe-fraction",
                str(config["full_free_probe_fraction"]),
                "--token-handoff-unforced-loss-weight",
                str(config["handoff_unforced_loss_weight"]),
                "--token-handoff-loss-normalization",
                str(config["handoff_loss_normalization"]),
                "--route-temperature-schedule", "staged_reheat",
                "--route-temperature-start", str(fixed),
                "--route-temperature-middle", str(fixed),
                "--route-temperature-end", str(fixed),
                "--route-temperature-step-offset", "0",
                "--route-temperature-total-steps", "1",
                ]
            )
            if config["handoff_allow_policy_override"]:
                command.append("--token-handoff-allow-policy-override")
        else:
            command.extend(
                [
                    "--route-temperature-schedule", "staged_reheat",
                    "--route-temperature-start", str(fixed),
                    "--route-temperature-middle", str(fixed),
                    "--route-temperature-end", str(fixed),
                    "--route-temperature-step-offset", "0",
                    "--route-temperature-total-steps", "1",
                ]
            )
    else:
        raise ValueError(f"unknown phase {phase}")
    command.extend(["--optimizer-train-components", *train_components])
    if resume_checkpoint is not None:
        command.extend(
            [
                "--resume-checkpoint", str(resume_checkpoint),
                "--resume-optimizer",
                "--resume-data-state", str(config["resume_data_state"]),
            ]
        )
    return command


class DriveMirror:
    def __init__(self, local_dir: Path, drive_dir: Path, config: dict) -> None:
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.log_interval = float(config["log_sync_seconds"])
        self.checkpoint_interval = float(config["checkpoint_sync_seconds"])
        now = time.monotonic()
        self.last_log = now
        self.last_checkpoint = now
        self.pending: list[str] = []

    def observe(self, line: str) -> None:
        self.pending.append(line)
        self.tick()

    def tick(self) -> None:
        """Honor wall-clock syncs even while the trainer emits no new line."""
        now = time.monotonic()
        if now - self.last_log >= self.log_interval:
            self.sync_log()
        if now - self.last_checkpoint >= self.checkpoint_interval:
            self.sync_checkpoint()

    def sync_log(self) -> None:
        if self.pending:
            self.drive_dir.mkdir(parents=True, exist_ok=True)
            with (self.drive_dir / "train.log").open(
                "a", encoding="utf-8"
            ) as output:
                output.write("".join(self.pending))
            self.pending.clear()
        self.last_log = time.monotonic()

    def sync_checkpoint(self, force: bool = False) -> None:
        source = self.local_dir / "latest.pt"
        if source.is_file() and (
            force
            or time.monotonic() - self.last_checkpoint
            >= self.checkpoint_interval
        ):
            _atomic_copy(source, self.drive_dir / "latest.pt")
            self.last_checkpoint = time.monotonic()

    def finalize(self) -> None:
        self.sync_log()
        self.sync_checkpoint(force=True)


def plateau_transition(
    state: dict, config: dict, drive_dir: Path, metrics: dict
) -> str | None:
    phase = state["phase"]
    if phase not in {PHASE_A, PHASE_B}:
        return None
    teacher = metrics.get("teacher")
    if not teacher or not math.isfinite(float(teacher["content_perplexity"])):
        raise RuntimeError("teacher phase ended without finite teacher content PPL")
    current = float(teacher["content_perplexity"])
    previous = state.get("phase_best_teacher_ppl")
    improved = previous is None or current < float(previous)
    if improved:
        state["phase_best_teacher_ppl"] = current
        state["phase_best_step"] = int(metrics["step"])
        _atomic_copy(
            drive_dir / "latest.pt",
            drive_dir / ("phase_a_best.pt" if phase == PHASE_A else "phase_b_best.pt"),
        )
    relative = None if previous is None else (float(previous) - current) / float(previous)
    state["last_plateau_relative_improvement"] = relative
    minimum_segments = int(config["plateau_min_segments"])
    max_steps = (
        int(config["coadapt_max_steps"])
        if phase == PHASE_A
        else int(config["route_form_max_steps"])
    )
    phase_work = int(metrics["step"]) - int(state["phase_entry_step"])
    plateau = (
        previous is not None
        and int(state["phase_segments"]) >= minimum_segments
        and relative < float(config["plateau_relative_improvement"])
    )
    capped = phase_work >= max_steps
    if not plateau and not capped:
        return None

    best_name = "phase_a_best.pt" if phase == PHASE_A else "phase_b_best.pt"
    probe_name = "phase_a_plateau_probe.pt" if phase == PHASE_A else "phase_b_plateau_probe.pt"
    _atomic_copy(drive_dir / "latest.pt", drive_dir / probe_name)
    _atomic_copy(drive_dir / best_name, drive_dir / "latest.pt")
    best_metrics = load_checkpoint_metrics(drive_dir / "latest.pt")
    state["retained_step"] = int(best_metrics["step"])
    state["phase_entry_step"] = int(best_metrics["step"])
    state["phase_segments"] = 0
    state["phase_best_teacher_ppl"] = (
        float(best_metrics["teacher"]["content_perplexity"])
        if phase == PHASE_A
        else None
    )
    state["phase_best_step"] = int(best_metrics["step"]) if phase == PHASE_A else None
    if phase == PHASE_A:
        state["phase"] = PHASE_B
        _atomic_copy(drive_dir / "latest.pt", drive_dir / "phase_b_best.pt")
    else:
        state["phase"] = PHASE_C
        state["fixed_route_temperature"] = 1.25
    return "plateau" if plateau else "safety_cap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--log-every-steps", type=int)
    parser.add_argument("--slot-isolation-slots", type=int)
    parser.add_argument(
        "--decode-mode",
        choices=DECODE_MODES,
        help="override the decode mode for all staged phases",
    )
    parser.add_argument(
        "--phase-a-decode-mode", choices=DECODE_MODES,
        help="decode mode during co-adaptation (default: teacher_forced)",
    )
    parser.add_argument(
        "--phase-b-decode-mode", choices=DECODE_MODES,
        help="decode mode during route/jury training (default: teacher_forced)",
    )
    parser.add_argument(
        "--phase-c-decode-mode", choices=DECODE_MODES,
        help="decode mode during node handoff (default: token_handoff)",
    )
    parser.add_argument(
        "--formation-objective",
        choices=("slot_ce", "blank_ce", "viterbi", "monotonic"),
        help="formation objective passed to the trainer (serial requires slot_ce)",
    )
    parser.add_argument("--handoff-min-teacher-accuracy", type=float)
    parser.add_argument("--handoff-min-frontier-accuracy", type=float)
    parser.add_argument("--handoff-advance-ratio", type=float)
    parser.add_argument("--handoff-retreat-ratio", type=float)
    parser.add_argument("--handoff-advance-patience", type=int)
    parser.add_argument("--handoff-retreat-patience", type=int)
    parser.add_argument("--handoff-full-probe-fraction", type=float)
    parser.add_argument("--handoff-unforced-loss-weight", type=float)
    parser.add_argument(
        "--handoff-loss-normalization",
        choices=("active_slots", "all_slots", "none"),
    )
    parser.add_argument(
        "--handoff-allow-policy-override",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.log_every_steps is not None and args.log_every_steps <= 0:
        parser.error("--log-every-steps must be positive")
    if (
        args.slot_isolation_slots is not None
        and not 0 <= args.slot_isolation_slots <= 128
    ):
        parser.error("--slot-isolation-slots must be between 0 and 128")
    for name, value in (
        ("--handoff-min-teacher-accuracy", args.handoff_min_teacher_accuracy),
        ("--handoff-min-frontier-accuracy", args.handoff_min_frontier_accuracy),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    if (
        args.handoff_advance_ratio is not None
        and not 0.75 < args.handoff_advance_ratio <= 1.0
    ):
        parser.error("--handoff-advance-ratio must be greater than 0.75 and at most 1")
    if args.handoff_retreat_ratio is not None and not 0.0 < args.handoff_retreat_ratio <= 1.0:
        parser.error("--handoff-retreat-ratio must be greater than 0 and at most 1")
    for name, value in (
        ("--handoff-advance-patience", args.handoff_advance_patience),
        ("--handoff-retreat-patience", args.handoff_retreat_patience),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if args.handoff_full_probe_fraction is not None and not 0.0 <= args.handoff_full_probe_fraction <= 1.0:
        parser.error("--handoff-full-probe-fraction must be between 0 and 1")
    if args.handoff_unforced_loss_weight is not None and args.handoff_unforced_loss_weight < 0:
        parser.error("--handoff-unforced-loss-weight cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("handoff_retreat_ratio", 0.75)
    config.setdefault("handoff_unforced_loss_weight", 2.0)
    config.setdefault("handoff_loss_normalization", "active_slots")
    config.setdefault("handoff_allow_policy_override", True)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.log_every_steps is not None:
        config["log_every_steps"] = args.log_every_steps
    if args.slot_isolation_slots is not None:
        config["slot_isolation_slots"] = args.slot_isolation_slots
    default_decode_modes = {
        "phase_a_decode_mode": "teacher_forced",
        "phase_b_decode_mode": "teacher_forced",
        "phase_c_decode_mode": "token_handoff",
    }
    for key, default in default_decode_modes.items():
        config[key] = config.get(key, default)
    if args.decode_mode is not None:
        for key in default_decode_modes:
            config[key] = args.decode_mode
    for key in default_decode_modes:
        arg_name = key.replace("phase_", "phase_")
        value = getattr(args, arg_name)
        if value is not None:
            config[key] = value
    config["formation_objective"] = config.get("formation_objective", "slot_ce")
    if args.formation_objective is not None:
        config["formation_objective"] = args.formation_objective
    if any(
        config[key] == "serial"
        for key in default_decode_modes
    ) and config["formation_objective"] != "slot_ce":
        raise ValueError("--decode-mode serial requires --formation-objective slot_ce")
    if args.handoff_min_teacher_accuracy is not None:
        config["handoff_min_teacher_accuracy"] = (
            args.handoff_min_teacher_accuracy
        )
    if args.handoff_min_frontier_accuracy is not None:
        config["handoff_min_frontier_accuracy"] = (
            args.handoff_min_frontier_accuracy
        )
    if args.handoff_advance_ratio is not None:
        config["handoff_advance_ratio"] = args.handoff_advance_ratio
    overrides = {
        "handoff_retreat_ratio": args.handoff_retreat_ratio,
        "handoff_advance_patience": args.handoff_advance_patience,
        "handoff_retreat_patience": args.handoff_retreat_patience,
        "full_free_probe_fraction": args.handoff_full_probe_fraction,
        "handoff_unforced_loss_weight": args.handoff_unforced_loss_weight,
        "handoff_loss_normalization": args.handoff_loss_normalization,
        "handoff_allow_policy_override": args.handoff_allow_policy_override,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    parameter_count = preflight_parameter_count()
    drive_dir = Path(config["drive_run_root"]) / config["run_name"]
    drive_dir.mkdir(parents=True, exist_ok=True)
    state_path = drive_dir / STATE_NAME
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = fresh_state(config)
        _atomic_json(state_path, state)
    if state["phase"] == COMPLETE:
        print(f"training_complete step={state['retained_step']}", flush=True)
        return

    resume = drive_dir / "latest.pt"
    resume_metrics = load_checkpoint_metrics(resume) if resume.is_file() else None
    resume_step = 0 if resume_metrics is None else int(resume_metrics["step"])
    state["retained_step"] = resume_step
    additional = min(
        int(config["segment_steps"]),
        int(config["target_steps"]) - resume_step,
    )
    if additional <= 0:
        state["phase"] = COMPLETE
        state["status"] = COMPLETE
        _atomic_json(state_path, state)
        print(f"training_complete step={resume_step}", flush=True)
        return

    segment_index = int(state["segment_index"]) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = (
        Path(config["local_work_root"])
        / config["run_name"]
        / f"segment_{segment_index:04d}_{state['phase']}_{stamp}"
    )
    local_dir.mkdir(parents=True, exist_ok=False)
    command = build_segment_command(
        config=config,
        phase=state["phase"],
        steps=additional,
        checkpoint_dir=local_dir,
        resume_checkpoint=resume if resume.is_file() else None,
        route_temperature=state.get("fixed_route_temperature"),
    )
    _atomic_json(
        local_dir / "RUN_SPEC.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": parameter_count,
            "phase": state["phase"],
            "resume_step": resume_step,
            "additional_steps": additional,
            "config": config,
            "command": command,
        },
    )
    print(
        f"segment_start index={segment_index} phase={state['phase']} "
        f"resume={resume_step} additional={additional} parameters={parameter_count}",
        flush=True,
    )
    print("command=" + " ".join(command), flush=True)
    if args.dry_run:
        return

    mirror = DriveMirror(local_dir, drive_dir, config)
    return_code = -1
    try:
        with (local_dir / "train.log").open(
            "w", encoding="utf-8", buffering=1
        ) as output:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="backslashreplace",
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while process.poll() is None:
                ready = selector.select(timeout=30.0)
                if not ready:
                    mirror.tick()
                    continue
                line = process.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    output.write(line)
                    mirror.observe(line)
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                mirror.observe(line)
            selector.close()
            return_code = process.wait()
    finally:
        mirror.finalize()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    metrics = load_checkpoint_metrics(drive_dir / "latest.pt")
    state["attempted_steps"] = int(state["attempted_steps"]) + (
        int(metrics["step"]) - resume_step
    )
    state["retained_step"] = int(metrics["step"])
    state["segment_index"] = segment_index
    state["phase_segments"] = int(state["phase_segments"]) + 1
    phase_before = state["phase"]
    transition = plateau_transition(state, config, drive_dir, metrics)
    if state["retained_step"] >= int(config["target_steps"]):
        state["phase"] = COMPLETE
        state["status"] = COMPLETE
    state["history"].append(
        {
            "segment_index": segment_index,
            "phase": phase_before,
            "metrics": metrics,
            "transition": transition,
            "retained_step": state["retained_step"],
            "attempted_steps": state["attempted_steps"],
        }
    )
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, state)
    _atomic_copy(args.config, drive_dir / "GATOR100M_COLAB_CONFIG.json")
    source_spec = local_dir / "RUN_SPEC.json"
    _atomic_copy(source_spec, drive_dir / "LAST_RUN_SPEC.json")
    shutil.rmtree(local_dir, ignore_errors=True)
    print(
        f"segment_complete index={segment_index} phase_before={phase_before} "
        f"phase_after={state['phase']} retained={state['retained_step']} "
        f"attempted={state['attempted_steps']} transition={transition}",
        flush=True,
    )


if __name__ == "__main__":
    main()
