"""Launch and monitor configurable GTR-L EvoToken training in Colab.

The module is intentionally importable directly from ``gtrl_evotoken_bundle.zip``.
A notebook only needs to mount Drive, put the bundle on ``sys.path``, and call
``main([...])``. The same interface is available from the command line:

    python -m gtrl_evotoken_colab --bundle PATH --output-slots 4

Every setting in :class:`RunConfig` is exposed as a kebab-case CLI flag and can
also be supplied in a JSON file with ``--config``. Command-line values override
the JSON file. The launcher extracts the bundle, runs focused tests, starts a
detached worker from this same module, and mirrors logs/checkpoints to Drive.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence
import zipfile


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/gtrl_evotoken_relay"
DEFAULT_BUNDLE_NAME = "gtrl_evotoken_bundle.zip"
WORKER_CONFIG_ENV = "GTRL_EVOTOKEN_WORKER_CONFIG"
LEGACY_WORKER_PATTERN = "gtrl_evotoken_relay_worker.py"
MODULE_WORKER_MARKER = "--worker-config"


@dataclass
class RunConfig:
    # Experiment identity. A custom architecture or objective should normally
    # receive a custom tag so it cannot resume a shape-incompatible checkpoint.
    experiment_mode: str = "relay"
    experiment_tag: str | None = None
    experiment_version: str | None = None

    # Paths. Optional paths are derived after identity is resolved.
    drive_root: str = DEFAULT_DRIVE_ROOT
    bundle: str | None = None
    code_dir: str = "/content/GTR_evotoken_relay"
    data_dir: str = "/content/tinystories"
    local_root: str = "/content"
    drive_log_dir: str | None = None
    local_log: str | None = None
    local_checkpoint: str | None = None
    drive_checkpoint: str | None = None
    initial_model_checkpoint: str | None = None
    progress_archive: str | None = None
    pid_file: str = "/content/gtrl_evotoken_relay_worker.pid"
    local_config: str | None = None
    drive_config: str | None = None

    # TinyStories/tokenizer.
    repository: str = "roneneldan/TinyStories"
    repository_type: str = "dataset"
    revision: str = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    train_filename: str = "TinyStories-train.txt"
    valid_filename: str = "TinyStories-valid.txt"
    train_chars: int = 80_000_000
    valid_chars: int = 4_000_000
    vocab: int = 2_048
    reserved_blank_tokens: int = 1
    excluded_whitespace_bytes: tuple[int, ...] = (9, 10, 11, 12, 13, 32)

    # Sequence/model shape.
    observed: int = 128
    output_slots: int = 4
    context_length: int | None = None
    streams: int = 10
    nodes: int = 40
    width: int = 16
    relays: int = 4
    degree: int = 6
    buses: int = 4
    mailbox_channels: int = 4
    mailbox_width: int = 8
    basis_rank: int = 4
    address_width: int = 4
    router_rank: int = 8
    conv_kernel: int = 9
    scan_chunk: int = 64
    scan_normalise: bool = False
    graph_layout: str = "small_world"
    graph_seed: int = 0
    graph_rewire: float = 0.5
    ingress_temperature: float = 1.0
    private_stream_io: bool = True
    private_stream_rank: int = 4
    heterogeneous_streams: bool = False
    stream_width_min_fraction: float = 0.25
    update_gate_bias: float = 0.0
    readout_bias: bool = True
    tie_token_embeddings: bool = False
    hard_routing: bool = True
    router_occupancy: bool = False
    linear_attention: bool = True
    linear_attention_key_dim: int = 8
    controlled_writing: bool = True
    causal: bool = False
    block_causal: bool = False

    # EvoToken trajectory.
    trajectory_steps: int = 4
    promotion_counts: tuple[int, ...] | None = None
    top_k: int = 3
    train_alpha_min: float = 0.5
    train_alpha_max: float = 1.0
    inference_alpha: float = 0.7
    feedback_scale: float | None = None
    visible_token_loss_weight: float = 0.0
    temperature: float = 1.0

    # Position-starvation ablations. Defaults preserve the v5 objective.
    slot_permutation_mode: str = "none"
    promotion_schedule_mode: str = "random"
    inference_promotion_mode: str = "confidence"
    position_objective_coordinates: str = "logical"
    tail_loss_slots: int = 0
    position_balance_gamma: float = 0.0
    position_balance_decay: float = 0.99
    position_balance_epsilon: float = 0.02
    position_balance_max_ratio: float = 3.0
    worst_position_weight: float = 0.0
    worst_position_temperature: float = 0.5
    position_gradient_every_evals: int = 0
    position_gradient_max_slots: int = 8
    position_gradient_coordinates: str = "logical"

    # Optimisation/runtime.
    total_steps: int = 10_000_000
    eval_every: int = 300
    checkpoint_every: int = 30_000
    batch_size: int = 16
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-4
    lr_decay_steps: int = 1_000_000
    weight_decay: float = 1e-4
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    gradient_clip: float = 1.0
    seed: int = 0
    eval_seed: int = 7
    device: str = "cuda"
    amp_dtype: str = "bf16"
    compile: bool = True
    compile_fullgraph: bool = True
    compile_mode: str | None = None
    compile_backend: str | None = None
    resume_source: str = "auto"
    checkpoint_on_exit: bool = True
    install_missing_dependencies: bool = True

    # Degeneracy diagnostics.
    collapse_warmup_steps: int = 10_000
    collapse_modal_share_threshold: float = 0.40
    collapse_max_unigram_lift: float = 0.02
    collapse_position_margin: float = 0.01
    collapse_min_weak_positions: int | None = None
    collapse_patience_evals: int = 5

    # Launcher/monitor.
    preflight: bool = True
    stop_existing: bool = True
    monitor: bool = True
    interval_seconds: float = 900.0
    startup_wait_seconds: float = 5.0
    stop_wait_seconds: float = 30.0
    digest_lines: int = 10
    mirror_samples: bool = False


_OPTIONAL_PARSERS: dict[str, Any] = {
    "experiment_tag": str,
    "experiment_version": str,
    "bundle": str,
    "drive_log_dir": str,
    "local_log": str,
    "local_checkpoint": str,
    "drive_checkpoint": str,
    "initial_model_checkpoint": str,
    "progress_archive": str,
    "local_config": str,
    "drive_config": str,
    "context_length": int,
    "promotion_counts": lambda value: _parse_int_tuple(value),
    "feedback_scale": float,
    "compile_mode": str,
    "compile_backend": str,
    "collapse_min_weak_positions": int,
}

_TUPLE_FIELDS = {
    "excluded_whitespace_bytes",
    "promotion_counts",
}


def _parse_int_tuple(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        return tuple(int(item.strip()) for item in stripped.split(","))
    return tuple(int(item) for item in value)


def balanced_promotion_counts(
    output_slots: int,
    steps: int,
) -> tuple[int, ...]:
    if output_slots <= 0 or steps <= 0:
        raise ValueError("output_slots and trajectory_steps must be positive")
    base, remainder = divmod(output_slots, steps)
    counts = [base] * steps
    for index in range(remainder):
        counts[index * steps // remainder] += 1
    return tuple(counts)


def _default_feedback_scale(mode: str) -> float:
    return {
        "relay": 0.5,
        "relay_no_feedback": 0.0,
        "full_pass": 1.0,
    }[mode]


def resolve_config(config: RunConfig) -> RunConfig:
    if config.experiment_mode not in {
        "relay",
        "relay_no_feedback",
        "full_pass",
    }:
        raise ValueError(f"unknown experiment_mode={config.experiment_mode!r}")
    if config.resume_source not in {"auto", "local", "drive", "none"}:
        raise ValueError("resume_source must be auto, local, drive, or none")
    if config.amp_dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("amp_dtype must be bf16, fp16, or fp32")
    if config.slot_permutation_mode not in {"none", "rotate", "shuffle"}:
        raise ValueError(
            "slot_permutation_mode must be none, rotate, or shuffle"
        )
    if config.promotion_schedule_mode not in {"random", "rotating", "fixed"}:
        raise ValueError(
            "promotion_schedule_mode must be random, rotating, or fixed"
        )
    if config.inference_promotion_mode not in {
        "confidence",
        "rotating",
        "fixed",
        "tail_first",
    }:
        raise ValueError(
            "inference_promotion_mode must be confidence, rotating, fixed, "
            "or tail_first"
        )
    if config.position_objective_coordinates not in {"logical", "physical"}:
        raise ValueError(
            "position_objective_coordinates must be logical or physical"
        )
    if config.position_gradient_coordinates not in {"logical", "physical"}:
        raise ValueError(
            "position_gradient_coordinates must be logical or physical"
        )

    positive_integer_fields = (
        "train_chars",
        "valid_chars",
        "vocab",
        "observed",
        "output_slots",
        "streams",
        "nodes",
        "width",
        "relays",
        "degree",
        "buses",
        "mailbox_channels",
        "mailbox_width",
        "basis_rank",
        "address_width",
        "router_rank",
        "conv_kernel",
        "scan_chunk",
        "private_stream_rank",
        "linear_attention_key_dim",
        "trajectory_steps",
        "top_k",
        "total_steps",
        "eval_every",
        "checkpoint_every",
        "batch_size",
        "eval_batch_size",
        "lr_decay_steps",
        "position_gradient_max_slots",
        "collapse_patience_evals",
        "digest_lines",
    )
    for name in positive_integer_fields:
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "learning_rate",
        "minimum_learning_rate",
        "gradient_clip",
        "temperature",
        "ingress_temperature",
        "optimizer_epsilon",
        "interval_seconds",
        "startup_wait_seconds",
        "stop_wait_seconds",
    ):
        if float(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0 <= config.tail_loss_slots <= config.output_slots:
        raise ValueError("tail_loss_slots must be within output slots")
    if config.position_balance_gamma < 0:
        raise ValueError("position_balance_gamma cannot be negative")
    if not 0.0 <= config.position_balance_decay < 1.0:
        raise ValueError("position_balance_decay must be in [0, 1)")
    if config.position_balance_epsilon <= 0:
        raise ValueError("position_balance_epsilon must be positive")
    if config.position_balance_max_ratio < 1.0:
        raise ValueError("position_balance_max_ratio must be at least one")
    if not 0.0 <= config.worst_position_weight <= 1.0:
        raise ValueError("worst_position_weight must be in [0, 1]")
    if config.worst_position_temperature <= 0:
        raise ValueError("worst_position_temperature must be positive")
    if config.position_gradient_every_evals < 0:
        raise ValueError("position_gradient_every_evals cannot be negative")
    if not 0.0 <= config.graph_rewire <= 1.0:
        raise ValueError("graph_rewire must be in [0, 1]")
    if config.graph_layout not in {"ring", "small_world", "dense"}:
        raise ValueError("graph_layout must be ring, small_world, or dense")
    if not 0.0 < config.stream_width_min_fraction <= 1.0:
        raise ValueError("stream_width_min_fraction must be in (0, 1]")
    if config.degree > config.nodes:
        raise ValueError("degree cannot exceed nodes")
    if config.address_width < 2:
        raise ValueError("address_width must be at least two")
    if config.causal and config.block_causal:
        raise ValueError("causal and block_causal are mutually exclusive")
    if config.causal and config.slot_permutation_mode != "none":
        raise ValueError("slot permutation requires a bidirectional output block")
    if not config.causal and config.conv_kernel % 2 == 0:
        raise ValueError("conv_kernel must be odd when causal is false")
    if config.block_causal and config.observed % config.output_slots != 0:
        raise ValueError(
            "block_causal observed length must be divisible by output_slots"
        )
    training_window = config.observed + config.output_slots
    if config.context_length is None:
        config.context_length = training_window
    elif config.context_length < training_window:
        raise ValueError(
            "context_length cannot be shorter than observed + output_slots"
        )
    for name in ("optimizer_beta1", "optimizer_beta2"):
        value = float(getattr(config, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if config.minimum_learning_rate > config.learning_rate:
        raise ValueError(
            "minimum_learning_rate cannot exceed learning_rate"
        )
    if not 0.0 <= config.train_alpha_min <= config.train_alpha_max <= 1.0:
        raise ValueError("training alpha bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= config.inference_alpha <= 1.0:
        raise ValueError("inference_alpha must be in [0, 1]")
    if not 0.0 <= config.visible_token_loss_weight <= 1.0:
        raise ValueError("visible_token_loss_weight must be in [0, 1]")
    if config.experiment_mode != "full_pass" and (
        config.relays != config.trajectory_steps
    ):
        raise ValueError(
            "relay modes require relays == trajectory_steps"
        )

    counts = (
        balanced_promotion_counts(
            config.output_slots,
            config.trajectory_steps,
        )
        if config.promotion_counts is None
        else _parse_int_tuple(config.promotion_counts)
    )
    if len(counts) != config.trajectory_steps:
        raise ValueError(
            "promotion_counts must contain one count per trajectory step"
        )
    if any(value < 0 for value in counts):
        raise ValueError("promotion counts cannot be negative")
    if sum(counts) != config.output_slots:
        raise ValueError("promotion counts must sum to output_slots")
    config.promotion_counts = counts

    if config.feedback_scale is None:
        config.feedback_scale = _default_feedback_scale(
            config.experiment_mode
        )
    if config.feedback_scale < 0:
        raise ValueError("feedback_scale cannot be negative")
    if config.collapse_min_weak_positions is None:
        config.collapse_min_weak_positions = max(
            1, config.output_slots // 2
        )
    if not 1 <= config.collapse_min_weak_positions <= config.output_slots:
        raise ValueError(
            "collapse_min_weak_positions must be within output slots"
        )

    tying_tag = "_tied" if config.tie_token_embeddings else ""
    block_tag = "_block" if config.block_causal else ""
    context_tag = (
        ""
        if config.context_length == training_window
        else f"_ctx{config.context_length}"
    )
    ablation_parts: list[str] = []
    if config.slot_permutation_mode != "none":
        ablation_parts.append(
            "permrot"
            if config.slot_permutation_mode == "rotate"
            else "permshuf"
        )
    if config.promotion_schedule_mode != "random":
        ablation_parts.append(
            "promrot"
            if config.promotion_schedule_mode == "rotating"
            else "promfix"
        )
    if config.inference_promotion_mode != "confidence":
        ablation_parts.append("infer" + config.inference_promotion_mode.replace("_", ""))
    if config.position_balance_gamma > 0:
        ablation_parts.append("pbal")
    if config.worst_position_weight > 0:
        ablation_parts.append(
            f"worst{round(config.worst_position_weight * 100)}"
        )
    if config.tail_loss_slots > 0:
        ablation_parts.append(f"tail{config.tail_loss_slots}")
    if (
        config.position_objective_coordinates == "physical"
        and (
            config.position_balance_gamma > 0
            or config.tail_loss_slots > 0
        )
    ):
        ablation_parts.append("phys")
    position_ablation_identity = {
        "slot_permutation_mode": config.slot_permutation_mode,
        "promotion_schedule_mode": config.promotion_schedule_mode,
        "inference_promotion_mode": config.inference_promotion_mode,
        "position_objective_coordinates": config.position_objective_coordinates,
        "tail_loss_slots": config.tail_loss_slots,
        "position_balance_gamma": config.position_balance_gamma,
        "position_balance_decay": config.position_balance_decay,
        "position_balance_epsilon": config.position_balance_epsilon,
        "position_balance_max_ratio": config.position_balance_max_ratio,
        "worst_position_weight": config.worst_position_weight,
        "worst_position_temperature": config.worst_position_temperature,
    }
    default_position_ablation_identity = {
        "slot_permutation_mode": "none",
        "promotion_schedule_mode": "random",
        "inference_promotion_mode": "confidence",
        "position_objective_coordinates": "logical",
        "tail_loss_slots": 0,
        "position_balance_gamma": 0.0,
        "position_balance_decay": 0.99,
        "position_balance_epsilon": 0.02,
        "position_balance_max_ratio": 3.0,
        "worst_position_weight": 0.0,
        "worst_position_temperature": 0.5,
    }
    if position_ablation_identity != default_position_ablation_identity:
        signature = hashlib.sha256(
            json.dumps(
                position_ablation_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:8]
        ablation_parts.append(f"a{signature}")
    ablation_tag = "" if not ablation_parts else "_" + "_".join(ablation_parts)
    tag = (
        config.experiment_tag
        or f"cts_masked_s{config.output_slots}{tying_tag}{block_tag}{context_tag}{ablation_tag}_v5"
    )
    version = (
        config.experiment_version
        or "gtrl-130k-evotoken-relay-"
        + tag.replace("_", "-")
    )
    config.experiment_tag = tag
    config.experiment_version = version

    drive_root = Path(config.drive_root)
    local_root = Path(config.local_root)
    code_dir = Path(config.code_dir)
    forbidden_code_dirs = {
        Path("/"),
        Path("/content"),
        Path.home(),
        drive_root,
        local_root,
    }
    if code_dir in forbidden_code_dirs:
        raise ValueError(
            f"code_dir is too broad for replacement: {code_dir}"
        )
    stem = f"{config.experiment_mode}_{tag}"
    local_stem = f"gtrl_evotoken_relay_{stem}"
    config.bundle = str(
        Path(config.bundle)
        if config.bundle is not None
        else drive_root / DEFAULT_BUNDLE_NAME
    )
    config.drive_log_dir = str(
        Path(config.drive_log_dir)
        if config.drive_log_dir is not None
        else drive_root / "run_logs"
    )
    config.local_log = str(
        Path(config.local_log)
        if config.local_log is not None
        else local_root / f"{local_stem}.log"
    )
    config.local_checkpoint = str(
        Path(config.local_checkpoint)
        if config.local_checkpoint is not None
        else local_root / f"{local_stem}_checkpoint.pt"
    )
    config.drive_checkpoint = str(
        Path(config.drive_checkpoint)
        if config.drive_checkpoint is not None
        else drive_root / f"{stem}_checkpoint.pt"
    )
    config.progress_archive = str(
        Path(config.progress_archive)
        if config.progress_archive is not None
        else Path(config.drive_log_dir) / f"{stem}_progress.log"
    )
    config.local_config = str(
        Path(config.local_config)
        if config.local_config is not None
        else local_root / f"{local_stem}_config.json"
    )
    config.drive_config = str(
        Path(config.drive_config)
        if config.drive_config is not None
        else drive_root / "run_configs" / f"{stem}.json"
    )
    return config


def config_dict(config: RunConfig) -> dict[str, Any]:
    result = asdict(config)
    for name in _TUPLE_FIELDS:
        value = result.get(name)
        if value is not None:
            result[name] = list(value)
    return result


def config_from_dict(payload: dict[str, Any]) -> RunConfig:
    allowed = {field.name for field in fields(RunConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    values = dict(payload)
    for name in _TUPLE_FIELDS:
        if values.get(name) is not None:
            values[name] = _parse_int_tuple(values[name])
    return resolve_config(RunConfig(**values))


def _field_parser(field_name: str, default: Any) -> Any:
    if field_name in _OPTIONAL_PARSERS:
        return _OPTIONAL_PARSERS[field_name]
    if field_name in _TUPLE_FIELDS:
        return _parse_int_tuple
    if isinstance(default, bool):
        return None
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    return str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON RunConfig; explicit CLI flags override its values",
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved configuration without launching",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved JSON configuration before launching",
    )
    for field in fields(RunConfig):
        flag = "--" + field.name.replace("_", "-")
        default = field.default
        kwargs: dict[str, Any] = {
            "dest": field.name,
            "default": argparse.SUPPRESS,
        }
        parser_type = _field_parser(field.name, default)
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = parser_type
        parser.add_argument(flag, **kwargs)
    return parser


def parse_config(
    argv: Sequence[str] | None = None,
) -> tuple[RunConfig, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_config is not None:
        payload = json.loads(args.worker_config.read_text(encoding="utf-8"))
        return config_from_dict(payload), args

    payload: dict[str, Any] = {}
    if args.config is not None:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(RunConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        parser.error(f"unknown config keys: {unknown}")
    for name in allowed:
        if hasattr(args, name):
            payload[name] = getattr(args, name)
    try:
        config = config_from_dict(payload)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return config, args


def drive_safe_text(text: str, *, mirror_samples: bool) -> str:
    if mirror_samples:
        return text
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not line.startswith("sample=")
    )


def digest(log_path: Path, *, lines: int) -> str:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    selected = [
        line
        for line in text.splitlines()
        if line.startswith("step=")
        or line.startswith("position=")
        or line.startswith("pos_grad ")
        or line.startswith("checkpoint=")
        or line.startswith("sample=")
        or line.startswith("=>")
        or line.startswith("!!! COLLAPSE WARNING")
        or "Traceback" in line
    ]
    tail = selected[-lines:] if selected else ["(no progress lines yet)"]
    return (
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        + "\n".join(tail)
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def _write_config(config: RunConfig) -> None:
    text = json.dumps(config_dict(config), indent=2, sort_keys=True) + "\n"
    _atomic_text(text, Path(config.local_config))
    _atomic_text(text, Path(config.drive_config))


def _pid_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return ""
    try:
        return path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def _stop_pid(pid: int, *, wait_seconds: float) -> None:
    command = _pid_command(pid)
    if MODULE_WORKER_MARKER not in command:
        print(
            f"worker=stale_pid ignored pid={pid} command={command!r}",
            flush=True,
        )
        return
    print(f"worker=stop_requested pid={pid}", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            print(f"worker=stopped pid={pid}", flush=True)
            return
        time.sleep(0.25)
    raise TimeoutError(
        f"worker pid {pid} did not stop within {wait_seconds}s"
    )


def stop_existing_workers(config: RunConfig) -> None:
    pid_path = Path(config.pid_file)
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            _stop_pid(pid, wait_seconds=config.stop_wait_seconds)
        except (OSError, ValueError) as error:
            print(f"worker=pid_cleanup warning={error}", flush=True)
        finally:
            pid_path.unlink(missing_ok=True)
    subprocess.run(
        ["pkill", "-f", LEGACY_WORKER_PATTERN],
        check=False,
    )


def preserve_local_artifacts(config: RunConfig) -> None:
    local_root = Path(config.local_root)
    drive_root = Path(config.drive_root)
    drive_log_dir = Path(config.drive_log_dir)
    for checkpoint in local_root.glob(
        "gtrl_evotoken_relay_*_checkpoint.pt"
    ):
        drive_name = checkpoint.name.removeprefix(
            "gtrl_evotoken_relay_"
        )
        _atomic_copy(checkpoint, drive_root / drive_name)
    for log_path in local_root.glob("gtrl_evotoken_relay_*.log"):
        safe = drive_safe_text(
            log_path.read_text(encoding="utf-8"),
            mirror_samples=config.mirror_samples,
        )
        _atomic_text(safe, drive_log_dir / log_path.name)


def extract_bundle(config: RunConfig) -> None:
    bundle = Path(config.bundle)
    if not bundle.is_file():
        raise FileNotFoundError(f"bundle is missing: {bundle}")
    code_dir = Path(config.code_dir)
    if code_dir.exists():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(code_dir)


def run_preflight(config: RunConfig) -> None:
    if not config.preflight:
        return
    print("Running focused EvoToken preflight...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_gator_linear_evotoken",
            "tests.test_gator_linear_block_cache",
            "tests.test_gator_linear."
            "LinearAttentionTest."
            "test_empty_nodes_have_zero_content_gradient",
            "tests.test_gtrl_evotoken_colab",
            "-q",
        ],
        cwd=config.code_dir,
        check=True,
    )


def worker_command(config: RunConfig) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "gtrl_evotoken_colab",
        "--worker-config",
        str(config.local_config),
    ]


def launch_worker(config: RunConfig) -> subprocess.Popen[bytes]:
    log_path = Path(config.local_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            worker_command(config),
            cwd=config.code_dir,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    Path(config.pid_file).write_text(
        f"{process.pid}\n", encoding="utf-8"
    )
    time.sleep(config.startup_wait_seconds)
    return_code = process.poll()
    if return_code is not None:
        tail = digest(log_path, lines=config.digest_lines)
        raise RuntimeError(
            f"worker exited during startup with code {return_code}\n{tail}"
        )
    print(
        f"worker=launched pid={process.pid} log={log_path}",
        flush=True,
    )
    return process


def sync_to_drive(
    config: RunConfig,
    *,
    checkpoint_signature: tuple[int, int] | None,
) -> tuple[int, int] | None:
    log_path = Path(config.local_log)
    if log_path.exists():
        safe = drive_safe_text(
            log_path.read_text(encoding="utf-8"),
            mirror_samples=config.mirror_samples,
        )
        _atomic_text(
            safe,
            Path(config.drive_log_dir) / log_path.name,
        )
    checkpoint = Path(config.local_checkpoint)
    if checkpoint.exists():
        stat = checkpoint.stat()
        signature_value = (stat.st_mtime_ns, stat.st_size)
        if signature_value != checkpoint_signature:
            _atomic_copy(checkpoint, Path(config.drive_checkpoint))
            checkpoint_signature = signature_value
    return checkpoint_signature


def monitor_worker(
    config: RunConfig,
    process: subprocess.Popen[bytes],
) -> None:
    log_path = Path(config.local_log)
    archive = Path(config.progress_archive)
    signature: tuple[int, int] | None = None
    print(
        f"Mirroring progress every {config.interval_seconds:g}s "
        f"to {config.drive_root}\n"
        "Interrupting this foreground monitor sends SIGTERM to the worker "
        "and waits for it to exit before syncing final artifacts.",
        flush=True,
    )
    try:
        while True:
            text = digest(log_path, lines=config.digest_lines)
            print(text, flush=True)
            _atomic_text(
                drive_safe_text(
                    text + "\n",
                    mirror_samples=config.mirror_samples,
                ),
                archive,
            )
            signature = sync_to_drive(
                config,
                checkpoint_signature=signature,
            )
            return_code = process.poll()
            if return_code is not None:
                print(
                    f"worker=exited code={return_code}; "
                    "final artifacts synced",
                    flush=True,
                )
                break
            time.sleep(config.interval_seconds)
    except KeyboardInterrupt:
        print(
            f"monitor=interrupted stopping_worker pid={process.pid}",
            flush=True,
        )
        os.kill(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + config.stop_wait_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.25)
        else:
            print(
                f"worker=stop_timeout pid={process.pid}; "
                f"did not exit within {config.stop_wait_seconds}s",
                flush=True,
            )
        signature = sync_to_drive(config, checkpoint_signature=signature)
        print(
            f"monitor=interrupted worker_stopped pid={process.pid} "
            f"code={process.poll()}",
            flush=True,
        )


def launch(config: RunConfig) -> None:
    Path(config.drive_root).mkdir(parents=True, exist_ok=True)
    Path(config.drive_log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.drive_config).parent.mkdir(parents=True, exist_ok=True)
    Path(config.local_checkpoint).parent.mkdir(
        parents=True, exist_ok=True
    )
    Path(config.pid_file).parent.mkdir(parents=True, exist_ok=True)
    if config.stop_existing:
        stop_existing_workers(config)
    preserve_local_artifacts(config)
    extract_bundle(config)
    _write_config(config)
    run_preflight(config)
    process = launch_worker(config)
    if config.monitor:
        monitor_worker(config, process)


WORKER_SOURCE = r"""
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


CONFIG_PATH = Path(os.environ["GTRL_EVOTOKEN_WORKER_CONFIG"])
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
sys.path.insert(0, CFG["code_dir"])

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    if not CFG["install_missing_dependencies"]:
        raise
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
        check=True,
    )
    from huggingface_hub import hf_hub_download

import torch
from torch.nn import functional as F

from gator.tokenization import CompactTokenizer, _PIECE_PATTERN
from gator_linear import (
    EvoTokenRefiner,
    GatorLinearConfig,
    GatorLinearPrefixConfig,
    GatorLinearPrefixModel,
    masked_trajectory_cross_entropy,
    position_gradient_statistics,
    sample_training_schedule,
)


EXPERIMENT_VERSION = CFG["experiment_version"]
EXPERIMENT_TAG = CFG["experiment_tag"]
EXPERIMENT_MODE = CFG["experiment_mode"]
REPOSITORY = CFG["repository"]
REPOSITORY_TYPE = CFG["repository_type"]
REVISION = CFG["revision"]
TRAIN_FILENAME = CFG["train_filename"]
VALID_FILENAME = CFG["valid_filename"]
DATA_DIR = Path(CFG["data_dir"])
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOCAB = CFG["vocab"]
OUTPUT_SLOTS = CFG["output_slots"]
OBSERVED = CFG["observed"]
TRAIN_WINDOW = OBSERVED + OUTPUT_SLOTS
CONTEXT_LENGTH = CFG["context_length"]
TRAIN_CHARS = CFG["train_chars"]
VALID_CHARS = CFG["valid_chars"]
RESERVED_BLANK_TOKENS = CFG["reserved_blank_tokens"]
EXCLUDED_WHITESPACE_BYTES = tuple(CFG["excluded_whitespace_bytes"])

STREAMS = CFG["streams"]
NODES = CFG["nodes"]
WIDTH = CFG["width"]
RELAYS = CFG["relays"]
DEGREE = CFG["degree"]
BUSES = CFG["buses"]
MAILBOX_CHANNELS = CFG["mailbox_channels"]
MAILBOX_WIDTH = CFG["mailbox_width"]
BASIS_RANK = CFG["basis_rank"]
ADDRESS_WIDTH = CFG["address_width"]
ROUTER_RANK = CFG["router_rank"]
CONV_KERNEL = CFG["conv_kernel"]
SCAN_CHUNK = CFG["scan_chunk"]
SCAN_NORMALISE = CFG["scan_normalise"]
GRAPH_LAYOUT = CFG["graph_layout"]
GRAPH_SEED = CFG["graph_seed"]
GRAPH_REWIRE = CFG["graph_rewire"]
INGRESS_TEMPERATURE = CFG["ingress_temperature"]
PRIVATE_STREAM_IO = CFG["private_stream_io"]
PRIVATE_STREAM_RANK = CFG["private_stream_rank"]
HETEROGENEOUS_STREAMS = CFG["heterogeneous_streams"]
STREAM_WIDTH_MIN_FRACTION = CFG["stream_width_min_fraction"]
UPDATE_GATE_BIAS = CFG["update_gate_bias"]
READOUT_BIAS = CFG["readout_bias"]
TIE_TOKEN_EMBEDDINGS = CFG["tie_token_embeddings"]
HARD_ROUTING = CFG["hard_routing"]
ROUTER_OCCUPANCY = CFG["router_occupancy"]
LINEAR_ATTENTION = CFG["linear_attention"]
LINEAR_ATTENTION_KEY_DIM = CFG["linear_attention_key_dim"]
CONTROLLED_WRITING = CFG["controlled_writing"]
CAUSAL = CFG["causal"]
BLOCK_CAUSAL = CFG["block_causal"]

TRAJECTORY_STEPS = CFG["trajectory_steps"]
PROMOTION_COUNTS = tuple(CFG["promotion_counts"])
TOP_K = CFG["top_k"]
TRAIN_ALPHA_MIN = CFG["train_alpha_min"]
TRAIN_ALPHA_MAX = CFG["train_alpha_max"]
INFERENCE_ALPHA = CFG["inference_alpha"]
FEEDBACK_SCALE = CFG["feedback_scale"]
VISIBLE_TOKEN_LOSS_WEIGHT = CFG["visible_token_loss_weight"]
TEMPERATURE = CFG["temperature"]
SLOT_PERMUTATION_MODE = CFG["slot_permutation_mode"]
PROMOTION_SCHEDULE_MODE = CFG["promotion_schedule_mode"]
INFERENCE_PROMOTION_MODE = CFG["inference_promotion_mode"]
POSITION_OBJECTIVE_COORDINATES = CFG["position_objective_coordinates"]
TAIL_LOSS_SLOTS = CFG["tail_loss_slots"]
POSITION_BALANCE_GAMMA = CFG["position_balance_gamma"]
POSITION_BALANCE_DECAY = CFG["position_balance_decay"]
POSITION_BALANCE_EPSILON = CFG["position_balance_epsilon"]
POSITION_BALANCE_MAX_RATIO = CFG["position_balance_max_ratio"]
WORST_POSITION_WEIGHT = CFG["worst_position_weight"]
WORST_POSITION_TEMPERATURE = CFG["worst_position_temperature"]
POSITION_GRADIENT_EVERY_EVALS = CFG["position_gradient_every_evals"]
POSITION_GRADIENT_MAX_SLOTS = CFG["position_gradient_max_slots"]
POSITION_GRADIENT_COORDINATES = CFG["position_gradient_coordinates"]
POSITION_ABLATION_SIGNATURE = {
    "slot_permutation_mode": SLOT_PERMUTATION_MODE,
    "promotion_schedule_mode": PROMOTION_SCHEDULE_MODE,
    "inference_promotion_mode": INFERENCE_PROMOTION_MODE,
    "position_objective_coordinates": POSITION_OBJECTIVE_COORDINATES,
    "tail_loss_slots": TAIL_LOSS_SLOTS,
    "position_balance_gamma": POSITION_BALANCE_GAMMA,
    "position_balance_decay": POSITION_BALANCE_DECAY,
    "position_balance_epsilon": POSITION_BALANCE_EPSILON,
    "position_balance_max_ratio": POSITION_BALANCE_MAX_RATIO,
    "worst_position_weight": WORST_POSITION_WEIGHT,
    "worst_position_temperature": WORST_POSITION_TEMPERATURE,
}
LEGACY_POSITION_ABLATION_SIGNATURE = {
    "slot_permutation_mode": "none",
    "promotion_schedule_mode": "random",
    "inference_promotion_mode": "confidence",
    "position_objective_coordinates": "logical",
    "tail_loss_slots": 0,
    "position_balance_gamma": 0.0,
    "position_balance_decay": 0.99,
    "position_balance_epsilon": 0.02,
    "position_balance_max_ratio": 3.0,
    "worst_position_weight": 0.0,
    "worst_position_temperature": 0.5,
}
WARMSTART_MODEL_FIELDS = (
    "vocab",
    "context_length",
    "streams",
    "nodes",
    "width",
    "relays",
    "degree",
    "buses",
    "mailbox_channels",
    "mailbox_width",
    "basis_rank",
    "address_width",
    "router_rank",
    "conv_kernel",
    "scan_chunk",
    "scan_normalise",
    "graph_layout",
    "graph_seed",
    "graph_rewire",
    "ingress_temperature",
    "private_stream_io",
    "private_stream_rank",
    "heterogeneous_streams",
    "stream_width_min_fraction",
    "update_gate_bias",
    "readout_bias",
    "tie_token_embeddings",
    "hard_routing",
    "router_occupancy",
    "linear_attention",
    "linear_attention_key_dim",
    "controlled_writing",
    "causal",
    "block_causal",
)

TOTAL_STEPS = CFG["total_steps"]
EVAL_EVERY = CFG["eval_every"]
CHECKPOINT_EVERY = CFG["checkpoint_every"]
BATCH_SIZE = CFG["batch_size"]
EVAL_BATCH_SIZE = CFG["eval_batch_size"]
LEARNING_RATE = CFG["learning_rate"]
MIN_LEARNING_RATE = CFG["minimum_learning_rate"]
LR_DECAY_STEPS = CFG["lr_decay_steps"]
WEIGHT_DECAY = CFG["weight_decay"]
OPTIMIZER_BETAS = (
    CFG["optimizer_beta1"],
    CFG["optimizer_beta2"],
)
OPTIMIZER_EPSILON = CFG["optimizer_epsilon"]
GRADIENT_CLIP = CFG["gradient_clip"]
SEED = CFG["seed"]
EVAL_SEED = CFG["eval_seed"]
COMPILE = CFG["compile"]
COMPILE_FULLGRAPH = CFG["compile_fullgraph"]
COMPILE_MODE = CFG["compile_mode"]
COMPILE_BACKEND = CFG["compile_backend"]
DEVICE = CFG["device"]
AMP_DTYPE_NAME = CFG["amp_dtype"]
AMP_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}[AMP_DTYPE_NAME]
AUTOCAST_ENABLED = AMP_DTYPE_NAME != "fp32"

COLLAPSE_WARMUP_STEPS = CFG["collapse_warmup_steps"]
COLLAPSE_MODAL_SHARE_THRESHOLD = CFG[
    "collapse_modal_share_threshold"
]
COLLAPSE_MAX_UNIGRAM_LIFT = CFG["collapse_max_unigram_lift"]
COLLAPSE_POSITION_MARGIN = CFG["collapse_position_margin"]
COLLAPSE_MIN_WEAK_POSITIONS = CFG[
    "collapse_min_weak_positions"
]
COLLAPSE_PATIENCE_EVALS = CFG["collapse_patience_evals"]

LOCAL_CHECKPOINT = Path(CFG["local_checkpoint"])
DRIVE_CHECKPOINT = Path(CFG["drive_checkpoint"])
RESUME_SOURCE = CFG["resume_source"]
CHECKPOINT_ON_EXIT = CFG["checkpoint_on_exit"]
INITIAL_MODEL_CHECKPOINT = (
    None
    if CFG["initial_model_checkpoint"] is None
    else Path(CFG["initial_model_checkpoint"])
)

def download(filename):
    return Path(
        hf_hub_download(
            repo_id=REPOSITORY,
            filename=filename,
            repo_type=REPOSITORY_TYPE,
            revision=REVISION,
            local_dir=str(DATA_DIR),
        )
    )


print("Downloading TinyStories...", flush=True)
train_path = download(TRAIN_FILENAME)
valid_path = download(VALID_FILENAME)
with open(train_path, "r", encoding="utf-8", errors="ignore") as handle:
    train_text = handle.read(TRAIN_CHARS)
with open(valid_path, "r", encoding="utf-8", errors="ignore") as handle:
    valid_text = handle.read(VALID_CHARS)
print(
    f"train_chars={len(train_text)} valid_chars={len(valid_text)}",
    flush=True,
)

print("Training tokenizer (whitespace excluded)...", flush=True)
counts = Counter()
for piece in _PIECE_PATTERN.findall(train_text):
    if not piece.isspace():
        counts[piece] += 1
capacity = (
    VOCAB
    - CompactTokenizer.LEARNED_OFFSET
    - RESERVED_BLANK_TOKENS
)
ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
if len(ranked) < capacity:
    raise ValueError(
        f"corpus has only {len(ranked)} unique non-whitespace pieces, "
        f"need {capacity}"
    )
tokenizer = CompactTokenizer(piece for piece, _ in ranked[:capacity])
if tokenizer.vocab_size != VOCAB:
    raise AssertionError(
        f"tokenizer vocabulary is {tokenizer.vocab_size}, expected {VOCAB}"
    )


def encode_no_whitespace(text):
    result = []
    for piece in _PIECE_PATTERN.findall(text):
        if piece.isspace():
            continue
        token_id = tokenizer._piece_to_id.get(piece)
        if token_id is not None:
            result.append(token_id)
        else:
            result.extend(
                tokenizer.BYTE_OFFSET + byte
                for byte in piece.encode("utf-8")
            )
    return result


print("Tokenizing corpus (whitespace stripped)...", flush=True)
device = torch.device(DEVICE)
train_ids = torch.tensor(
    encode_no_whitespace(train_text), dtype=torch.long, device=device
)
valid_ids = torch.tensor(
    encode_no_whitespace(valid_text), dtype=torch.long, device=device
)
print(
    f"train_tokens={len(train_ids)} valid_tokens={len(valid_ids)}",
    flush=True,
)

# Define legality from tokenizer semantics, never from validation occurrence.
# All learned pieces are non-whitespace by construction. Byte fallback remains
# legal except for the ASCII whitespace bytes removed by this experiment.
valid_token_mask = torch.zeros(VOCAB, dtype=torch.bool, device=device)
valid_token_mask[
    tokenizer.BYTE_OFFSET : tokenizer.LEARNED_OFFSET
] = True
whitespace_byte_ids = torch.tensor(
    [
        tokenizer.BYTE_OFFSET + byte
        for byte in EXCLUDED_WHITESPACE_BYTES
    ],
    dtype=torch.long,
    device=device,
)
valid_token_mask[whitespace_byte_ids] = False
valid_token_mask[
    tokenizer.LEARNED_OFFSET : tokenizer.learned_end
] = True
if not bool(valid_token_mask[train_ids].all()):
    raise AssertionError("training stream contains an illegal target token")
if not bool(valid_token_mask[valid_ids].all()):
    raise AssertionError("validation stream contains an illegal target token")


def make_batch(ids, batch_size, generator=None):
    max_start = ids.shape[0] - TRAIN_WINDOW - 1
    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
        device=ids.device,
        generator=generator,
    )
    offsets = torch.arange(TRAIN_WINDOW, device=ids.device)
    windows = ids[starts.unsqueeze(1) + offsets.unsqueeze(0)]
    return windows[:, :OBSERVED], windows[:, OBSERVED:]


def legal_argmax(logits):
    return logits.masked_fill(
        ~valid_token_mask.view(*([1] * (logits.ndim - 1)), -1),
        -torch.inf,
    ).argmax(-1)


def sample_slot_mapping():
    identity = torch.arange(OUTPUT_SLOTS, device=device)
    if SLOT_PERMUTATION_MODE == "none":
        return None, identity, identity
    if SLOT_PERMUTATION_MODE == "rotate":
        offset = torch.randint(OUTPUT_SLOTS, (), device=device)
        physical_to_logical = (identity + offset) % OUTPUT_SLOTS
    else:
        physical_to_logical = torch.randperm(OUTPUT_SLOTS, device=device)
    logical_to_physical = physical_to_logical.argsort()
    return physical_to_logical, physical_to_logical, logical_to_physical


def training_promotion_order(batch_size):
    if PROMOTION_SCHEDULE_MODE == "random":
        return None
    base = torch.arange(OUTPUT_SLOTS, device=device).view(1, -1)
    if PROMOTION_SCHEDULE_MODE == "fixed":
        return base.expand(batch_size, -1)
    offsets = (
        completed + torch.arange(batch_size, device=device)
    ).remainder(OUTPUT_SLOTS)
    return (base + offsets.view(-1, 1)).remainder(OUTPUT_SLOTS)


def promotion_schedule_from_order(order):
    schedule = torch.zeros(
        TRAJECTORY_STEPS,
        order.shape[0],
        OUTPUT_SLOTS,
        device=order.device,
        dtype=torch.bool,
    )
    cursor = 0
    for stage, count in enumerate(PROMOTION_COUNTS):
        if count:
            selected = order[:, cursor : cursor + count]
            schedule[stage].scatter_(1, selected, True)
        cursor += count
    return schedule


def evaluation_promotion_schedule(batch_size):
    if INFERENCE_PROMOTION_MODE == "confidence":
        return None
    base = torch.arange(OUTPUT_SLOTS, device=device).view(1, -1)
    if INFERENCE_PROMOTION_MODE == "fixed":
        order = base.expand(batch_size, -1)
    elif INFERENCE_PROMOTION_MODE == "tail_first":
        order = base.flip(1).expand(batch_size, -1)
    else:
        offsets = torch.arange(batch_size, device=device).remainder(
            OUTPUT_SLOTS
        )
        order = (base + offsets.view(-1, 1)).remainder(OUTPUT_SLOTS)
    return promotion_schedule_from_order(order)


def promotion_stage_by_position(stage_logits, schedule):
    batch = stage_logits.shape[1]
    promoted_at = torch.full(
        (batch, OUTPUT_SLOTS),
        -1,
        device=stage_logits.device,
        dtype=torch.long,
    )
    pure = torch.zeros_like(promoted_at, dtype=torch.bool)
    legal = valid_token_mask.view(1, 1, -1)
    for stage, count in enumerate(PROMOTION_COUNTS):
        if schedule is None:
            filtered = stage_logits[stage].detach().float().masked_fill(
                ~legal, -torch.inf
            )
            top_value = filtered.max(dim=-1).values
            confidence = torch.exp(
                top_value - torch.logsumexp(filtered, dim=-1)
            )
            selected = confidence.masked_fill(pure, -torch.inf).topk(
                count, dim=-1
            ).indices
            promoted = torch.zeros_like(pure).scatter_(1, selected, True)
        else:
            promoted = schedule[stage]
        promoted_at = torch.where(
            promoted,
            torch.full_like(promoted_at, stage + 1),
            promoted_at,
        )
        pure = pure | promoted
    return promoted_at.float().mean(dim=0)


def position_loss_weights(physical_to_logical):
    if POSITION_OBJECTIVE_COORDINATES == "logical":
        accuracy = logical_accuracy_ema
    else:
        accuracy = physical_accuracy_ema
    coordinate_weights = torch.ones(
        OUTPUT_SLOTS, device=device, dtype=torch.float32
    )
    if POSITION_BALANCE_GAMMA > 0 and position_accuracy_updates > 0:
        reference = accuracy.max()
        coordinate_weights = (
            (reference + POSITION_BALANCE_EPSILON)
            / (accuracy + POSITION_BALANCE_EPSILON)
        ).pow(POSITION_BALANCE_GAMMA)
        coordinate_weights = coordinate_weights.clamp(
            1.0, POSITION_BALANCE_MAX_RATIO
        )
    if TAIL_LOSS_SLOTS:
        active = torch.arange(OUTPUT_SLOTS, device=device) >= (
            OUTPUT_SLOTS - TAIL_LOSS_SLOTS
        )
        coordinate_weights = coordinate_weights * active
    if POSITION_OBJECTIVE_COORDINATES == "logical":
        return coordinate_weights.index_select(0, physical_to_logical)
    return coordinate_weights


torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

core = GatorLinearConfig(
    vocab_size=VOCAB,
    streams=STREAMS,
    nodes=NODES,
    width=WIDTH,
    relays=RELAYS,
    degree=DEGREE,
    buses=BUSES,
    mailbox_channels=MAILBOX_CHANNELS,
    mailbox_width=MAILBOX_WIDTH,
    basis_rank=BASIS_RANK,
    address_width=ADDRESS_WIDTH,
    router_rank=ROUTER_RANK,
    conv_kernel=CONV_KERNEL,
    scan_chunk=SCAN_CHUNK,
    context_length=CONTEXT_LENGTH,
    scan_normalise=SCAN_NORMALISE,
    graph_layout=GRAPH_LAYOUT,
    graph_seed=GRAPH_SEED,
    graph_rewire=GRAPH_REWIRE,
    ingress_temperature=INGRESS_TEMPERATURE,
    private_stream_io=PRIVATE_STREAM_IO,
    private_stream_rank=PRIVATE_STREAM_RANK,
    heterogeneous_streams=HETEROGENEOUS_STREAMS,
    stream_width_min_fraction=STREAM_WIDTH_MIN_FRACTION,
    update_gate_bias=UPDATE_GATE_BIAS,
    readout_bias=READOUT_BIAS,
    tie_token_embeddings=TIE_TOKEN_EMBEDDINGS,
    hard_routing=HARD_ROUTING,
    router_occupancy=ROUTER_OCCUPANCY,
    linear_attention=LINEAR_ATTENTION,
    linear_attention_key_dim=LINEAR_ATTENTION_KEY_DIM,
    controlled_writing=CONTROLLED_WRITING,
    causal=CAUSAL,
    block_causal=BLOCK_CAUSAL,
)
model = GatorLinearPrefixModel(
    GatorLinearPrefixConfig(core=core)
).to(device)

refiner_mode = "full_pass" if EXPERIMENT_MODE == "full_pass" else "relay"
feedback_scale = FEEDBACK_SCALE
refiner = EvoTokenRefiner(
    model,
    mask_token_id=tokenizer.MASK_ID,
    valid_token_mask=valid_token_mask,
    steps=TRAJECTORY_STEPS,
    top_k=TOP_K,
    mode=refiner_mode,
    feedback_scale=feedback_scale,
).to(device)

n_params = sum(parameter.numel() for parameter in model.parameters())
print(
    f"model params={n_params:,} ({n_params/1e3:.1f}k) "
    f"params_per_slot={n_params/OUTPUT_SLOTS:,.1f} "
    f"slots={OUTPUT_SLOTS} promotions={PROMOTION_COUNTS} "
    f"train_window={TRAIN_WINDOW} context_length={CONTEXT_LENGTH} "
    f"mode={EXPERIMENT_MODE} feedback_scale={feedback_scale} "
    f"tied_embeddings={core.tie_token_embeddings} "
    f"causal={core.causal} block_causal={core.block_causal}",
    flush=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=OPTIMIZER_BETAS,
    eps=OPTIMIZER_EPSILON,
)
completed = 0
best_refined = 0.0
best_one_pass = 0.0
collapse_streak = 0
physical_accuracy_ema = torch.zeros(
    OUTPUT_SLOTS, device=device, dtype=torch.float32
)
logical_accuracy_ema = torch.zeros_like(physical_accuracy_ema)
position_accuracy_updates = 0
print(
    f"objective=cts_masked_only lr={LEARNING_RATE:.2e} "
    f"visible_weight={VISIBLE_TOKEN_LOSS_WEIGHT:.2f} "
    f"slot_permutation={SLOT_PERMUTATION_MODE} "
    f"train_promotions={PROMOTION_SCHEDULE_MODE} "
    f"infer_promotions={INFERENCE_PROMOTION_MODE} "
    f"position_coordinates={POSITION_OBJECTIVE_COORDINATES} "
    f"tail_loss_slots={TAIL_LOSS_SLOTS} "
    f"balance_gamma={POSITION_BALANCE_GAMMA:.3f} "
    f"worst_weight={WORST_POSITION_WEIGHT:.3f} "
    f"amp={AMP_DTYPE_NAME} compile={COMPILE}",
    flush=True,
)


def restore_checkpoint():
    global completed, best_refined, best_one_pass, collapse_streak
    global position_accuracy_updates
    local_exists = LOCAL_CHECKPOINT.exists()
    drive_exists = DRIVE_CHECKPOINT.exists()
    if RESUME_SOURCE == "none":
        local_exists = False
        drive_exists = False
    elif RESUME_SOURCE == "local":
        drive_exists = False
    elif RESUME_SOURCE == "drive":
        local_exists = False
    print(
        f"checkpoint=scan mode={EXPERIMENT_MODE} "
        f"local={local_exists} drive={drive_exists}",
        flush=True,
    )
    if local_exists:
        checkpoint = LOCAL_CHECKPOINT
        source = "local"
    elif drive_exists:
        checkpoint = DRIVE_CHECKPOINT
        source = "drive"
    else:
        if INITIAL_MODEL_CHECKPOINT is None:
            print(
                "checkpoint=resume_none starting_step=0",
                flush=True,
            )
            return
        if not INITIAL_MODEL_CHECKPOINT.is_file():
            raise FileNotFoundError(
                "initial_model_checkpoint does not exist: "
                f"{INITIAL_MODEL_CHECKPOINT}"
            )
        print(
            "checkpoint=warmstart_found "
            f"path={INITIAL_MODEL_CHECKPOINT}",
            flush=True,
        )
        source_state = torch.load(
            INITIAL_MODEL_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
        if source_state["tokenizer"] != tokenizer.to_dict():
            raise ValueError(
                "warm-start checkpoint tokenizer does not match rebuilt tokenizer"
            )
        source_config = source_state.get("run_config")
        if not isinstance(source_config, dict):
            raise ValueError(
                "warm-start checkpoint lacks run_config; model architecture "
                "cannot be validated safely"
            )
        architecture_mismatches = {
            name: (source_config.get(name, "<missing>"), CFG[name])
            for name in WARMSTART_MODEL_FIELDS
            if source_config.get(name, "<missing>") != CFG[name]
        }
        if architecture_mismatches:
            raise ValueError(
                "warm-start checkpoint model architecture mismatch: "
                f"{architecture_mismatches}"
            )
        model.load_state_dict(source_state["model"])
        print(
            "checkpoint=warmstart_loaded "
            f"source_step={int(source_state.get('completed', 0))} "
            "starting_step=0 optimizer=fresh ema=fresh",
            flush=True,
        )
        return
    print(
        f"checkpoint=resume_found source={source} path={checkpoint}",
        flush=True,
    )
    state = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    if state["experiment_version"] != EXPERIMENT_VERSION:
        raise ValueError(
            "checkpoint experiment version mismatch: "
            f"{state['experiment_version']!r}"
        )
    if state["experiment_mode"] != EXPERIMENT_MODE:
        raise ValueError(
            f"checkpoint mode={state['experiment_mode']!r}, "
            f"worker mode={EXPERIMENT_MODE!r}"
        )
    checkpoint_tied = bool(
        state.get("run_config", {}).get("tie_token_embeddings", False)
    )
    if checkpoint_tied != TIE_TOKEN_EMBEDDINGS:
        raise ValueError(
            "checkpoint tie_token_embeddings mismatch: "
            f"checkpoint={checkpoint_tied}, worker={TIE_TOKEN_EMBEDDINGS}"
        )
    checkpoint_block_causal = bool(
        state.get("run_config", {}).get("block_causal", False)
    )
    if checkpoint_block_causal != BLOCK_CAUSAL:
        raise ValueError(
            "checkpoint block_causal mismatch: "
            f"checkpoint={checkpoint_block_causal}, worker={BLOCK_CAUSAL}"
        )
    checkpoint_ablation_signature = state.get(
        "position_ablation_signature",
        LEGACY_POSITION_ABLATION_SIGNATURE,
    )
    if checkpoint_ablation_signature != POSITION_ABLATION_SIGNATURE:
        raise ValueError(
            "checkpoint position-ablation signature mismatch: "
            f"checkpoint={checkpoint_ablation_signature}, "
            f"worker={POSITION_ABLATION_SIGNATURE}"
        )
    if state["tokenizer"] != tokenizer.to_dict():
        raise ValueError("checkpoint tokenizer does not match rebuilt tokenizer")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    completed = int(state["completed"])
    best_refined = float(state.get("best_refined", 0.0))
    best_one_pass = float(state.get("best_one_pass", 0.0))
    collapse_streak = int(state.get("collapse_streak", 0))
    position_state = state.get("position_objective_state")
    if position_state is None:
        if POSITION_BALANCE_GAMMA > 0 and completed > 0:
            raise ValueError(
                "position-balanced resume requires checkpoint EMA state"
            )
    else:
        physical = torch.as_tensor(
            position_state["physical_accuracy_ema"],
            device=device,
            dtype=torch.float32,
        )
        logical = torch.as_tensor(
            position_state["logical_accuracy_ema"],
            device=device,
            dtype=torch.float32,
        )
        if physical.shape != (OUTPUT_SLOTS,) or logical.shape != (OUTPUT_SLOTS,):
            raise ValueError("checkpoint position-accuracy EMA has wrong shape")
        physical_accuracy_ema.copy_(physical)
        logical_accuracy_ema.copy_(logical)
        position_accuracy_updates = int(position_state["updates"])
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all(
        [rng.cpu() for rng in state["cuda_rng"]]
    )
    random.setstate(state["python_rng"])
    print(
        f"resumed completed={completed} best_refined={best_refined:.4f} "
        f"best_one_pass={best_one_pass:.4f} "
        f"collapse_streak={collapse_streak}",
        flush=True,
    )


def save_checkpoint():
    state = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_mode": EXPERIMENT_MODE,
        "completed": completed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_refined": best_refined,
        "best_one_pass": best_one_pass,
        "collapse_streak": collapse_streak,
        "position_ablation_signature": POSITION_ABLATION_SIGNATURE,
        "position_objective_state": {
            "physical_accuracy_ema": physical_accuracy_ema.detach().cpu(),
            "logical_accuracy_ema": logical_accuracy_ema.detach().cpu(),
            "updates": position_accuracy_updates,
        },
        "tokenizer": tokenizer.to_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "run_config": CFG,
        "config": {
            "objective": "cts_masked_only",
            "visible_token_loss_weight": VISIBLE_TOKEN_LOSS_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "lr_decay_steps": LR_DECAY_STEPS,
            "collapse_modal_share_threshold": COLLAPSE_MODAL_SHARE_THRESHOLD,
            "collapse_max_unigram_lift": COLLAPSE_MAX_UNIGRAM_LIFT,
            "trajectory_steps": TRAJECTORY_STEPS,
            "promotion_counts": PROMOTION_COUNTS,
            "top_k": TOP_K,
            "feedback_scale": feedback_scale,
            "output_slots": OUTPUT_SLOTS,
        },
    }
    temporary = LOCAL_CHECKPOINT.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(LOCAL_CHECKPOINT)
    print(
        f"checkpoint=local step={completed} path={LOCAL_CHECKPOINT}",
        flush=True,
    )


restore_checkpoint()

train_refiner = refiner
if COMPILE:
    print("Compiling refinement graph...", flush=True)
    compile_kwargs = {"fullgraph": COMPILE_FULLGRAPH}
    if COMPILE_MODE is not None:
        compile_kwargs["mode"] = COMPILE_MODE
    if COMPILE_BACKEND is not None:
        compile_kwargs["backend"] = COMPILE_BACKEND
    train_refiner = torch.compile(refiner, **compile_kwargs)


def decode_sample(token_ids):
    return tokenizer.decode(
        token_ids.detach().cpu().tolist(),
        show_roles=False,
        diagnostic=False,
    )


def evaluate():
    global best_refined, best_one_pass
    generator = torch.Generator(device=device).manual_seed(EVAL_SEED)
    observed, targets = make_batch(
        valid_ids, EVAL_BATCH_SIZE, generator=generator
    )
    initial_ids = torch.full(
        (EVAL_BATCH_SIZE, OUTPUT_SLOTS),
        tokenizer.MASK_ID,
        device=device,
        dtype=torch.long,
    )
    initial_pure = torch.zeros_like(initial_ids, dtype=torch.bool)
    alpha = torch.tensor(INFERENCE_ALPHA, device=device)
    evaluation_schedule = evaluation_promotion_schedule(EVAL_BATCH_SIZE)
    evaluation_position_ids = (
        torch.arange(OUTPUT_SLOTS, device=device)
        if SLOT_PERMUTATION_MODE != "none"
        else None
    )
    refinement_promotion_args = (
        {"inference_promotion_counts": PROMOTION_COUNTS}
        if evaluation_schedule is None
        else {"promotion_schedule": evaluation_schedule}
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=AMP_DTYPE,
        enabled=AUTOCAST_ENABLED,
    ):
        stage_logits, decoded, retained = train_refiner(
            observed,
            initial_ids,
            initial_pure,
            alpha=alpha,
            output_position_ids=evaluation_position_ids,
            temperature=TEMPERATURE,
            **refinement_promotion_args,
        )
        mask_embeddings = F.embedding(
            initial_ids, model.token_embedding.weight
        )
        one_pass_logits = model(
            observed,
            output_slots=OUTPUT_SLOTS,
            output_embeddings=mask_embeddings,
            output_position_ids=evaluation_position_ids,
            temperature=TEMPERATURE,
        )

    correct = decoded.eq(targets)
    refined_accuracy = float(correct.float().mean())
    exact = float(correct.all(dim=1).float().mean())
    by_position = correct.float().mean(dim=0).tolist()
    one_pass = legal_argmax(one_pass_logits)
    one_pass_accuracy = float(one_pass.eq(targets).float().mean())
    best_refined = max(best_refined, refined_accuracy)
    best_one_pass = max(best_one_pass, one_pass_accuracy)

    stage_predictions = legal_argmax(stage_logits)
    stage_top3 = stage_logits.float().masked_fill(
        ~valid_token_mask.view(1, 1, 1, -1), -torch.inf
    ).topk(TOP_K, dim=-1).indices
    one_pass_top3 = one_pass_logits.float().masked_fill(
        ~valid_token_mask.view(1, 1, -1), -torch.inf
    ).topk(TOP_K, dim=-1).indices
    stage_top3_hits = stage_top3.eq(
        targets.unsqueeze(0).unsqueeze(-1)
    ).any(dim=-1).float().mean(dim=(1, 2)).tolist()
    one_pass_top3_hit = float(
        one_pass_top3.eq(targets.unsqueeze(-1)).any(dim=-1).float().mean()
    )
    sample_index = 0
    sample = {
        "one_pass": decode_sample(one_pass[sample_index]),
        "stages": [
            decode_sample(stage[sample_index])
            for stage in stage_predictions
        ],
        "refined": decode_sample(decoded[sample_index]),
        "stage_matches": stage_predictions[:, sample_index]
        .eq(targets[sample_index])
        .sum(dim=-1)
        .tolist(),
        "refined_matches": int(correct[sample_index].sum()),
    }
    stage_accuracies = (
        stage_predictions.eq(targets.unsqueeze(0))
        .float()
        .mean(dim=(1, 2))
        .tolist()
    )
    if stage_predictions.shape[0] > 1:
        revisions = stage_predictions[1:].ne(stage_predictions[:-1])
        revision_rate = float(revisions.float().mean())
        corrections = (
            revisions
            & stage_predictions[:-1].ne(targets.unsqueeze(0))
            & stage_predictions[1:].eq(targets.unsqueeze(0))
        )
        correction_rate = float(
            corrections.sum().float() / revisions.sum().clamp_min(1)
        )
    else:
        revision_rate = 0.0
        correction_rate = 0.0
    modal_counts = torch.bincount(decoded.reshape(-1), minlength=VOCAB)
    modal_token_id = int(modal_counts.argmax())
    modal_share = float(modal_counts[modal_token_id] / decoded.numel())
    modal_target_by_position = (
        targets.eq(modal_token_id).float().mean(dim=0).tolist()
    )
    modal_target_share = float(targets.eq(modal_token_id).float().mean())
    modal_lift = refined_accuracy - modal_target_share
    weak_positions = sum(
        accuracy <= baseline + COLLAPSE_POSITION_MARGIN
        for accuracy, baseline in zip(by_position, modal_target_by_position)
    )
    modal_token = decode_sample(
        targets.new_tensor([modal_token_id], dtype=torch.long)
    )
    final_probabilities = stage_logits[-1].float().softmax(-1)
    invalid_mass = float(
        final_probabilities[..., ~valid_token_mask].sum(-1).mean()
    )
    promotion_stages = promotion_stage_by_position(
        stage_logits, evaluation_schedule
    ).tolist()
    return {
        "refined_accuracy": refined_accuracy,
        "one_pass_accuracy": one_pass_accuracy,
        "exact": exact,
        "by_position": by_position,
        "stage_accuracies": stage_accuracies,
        "stage_top3_hits": stage_top3_hits,
        "one_pass_top3_hit": one_pass_top3_hit,
        "revision_rate": revision_rate,
        "correction_rate": correction_rate,
        "modal_share": modal_share,
        "modal_token": modal_token,
        "modal_target_share": modal_target_share,
        "modal_lift": modal_lift,
        "weak_positions": weak_positions,
        "retained_mass": float(retained.mean()),
        "invalid_mass": invalid_mass,
        "promotion_stages": promotion_stages,
        "sample": sample,
    }


def scheduled_learning_rate():
    progress = min(1.0, completed / LR_DECAY_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + (
        LEARNING_RATE - MIN_LEARNING_RATE
    ) * cosine


def parameter_grad_norm(parameters):
    total = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return float(total.sqrt())


def gradient_group_norms():
    relay_parameters = {
        id(parameter)
        for relay in model.core.relays
        for parameter in relay.parameters()
    }
    named = list(model.named_parameters())
    groups = {
        "shell": [
            parameter
            for _, parameter in named
            if id(parameter) not in relay_parameters
        ],
        "attention": [
            parameter
            for name, parameter in named
            if ".attn_" in name
        ],
        "routing": [
            parameter
            for name, parameter in named
            if any(
                marker in name
                for marker in (
                    ".router_",
                    ".edge_",
                    ".bus_",
                    "ingress",
                )
            )
        ],
    }
    for index, relay in enumerate(model.core.relays, start=1):
        groups[f"relay{index}"] = list(relay.parameters())
    return {
        name: parameter_grad_norm(parameters)
        for name, parameters in groups.items()
    }


def diagnostic_position_indices():
    count = min(OUTPUT_SLOTS, POSITION_GRADIENT_MAX_SLOTS)
    if count == OUTPUT_SLOTS:
        return tuple(range(OUTPUT_SLOTS))
    if count == 1:
        return (OUTPUT_SLOTS - 1,)
    return tuple(
        round(index * (OUTPUT_SLOTS - 1) / (count - 1))
        for index in range(count)
    )


relay_parameter_ids = {
    id(parameter)
    for relay in model.core.relays
    for parameter in relay.parameters()
}
named_trainable_parameters = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]
POSITION_GRADIENT_GROUPS = [
    (
        "shared",
        tuple(
            parameter
            for _, parameter in named_trainable_parameters
            if id(parameter) not in relay_parameter_ids
        ),
    )
]
POSITION_GRADIENT_GROUPS.extend(
    (f"relay{index}", tuple(relay.parameters()))
    for index, relay in enumerate(model.core.relays, start=1)
)
POSITION_GRADIENT_POSITIONS = diagnostic_position_indices()


def position_gradient_diagnostics(physical_losses, logical_to_physical):
    ordered_losses = (
        physical_losses.index_select(0, logical_to_physical)
        if POSITION_GRADIENT_COORDINATES == "logical"
        else physical_losses
    )
    selected = ordered_losses[list(POSITION_GRADIENT_POSITIONS)]
    norms, cos_prefix = position_gradient_statistics(
        selected,
        tuple(parameters for _, parameters in POSITION_GRADIENT_GROUPS),
    )
    return {
        "axis": POSITION_GRADIENT_COORDINATES,
        "positions": POSITION_GRADIENT_POSITIONS,
        "losses": selected.detach().float().cpu().tolist(),
        "norms": norms.cpu().tolist(),
        "cos_prefix": cos_prefix.cpu().tolist(),
    }


def print_position_gradient_diagnostics(diagnostics):
    positions = ",".join(str(value) for value in diagnostics["positions"])
    losses = ",".join(f"{value:.4f}" for value in diagnostics["losses"])
    for group_index, (group_name, parameters) in enumerate(
        POSITION_GRADIENT_GROUPS
    ):
        norms = diagnostics["norms"][group_index]
        mean_norm = sum(norms) / max(len(norms), 1)
        relative = [
            value / max(mean_norm, 1e-30) for value in norms
        ]
        norm_text = ",".join(f"{value:.3e}" for value in norms)
        relative_text = ",".join(f"{value:.2f}" for value in relative)
        cosine_text = ",".join(
            "nan" if math.isnan(value) else f"{value:+.3f}"
            for value in diagnostics["cos_prefix"][group_index]
        )
        parameter_count = sum(parameter.numel() for parameter in parameters)
        print(
            f"pos_grad step={completed} axis={diagnostics['axis']} "
            f"group={group_name} params={parameter_count} "
            f"positions=[{positions}] loss=[{losses}] "
            f"norm=[{norm_text}] relative=[{relative_text}] "
            f"cos_prefix=[{cosine_text}]",
            flush=True,
        )


stop_requested = False
stop_signal = None


def request_stop(signum, _frame):
    global stop_requested, stop_signal
    stop_requested = True
    stop_signal = signum
    print(
        f"checkpoint=stop_requested signal={signum} step={completed}",
        flush=True,
    )


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

started = time.time()
last_train = {}
last_grad_groups = {}
last_position_gradient = None
while completed < TOTAL_STEPS and not stop_requested:
    current_lr = scheduled_learning_rate()
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = current_lr
    phase = "cts_masked"

    observed, logical_targets = make_batch(train_ids, BATCH_SIZE)
    (
        output_position_ids,
        physical_to_logical,
        logical_to_physical,
    ) = sample_slot_mapping()
    targets = logical_targets.index_select(1, physical_to_logical)
    physical_position_weights = position_loss_weights(
        physical_to_logical
    )
    promotion_order = training_promotion_order(BATCH_SIZE)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=AMP_DTYPE,
        enabled=AUTOCAST_ENABLED,
    ):
        (
            initial_ids,
            initial_pure,
            promotions,
            promotion_counts,
        ) = sample_training_schedule(
            targets,
            mask_token_id=tokenizer.MASK_ID,
            steps=TRAJECTORY_STEPS,
            promotion_counts=PROMOTION_COUNTS,
            slot_order=promotion_order,
        )
        alpha = torch.empty(
            TRAJECTORY_STEPS,
            BATCH_SIZE,
            device=device,
        ).uniform_(TRAIN_ALPHA_MIN, TRAIN_ALPHA_MAX)
        stage_logits, decoded_train, retained = train_refiner(
            observed,
            initial_ids,
            initial_pure,
            alpha=alpha,
            promotion_schedule=promotions,
            output_position_ids=output_position_ids,
            temperature=TEMPERATURE,
        )
        loss, token_loss = masked_trajectory_cross_entropy(
            stage_logits,
            targets,
            initial_pure,
            visible_weight=VISIBLE_TOKEN_LOSS_WEIGHT,
            position_weights=(
                physical_position_weights
                if POSITION_BALANCE_GAMMA > 0 or TAIL_LOSS_SLOTS > 0
                else None
            ),
            worst_position_weight=WORST_POSITION_WEIGHT,
            worst_position_temperature=WORST_POSITION_TEMPERATURE,
        )
        initially_masked = (~initial_pure).unsqueeze(0).expand_as(
            token_loss
        )
        initially_visible = initial_pure.unsqueeze(0).expand_as(
            token_loss
        )
        masked_float = initially_masked.to(token_loss.dtype)
        stage_masked_ce = (
            (token_loss * masked_float).sum(dim=(1, 2))
            / masked_float.sum(dim=(1, 2)).clamp_min(1.0)
        )
        position_masked_ce = (
            (token_loss * masked_float).sum(dim=(0, 1))
            / masked_float.sum(dim=(0, 1)).clamp_min(1.0)
        )
        logical_position_ce = position_masked_ce.index_select(
            0, logical_to_physical
        )
        logical_position_weights = physical_position_weights.index_select(
            0, logical_to_physical
        )
        promotion_stage_numbers = torch.arange(
            1,
            TRAJECTORY_STEPS + 1,
            device=device,
            dtype=torch.float32,
        ).view(-1, 1, 1)
        physical_promotion_stages = (
            promotions.float() * promotion_stage_numbers
        ).sum(dim=0).mean(dim=0)
        logical_promotion_stages = physical_promotion_stages.index_select(
            0, logical_to_physical
        )
        visible_ce = (
            float(token_loss[initially_visible].mean().detach())
            if bool(initially_visible.any())
            else float("nan")
        )
        last_train = {
            "masked_ce": float(
                token_loss[initially_masked].mean().detach()
            ),
            "visible_ce": visible_ce,
            "promotion_counts": promotion_counts,
            "retained_mass": float(retained.mean().detach()),
            "stage_ce": [
                float(value) for value in stage_masked_ce.detach()
            ],
            "physical_position_ce": position_masked_ce.detach().tolist(),
            "logical_position_ce": logical_position_ce.detach().tolist(),
            "physical_position_weights": (
                physical_position_weights.detach().tolist()
            ),
            "logical_position_weights": (
                logical_position_weights.detach().tolist()
            ),
            "physical_to_logical": physical_to_logical.detach().tolist(),
            "physical_promotion_stages": (
                physical_promotion_stages.detach().tolist()
            ),
            "logical_promotion_stages": (
                logical_promotion_stages.detach().tolist()
            ),
        }

    should_evaluate = (completed + 1) % EVAL_EVERY == 0
    evaluation_index = (completed + 1) // EVAL_EVERY
    should_measure_position_gradients = (
        should_evaluate
        and POSITION_GRADIENT_EVERY_EVALS > 0
        and evaluation_index % POSITION_GRADIENT_EVERY_EVALS == 0
    )
    if should_measure_position_gradients:
        last_position_gradient = position_gradient_diagnostics(
            position_masked_ce,
            logical_to_physical,
        )
    else:
        last_position_gradient = None
    loss.backward()
    if should_evaluate:
        last_grad_groups = gradient_group_norms()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
    )
    clip_scale = min(
        1.0,
        GRADIENT_CLIP / max(float(gradient_norm), 1e-12),
    )
    optimizer.step()
    with torch.no_grad():
        physical_correct = decoded_train.eq(targets).float()
        logical_correct = physical_correct.index_select(
            1, logical_to_physical
        )
        physical_batch_accuracy = physical_correct.mean(dim=0)
        logical_batch_accuracy = logical_correct.mean(dim=0)
        if position_accuracy_updates == 0:
            physical_accuracy_ema.copy_(physical_batch_accuracy)
            logical_accuracy_ema.copy_(logical_batch_accuracy)
        else:
            update_weight = 1.0 - POSITION_BALANCE_DECAY
            physical_accuracy_ema.lerp_(
                physical_batch_accuracy, update_weight
            )
            logical_accuracy_ema.lerp_(
                logical_batch_accuracy, update_weight
            )
        position_accuracy_updates += 1
    completed += 1

    if should_evaluate:
        metrics = evaluate()
        collapse_candidate = (
            completed >= COLLAPSE_WARMUP_STEPS
            and metrics["modal_share"] >= COLLAPSE_MODAL_SHARE_THRESHOLD
            and metrics["modal_lift"] <= COLLAPSE_MAX_UNIGRAM_LIFT
            and metrics["weak_positions"] >= COLLAPSE_MIN_WEAK_POSITIONS
        )
        collapse_streak = collapse_streak + 1 if collapse_candidate else 0
        elapsed = time.time() - started
        stage_ce = ",".join(
            f"{value:.3f}" for value in last_train["stage_ce"]
        )
        stage_acc = ",".join(
            f"{value:.3f}" for value in metrics["stage_accuracies"]
        )
        stage_top3 = ",".join(
            f"{value:.3f}" for value in metrics["stage_top3_hits"]
        )
        grad_groups = ",".join(
            f"{name}:{value:.1f}"
            for name, value in last_grad_groups.items()
        )
        by_position = " ".join(
            f"{value:.2f}" for value in metrics["by_position"]
        )
        print(
            f"step={completed:7d}/{TOTAL_STEPS} phase={phase} "
            f"t={elapsed/60:.1f}m loss={float(loss.detach()):.4f} "
            f"lr={current_lr:.2e} visible_w={VISIBLE_TOKEN_LOSS_WEIGHT:.2f} "
            f"masked_ce={last_train['masked_ce']:.4f} "
            f"visible_ce={last_train['visible_ce']:.4f} "
            f"grad={float(gradient_norm):.3f} clip_scale={clip_scale:.3e} "
            f"grad_groups=[{grad_groups}] "
            f"refined_acc={metrics['refined_accuracy']:.4f} "
            f"onepass_acc={metrics['one_pass_accuracy']:.4f} "
            f"onepass_top3={metrics['one_pass_top3_hit']:.4f} "
            f"best={best_refined:.4f} exact={metrics['exact']:.4f} "
            f"modal={metrics['modal_share']:.3f} "
            f"modal_token={metrics['modal_token']!r} "
            f"modal_base={metrics['modal_target_share']:.3f} "
            f"modal_lift={metrics['modal_lift']:+.3f} "
            f"weak_pos={metrics['weak_positions']}/{OUTPUT_SLOTS} "
            f"collapse_streak={collapse_streak} "
            f"revise={metrics['revision_rate']:.3f} "
            f"correct_rev={metrics['correction_rate']:.3f} "
            f"retained={metrics['retained_mass']:.3f} "
            f"invalid_mass={metrics['invalid_mass']:.3f} "
            f"promotions={last_train['promotion_counts']} "
            f"stage_ce=[{stage_ce}] stage_acc=[{stage_acc}] "
            f"stage_top3=[{stage_top3}] by_pos=[{by_position}]",
            flush=True,
        )
        logical_train_ce = " ".join(
            f"{value:.3f}" for value in last_train["logical_position_ce"]
        )
        physical_train_ce = " ".join(
            f"{value:.3f}" for value in last_train["physical_position_ce"]
        )
        logical_weights = " ".join(
            f"{value:.2f}"
            for value in last_train["logical_position_weights"]
        )
        physical_weights = " ".join(
            f"{value:.2f}"
            for value in last_train["physical_position_weights"]
        )
        logical_ema = " ".join(
            f"{value:.3f}" for value in logical_accuracy_ema.tolist()
        )
        physical_ema = " ".join(
            f"{value:.3f}" for value in physical_accuracy_ema.tolist()
        )
        promotion_stages = " ".join(
            f"{value:.2f}" for value in metrics["promotion_stages"]
        )
        train_promotions_logical = " ".join(
            f"{value:.2f}"
            for value in last_train["logical_promotion_stages"]
        )
        train_promotions_physical = " ".join(
            f"{value:.2f}"
            for value in last_train["physical_promotion_stages"]
        )
        print(
            f"position=map_physical_to_logical="
            f"{last_train['physical_to_logical']} "
            f"train_ce_logical=[{logical_train_ce}] "
            f"train_ce_physical=[{physical_train_ce}] "
            f"weights_logical=[{logical_weights}] "
            f"weights_physical=[{physical_weights}] "
            f"ema_logical=[{logical_ema}] "
            f"ema_physical=[{physical_ema}] "
            f"train_promote_logical=[{train_promotions_logical}] "
            f"train_promote_physical=[{train_promotions_physical}] "
            f"infer_promotion={INFERENCE_PROMOTION_MODE} "
            f"infer_promote_stage=[{promotion_stages}]",
            flush=True,
        )
        if last_position_gradient is not None:
            print_position_gradient_diagnostics(last_position_gradient)
        sample = metrics["sample"]
        print(
            f"sample=one_pass={sample['one_pass']!r} "
            f"stages={sample['stages']!r} "
            f"refined={sample['refined']!r} "
            f"stage_matches={sample['stage_matches']!r}/{OUTPUT_SLOTS} "
            f"refined_matches={sample['refined_matches']}/{OUTPUT_SLOTS}",
            flush=True,
        )
        if collapse_candidate and (
            collapse_streak == COLLAPSE_PATIENCE_EVALS
            or collapse_streak > COLLAPSE_PATIENCE_EVALS
            and collapse_streak % 10 == 0
        ):
            print(
                "!!! COLLAPSE WARNING "
                f"streak={collapse_streak} "
                f"token={metrics['modal_token']!r} "
                f"prediction_share={metrics['modal_share']:.3f} "
                f"target_baseline={metrics['modal_target_share']:.3f} "
                f"accuracy_lift={metrics['modal_lift']:+.3f} "
                f"weak_positions={metrics['weak_positions']}/{OUTPUT_SLOTS}",
                flush=True,
            )


    if completed % CHECKPOINT_EVERY == 0:
        save_checkpoint()

if CHECKPOINT_ON_EXIT:
    save_checkpoint()
status = "STOPPED" if stop_requested else "COMPLETE"
print(
    f"=> {status} mode={EXPERIMENT_MODE} steps={completed} "
    f"signal={stop_signal} "
    f"best_refined={best_refined:.4f} "
    f"best_one_pass={best_one_pass:.4f}",
    flush=True,
)

"""


def run_worker(config: RunConfig) -> None:
    os.environ[WORKER_CONFIG_ENV] = str(config.local_config)
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(config.code_dir) / "gtrl_evotoken_colab.py"),
    }
    exec(
        compile(WORKER_SOURCE, namespace["__file__"], "exec"),
        namespace,
        namespace,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config, args = parse_config(argv)
    if args.worker_config is not None:
        run_worker(config)
        return
    if args.print_config or args.dry_run:
        print(
            json.dumps(config_dict(config), indent=2, sort_keys=True),
            flush=True,
        )
    if args.dry_run:
        return
    launch(config)


if __name__ == "__main__":
    main()
