"""Run the frozen 100M node-heavy Gator with low-write Drive mirroring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from gator_convergent import ConvergentGator
from run_gator_140m_colab import (
    DEFAULT_DRIVE_ROOT,
    DEFAULT_LOCAL_ROOT,
    DriveMirror,
    build_command as build_140m_command,
)


EXPECTED_PARAMETERS = 99_979_407
VOCAB_SIZE = 32_768
EMBEDDING_DIM = 256
D_MODEL = 256
N_NODES = 1_337
LANES = 10
DEGREE = 3
RELAYS = 2
ROUTER_RANK = 96
ACTIVE_NODE_TOP_K = 32


def preflight_parameter_count() -> int:
    """Return the exact frozen node-heavy architecture count."""

    actual = ConvergentGator.shared_language_parameter_count_for(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        n_nodes=N_NODES,
        d_model=D_MODEL,
        lanes=LANES,
        degree=DEGREE,
        router_mode="shared_low_rank",
        router_rank=ROUTER_RANK,
        ingress_mode="learned",
        terminal_vote_mode="prompt",
    )
    if actual != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "Gator-100M architecture drift: "
            f"expected {EXPECTED_PARAMETERS:,}, found {actual:,}"
        )
    return actual


def _replace_cli_value(command: list[str], flag: str, value: object) -> None:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise RuntimeError(f"base command is missing {flag}") from error
    command[index + 1] = str(value)


def build_command(
    *,
    steps: int,
    batch_size: int,
    eval_examples: int,
    log_every: int,
    checkpoint_dir: Path,
    compile_model: bool,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the frozen dense-matrix 100M mixed-corpus command."""

    command = build_140m_command(
        steps=steps,
        batch_size=batch_size,
        eval_examples=eval_examples,
        log_every=log_every,
        checkpoint_dir=checkpoint_dir,
        compile_model=compile_model,
        route_backend="dense",
        packed_memory_backend="matrix",
        packed_memory_feature_chunk=32,
        python_executable=python_executable,
    )
    _replace_cli_value(command, "--embedding-dim", EMBEDDING_DIM)
    _replace_cli_value(command, "--d-model", D_MODEL)
    _replace_cli_value(command, "--nodes", N_NODES)
    return command


def write_run_spec(
    path: Path,
    *,
    run_name: str,
    command: list[str],
    args: argparse.Namespace,
) -> None:
    path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": EXPECTED_PARAMETERS,
                "architecture": {
                    "vocab_size": VOCAB_SIZE,
                    "embedding_dim": EMBEDDING_DIM,
                    "d_model": D_MODEL,
                    "nodes": N_NODES,
                    "lanes": LANES,
                    "degree": DEGREE,
                    "relays": RELAYS,
                    "router_mode": "shared_low_rank",
                    "router_rank": ROUTER_RANK,
                    "ingress_mode": "learned",
                    "active_node_top_k": ACTIVE_NODE_TOP_K,
                    "route_backend": "dense",
                    "packed_memory_backend": "matrix",
                    "terminal_vote_mode": "prompt",
                },
                "runtime": {
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "eval_examples": args.eval_examples,
                    "log_every": args.log_every,
                    "compile": args.compile,
                    "drive_sync_seconds": args.drive_sync_seconds,
                    "drive_log_updates": args.drive_log_updates,
                },
                "command": command,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--drive-sync-seconds", type=float, default=900.0)
    parser.add_argument("--drive-log-updates", type=int, default=20)
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive")
    if args.eval_examples <= 0 or args.log_every <= 0:
        parser.error("--eval-examples and --log-every must be positive")
    if args.drive_sync_seconds <= 0 or args.drive_log_updates <= 0:
        parser.error("Drive sync controls must be positive")
    return args


def main() -> None:
    args = parse_args()
    parameter_count = preflight_parameter_count()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        "gator100m_d256_n1337_dense_matrix_seed0_" + timestamp
    )
    local_dir = args.local_root / run_name
    drive_dir = args.drive_root / run_name
    if local_dir.exists() or drive_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_name}")
    local_dir.mkdir(parents=True)
    drive_dir.mkdir(parents=True)

    command = build_command(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_examples=args.eval_examples,
        log_every=args.log_every,
        checkpoint_dir=local_dir,
        compile_model=args.compile,
    )
    write_run_spec(
        local_dir / "RUN_SPEC.json",
        run_name=run_name,
        command=command,
        args=args,
    )
    mirror = DriveMirror(
        local_dir=local_dir,
        drive_dir=drive_dir,
        sync_seconds=args.drive_sync_seconds,
        log_updates_per_sync=args.drive_log_updates,
    )
    mirror.sync_small_metadata()

    print(
        f"run={run_name} parameters={parameter_count} "
        f"batch={args.batch_size} steps={args.steps} compile={args.compile} "
        f"width={D_MODEL} nodes={N_NODES} route=dense memory=matrix",
        flush=True,
    )
    print(f"local_checkpoint_dir={local_dir}", flush=True)
    print(f"drive_mirror_dir={drive_dir}", flush=True)
    print(
        f"drive_log_sync=every_{args.drive_log_updates}_reports_or_"
        f"{args.drive_sync_seconds:.0f}s "
        f"drive_latest_sync={args.drive_sync_seconds:.0f}s",
        flush=True,
    )
    print("command=" + " ".join(command), flush=True)

    return_code = -1
    try:
        with mirror.local_log.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as output:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                mirror.observe(line)
            return_code = process.wait()
    finally:
        mirror.finalize()

    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"complete drive_mirror_dir={drive_dir}", flush=True)


if __name__ == "__main__":
    main()
