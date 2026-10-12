"""Run the frozen 140M shared-routing Gator with low-write Drive mirroring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gator_convergent import ConvergentGator


EXPECTED_PARAMETERS = 140_193_699
DEFAULT_DRIVE_ROOT = Path(
    "/content/drive/MyDrive/GTR/experimental_checkpoints"
)
DEFAULT_LOCAL_ROOT = Path("/content/GTR_local_checkpoints")


def preflight_parameter_count() -> int:
    """Return the exact frozen architecture count or reject drift."""

    actual = ConvergentGator.shared_language_parameter_count_for(
        vocab_size=32_768,
        embedding_dim=768,
        n_nodes=154,
        d_model=768,
        lanes=10,
        degree=3,
        router_mode="shared_low_rank",
        router_rank=96,
        ingress_mode="learned",
        terminal_vote_mode="slot",
    )
    if actual != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "Gator-140M architecture drift: "
            f"expected {EXPECTED_PARAMETERS:,}, found {actual:,}"
        )
    return actual


def build_command(
    *,
    steps: int,
    batch_size: int,
    eval_examples: int,
    log_every: int,
    checkpoint_dir: Path,
    compile_model: bool,
    route_backend: str = "dense",
    packed_memory_backend: str = "matrix",
    packed_memory_reader_backend: str = "torch",
    packed_memory_writer_backend: str = "torch",
    packed_memory_feature_chunk: int = 32,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the frozen mixed-corpus teacher-forced sanity command."""

    command = [
        python_executable,
        "-u",
        "gator_language_train.py",
        "--data",
        "/content/drive/MyDrive/GTR/training_data/smoltalk_everyday.jsonl",
        "--tinystories-data",
        "/content/drive/MyDrive/GTR/training_data/TinyStories-train.txt",
        "--tinystories-validation-data",
        "/content/drive/MyDrive/GTR/training_data/TinyStories-valid.txt",
        "--tinystories-mix-fraction",
        "0.40",
        "--tinystories-validation-stories",
        "512",
        "--tinystories-tokenizer-stories",
        "50000",
        "--tinystories-shuffle-buffer",
        "16384",
        "--lmsys-data",
        "/content/drive/MyDrive/GTR/training_data/lmsys_chat_1m.jsonl",
        "--lmsys-source-weight",
        "4",
        "--lmsys-load-mode",
        "stream",
        "--lmsys-shuffle-buffer",
        "16384",
        "--lmsys-validation-conversations",
        "512",
        "--lmsys-max-characters",
        "8192",
        "--lmsys-min-ascii-letter-ratio",
        "0.9",
        "--lmsys-min-response-characters",
        "8",
        "--lmsys-require-user-prompt",
        "--chat-data",
        "/content/drive/MyDrive/Sonic/training_data/chat/tulu3_sft.jsonl",
        "--chat-source-weight",
        "3",
        "--chat-data",
        (
            "/content/drive/MyDrive/Sonic/training_data/chat/"
            "smoltalk_preference.jsonl"
        ),
        "--chat-source-weight",
        "2",
        "--chat-max-characters",
        "8192",
        "--chat-shuffle-buffer",
        "8192",
        "--streaming-weight",
        "1.0",
        "--streaming-exhaustion",
        "restart",
        "--primary-stream-weight",
        "1",
        "--primary-shuffle-buffer",
        "8192",
        "--require-complete-context",
        "--trim-context-turns",
        "--deduplicate-examples",
        "--skip-first-assistant",
        "--architecture",
        "convergent",
        "--vocab-size",
        "32768",
        "--embedding-dim",
        "768",
        "--d-model",
        "768",
        "--nodes",
        "154",
        "--lanes",
        "10",
        "--graph-degree",
        "3",
        "--relays",
        "2",
        "--connection-layout",
        "small_world",
        "--connection-seed",
        "0",
        "--connection-rewire",
        "0.5",
        "--router-mode",
        "shared_low_rank",
        "--router-rank",
        "96",
        "--ingress-mode",
        "learned",
        "--active-node-top-k",
        "32",
        "--route-backend",
        route_backend,
        "--packed-memory-backend",
        packed_memory_backend,
        "--packed-memory-reader-backend",
        packed_memory_reader_backend,
        "--packed-memory-writer-backend",
        packed_memory_writer_backend,
        "--packed-memory-feature-chunk",
        str(packed_memory_feature_chunk),
        "--terminal-vote-mode",
        "slot",
        "--decode-mode",
        "teacher_forced",
        "--formation-objective",
        "slot_ce",
        "--context-tokens",
        "512",
        "--response-tokens",
        "128",
        "--optimizer",
        "adamw",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--gradient-clip",
        "1.0",
        "--scheduler",
        "constant",
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        "1",
        "--eval-examples",
        str(eval_examples),
        "--log-every",
        str(log_every),
        "--logging-slim",
        "--batch-pipeline",
        "prefetch",
        "--prefetch-batches",
        "20",
        "--float32-matmul-precision",
        "high",
        "--cudnn-tf32",
        "--seed",
        "0",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if compile_model:
        command.append("--compile")
    return command


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy one stable local artifact to Drive without exposing partial bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".uploading")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class DriveMirror:
    """Batch text appends and rate-limit large checkpoint copies."""

    def __init__(
        self,
        *,
        local_dir: Path,
        drive_dir: Path,
        sync_seconds: float,
        log_updates_per_sync: int,
        clock=time.monotonic,
    ) -> None:
        if sync_seconds <= 0:
            raise ValueError("sync_seconds must be positive")
        if log_updates_per_sync <= 0:
            raise ValueError("log_updates_per_sync must be positive")
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.sync_seconds = float(sync_seconds)
        self.log_updates_per_sync = int(log_updates_per_sync)
        self.clock = clock
        self.local_log = local_dir / "train.log"
        self.drive_log = drive_dir / "train.log"
        self.pending_lines: list[str] = []
        self.pending_updates = 0
        now = self.clock()
        self.last_log_sync = now
        self.last_checkpoint_sync = now
        self.last_latest_signature: tuple[int, int] | None = None

    def observe(self, line: str) -> None:
        self.pending_lines.append(line)
        if line.startswith("step="):
            self.pending_updates += 1
        now = self.clock()
        if (
            self.pending_updates >= self.log_updates_per_sync
            or now - self.last_log_sync >= self.sync_seconds
        ):
            self.sync_log(now=now)
        if now - self.last_checkpoint_sync >= self.sync_seconds:
            self.sync_latest(now=now)

    def sync_log(self, *, now: float | None = None, force: bool = False) -> None:
        if not self.pending_lines and not force:
            return
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        if self.pending_lines:
            with self.drive_log.open(
                "a",
                encoding="utf-8",
                buffering=1,
            ) as output:
                output.write("".join(self.pending_lines))
            self.pending_lines.clear()
        self.pending_updates = 0
        self.last_log_sync = self.clock() if now is None else now

    def sync_latest(self, *, now: float | None = None) -> bool:
        source = self.local_dir / "latest.pt"
        if not source.is_file():
            self.last_checkpoint_sync = (
                self.clock() if now is None else now
            )
            return False
        stat = source.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == self.last_latest_signature:
            self.last_checkpoint_sync = (
                self.clock() if now is None else now
            )
            return False
        _atomic_copy(source, self.drive_dir / source.name)
        self.last_latest_signature = signature
        self.last_checkpoint_sync = self.clock() if now is None else now
        return True

    def sync_small_metadata(self) -> None:
        for name in ("tokenizer.json", "RUN_SPEC.json"):
            source = self.local_dir / name
            if source.is_file():
                _atomic_copy(source, self.drive_dir / name)

    def finalize(self) -> None:
        self.sync_log(force=True)
        self.sync_small_metadata()
        for source in sorted(self.local_dir.glob("*.pt")):
            _atomic_copy(source, self.drive_dir / source.name)


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
                    "vocab_size": 32_768,
                    "embedding_dim": 768,
                    "d_model": 768,
                    "nodes": 154,
                    "lanes": 10,
                    "degree": 3,
                    "relays": 2,
                    "router_mode": "shared_low_rank",
                    "router_rank": 96,
                    "ingress_mode": "learned",
                    "active_node_top_k": 32,
                    "route_backend": args.route_backend,
                    "packed_memory_backend": args.packed_memory_backend,
                    "packed_memory_reader_backend": (
                        args.packed_memory_reader_backend
                    ),
                    "packed_memory_writer_backend": (
                        args.packed_memory_writer_backend
                    ),
                    "packed_memory_feature_chunk": (
                        args.packed_memory_feature_chunk
                    ),
                    "terminal_vote_mode": "slot",
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


def default_run_name(args: argparse.Namespace, timestamp: str) -> str:
    """Name a run with every execution backend that changes its result."""

    return (
        "gator140m_teacherforced_shared_r96_"
        f"{args.route_backend}_{args.packed_memory_backend}_"
        f"reader-{args.packed_memory_reader_backend}_"
        f"writer-{args.packed_memory_writer_backend}_seed0_{timestamp}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--drive-sync-seconds", type=float, default=900.0)
    parser.add_argument("--drive-log-updates", type=int, default=20)
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--route-backend",
        choices=("dense", "packed"),
        default="dense",
    )
    parser.add_argument(
        "--packed-memory-backend",
        choices=("matrix", "convolution", "fft", "fft_opaque"),
        default="matrix",
    )
    parser.add_argument(
        "--packed-memory-reader-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--packed-memory-writer-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--packed-memory-feature-chunk",
        type=int,
        default=32,
    )
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
    if args.packed_memory_feature_chunk <= 0:
        parser.error("--packed-memory-feature-chunk must be positive")
    if (
        args.route_backend != "packed"
        and args.packed_memory_backend != "matrix"
    ):
        parser.error(
            "convolution/fft/fft_opaque memory requires --route-backend packed"
        )
    if (
        args.packed_memory_reader_backend == "triton"
        and args.packed_memory_backend == "matrix"
    ):
        parser.error(
            "the Triton packed reader requires convolution, fft, or "
            "fft_opaque memory"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and (
            args.route_backend != "packed"
            or args.packed_memory_backend != "fft_opaque"
        )
    ):
        parser.error(
            "the Triton packed writer currently requires --route-backend "
            "packed and --packed-memory-backend fft_opaque"
        )
    if (
        args.packed_memory_writer_backend == "triton"
        and args.packed_memory_feature_chunk > 128
    ):
        parser.error(
            "the Triton packed writer supports feature chunks up to 128"
        )
    if args.compile and args.packed_memory_backend == "fft":
        parser.error(
            "compiled FFT receiver memory is unsupported by the current "
            "TorchInductor complex64 backend; use fft_opaque to keep cuFFT "
            "behind a real-valued custom-op boundary, convolution with "
            "--compile, or raw fft with --no-compile"
        )
    return args


def main() -> None:
    args = parse_args()
    parameter_count = preflight_parameter_count()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or default_run_name(args, timestamp)
    local_dir = args.local_root / run_name
    drive_dir = args.drive_root / run_name
    if local_dir.exists() or drive_dir.exists():
        raise FileExistsError(
            f"refusing to reuse run directory: {run_name}"
        )
    local_dir.mkdir(parents=True)
    drive_dir.mkdir(parents=True)

    command = build_command(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_examples=args.eval_examples,
        log_every=args.log_every,
        checkpoint_dir=local_dir,
        compile_model=args.compile,
        route_backend=args.route_backend,
        packed_memory_backend=args.packed_memory_backend,
        packed_memory_reader_backend=args.packed_memory_reader_backend,
        packed_memory_writer_backend=args.packed_memory_writer_backend,
        packed_memory_feature_chunk=args.packed_memory_feature_chunk,
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
        f"route={args.route_backend} memory={args.packed_memory_backend} "
        f"reader={args.packed_memory_reader_backend} "
        f"writer={args.packed_memory_writer_backend}",
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
    local_log = mirror.local_log
    try:
        with local_log.open(
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
