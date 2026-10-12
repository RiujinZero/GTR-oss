"""Run the TinyStories-first causal or accuracy-gated Gator experiment.

The runner deliberately exposes only payload width as the architectural
variable.  It writes the complete command and a live copy of stdout into the
checkpoint directory so a successful run can be reconstructed verbatim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence, TextIO

from gator_convergent import ConvergentLanguageConfig, ConvergentLanguageModel


EXPECTED_PARAMETERS = {
    20: 130_442,
    400: 14_360_682,
}
DEFAULT_CHECKPOINT_ROOT = Path(
    "/content/drive/MyDrive/GTR/experimental_checkpoints/"
    "teacher_forced_language_controls"
)


def model_config(width: int) -> ConvergentLanguageConfig:
    """Return the architecture shared by the D20 and D400 controls."""

    return ConvergentLanguageConfig(
        vocab_size=2048,
        d_model=width,
        embedding_dim=32,
        n_nodes=66,
        degree=3,
        relays=2,
        lanes=10,
        mask_token_id=3,
        blank_token_id=2047,
        forbidden_output_token_ids=(0, 1, 3, 4, 5),
        connection_layout="small_world",
        connection_seed=0,
        connection_rewire=0.5,
        terminal_vote_mode="slot",
    )


def preflight_parameter_count(width: int) -> int:
    """Reject accidental architecture drift before the corpus is loaded."""

    if width not in EXPECTED_PARAMETERS:
        supported = ", ".join(str(value) for value in EXPECTED_PARAMETERS)
        raise ValueError(
            f"unsupported width {width}; frozen controls are {supported}"
        )
    actual = ConvergentLanguageModel(model_config(width)).parameter_count
    expected = EXPECTED_PARAMETERS[width]
    if actual != expected:
        raise RuntimeError(
            "teacher-forced control drift: "
            f"D{width} expected {expected:,} parameters, found {actual:,}"
        )
    return actual


def default_run_name(
    width: int,
    steps: int,
    decode_mode: str = "graduated_handoff",
    chunk_size: int = 8,
) -> str:
    mode_label = {
        "teacher_forced": "teacherforced",
        "handoff": "handoff60to90",
        "graduated_handoff": "graduated_u5_teacher",
        "token_handoff": "tokenwise_teacher_decay",
        "prefix_denoise": "prefixdenoise95to50_contextgate",
        "chunk_parallel": f"nextchunk{chunk_size}",
    }[decode_mode]
    output_label = (
        f"o{chunk_size}_ref128"
        if decode_mode == "chunk_parallel"
        else "o128"
    )
    return (
        f"{mode_label}_d{width}_n66_l10_k3_r2_"
        f"c512_{output_label}_b8_rw50_seed0_{steps // 1000}k"
    )


def validate_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser()
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            "refusing to overwrite or implicitly resume an existing run: "
            f"{checkpoint_dir}"
        )
    return checkpoint_dir


def build_command(
    *,
    width: int,
    steps: int,
    log_every: int,
    batch_size: int = 8,
    eval_batch_size: int = 4,
    eval_examples: int = 0,
    checkpoint_dir: Path,
    compile_model: bool,
    decode_mode: str = "graduated_handoff",
    chunk_size: int = 8,
    resume_checkpoint: Path | None = None,
    token_handoff_advance_ratio: float = 0.95,
    token_handoff_retreat_ratio: float = 0.75,
    token_handoff_advance_patience: int = 3,
    token_handoff_retreat_patience: int = 2,
    token_handoff_full_probe_fraction: float = 0.02,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the frozen mixed-corpus teacher-forced command."""

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
        "--streaming-weight",
        "1.0",
        "--streaming-exhaustion",
        "restart",
        "--primary-stream-weight",
        "1",
        "--primary-shuffle-buffer",
        "8192",
        "--lmsys-source-weight",
        "4",
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
        "--lmsys-load-mode",
        "stream",
        "--lmsys-shuffle-buffer",
        "16384",
        "--chat-shuffle-buffer",
        "8192",
        "--lmsys-validation-conversations",
        "512",
        "--lmsys-max-characters",
        "8192",
        "--chat-max-characters",
        "8192",
        "--lmsys-min-ascii-letter-ratio",
        "0.9",
        "--lmsys-min-response-characters",
        "8",
        "--lmsys-require-user-prompt",
        "--require-complete-context",
        "--trim-context-turns",
        "--deduplicate-examples",
        "--architecture",
        "convergent",
        "--decode-mode",
        decode_mode,
        "--chunk-size",
        str(chunk_size),
        "--formation-objective",
        "slot_ce",
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(eval_batch_size),
        "--eval-examples",
        str(eval_examples),
        "--log-every",
        str(log_every),
        "--logging-slim",
        "--vocab-size",
        "2048",
        "--context-tokens",
        "512",
        "--response-tokens",
        "128",
        "--embedding-dim",
        "32",
        "--d-model",
        str(width),
        "--nodes",
        "66",
        "--graph-degree",
        "3",
        "--lanes",
        "10",
        "--terminal-vote-mode",
        "slot",
        "--relays",
        "2",
        "--connection-layout",
        "small_world",
        "--connection-seed",
        "0",
        "--connection-rewire",
        "0.5",
        "--skip-first-assistant",
        "--learning-rate",
        "0.003",
        "--scheduler",
        "cosine",
        "--scheduler-total-steps",
        str(steps),
        "--minimum-lr",
        "0.0001",
        "--seed",
        "0",
        "--batch-pipeline",
        "prefetch",
        "--prefetch-batches",
        "20",
        "--float32-matmul-precision",
        "high",
        "--cudnn-tf32",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if decode_mode in {
        "handoff",
        "graduated_handoff",
        "prefix_denoise",
    }:
        command.extend(
            [
                "--handoff-start-content-accuracy",
                "0.60",
                "--handoff-end-content-accuracy",
                "0.90",
                "--handoff-accuracy-ema-decay",
                "0.50",
            ]
        )
    if decode_mode == "graduated_handoff":
        command.extend(
            [
                "--graduated-baseline-fraction",
                "0.05",
                "--graduated-accuracy-step",
                "0.05",
                "--graduated-release-accuracy",
                "0.90",
            ]
        )
    if decode_mode == "token_handoff":
        command.extend(
            [
                "--token-handoff-start-removed",
                "1",
                "--token-handoff-max-removed",
                "0",
                "--token-handoff-advance-ratio",
                str(token_handoff_advance_ratio),
                "--token-handoff-retreat-ratio",
                str(token_handoff_retreat_ratio),
                "--token-handoff-min-teacher-accuracy",
                "0.15",
                "--token-handoff-advance-patience",
                str(token_handoff_advance_patience),
                "--token-handoff-retreat-patience",
                str(token_handoff_retreat_patience),
                "--token-handoff-full-probe-fraction",
                str(token_handoff_full_probe_fraction),
                "--token-handoff-unforced-loss-weight",
                "1.0",
            ]
        )
    if decode_mode == "prefix_denoise":
        command.extend(
            [
                "--prefix-retention-start",
                "0.95",
                "--prefix-retention-floor",
                "0.50",
                "--prefix-denoise-warmup-steps",
                "100000",
                "--prefix-zero-row-fraction",
                "0.05",
                "--prefix-short-row-fraction",
                "0.10",
                "--prefix-short-max-tokens",
                "8",
                "--prefix-span-fraction",
                "0.50",
                "--prefix-context-utility-start",
                "0.02",
                "--prefix-context-utility-end",
                "0.10",
            ]
        )
    if compile_model:
        command.append("--compile")
    if resume_checkpoint is not None:
        command.extend(
            [
                "--resume-checkpoint",
                str(resume_checkpoint),
                "--resume-optimizer",
                "--resume-data-state",
                "auto",
            ]
        )
    return command


def write_run_spec(
    checkpoint_dir: Path,
    *,
    width: int,
    steps: int,
    parameter_count: int,
    command: Sequence[str],
    decode_mode: str = "graduated_handoff",
    chunk_size: int = 8,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Freeze the exact experiment before compilation begins."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "RUN_SPEC.json"
    path.write_text(
        json.dumps(
            {
                "experiment": (
                    "teacher-forced Gator language capacity control"
                    if decode_mode == "teacher_forced"
                    else (
                        "accuracy-gated teacher-to-parallel Gator handoff"
                        if decode_mode == "handoff"
                        else (
                            "parallel next-chunk language formation"
                            if decode_mode == "chunk_parallel"
                            else (
                                "graduated unsupported-to-teacher cadence"
                                if decode_mode == "graduated_handoff"
                                else (
                                    "one-token-at-a-time teacher-mask "
                                    "curriculum"
                                    if decode_mode == "token_handoff"
                                    else (
                                        "causal prefix-denoising Gator "
                                        "language curriculum"
                                    )
                                )
                            )
                        )
                    )
                ),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "launched",
                "controlled_variable": "d_model",
                "d_model": width,
                "parameters": parameter_count,
                "decode_mode": decode_mode,
                "chunk_size": chunk_size,
                "causal_semantics": (
                    True
                    if decode_mode == "teacher_forced"
                    else (
                        "whole-batch mixture of causal teacher forcing "
                        "and noncausal blank formation"
                        if decode_mode in {
                            "handoff",
                            "graduated_handoff",
                        }
                        else (
                            "gold completed chunks in context; current "
                            "chunk is entirely blank and noncausal"
                            if decode_mode == "chunk_parallel"
                            else (
                                "causal token/span prefix corruption with "
                                "zero-prefix and short-prefix rows"
                                if decode_mode == "prefix_denoise"
                                else (
                                    "causal shifted targets with exactly one "
                                    "additional teacher token removed per "
                                    "mastered level"
                                )
                            )
                        )
                    )
                ),
                "parallel_training_predictions": (
                    chunk_size
                    if decode_mode == "chunk_parallel"
                    else 128
                ),
                "novel_inference": (
                    "autoregressive"
                    if decode_mode == "teacher_forced"
                    else (
                        "target is simultaneous blank formation after handoff"
                        if decode_mode in {
                            "handoff",
                            "graduated_handoff",
                        }
                        else (
                            f"{chunk_size}-token chunks committed "
                            "sequentially; tokens inside each chunk form "
                            "simultaneously"
                            if decode_mode == "chunk_parallel"
                            else (
                                "progressively unsupported simultaneous causal "
                                "response formation"
                            )
                        )
                    )
                ),
                "steps": steps,
                "resume_checkpoint": (
                    None
                    if resume_checkpoint is None
                    else str(resume_checkpoint)
                ),
                "corpus_order": [
                    "roneneldan/TinyStories",
                    "smoltalk_everyday",
                    "lmsys_chat_1m",
                    "tulu3_sft",
                    "smoltalk_preference",
                ],
                "corpus_shares": {
                    "roneneldan/TinyStories": 0.40,
                    "smoltalk_everyday": 0.06,
                    "lmsys_chat_1m": 0.24,
                    "tulu3_sft": 0.18,
                    "smoltalk_preference": 0.12,
                },
                "tinystories_curriculum": (
                    "long reference continuations sampled as aligned "
                    f"{chunk_size}-token next chunks"
                    if decode_mode == "chunk_parallel"
                    else "two-sentence prefix/continuation pairs per 64 slots"
                ),
                "command": list(command),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def stream_and_log(
    command: Sequence[str],
    log_file: TextIO,
    *,
    expected_decode_mode: str = "graduated_handoff",
) -> None:
    """Mirror merged stdout/stderr to both the notebook and Drive."""

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    mode_verified = False
    parameters_verified = False
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
        if "decode_mode=" in line:
            if f"decode_mode={expected_decode_mode}" not in line:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError(
                    "trainer reported the wrong decode mode; terminated"
                )
            mode_verified = True
        if line.startswith("run ") and "parameters=" in line:
            parameters_verified = True
        if line.startswith("step=") and not (
            mode_verified and parameters_verified
        ):
            process.terminate()
            process.wait(timeout=30)
            raise RuntimeError(
                "training reached a step before mode/parameter verification"
            )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, choices=(20, 400), default=20)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=0,
        help=(
            "validation examples per teacher/parallel evaluation; "
            "0 evaluates the complete validation set"
        ),
    )
    parser.add_argument(
        "--decode-mode",
        choices=(
            "teacher_forced",
            "handoff",
            "graduated_handoff",
            "token_handoff",
            "prefix_denoise",
            "chunk_parallel",
        ),
        default="graduated_handoff",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("--batch-size and --eval-batch-size must be positive")
    if args.eval_examples < 0:
        parser.error("--eval-examples cannot be negative")
    if args.chunk_size <= 0 or args.chunk_size > 128:
        parser.error("--chunk-size must be in [1, 128]")

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = (
            DEFAULT_CHECKPOINT_ROOT
            / default_run_name(
                args.d_model,
                args.steps,
                args.decode_mode,
                args.chunk_size,
            )
        )
    checkpoint_dir = validate_checkpoint_dir(checkpoint_dir)
    parameter_count = preflight_parameter_count(args.d_model)
    command = build_command(
        width=args.d_model,
        steps=args.steps,
        log_every=args.log_every,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_examples=args.eval_examples,
        checkpoint_dir=checkpoint_dir,
        compile_model=not args.no_compile,
        decode_mode=args.decode_mode,
        chunk_size=args.chunk_size,
        resume_checkpoint=args.resume_checkpoint,
    )
    spec = write_run_spec(
        checkpoint_dir,
        width=args.d_model,
        steps=args.steps,
        parameter_count=parameter_count,
        command=command,
        decode_mode=args.decode_mode,
        chunk_size=args.chunk_size,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(
        "LANGUAGE CONTROL PREFLIGHT PASSED "
        f"decode_mode={args.decode_mode} "
        f"d_model={args.d_model} parameters={parameter_count:,}",
        flush=True,
    )
    print(f"checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"run_spec={spec}", flush=True)
    print("command=" + " ".join(command), flush=True)
    with (checkpoint_dir / "train.log").open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        stream_and_log(
            command,
            log_file,
            expected_decode_mode=args.decode_mode,
        )


if __name__ == "__main__":
    main()
