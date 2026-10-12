"""Launch the guarded D30 slot-voter language A/B on Colab.

This runner is intentionally narrow.  It exists to make the formation-control
experiment hard to confuse with the completed D30/global control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from gator_convergent import (
    ConvergentLanguageConfig,
    ConvergentLanguageModel,
)


EXPECTED_PARAMETERS = 186_722
EXPECTED_MODE = "slot"
DEFAULT_CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/GTR/experimental_checkpoints/"
    "convergent_scaling/"
    "formation_slotjury_v1_d30_n66_l10_r2_seed0_300000steps"
)


def slot_model_config() -> ConvergentLanguageConfig:
    """Return the exact architecture used by the guarded A/B."""

    return ConvergentLanguageConfig(
        vocab_size=2048,
        d_model=30,
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
        terminal_vote_mode=EXPECTED_MODE,
    )


def preflight_parameter_count() -> int:
    """Instantiate the exact model and reject any source/config drift."""

    model = ConvergentLanguageModel(slot_model_config())
    parameter_count = model.parameter_count
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "slot-voter source/config drift: "
            f"expected {EXPECTED_PARAMETERS:,} parameters, "
            f"found {parameter_count:,}"
        )
    return parameter_count


def validate_checkpoint_dir(checkpoint_dir: Path) -> Path:
    """Require an unmistakable, unused slot-voter checkpoint directory."""

    checkpoint_dir = checkpoint_dir.expanduser()
    if "slot" not in checkpoint_dir.name.lower():
        raise ValueError(
            "refusing ambiguous checkpoint directory without 'slot' in "
            f"its final component: {checkpoint_dir}"
        )
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            "refusing to overwrite or implicitly resume an existing run: "
            f"{checkpoint_dir}"
        )
    return checkpoint_dir


def build_command(
    *,
    checkpoint_dir: Path,
    steps: int,
    log_every: int,
    compile_model: bool,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the exact trainer command for the D30 slot-voter A/B."""

    command = [
        python_executable,
        "-u",
        "gator_language_train.py",
        "--data",
        "/content/drive/MyDrive/GTR/training_data/smoltalk_everyday.jsonl",
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
        "512",
        "--chat-max-characters",
        "512",
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
        "--steps",
        str(steps),
        "--batch-size",
        "32",
        "--eval-batch-size",
        "64",
        "--log-every",
        str(log_every),
        "--logging-slim",
        "--vocab-size",
        "2048",
        "--context-tokens",
        "64",
        "--response-tokens",
        "32",
        "--embedding-dim",
        "32",
        "--d-model",
        "30",
        "--nodes",
        "66",
        "--graph-degree",
        "3",
        "--lanes",
        "10",
        "--terminal-vote-mode",
        EXPECTED_MODE,
        "--relays",
        "2",
        "--connection-layout",
        "small_world",
        "--connection-seed",
        "0",
        "--connection-rewire",
        "0.5",
        "--skip-first-assistant",
        "--scheduler",
        "cosine",
        "--scheduler-total-steps",
        "300000",
        "--minimum-lr",
        "0.0001",
        "--learning-rate",
        "0.003",
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
    if compile_model:
        command.append("--compile")
    return command


def write_run_spec(
    checkpoint_dir: Path,
    *,
    command: Sequence[str],
    parameter_count: int,
) -> Path:
    """Create a visible Drive marker before data loading or compilation."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "RUN_SPEC.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "D30 slot-conditioned terminal voting",
                "expected_parameters": parameter_count,
                "terminal_vote_mode": EXPECTED_MODE,
                "steps": 300_000,
                "command": list(command),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_guarded(command: Sequence[str]) -> None:
    """Stream training and terminate immediately on a wrong startup mode."""

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    mode_verified = False
    for line in process.stdout:
        print(line, end="", flush=True)
        if "terminal_vote_mode=" in line:
            if f"terminal_vote_mode={EXPECTED_MODE}" not in line:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError(
                    "trainer reported the wrong terminal vote mode; "
                    "the process was terminated"
                )
            mode_verified = True
        if line.startswith("step=") and not mode_verified:
            process.terminate()
            process.wait(timeout=30)
            raise RuntimeError(
                "training reached a step log before the slot mode was verified"
            )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if not mode_verified:
        raise RuntimeError(
            "trainer exited without reporting terminal_vote_mode=slot"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")

    checkpoint_dir = validate_checkpoint_dir(args.checkpoint_dir)
    parameter_count = preflight_parameter_count()
    command = build_command(
        checkpoint_dir=checkpoint_dir,
        steps=args.steps,
        log_every=args.log_every,
        compile_model=not args.no_compile,
    )
    spec_path = write_run_spec(
        checkpoint_dir,
        command=command,
        parameter_count=parameter_count,
    )
    print(
        "SAFE SLOT PREFLIGHT PASSED "
        f"parameters={parameter_count} "
        f"terminal_vote_mode={EXPECTED_MODE}",
        flush=True,
    )
    print(f"checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"run_spec={spec_path}", flush=True)
    print("command=" + " ".join(command), flush=True)
    run_guarded(command)


if __name__ == "__main__":
    main()
