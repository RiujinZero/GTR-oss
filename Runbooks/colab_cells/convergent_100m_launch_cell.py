# Paste this entire cell into Colab. Edit only the PATHS section.
#
# Convergent Gator (--architecture convergent), 97.7M params, the config
# currently in production (resumed lineage gator100m_serial_direct_golden_v2,
# TARGET_STEPS=3,500,000). See ../CONVERGENT_RUNBOOK.md for the full
# rationale behind every flag below.
#
# gator_language_train.py's own --steps flag is a PER-INVOCATION segment
# length, not an absolute target (completed_step = start_step + local_step).
# This cell loops, resuming from the checkpoint's own recorded step count
# each time, until TARGET_STEPS is reached or the Colab runtime disconnects
# (rerun the same cell to resume from the last Drive-synced checkpoint).

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import torch

# ----------------------------- PATHS ---------------------------------
BUNDLE_ZIP = Path("/content/drive/MyDrive/GTR/colab_bundles/gator_colab_bundle.zip")
CODE_DIR = Path("/content/GTR_convergent_100m")
DRIVE_RUN_ROOT = Path("/content/drive/MyDrive/GTR/experimental_checkpoints/gator100m_serial_direct_golden_v2")
LOCAL_CHECKPOINT_DIR = Path("/content/gator100m_convergent_work/checkpoints")
LOCAL_LOG_PATH = Path("/content/gator100m_convergent_work/train.log")

SMOLTALK_EVERYDAY = Path("/content/drive/MyDrive/GTR/training_data/smoltalk_everyday.jsonl")
TINYSTORIES_TRAIN = Path("/content/drive/MyDrive/GTR/training_data/TinyStories-train.txt")
TINYSTORIES_VALIDATION = Path("/content/drive/MyDrive/GTR/training_data/TinyStories-valid.txt")
LMSYS_CHAT = Path("/content/drive/MyDrive/GTR/training_data/lmsys_chat_1m.jsonl")
TULU3_SFT = Path("/content/drive/MyDrive/Sonic/training_data/chat/tulu3_sft.jsonl")
SMOLTALK_PREFERENCES = Path("/content/drive/MyDrive/Sonic/training_data/chat/smoltalk_preference.jsonl")
WIKIPEDIA = Path("/content/drive/MyDrive/Sonic/training_data/Knowledge/wikipedia_en.jsonl")
OPENSCIENCE = Path("/content/drive/MyDrive/GTR/training_data/science/openscience.jsonl")

RESUME_CHECKPOINT = DRIVE_RUN_ROOT / "checkpoints" / "latest.pt"  # None-able below

TARGET_STEPS = 3_500_000
SEGMENT_STEPS = 10_000       # steps per gator_language_train.py invocation
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LOG_EVERY = 2_000
EVAL_EXAMPLES = 16
COMPILE = True
CHECKPOINT_SYNC_SECONDS = 3600
LOG_SYNC_SECONDS = 900
# ---------------------------------------------------------------------

if not BUNDLE_ZIP.is_file():
    raise FileNotFoundError(f"Bundle not found: {BUNDLE_ZIP}")
for name, path in (
    ("smoltalk_everyday", SMOLTALK_EVERYDAY),
    ("tinystories_train", TINYSTORIES_TRAIN),
    ("tinystories_validation", TINYSTORIES_VALIDATION),
    ("lmsys_chat", LMSYS_CHAT),
    ("tulu3_sft", TULU3_SFT),
    ("smoltalk_preferences", SMOLTALK_PREFERENCES),
    ("wikipedia", WIKIPEDIA),
    ("openscience", OPENSCIENCE),
):
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")

if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_ZIP) as archive:
    archive.extractall(CODE_DIR)

LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_RUN_ROOT.mkdir(parents=True, exist_ok=True)


def _common_flags(checkpoint_dir: Path) -> list[str]:
    flags = [
        sys.executable, "-u", "gator_language_train.py",
        # --- turn-based chat streams (60% of total mix) ---
        "--data", str(SMOLTALK_EVERYDAY),
        "--primary-stream-weight", "1",
        "--primary-shuffle-buffer", "8192",
        "--lmsys-data", str(LMSYS_CHAT),
        "--lmsys-source-weight", "4",
        "--lmsys-load-mode", "stream",
        "--lmsys-shuffle-buffer", "16384",
        "--lmsys-validation-conversations", "512",
        "--lmsys-max-characters", "8192",
        "--lmsys-min-ascii-letter-ratio", "0.9",
        "--lmsys-min-response-characters", "8",
        "--lmsys-require-user-prompt",
        "--chat-data", str(TULU3_SFT),
        "--chat-source-weight", "3",
        "--chat-data", str(SMOLTALK_PREFERENCES),
        "--chat-source-weight", "2",
        "--chat-max-characters", "8192",
        "--chat-shuffle-buffer", "8192",
        # --- prose streams: TinyStories 10%, Wikipedia 15%, OpenScience 15% ---
        "--tinystories-data", str(TINYSTORIES_TRAIN),
        "--tinystories-validation-data", str(TINYSTORIES_VALIDATION),
        "--tinystories-mix-fraction", "0.10",
        "--tinystories-validation-stories", "512",
        "--tinystories-tokenizer-stories", "50000",
        "--tinystories-shuffle-buffer", "16384",
        "--prose-data", str(WIKIPEDIA),
        "--prose-source-weight", "2.5",
        "--prose-data", str(OPENSCIENCE),
        "--prose-source-weight", "2.5",
        "--streaming-weight", "1.0",
        "--streaming-exhaustion", "restart",
        "--require-complete-context", "--trim-context-turns",
        "--deduplicate-examples", "--skip-first-assistant",
        # --- 100M architecture ---
        "--architecture", "convergent",
        "--vocab-size", "20000",
        "--embedding-dim", "200",
        "--d-model", "379",
        "--nodes", "643",
        "--lanes", "1",
        "--graph-degree", "4",
        "--relays", "2",
        "--connection-layout", "small_world",
        "--connection-seed", "0",
        "--connection-rewire", "0.5",
        "--router-mode", "shared_low_rank",
        "--router-rank", "96",
        "--ingress-mode", "learned",
        "--active-node-top-k", "32",
        "--route-backend", "dense",
        "--terminal-vote-mode", "prompt",
        "--formation-objective", "slot_ce",
        "--context-tokens", "512",
        "--response-tokens", "128",
        # --- serial decode: the plateau-instability study's clean fix ---
        "--decode-mode", "serial",
        "--optimizer-groups", "monolithic",
        "--learning-rate", "3e-4",
        "--weight-decay", "0.0001",
        "--optimizer-epsilon", "1e-8",
        "--gradient-clip", "1.0",
        "--scheduler", "constant",
        "--batch-size", str(BATCH_SIZE),
        "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--eval-examples", str(EVAL_EXAMPLES),
        "--log-every", str(LOG_EVERY),
        "--logging-slim",
        "--seed", "0",
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    if COMPILE:
        flags.append("--compile")
    return flags


def _current_step(checkpoint_dir: Path) -> int:
    latest = checkpoint_dir / "latest.pt"
    if not latest.is_file():
        return 0
    try:
        state = torch.load(latest, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError) as error:
        raise RuntimeError(
            f"local checkpoint at {latest} appears truncated/corrupt "
            f"({error}); restore from a Drive backup (e.g. best.pt) into "
            f"{checkpoint_dir} before rerunning this cell"
        ) from error
    return int(state.get("step", 0))


def _sync_to_drive() -> None:
    dest = DRIVE_RUN_ROOT / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    for item in LOCAL_CHECKPOINT_DIR.glob("*"):
        _copy_checkpoint_verified(item, dest / item.name)
    if LOCAL_LOG_PATH.is_file():
        shutil.copy2(LOCAL_LOG_PATH, DRIVE_RUN_ROOT / "train.log")


import os
os.chdir(CODE_DIR)


def _copy_checkpoint_verified(src: Path, dst: Path, attempts: int = 3) -> None:
    # drive.mount()'s FUSE layer can let shutil.copy2 return before the
    # underlying Drive sync has actually flushed every byte, silently
    # leaving a truncated local file (torch.load then dies with
    # "EOFError: Ran out of input"). Verify size and retry before trusting it.
    expected_size = src.stat().st_size
    for attempt in range(1, attempts + 1):
        shutil.copy2(src, dst)
        actual_size = dst.stat().st_size
        if actual_size == expected_size:
            return
        print(
            f"checkpoint copy attempt {attempt} truncated "
            f"(expected {expected_size}, got {actual_size}); retrying",
            flush=True,
        )
    raise RuntimeError(
        f"checkpoint copy from {src} kept truncating after {attempts} attempts"
    )


# First run in a fresh Colab session: seed the local checkpoint dir from the
# last Drive-synced checkpoint (if any) so this cell resumes instead of
# restarting from scratch.
if not (LOCAL_CHECKPOINT_DIR / "latest.pt").is_file() and RESUME_CHECKPOINT.is_file():
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _copy_checkpoint_verified(RESUME_CHECKPOINT, LOCAL_CHECKPOINT_DIR / "latest.pt")

last_checkpoint_sync = 0.0
last_log_sync = 0.0
log_file = open(LOCAL_LOG_PATH, "a")

while True:
    completed = _current_step(LOCAL_CHECKPOINT_DIR)
    if completed >= TARGET_STEPS:
        print(f"reached target_steps={TARGET_STEPS}, stopping.", flush=True)
        break
    segment = min(SEGMENT_STEPS, TARGET_STEPS - completed)
    command = _common_flags(LOCAL_CHECKPOINT_DIR)
    command += ["--steps", str(segment)]
    if completed > 0:
        command += ["--resume-checkpoint", str(LOCAL_CHECKPOINT_DIR / "latest.pt")]
    print(
        f"=== launching segment: completed={completed} "
        f"target_this_segment={completed + segment} ===",
        flush=True,
    )
    process = subprocess.Popen(
        command, cwd=CODE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        now = time.time()
        if now - last_log_sync > LOG_SYNC_SECONDS:
            log_file.flush()
            _sync_to_drive()
            last_log_sync = now
        if now - last_checkpoint_sync > CHECKPOINT_SYNC_SECONDS:
            _sync_to_drive()
            last_checkpoint_sync = now
    process.wait()
    log_file.flush()
    _sync_to_drive()
    if process.returncode != 0:
        print(
            f"segment exited with code {process.returncode}; "
            "re-checking checkpoint step and retrying.",
            flush=True,
        )
        time.sleep(5)

print("done.", flush=True)
