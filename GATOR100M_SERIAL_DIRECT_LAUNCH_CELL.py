# Paste this entire cell into Colab. Edit only the PATHS section.
#
# This is a DIRECT alternative to GATOR100M_COLAB_LAUNCH_CELL.py / the
# gator100m_staged_colab.zip bundle. It bypasses run_gator_100m_staged_colab.py
# entirely (no Phase A/B/C staging, no route-temperature-freeze bug -
# gator_language_train.py's own internal route_temperature_schedule=legacy
# cooling schedule runs automatically every invocation) and calls
# gator_language_train.py directly with --decode-mode serial, which the
# 2026-07-23 plateau-instability study found eliminates the
# improve-then-regress pattern entirely at small scale (see
# Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md sections 6-7).
#
# Same 100M architecture (D_MODEL=368, N_NODES=626 -> ~1.7 nodes/width,
# EMBEDDING_DIM=288, ROUTER_RANK=96, LANES=10, DEGREE=4, RELAYS=2,
# ACTIVE_NODE_TOP_K=32, VOCAB_SIZE=32768) and the same full multi-corpus
# interleaved data mixture (smoltalk_everyday + TinyStories + lmsys_chat_1m +
# tulu3_sft + smoltalk_preference) as GATOR100M_COLAB_CONFIG.json /
# run_gator_100m_curriculum_colab.py, targeting 800,000 steps.
#
# gator_language_train.py's own --steps flag is a PER-INVOCATION segment
# length, not an absolute target (completed_step = start_step + local_step).
# This cell loops, resuming from the checkpoint's own recorded step count
# each time, until TARGET_STEPS is reached or the Colab runtime disconnects.

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import zipfile

import torch

# ----------------------------- PATHS ---------------------------------
BUNDLE_ZIP = Path("/content/drive/MyDrive/GTR/colab_bundles/gator100m_serial_direct.zip")
CODE_DIR = Path("/content/GTR_serial_direct")
DRIVE_RUN_ROOT = Path("/content/drive/MyDrive/GTR/experimental_checkpoints/gator100m_serial_direct")
LOCAL_CHECKPOINT_DIR = Path("/content/gator100m_serial_direct_work/checkpoints")
LOCAL_LOG_PATH = Path("/content/gator100m_serial_direct_work/train.log")

SMOLTALK_EVERYDAY = Path("/content/drive/MyDrive/GTR/training_data/smoltalk_everyday.jsonl")
TINYSTORIES_TRAIN = Path("/content/drive/MyDrive/GTR/training_data/TinyStories-train.txt")
TINYSTORIES_VALIDATION = Path("/content/drive/MyDrive/GTR/training_data/TinyStories-valid.txt")
LMSYS_CHAT = Path("/content/drive/MyDrive/GTR/training_data/lmsys_chat_1m.jsonl")
TULU3_SFT = Path("/content/drive/MyDrive/Sonic/training_data/chat/tulu3_sft.jsonl")
SMOLTALK_PREFERENCES = Path("/content/drive/MyDrive/Sonic/training_data/chat/smoltalk_preference.jsonl")

TARGET_STEPS = 800_000
SEGMENT_STEPS = 10_000       # steps per gator_language_train.py invocation
BATCH_SIZE = 1
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
        # --- full multi-corpus interleaved mixture, same as the staged bundle ---
        "--data", str(SMOLTALK_EVERYDAY),
        "--tinystories-data", str(TINYSTORIES_TRAIN),
        "--tinystories-validation-data", str(TINYSTORIES_VALIDATION),
        "--tinystories-mix-fraction", "0.40",
        "--tinystories-validation-stories", "512",
        "--tinystories-tokenizer-stories", "50000",
        "--tinystories-shuffle-buffer", "16384",
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
        "--streaming-weight", "1.0",
        "--streaming-exhaustion", "restart",
        "--primary-stream-weight", "1",
        "--primary-shuffle-buffer", "8192",
        "--require-complete-context", "--trim-context-turns",
        "--deduplicate-examples", "--skip-first-assistant",
        # --- same 100M architecture as run_gator_100m_curriculum_colab.py ---
        "--architecture", "convergent",
        "--vocab-size", "32768",
        "--embedding-dim", "288",
        "--d-model", "368",
        "--nodes", "626",
        "--lanes", "10",
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
        "--terminal-vote-mode", "slot",
        # --- lane-collapse counter-measures: winning_lanes was pinned at
        # 1.00 from step 2000-10000 in the first run of this bundle. Not yet
        # confirmed as THE root cause (a parallel finding shows a 50k-param
        # Gator using global-jury mode, which cannot lane-collapse, still
        # gets PPL in the hundreds on this same full corpus mixture -
        # pointing at a data/corpus issue as at least a second contributing
        # factor). Keeping these anyway since they're free/harmless and
        # address a real, separately-confirmed collapse mode.
        "--slot-isolation-slots", "128",
        "--lane-vote-noise-std", "0.1",
        "--lane-balance-loss-weight", "0.01",
        "--formation-objective", "slot_ce",
        "--context-tokens", "512",
        "--response-tokens", "128",
        # --- serial decode: the fix validated in the plateau-instability study ---
        "--decode-mode", "serial",
        # --- optimizer: monolithic, no staged/component-scaled multipliers -
        # the staged bundle's component_scaled LR multipliers exist to support
        # its Phase A/B/C freeze/rewind curriculum, which this direct script
        # does not implement. serial decode does not require them. ---
        "--optimizer-groups", "monolithic",
        "--learning-rate", "3e-4",
        "--weight-decay", "0.0001",
        "--optimizer-epsilon", "1e-8",
        "--gradient-clip", "1.0",
        "--scheduler", "constant",
        "--batch-size", str(BATCH_SIZE),
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
    state = torch.load(latest, map_location="cpu", weights_only=False)
    return int(state.get("step", 0))


def _sync_to_drive() -> None:
    dest = DRIVE_RUN_ROOT / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    for item in LOCAL_CHECKPOINT_DIR.glob("*"):
        target = dest / item.name
        shutil.copy2(item, target)
    if LOCAL_LOG_PATH.is_file():
        shutil.copy2(LOCAL_LOG_PATH, DRIVE_RUN_ROOT / "train.log")


import os
os.chdir(CODE_DIR)

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
