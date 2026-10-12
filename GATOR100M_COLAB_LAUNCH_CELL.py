# Paste this entire cell into Colab. Edit only the PATHS section.
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import zipfile

# ----------------------------- PATHS ---------------------------------
BUNDLE_ZIP = Path("/content/drive/MyDrive/GTR/colab_bundles/gator100m_staged_colab.zip")
CODE_DIR = Path("/content/GTR")
CONFIG_FILE = CODE_DIR / "GATOR100M_COLAB_CONFIG.json"
DRIVE_RUN_ROOT = Path("/content/drive/MyDrive/GTR/experimental_checkpoints")
LOCAL_WORK_ROOT = Path("/content/gator100m_work")

BATCH_SIZE = 1
LOG_EVERY_STEPS = 10000
SLOT_ISOLATION_SLOTS = 0  # 0 = all 626 nodes shared; 128 = one pool per output slot
HANDOFF_MIN_TEACHER_ACCURACY = 0.80
HANDOFF_MIN_FRONTIER_ACCURACY = 0.80
HANDOFF_ADVANCE_RATIO = 0.95
HANDOFF_RETREAT_RATIO = 0.75
HANDOFF_ADVANCE_PATIENCE = 3
HANDOFF_RETREAT_PATIENCE = 2
HANDOFF_FULL_PROBE_FRACTION = 0.02
HANDOFF_UNFORCED_LOSS_WEIGHT = 2.0
HANDOFF_LOSS_NORMALIZATION = "active_slots"
HANDOFF_ALLOW_POLICY_OVERRIDE = True

# Leave DECODE_MODE unset to keep the staged defaults. Set it to "serial" for
# a clean serial baseline, or use the per-phase overrides independently.
DECODE_MODE = None
PHASE_A_DECODE_MODE = None
PHASE_B_DECODE_MODE = None
PHASE_C_DECODE_MODE = None
FORMATION_OBJECTIVE = "slot_ce"

SMOLTALK_EVERYDAY = Path("/content/drive/MyDrive/GTR/training_data/smoltalk_everyday.jsonl")
TINYSTORIES_TRAIN = Path("/content/drive/MyDrive/GTR/training_data/TinyStories-train.txt")
TINYSTORIES_VALIDATION = Path("/content/drive/MyDrive/GTR/training_data/TinyStories-valid.txt")
LMSYS_CHAT = Path("/content/drive/MyDrive/GTR/training_data/lmsys_chat_1m.jsonl")
TULU3_SFT = Path("/content/drive/MyDrive/Sonic/training_data/chat/tulu3_sft.jsonl")
SMOLTALK_PREFERENCES = Path("/content/drive/MyDrive/Sonic/training_data/chat/smoltalk_preference.jsonl")
# ---------------------------------------------------------------------

if not BUNDLE_ZIP.is_file():
    raise FileNotFoundError(f"Bundle not found: {BUNDLE_ZIP}")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_ZIP) as archive:
    archive.extractall(CODE_DIR)

config = json.loads(CONFIG_FILE.read_text())
config.update({
    "drive_run_root": str(DRIVE_RUN_ROOT),
    "local_work_root": str(LOCAL_WORK_ROOT),
    "batch_size": BATCH_SIZE,
    "log_every_steps": LOG_EVERY_STEPS,
    "slot_isolation_slots": SLOT_ISOLATION_SLOTS,
    "handoff_min_teacher_accuracy": HANDOFF_MIN_TEACHER_ACCURACY,
    "handoff_min_frontier_accuracy": HANDOFF_MIN_FRONTIER_ACCURACY,
    "handoff_advance_ratio": HANDOFF_ADVANCE_RATIO,
    "handoff_retreat_ratio": HANDOFF_RETREAT_RATIO,
    "handoff_advance_patience": HANDOFF_ADVANCE_PATIENCE,
    "handoff_retreat_patience": HANDOFF_RETREAT_PATIENCE,
    "full_free_probe_fraction": HANDOFF_FULL_PROBE_FRACTION,
    "handoff_unforced_loss_weight": HANDOFF_UNFORCED_LOSS_WEIGHT,
    "handoff_loss_normalization": HANDOFF_LOSS_NORMALIZATION,
    "handoff_allow_policy_override": HANDOFF_ALLOW_POLICY_OVERRIDE,
    "formation_objective": FORMATION_OBJECTIVE,
    "smoltalk_everyday": str(SMOLTALK_EVERYDAY),
    "tinystories_train": str(TINYSTORIES_TRAIN),
    "tinystories_validation": str(TINYSTORIES_VALIDATION),
    "lmsys_chat": str(LMSYS_CHAT),
    "tulu3_sft": str(TULU3_SFT),
    "smoltalk_preferences": str(SMOLTALK_PREFERENCES),
})
CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")

for name in (
    "smoltalk_everyday", "tinystories_train", "tinystories_validation",
    "lmsys_chat", "tulu3_sft", "smoltalk_preferences",
):
    path = Path(config[name])
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")

os.chdir(CODE_DIR)
print("Configuration:")
print(CONFIG_FILE.read_text())
print("Starting browser-owned foreground training loop...")

while True:
    runner_args = [
        sys.executable, "-u",
        "run_gator_100m_staged_colab.py",
        "--config", str(CONFIG_FILE),
        "--batch-size", str(BATCH_SIZE),
        "--log-every-steps", str(LOG_EVERY_STEPS),
        "--slot-isolation-slots", str(SLOT_ISOLATION_SLOTS),
        "--handoff-min-teacher-accuracy", str(HANDOFF_MIN_TEACHER_ACCURACY),
        "--handoff-min-frontier-accuracy", str(HANDOFF_MIN_FRONTIER_ACCURACY),
        "--handoff-advance-ratio", str(HANDOFF_ADVANCE_RATIO),
        "--handoff-retreat-ratio", str(HANDOFF_RETREAT_RATIO),
        "--handoff-advance-patience", str(HANDOFF_ADVANCE_PATIENCE),
        "--handoff-retreat-patience", str(HANDOFF_RETREAT_PATIENCE),
        "--handoff-full-probe-fraction", str(HANDOFF_FULL_PROBE_FRACTION),
        "--handoff-unforced-loss-weight", str(HANDOFF_UNFORCED_LOSS_WEIGHT),
        "--handoff-loss-normalization", HANDOFF_LOSS_NORMALIZATION,
        "--formation-objective", FORMATION_OBJECTIVE,
    ]
    for flag, value in (
        ("--decode-mode", DECODE_MODE),
        ("--phase-a-decode-mode", PHASE_A_DECODE_MODE),
        ("--phase-b-decode-mode", PHASE_B_DECODE_MODE),
        ("--phase-c-decode-mode", PHASE_C_DECODE_MODE),
    ):
        if value is not None:
            runner_args.extend([flag, value])
    if not HANDOFF_ALLOW_POLICY_OVERRIDE:
        runner_args.append("--no-handoff-allow-policy-override")
    result = subprocess.run(runner_args, check=False)
    if result.returncode:
        raise RuntimeError(f"training segment failed with exit code {result.returncode}")
    state_path = DRIVE_RUN_ROOT / config["run_name"] / "STAGED_CURRICULUM_STATE.json"
    state = json.loads(state_path.read_text())
    if state["status"] == "complete":
        print(f"Training complete at retained step {state['retained_step']}")
        break
