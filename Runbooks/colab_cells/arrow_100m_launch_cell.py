# Paste this entire cell into Colab. Edit only the PATHS/CONFIG section.
#
# Arrow (--architecture gator), 4-cell / 100.2M-param config validated in the
# 2026-07-26 scaling session. See ../ARROW_RUNBOOK.md for the full rationale
# and known caveats (LR/schedule untested past ~1400-2000 steps, requires
# the bf16 fix from commits 29de4fa/5f1808c, decode_mode must be parallel or
# serial — chunk_parallel is not wired up for this architecture).
#
# This is a SHORT validation cell (STEPS below), not an unattended long-run
# loop like the convergent Colab cell — rerun with a higher --steps once
# you've confirmed stability past the point of the last known crash.

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# ----------------------------- PATHS ---------------------------------
BUNDLE_ZIP = Path("/content/drive/MyDrive/GTR/colab_bundles/gator_colab_bundle.zip")
CODE_DIR = Path("/content/GTR_arrow_100m")
CHECKPOINT_DIR = Path("/content/arrow_100m_ckpt")
DATA_PATH = Path("/content/drive/MyDrive/GTR/training_data/smoltalk_preference.jsonl")
LOG_PATH = Path("/content/arrow_100m.log")

STEPS = 2000       # short validation run; raise once stable past this point
BATCH_SIZE = 4
# ---------------------------------------------------------------------

if not BUNDLE_ZIP.is_file():
    raise FileNotFoundError(f"Bundle not found: {BUNDLE_ZIP}")
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")

if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_ZIP) as archive:
    archive.extractall(CODE_DIR)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable, "-u", "gator_language_train.py",
    "--data", str(DATA_PATH),
    "--require-complete-context", "--trim-context-turns", "--deduplicate-examples",
    "--architecture", "gator",
    "--vocab-size", "20000", "--context-tokens", "512", "--response-tokens", "128",
    "--d-model", "2432", "--embedding-dim", "200",
    "--cells", "4",
    "--lexical-mode", "communal",
    "--cell-input-mode", "feature_gated",
    "--cell-output-mode", "shared_pool",
    "--node-topology", "hybrid_hemispheric",
    "--formation-objective", "slot_ce",
    "--relays", "4",
    "--neighbors", "2",
    "--cell-transport-mode", "stewardship",
    "--stewardship-min-active", "2",
    "--stewardship-max-active", "0",
    "--stewardship-keep-mass", "0.9",
    "--stewardship-handoff-keep-mass", "0.9",
    "--stewardship-exploration", "0.01",
    "--stewardship-scouts", "1",
    "--stewardship-execution", "packed_auto",
    "--optimizer-groups", "monolithic",
    "--learning-rate", "3e-3", "--weight-decay", "0.0001",
    "--optimizer-epsilon", "1e-8", "--gradient-clip", "1.0", "--scheduler", "constant",
    "--batch-size", str(BATCH_SIZE),
    "--mixed-precision", "bf16",
    "--compile",
    "--eval-examples", "32",
    "--log-every", "100", "--seed", "0",
    "--checkpoint-dir", str(CHECKPOINT_DIR),
    "--decode-mode", "parallel",
    "--steps", str(STEPS),
]

log_file = open(LOG_PATH, "w")
process = subprocess.Popen(
    command, cwd=CODE_DIR, stdout=log_file, stderr=subprocess.STDOUT,
)
print(f"launched pid={process.pid}, log at {LOG_PATH}", flush=True)
print("tail the log with: !tail -f", str(LOG_PATH))
