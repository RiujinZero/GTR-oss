"""AdamW length=7 breakout trace, all levers + LOWER learning rate, on Colab: paste this whole file into one cell.

Same config as gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc
(--update-gate-bias 3.0 --no-readout-bias --router-occupancy --graph-layout
ring), which reached the highest valid_acc of any GTR-L AdamW run this
session (0.35 at step 37,500) but then stalled: positions 1-2 fanned out
to near-1.000, positions 3-7 never started moving, valid_acc flat for
~13,000 steps -- see Runbooks/GTRL_ADAMW_BREAKOUT_LOG.md.

ONLY CHANGE: --learning-rate 3e-3 -> 3e-4. User's read (2026-08-09): "The
run collapsed which looks like a learning rate mismatch to me" -- and,
directly: "We were using 3e-4 and 3e-5 towards the better runs." Every
length=7 AdamW run this session (this one included) used 3e-3, carried
over by citation from Papers/GATOR_LINEAR_DIAGNOSTIC.md's finding that
3e-3 solves LENGTH=2 at 1.000 -- that doc's own "Open items" section
flagged "LR sensitivity unmapped... worth knowing whether the window is
narrow" for length=7 specifically, and a repo-wide search (2026-08-09)
found the length=7 LR sweep it mentions as "running" (1e-3, 3e-4) was
never actually written up anywhere. 3e-3 at length=7 was inherited, never
validated. This run tests the user's recollection directly: same
architecture/levers, one order of magnitude lower LR.

Runs in the foreground (blocks this cell until it finishes) -- Colab is
more aggressive about reclaiming a runtime that has no actively-running
foreground cell, so a detached background subprocess risks the whole VM
getting killed even with the tab open. Logs sync to Drive every 900s via
a background thread (discrete open-write-close per sync, like
checkpoints -- not a held-open handle; kept infrequent so Drive doesn't
rate-limit/complain).

Setup (one time): gtrl_evotoken_bundle.zip already at
MyDrive/GTR-L/gtrl_evotoken_bundle.zip, must include the --router-occupancy
code (gator_linear_prefix_train.py commit 1b006ba) -- confirm this before
running, since an old bundle will silently ignore the flag or error out.
"""

import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from google.colab import drive  # type: ignore[import-not-found]

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/GTR-L")
BUNDLE = DRIVE_ROOT / "gtrl_evotoken_bundle.zip"
if not BUNDLE.exists():
    raise FileNotFoundError(f"Expected the bundle at {BUNDLE}")

CODE_DIR = Path("/content/GTR_len7_adamw_trace_gatebias3_noreadoutbias_routerocc_lr3e4")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

RUN_ROOT = DRIVE_ROOT / "gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc_lr3e4"
CHECKPOINT_DIR = RUN_ROOT / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STDOUT_LOG = RUN_ROOT / "train.stdout.log"
STDERR_LOG = RUN_ROOT / "train.stderr.log"

LOCAL_DIR = Path("/content/local_logs_gatebias3_noreadoutbias_routerocc_lr3e4")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_STDOUT = LOCAL_DIR / "train.stdout.log"
LOCAL_STDERR = LOCAL_DIR / "train.stderr.log"

args = [
    sys.executable, "-u", str(CODE_DIR / "gator_linear_prefix_train.py"),
    "--optimizer", "adamw",
    "--steps", "1000000",
    "--batch-size", "144",
    "--eval-batch-size", "512",
    "--prefix-episode-mode", "fresh",
    "--length", "7",
    "--streams", "10",
    "--nodes", "40",
    "--width", "8",
    "--relays", "4",
    "--degree", "6",
    "--buses", "4",
    "--mailbox-channels", "4",
    "--mailbox-width", "8",
    "--basis-rank", "4",
    "--address-width", "4",
    "--router-rank", "8",
    "--conv-kernel", "3",
    "--graph-layout", "ring",
    "--graph-seed", "0",
    "--graph-rewire", "0.5",
    "--ingress-temperature", "1.0",
    "--device", "auto",
    "--amp-dtype", "none",
    "--log-every", "500",
    "--checkpoint-dir", str(CHECKPOINT_DIR),
    "--hard-routing",
    "--no-scan-normalise",
    "--router-occupancy",
    "--update-gate-bias", "3.0",
    "--no-readout-bias",
    "--temperature-start", "1.0",
    "--temperature-min", "1.0",
    "--temperature-decay", "1.0",
    "--balanced-prefix-batches",
    "--prefix-loss-vocabulary", "valid",
    "--learning-rate", "3e-4",
    "--weight-decay", "1e-4",
    "--gradient-clip", "1.0",
    "--adamw-weight-trace-every", "500",
    "--adamw-weight-trace-dir", str(CHECKPOINT_DIR / "adamw_weight_trace"),
    "--seed", "0",
]

print(f"local stdout: {LOCAL_STDOUT} -> synced to {STDOUT_LOG}")
print(f"local stderr: {LOCAL_STDERR} -> synced to {STDERR_LOG}")
print(f"trace artifacts: {CHECKPOINT_DIR / 'adamw_weight_trace'}")
print("Running in the foreground -- this cell blocks until the run finishes.")
print("Logs sync to Drive every 900s via a background thread (discrete")
print("open-write-close per sync, like checkpoints -- not a held-open handle).")
print()

sync_stop = threading.Event()


def _sync_logs_to_drive():
    while not sync_stop.is_set():
        for local, remote in ((LOCAL_STDOUT, STDOUT_LOG), (LOCAL_STDERR, STDERR_LOG)):
            if local.exists():
                shutil.copy2(local, remote)
        sync_stop.wait(900)


sync_thread = threading.Thread(target=_sync_logs_to_drive, daemon=True)
sync_thread.start()

with LOCAL_STDOUT.open("w") as out, LOCAL_STDERR.open("w") as err:
    result = subprocess.run(
        args,
        cwd=str(CODE_DIR),
        stdout=out,
        stderr=err,
    )

sync_stop.set()
sync_thread.join(timeout=5)
for local, remote in ((LOCAL_STDOUT, STDOUT_LOG), (LOCAL_STDERR, STDERR_LOG)):
    if local.exists():
        shutil.copy2(local, remote)

print(f"process exited with code {result.returncode}")
