"""AdamW length=7 breakout trace, on Colab: paste this whole file into one cell.

Same config as the local WSL run (slot2_length7_adamw_trace_seed0):
nodes=40/width=8/relays=4/streams=10 (46,701 params), --length 7,
--optimizer adamw --learning-rate 3e-3 --no-scan-normalise --hard-routing,
--adamw-weight-trace-every 500. Runs in the foreground (blocks this cell
until it finishes) -- Colab is more aggressive about reclaiming a runtime
that has no actively-running foreground cell, so a detached background
subprocess risks the whole VM getting killed even with the tab open.
**Log sync fix (2026-08-09, applied retroactively to this base template):**
earlier versions of this cell piped the training subprocess's stdout/
stderr directly into a file opened on the Drive FUSE mount and held open
for the entire run. Drive doesn't appear to sync that kind of long-held,
continuously-appended file handle on its own (checkpoints sync fine
because torch.save does a discrete open-write-close per file) -- user had
to manually click the file in Colab's file browser to force a sync. Fix:
write the logs to a LOCAL file instead, and run a background thread that
copies the local file over the Drive path every 900s (15 min, kept
infrequent so Drive doesn't rate-limit/complain -- user's call) via a
fresh open-write-close (shutil.copy2), the same kind of discrete write
that makes checkpoints sync reliably. A final sync also runs after the
process exits so the last lines aren't lost.

Setup (one time): gtrl_evotoken_bundle.zip already at
MyDrive/GTR-L/gtrl_evotoken_bundle.zip (confirmed fresh, includes the
config-dump feature and AdamWWeightTracer as of 2026-08-08).
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

CODE_DIR = Path("/content/GTR_len7_adamw_trace")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

RUN_ROOT = DRIVE_ROOT / "gtrl_len7_adamw_trace_seed0"
CHECKPOINT_DIR = RUN_ROOT / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STDOUT_LOG = RUN_ROOT / "train.stdout.log"
STDERR_LOG = RUN_ROOT / "train.stderr.log"

LOCAL_DIR = Path("/content/local_logs_seed0")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_STDOUT = LOCAL_DIR / "train.stdout.log"
LOCAL_STDERR = LOCAL_DIR / "train.stderr.log"

args = [
    sys.executable, "-u", str(CODE_DIR / "gator_linear_prefix_train.py"),
    "--optimizer", "adamw",
    "--steps", "100000",
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
    "--graph-layout", "small_world",
    "--graph-seed", "0",
    "--graph-rewire", "0.5",
    "--ingress-temperature", "1.0",
    "--device", "auto",
    "--amp-dtype", "none",
    "--log-every", "500",
    "--checkpoint-dir", str(CHECKPOINT_DIR),
    "--hard-routing",
    "--no-scan-normalise",
    "--temperature-start", "1.0",
    "--temperature-min", "1.0",
    "--temperature-decay", "1.0",
    "--balanced-prefix-batches",
    "--prefix-loss-vocabulary", "valid",
    "--learning-rate", "3e-3",
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
