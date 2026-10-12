"""Procoim consolidation-gate A/B, CONTROL arm, on Colab: paste this whole file into one cell.

Same GTR-L architecture as the currently-running Colab AdamW trace
(gtrl_len7_adamw_trace_seed0_gatebias3_noreadoutbias_routerocc -- length=7,
ring layout, --update-gate-bias 3.0 --no-readout-bias --router-occupancy),
swapped to the Procoim optimizer using the standing-best Procoim search
config from _launch_250k_wsl.sh (directional-Adam, rotating_factorized
basis, past-reference, K=256 adaptive candidate directions), EXCEPT basis
rotation uses banks=8 instead of that config's forwards=500. Per the
Procoim memory (2026-08-07): at K=256, one directional-Adam bank already
costs 2*256=512 forwards -- past the 500-forward threshold in a single
attempt, so forwards=500 rotates wide-K groups almost every bank. That
was diagnosed as the cause of a whole-model stall under reference-guard
and reverted to banks=8; the past-reference config (this one) tolerated
it and got the standing-best result regardless, but for isolating the
new gate's own effect the shorter, noisier forwards=500 windows would
blunt its signal specifically on the wide-K groups doing the most
search. banks=8 gives the gate (and this control) meaningfully-sized
windows to compare. This is the CONTROL arm: the new consolidation-cosine
rotation gate (gator/optimizers/procoim.py) is left OFF (its default).

Run the matching TREATMENT cell
(gtrl_len7_procoim_consolidation_ab_treatment_colab_cell.py) alongside
this one -- same everything, only
--procoim-directional-basis-consolidation-cosine-margin added -- to see
whether deferring basis rotation during directionally-consistent windows
changes recruitment/plateau behavior versus this control.

Local WSL throughput at this config (length=7, K=256, 64 Dykstra
projection iterations) was far below expectation -- essentially zero
step=1000 progress after ~19 minutes of real CPU burn -- hence moving
this to Colab GPU instead of running it locally.

Runs in the foreground (blocks this cell until it finishes) -- Colab is
more aggressive about reclaiming a runtime with no actively-running
foreground cell. Logs sync to Drive every 900s via a background thread
(discrete open-write-close per sync, like checkpoints -- not a held-open
handle -- so Drive doesn't rate-limit/complain).

Setup (one time): gtrl_evotoken_bundle.zip already at
MyDrive/GTR-L/gtrl_evotoken_bundle.zip -- MUST be the bundle rebuilt
2026-08-09 with the consolidation-cosine gate in gator/optimizers/procoim.py
(format_version=19); an older bundle will error on the new CLI flags in
the treatment arm and silently run with the pre-gate optimizer here.
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

CODE_DIR = Path("/content/GTR_len7_procoim_consolidation_ab_control")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

RUN_ROOT = DRIVE_ROOT / "gtrl_len7_procoim_consolidation_ab_control_seed0"
CHECKPOINT_DIR = RUN_ROOT / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STDOUT_LOG = RUN_ROOT / "train.stdout.log"
STDERR_LOG = RUN_ROOT / "train.stderr.log"

LOCAL_DIR = Path("/content/local_logs_procoim_consolidation_ab_control")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_STDOUT = LOCAL_DIR / "train.stdout.log"
LOCAL_STDERR = LOCAL_DIR / "train.stderr.log"

args = [
    sys.executable, "-u", str(CODE_DIR / "gator_linear_prefix_train.py"),
    "--optimizer", "procoim",
    "--steps", "250000",
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
    "--procoim-slot-objective-reduction", "batch_mean",
    "--procoim-reduction-fraction", "0.0001",
    "--procoim-adaptive-reduction-fraction",
    "--procoim-reduction-fraction-decay-rate", "0.1",
    "--procoim-minimum-reduction-fraction", "1e-6",
    "--procoim-candidate-directions", "256",
    "--procoim-adaptive-candidate-directions",
    "--procoim-minimum-candidate-directions", "16",
    "--procoim-candidate-directions-growth", "2.0",
    "--procoim-max-candidate-banks", "16",
    "--procoim-max-accepted-updates", "64",
    "--procoim-probe-scale", "0.03",
    "--procoim-directional-adam",
    "--procoim-directional-adam-learning-rate", "0.03",
    "--procoim-directional-basis-mode", "rotating_factorized",
    "--procoim-directional-factor-rank", "1",
    "--procoim-directional-basis-rotation-banks", "8",
    "--procoim-past-reference-buffer-size", "64",
    "--procoim-past-reference-policy", "improve",
    "--procoim-past-reference-tolerance-fraction", "0.0",
    "--procoim-anchor-buffer-size", "0",
    "--procoim-anchor-sentinel-count", "0",
    "--procoim-consolidation-interval-examples", "0",
    "--procoim-parameter-groups",
    "routing", "receiver_memory", "stream_mix", "node_compute", "readout", "embedding",
    "--seed", "0",
]

print(f"local stdout: {LOCAL_STDOUT} -> synced to {STDOUT_LOG}")
print(f"local stderr: {LOCAL_STDERR} -> synced to {STDERR_LOG}")
print("CONTROL arm: consolidation-cosine gate OFF.")
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
