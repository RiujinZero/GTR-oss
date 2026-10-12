"""GTR-L levers-vs-control run: paste this whole file into one Colab cell.

Setup (one time):
  1. Upload gator_colab_bundle.zip to Google Drive at:
       MyDrive/gtrl/gator_colab_bundle.zip
  2. Open a new Colab notebook, select a GPU runtime (Runtime > Change
     runtime type), paste this file into a single cell, and run it.

What it does:
  * Mounts Drive and copies the bundle in from there (not from colab_cli's
    /content upload, which requires the CLI session to stay alive).
  * Launches two detached (setsid) training runs -- levers vs control, the
    pair isolating whether opening the residual gate and dropping the
    readout bias breaks the constant-prediction collapse -- so they survive
    this cell being interrupted or the kernel restarting.
  * Loops in the foreground, which is required: Colab reclaims a runtime it
    considers idle, and several runs today were lost because nothing was
    holding the foreground while training happened in detached processes.
  * Every 5 minutes, appends a progress digest AND copies the raw per-run
    logs and any checkpoints to Drive, so a lost runtime costs at most 5
    minutes, not the whole run.

Stop early any time by interrupting the cell (Runtime > Interrupt execution)
-- the training processes are session leaders and keep running; press
"Runtime > Restart session" only if you want to kill them too.
"""

import glob
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from google.colab import drive  # type: ignore[import-not-found]

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/gtrl_v4_hardrouting")
BUNDLE = DRIVE_ROOT / "gator_colab_bundle.zip"
DRIVE_LOG_DIR = DRIVE_ROOT / "run_logs"
DRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

if not BUNDLE.exists():
    raise FileNotFoundError(
        f"Expected the bundle at {BUNDLE}. Upload "
        "gator_colab_bundle.zip to Drive at MyDrive/gtrl/ first."
    )

CODE_DIR = Path("/content/GTR_run")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(CODE_DIR)

OUT = Path("/content/lev4")
OUT.mkdir(exist_ok=True)

BASE = (
    "--length 7 --batch-size 128 --eval-batch-size 1024 "
    "--steps 80000 --log-every 500 "
    "--streams 10 --nodes 40 --degree 6 --relays 4 --width 32 "
    "--conv-kernel 9 --temperature-decay 0.9997 "
    "--learning-rate 3e-3 --no-scan-normalise "
    "--route-entropy-loss-weight 0.15 --route-balance-loss-weight 0.5 "
    "--heterogeneous-streams --stream-width-min-fraction 0.125 "
    "--hard-routing"
)
# v4: --hard-routing added to BASE (applies to both arms equally). The
# soft-blend collapse (bus fully open, then, with the bus shut, occupancy
# collapsing onto one node -- both while loss sat exactly on ln(12)) is what
# straight-through hard commit exists to remove: the forward pass now
# genuinely routes to one node per (slot, stream, relay), so "blend toward
# the marginal" is no longer a reachable forward-pass value at all. A CPU
# smoke test on length=2 (which soft routing already solved at 1.000) shows
# hard routing reaches 0.997, escaping faster than soft (step 900 vs 2500)
# but with more oscillation on the way there -- worth watching at length=7.
#
# The pair under test is now secondary to that: does opening the residual
# gate and dropping the readout bias still help *on top of* hard routing?
# Both arms are otherwise identical, so any difference is attributable to
# the levers specifically.
CONFIGS = {
    "levers": "--update-gate-bias 3.0 --no-readout-bias",
    "control": "",
}

subprocess.run(["pkill", "-f", "gator_linear_prefix_train"], check=False)
time.sleep(3)

for name, extra in CONFIGS.items():
    cmd = (
        f"cd {CODE_DIR} && setsid nohup {sys.executable} -u "
        f"gator_linear_prefix_train.py {BASE} {extra} "
        f"--checkpoint-dir {OUT}/{name}_ckpt "
        f"> {OUT}/{name}.log 2>&1 < /dev/null &"
    )
    subprocess.run(["/bin/bash", "-c", cmd], check=True)
    print(f"launched {name}: {extra or '(control, no levers)'}", flush=True)

time.sleep(5)
alive = subprocess.run(
    ["pgrep", "-fc", "gator_linear_prefix_train"],
    capture_output=True, text=True,
).stdout.strip()
print(f"training processes running: {alive}\n", flush=True)


def digest() -> str:
    lines = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ==="]
    for name in CONFIGS:
        log = OUT / f"{name}.log"
        if not log.exists():
            lines.append(f"  {name:9s} (no log yet)")
            continue
        rows = [l for l in log.read_text().splitlines() if l.startswith("step=")]
        if not rows:
            lines.append(f"  {name:9s} starting")
            continue
        last = rows[-1]

        def field(pattern: str, default: str = "-") -> str:
            found = re.search(pattern, last)
            return found.group(1) if found else default

        best = max(
            float(m.group(1))
            for r in rows
            if (m := re.search(r" acc=([0-9.]+)", r))
        )
        lines.append(
            f"  {name:9s} step={field(r'step=(\d+)'):>6s} "
            f"acc={field(r' acc=([0-9.]+)')} best={best:.4f} "
            f"loss={field(r'loss=([0-9.]+)')} "
            f"agree={field(r'stream_agreement=([0-9.]+)')} "
            f"spread={field(r'stream_spread=([0-9.]+)')} "
            f"bus_gate={field(r'bus_gate=([0-9.]+)')}"
        )
    return "\n".join(lines)


def sync_to_drive() -> None:
    for log in glob.glob(str(OUT / "*.log")):
        shutil.copy2(log, DRIVE_LOG_DIR / Path(log).name)
    # Checkpoints are the expensive thing to lose; mirror the latest of each.
    for ckpt_dir in glob.glob(str(OUT / "*_ckpt")):
        name = Path(ckpt_dir).name
        dest = DRIVE_LOG_DIR / name
        dest.mkdir(exist_ok=True)
        for ckpt in glob.glob(f"{ckpt_dir}/*.pt"):
            shutil.copy2(ckpt, dest / Path(ckpt).name)


ARCHIVE = DRIVE_LOG_DIR / "progress.log"
INTERVAL_SECONDS = 300  # 5 minutes

print(
    f"Logging to Drive every {INTERVAL_SECONDS}s at {ARCHIVE}\n"
    "This cell must stay running (foreground) to keep the runtime alive.\n",
    flush=True,
)

while True:
    text = digest()
    print(text, flush=True)
    with ARCHIVE.open("a") as handle:
        handle.write(text + "\n")
    sync_to_drive()

    alive = int(
        subprocess.run(
            ["pgrep", "-fc", "gator_linear_prefix_train"],
            capture_output=True, text=True,
        ).stdout.strip() or 0
    )
    if alive == 0:
        print("\nall runs finished", flush=True)
        break
    time.sleep(INTERVAL_SECONDS)
