"""GTR-L quadratic escape-step test: paste this whole file into one Colab cell.

Setup (one time):
  1. Upload gator_colab_bundle.zip to Google Drive at:
       MyDrive/gtrl_quad/gator_colab_bundle.zip
  2. Open a new Colab notebook, select a GPU runtime (Runtime > Change
     runtime type), paste this file into a single cell, and run it.

What this tests: escape_step ~ 600*(length-1)^2, fit from two clean data
points (length=2 escaped@600, length=3 escaped@2400, both rank=64) and a
turbulence onset in length=4 that landed almost exactly on the formula's
prediction of 5400. Each length below gets a budget of predicted_escape*1.10
so there is real headroom to confirm or refute the fit instead of timing out
exactly at the predicted point with no room to see what happens next.

What it does, same pattern as the earlier levers-vs-control Drive cell:
  * Mounts Drive and pulls the bundle from there, not from colab_cli's
    /content upload (which needs the CLI session to stay alive -- this run
    died mid-flight once already today for exactly that reason).
  * Runs the four lengths SEQUENTIALLY in one detached (setsid) process, so
    it survives this cell being interrupted or the kernel restarting.
  * Holds the foreground in a loop -- required, Colab reclaims an
    apparently-idle runtime -- and every 5 minutes copies the raw log and a
    parsed progress digest to Drive, so a lost runtime costs at most 5
    minutes of this length's progress, not the whole multi-hour run.

Stop early any time by interrupting the cell (Runtime > Interrupt execution)
-- the training process is a session leader and keeps running; only
"Runtime > Restart session" kills it too.
"""

import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from google.colab import drive  # type: ignore[import-not-found]

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/gtrl_quad")
BUNDLE = DRIVE_ROOT / "gator_colab_bundle.zip"
DRIVE_LOG_DIR = DRIVE_ROOT / "run_logs"
DRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

if not BUNDLE.exists():
    raise FileNotFoundError(
        f"Expected the bundle at {BUNDLE}. Upload "
        "gator_colab_bundle.zip to Drive at MyDrive/gtrl_quad/ first."
    )

CODE_DIR = Path("/content/GTR_quad_run")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(CODE_DIR)

LOG = Path("/content/quad_train.log")

# The exact training script, written to disk and launched detached so it
# survives this cell's interruption. Deliberately bypasses
# gator_linear_prefix_train.py's CLI: that auto-computes context_length and
# anneals temperature, neither of which the validated smoke-test harness
# (fixed temperature=1.0, context_length=9) used -- comparing the CLI
# trainer's output to that harness was compared apples to oranges once
# already today.
TRAIN_SCRIPT = r'''
import sys, time
sys.path.insert(0, "/content/GTR_quad_run")
import torch
from torch.nn import functional as F
from gator_linear import (GatorLinearConfig, GatorLinearPrefixConfig,
                          GatorLinearPrefixModel)
from parallel_prefix_benchmark import make_prefix_batch

device = torch.device("cuda")
EVAL_EVERY = 300
PLAN = {4: (6000, 5400), 5: (10500, 9600), 6: (16500, 15000), 7: (23700, 21600)}

def run(length, steps, predicted):
    torch.manual_seed(0)
    core = GatorLinearConfig(vocab_size=16, streams=10, nodes=40, width=32,
        relays=4, degree=6, conv_kernel=9, context_length=9,
        scan_normalise=False, hard_routing=True, basis_rank=64)
    model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best, escaped_at, loss_history = 0.0, None, []
    for step in range(steps):
        observed, targets = make_prefix_batch(128, length, device)
        logits = model(observed, output_slots=length, temperature=1.0)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_history.append(float(loss))
        if (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                ob, tg = make_prefix_batch(512, length, device,
                    generator=torch.Generator(device=device).manual_seed(7))
                acc = float(model(ob, output_slots=length, temperature=1.0)
                            .argmax(-1).eq(tg).float().mean())
            best = max(best, acc)
            if escaped_at is None and acc > 2.0/12.0:
                escaped_at = step + 1
            recent = loss_history[-EVAL_EVERY:]
            volatility = max(recent) - min(recent)
            print(f"    len={length} step={step+1:6d}/{steps} "
                  f"predicted={predicted} loss={float(loss):.4f} acc={acc:.3f} "
                  f"best={best:.3f} grad_norm={float(grad_norm):.3f} "
                  f"loss_volatility={volatility:.3f}", flush=True)
    return {"length": length, "best": best, "escaped_at": escaped_at,
            "predicted": predicted}

results = []
for length, (steps, predicted) in PLAN.items():
    print(f"\\n=== length={length} budget={steps} predicted_escape={predicted} ===",
          flush=True)
    r = run(length, steps, predicted)
    results.append(r)
    tag = "ESCAPED" if r["escaped_at"] else "WALL"
    print(f"  => length={length} best={r['best']:.3f} "
          f"escaped_at={r['escaped_at']} predicted={predicted} {tag}", flush=True)

print("\\n\\n=== SUMMARY: escape_step ~ 600*(length-1)^2 ===", flush=True)
for r in results:
    tag = "ESCAPED" if r["escaped_at"] else "WALL"
    print(f"length={r['length']} predicted={r['predicted']:6d} "
          f"actual={r['escaped_at']} best={r['best']:.3f} {tag}", flush=True)
'''

script_path = Path("/content/quad_train_worker.py")
script_path.write_text(TRAIN_SCRIPT)

subprocess.run(["pkill", "-f", "quad_train_worker.py"], check=False)
time.sleep(3)
cmd = (
    f"cd {CODE_DIR} && setsid nohup {sys.executable} -u {script_path} "
    f"> {LOG} 2>&1 < /dev/null &"
)
subprocess.run(["/bin/bash", "-c", cmd], check=True)
time.sleep(5)
alive = subprocess.run(
    ["pgrep", "-fc", "quad_train_worker.py"],
    capture_output=True, text=True,
).stdout.strip()
print(f"training process running: {alive}\n", flush=True)


def digest() -> str:
    text = LOG.read_text() if LOG.exists() else ""
    lines = [l for l in text.splitlines() if l.startswith("    len=") or "=>" in l]
    tail = lines[-6:] if lines else ["(no step lines yet)"]
    return f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n" + "\n".join(tail)


def sync_to_drive() -> None:
    if LOG.exists():
        shutil.copy2(LOG, DRIVE_LOG_DIR / LOG.name)


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
            ["pgrep", "-fc", "quad_train_worker.py"],
            capture_output=True, text=True,
        ).stdout.strip() or 0
    )
    if alive == 0:
        print("\nrun finished", flush=True)
        break
    time.sleep(INTERVAL_SECONDS)
