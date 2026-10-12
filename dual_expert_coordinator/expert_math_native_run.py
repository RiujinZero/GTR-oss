from pathlib import Path
import subprocess
import sys
import zipfile
import shutil

BUNDLE_ZIP = Path("/content/gator100m_serial_direct.zip")
CODE_DIR = Path("/content/GTR_expert_math_native")
CHECKPOINT_PATH = Path("/content/expert_math_native_work/checkpoint.pt")
LOG_PATH = Path("/content/expert_math_native_work/train.log")

if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_ZIP) as archive:
    archive.extractall(CODE_DIR)
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable, "-u", "gator_convergent.py",
    "--steps", "10000",
    "--batch-size", "128",
    "--nodes", "51",
    "--width", "30",
    "--degree", "3",
    "--router-mode", "private_full",
    "--router-rank", "16",
    "--ingress-mode", "fixed",
    "--route-backend", "dense",
    "--connection-layout", "ring",
    "--connection-seed", "0",
    "--relays", "4",
    "--lanes", "8",
    "--log-every", "500",
    "--seed", "0",
    "--checkpoint", str(CHECKPOINT_PATH),
    "--compile",
]

log_file = open(LOG_PATH, "w")
process = subprocess.Popen(
    command, cwd=CODE_DIR, stdout=log_file, stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"launched pid={process.pid}", flush=True)
