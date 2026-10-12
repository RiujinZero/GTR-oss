from pathlib import Path
import subprocess
import sys
import zipfile
import shutil

BUNDLE_ZIP = Path("/content/gator100m_serial_direct.zip")
CODE_DIR = Path("/content/GTR_expert_language")
CHECKPOINT_DIR = Path("/content/expert_language_work/checkpoints")
LOG_PATH = Path("/content/expert_language_work/train.log")

if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_ZIP) as archive:
    archive.extractall(CODE_DIR)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable, "-u", "gator_language_train.py",
    "--data", "/content/smoltalk_preference_valid.jsonl",
    "--require-complete-context", "--trim-context-turns", "--deduplicate-examples",
    "--architecture", "convergent",
    "--vocab-size", "2000", "--context-tokens", "256", "--response-tokens", "128",
    "--embedding-dim", "20", "--d-model", "31", "--nodes", "53",
    "--lanes", "8", "--graph-degree", "4", "--relays", "4",
    "--connection-layout", "small_world", "--connection-seed", "0", "--connection-rewire", "1.0",
    "--router-mode", "shared_low_rank", "--router-rank", "16",
    "--ingress-mode", "learned",
    "--route-backend", "dense",
    "--terminal-vote-mode", "slot",
    "--formation-objective", "slot_ce",
    "--decode-mode", "serial",
    "--optimizer-groups", "monolithic",
    "--learning-rate", "3e-4", "--weight-decay", "0.0001",
    "--optimizer-epsilon", "1e-8", "--gradient-clip", "1.0", "--scheduler", "constant",
    "--batch-size", "16",
    "--mixed-precision", "bf16",
    "--eval-examples", "128",
    "--log-every", "500", "--seed", "0",
    "--checkpoint-dir", str(CHECKPOINT_DIR),
    "--compile",
    "--steps", "20000",
]

log_file = open(LOG_PATH, "w")
process = subprocess.Popen(
    command, cwd=CODE_DIR, stdout=log_file, stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"launched pid={process.pid}", flush=True)
