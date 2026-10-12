"""GTR-L 130k-param language, whitespace-stripped, long run: paste this
whole file into one Colab cell.

Setup (one time):
  1. Upload gator_colab_bundle.zip to Google Drive at:
       MyDrive/gtrl_nows/gator_colab_bundle.zip
  2. Open a new Colab notebook, select a GPU runtime (Runtime > Change
     runtime type), paste this file into a single cell, and run it.

What this is: the whitespace-stripped 130k-param TinyStories validation,
promoted to a longer run after the 9000-step comparison showed it never
re-plateaued the way the whitespace-included baseline did (modal_share kept
dropping past where the baseline stalled at ~0.80, by_pos showed a smooth
gradient across all 8 positions instead of a sharp "4 learn, 4 frozen"
split). Config is unchanged (vocab=2048, width=16/nodes=40=129k params,
context=136 = 128 observed + 8 output slots, controlled_writing/
linear_attention/hard_routing on) except:
  - Whitespace pieces are excluded from tokenizer training and skipped when
    building the token stream -- the model never sees or predicts a
    whitespace token, removing the "free" high-frequency floor that was
    inflating the baseline's raw accuracy (~30 points of the baseline's
    ~41% was traceable to guessing the modal whitespace token correctly,
    not real learning).
  - Steps raised from 9000 to 10,000,000 (effectively open-ended -- stop
    the cell whenever), and the training text slice enlarged from 20M to
    80M characters so a run this much longer sees meaningfully more
    distinct text instead of recycling the same ~5.8M-token stream.
  - torch.compile + bf16 autocast are both on. Measured on a 100M-param
    config: 27-47% less peak memory (growing with context) and 2.7-3.8x
    more steps/sec than plain eager fp32. Loss/accuracy trajectories were
    verified to match eager fp32 to 3-4 significant figures on a smoke
    test, so this isn't expected to change what the run learns, only how
    fast and how much memory it takes to learn it.

Runs detached (setsid) so it survives this cell being interrupted, and
logs a progress digest to Drive every 5 minutes -- required, Colab
reclaims an apparently-idle runtime, and the WSL/Colab CLI sessions this
whole investigation ran on have died unpredictably many times.
"""

import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

from google.colab import drive  # type: ignore[import-not-found]

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/gtrl_nows")
BUNDLE = DRIVE_ROOT / "gator_colab_bundle.zip"
DRIVE_LOG_DIR = DRIVE_ROOT / "run_logs"
DRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

if not BUNDLE.exists():
    raise FileNotFoundError(
        f"Expected the bundle at {BUNDLE}. Upload "
        "gator_colab_bundle.zip to Drive at MyDrive/gtrl_nows/ first."
    )

CODE_DIR = Path("/content/GTR_nows_long")
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
CODE_DIR.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as zf:
    zf.extractall(CODE_DIR)

LOG = Path("/content/nows_long_train.log")

TRAIN_SCRIPT = r'''
import subprocess, sys, time
sys.path.insert(0, "/content/GTR_nows_long")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download

import torch
from torch.nn import functional as F
from pathlib import Path
from collections import Counter

from gator.tokenization import CompactTokenizer, _PIECE_PATTERN
from gator_linear import GatorLinearConfig, GatorLinearPrefixConfig, GatorLinearPrefixModel

REPOSITORY = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATA_DIR = Path("/content/tinystories")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOCAB = 2048
OUTPUT_SLOTS = 8
OBSERVED = 128
CONTEXT_LENGTH = OBSERVED + OUTPUT_SLOTS
TRAIN_CHARS = 80_000_000
VALID_CHARS = 4_000_000

STEPS = 10_000_000  # open-ended -- stop the cell whenever
EVAL_EVERY = 300
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 256
SEED = 0

def download(filename):
    return Path(hf_hub_download(repo_id=REPOSITORY, filename=filename,
        repo_type="dataset", revision=REVISION, local_dir=str(DATA_DIR)))

print("Downloading TinyStories...", flush=True)
train_path = download("TinyStories-train.txt")
valid_path = download("TinyStories-valid.txt")

with open(train_path, "r", encoding="utf-8", errors="ignore") as handle:
    train_text = handle.read(TRAIN_CHARS)
with open(valid_path, "r", encoding="utf-8", errors="ignore") as handle:
    valid_text = handle.read(VALID_CHARS)
print(f"train_chars={len(train_text)} valid_chars={len(valid_text)}", flush=True)

print("Training tokenizer (whitespace excluded)...", flush=True)
counts = Counter()
for piece in _PIECE_PATTERN.findall(train_text):
    if not piece.isspace():
        counts[piece] += 1
reserved_blank = 1
capacity = VOCAB - CompactTokenizer.LEARNED_OFFSET - reserved_blank
ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
if len(ranked) < capacity:
    raise ValueError(f"corpus has only {len(ranked)} unique non-whitespace pieces, need {capacity}")
tokenizer = CompactTokenizer(piece for piece, _ in ranked[:capacity])
assert tokenizer.vocab_size == VOCAB

def encode_no_whitespace(text):
    result = []
    for piece in _PIECE_PATTERN.findall(text):
        if piece.isspace():
            continue
        token_id = tokenizer._piece_to_id.get(piece)
        if token_id is not None:
            result.append(token_id)
        else:
            result.extend(tokenizer.BYTE_OFFSET + b for b in piece.encode("utf-8"))
    return result

print("Tokenizing corpus (whitespace stripped)...", flush=True)
device = torch.device("cuda")
train_ids = torch.tensor(encode_no_whitespace(train_text), dtype=torch.long).to(device)
valid_ids = torch.tensor(encode_no_whitespace(valid_text), dtype=torch.long).to(device)
print(f"train_tokens={len(train_ids)} valid_tokens={len(valid_ids)}", flush=True)

def make_batch(ids, batch_size, generator=None):
    max_start = ids.shape[0] - CONTEXT_LENGTH - 1
    starts = torch.randint(0, max_start, (batch_size,), device=ids.device, generator=generator)
    offsets = torch.arange(CONTEXT_LENGTH, device=ids.device)
    windows = ids[starts.unsqueeze(1) + offsets.unsqueeze(0)]
    return windows[:, :OBSERVED], windows[:, OBSERVED:]

torch.manual_seed(SEED)
core = GatorLinearConfig(vocab_size=VOCAB, streams=10, nodes=40, width=16, relays=4,
    degree=6, buses=4, mailbox_channels=4, mailbox_width=8, basis_rank=4,
    address_width=4, router_rank=8, conv_kernel=9, context_length=CONTEXT_LENGTH,
    scan_normalise=False, hard_routing=True, linear_attention=True)
model = GatorLinearPrefixModel(GatorLinearPrefixConfig(core=core)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params={n_params:,} ({n_params/1e3:.1f}k)", flush=True)

# torch.compile + bf16 autocast: measured on the 100M-scale grid at
# ~27-47% memory savings (growing with context) and 2.7-3.8x speedup over
# plain eager fp32. Compile specializes per input shape and only trains at
# a fixed (BATCH_SIZE, CONTEXT_LENGTH) shape here plus one eval shape
# (EVAL_BATCH_SIZE) -- two shapes total, well under dynamo's default
# recompile cap, so this run compiles once per shape and keeps the
# speedup for its whole (open-ended) duration, unlike a benchmark that
# sweeps many shapes and can silently fall back to eager after the cap.
train_model = torch.compile(model)
AMP_DTYPE = torch.bfloat16

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

best, escaped_at, best_by_position = 0.0, None, None
started = time.time()
for step in range(STEPS):
    observed, targets = make_batch(train_ids, BATCH_SIZE)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        logits = train_model(observed, output_slots=OUTPUT_SLOTS, temperature=1.0)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % EVAL_EVERY == 0:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            ob, tg = make_batch(valid_ids, EVAL_BATCH_SIZE,
                generator=torch.Generator(device=device).manual_seed(7))
            pred = train_model(ob, output_slots=OUTPUT_SLOTS, temperature=1.0).argmax(-1)
            correct = pred.eq(tg).float()
            acc = float(correct.mean())
            by_position = correct.mean(dim=0).tolist()
            modal_share = float(torch.bincount(pred.reshape(-1), minlength=VOCAB).max() / pred.numel())
        if acc > best:
            best = acc
            best_by_position = by_position
        if escaped_at is None and acc > 2.0 / VOCAB:
            escaped_at = step + 1
        elapsed = time.time() - started
        pos_str = " ".join(f"{p:.2f}" for p in by_position)
        print(f"step={step+1:6d}/{STEPS} t={elapsed/60:.1f}m loss={float(loss):.4f} "
              f"acc={acc:.4f} best={best:.4f} modal_share={modal_share:.3f} "
              f"by_pos=[{pos_str}]", flush=True)

tag = "ESCAPED" if escaped_at else "WALL"
pos_str = " ".join(f"{p:.2f}" for p in (best_by_position or []))
print(f"\n=> nowhitespace_long vocab={VOCAB} width=16 nodes=40 params={n_params} "
      f"steps={STEPS} best={best:.4f} escaped_at={escaped_at} {tag} by_pos=[{pos_str}]", flush=True)
'''

script_path = Path("/content/nows_long_worker.py")
script_path.write_text(TRAIN_SCRIPT)

subprocess.run(["pkill", "-f", "nows_long_worker.py"], check=False)
time.sleep(3)
cmd = (
    f"cd {CODE_DIR} && setsid nohup {sys.executable} -u {script_path} "
    f"> {LOG} 2>&1 < /dev/null &"
)
subprocess.run(["/bin/bash", "-c", cmd], check=True)
time.sleep(5)
alive = subprocess.run(
    ["pgrep", "-fc", "nows_long_worker.py"],
    capture_output=True, text=True,
).stdout.strip()
print(f"training process running: {alive}\n", flush=True)


def digest() -> str:
    text = LOG.read_text() if LOG.exists() else ""
    lines = [l for l in text.splitlines() if l.startswith("step=") or "=>" in l]
    tail = lines[-8:] if lines else ["(no step lines yet)"]
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
            ["pgrep", "-fc", "nows_long_worker.py"],
            capture_output=True, text=True,
        ).stdout.strip() or 0
    )
    if alive == 0:
        print("\nrun finished", flush=True)
        break
    time.sleep(INTERVAL_SECONDS)
