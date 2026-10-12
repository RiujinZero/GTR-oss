#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/user/Documents/GTR
ROOT="/mnt/c/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/arw_10k_trace_seed0"
mkdir -p "$ROOT/checkpoint"

# Rerun of the known-good GTR-ARW 10k-param mastery checkpoint (see memory
# gtr-arw-10k-mastery-reference / experimental_checkpoints/README.md), this
# time with AdamWWeightTracer on -- decided 2026-08-08 after the GTR-L
# length=7 AdamW trace run (slot2_length7_adamw_trace_seed0) got stuck in a
# disturbed regime around step 16.5k with no exact-match by step 28k, while
# GTR-ARW is *known* to master this exact task (length=7, 16-symbol vocab)
# within 20k steps at seed 0 (99.8% token / 98.8% exact). We already have
# the GTR-L trace data to dissect what went wrong there; this run gets the
# equivalent trace for an architecture that actually succeeds, for direct
# comparison.
#
# Exact reproducible command per memory: `python gator_cells_vectorized.py
# --steps 20000 --seed 0 --compile` -- every other value is the script's
# own default (cells=10, nodes=4, relays=4, neighbors=4, batch-size=128).
# No carry-state/persistent-active/reference-dense flags (those are
# confirmed-different, weaker variants, not stronger ones).
#
# DEVIATION FROM THE REFERENCE COMMAND: --compile dropped. The original
# verified run used --compile on CUDA/Colab (per
# experimental_checkpoints/README.md's own entry for this checkpoint); on
# this local WSL CPU environment, torch.compile's CPU codegen path requires
# a working C++ compiler and none is installed here (g++ not found),
# crashing immediately with InvalidCxxCompiler. At 10,000 params, eager
# mode on CPU should still be fast enough without it -- flagged here so
# this run is never mistaken for a bit-for-bit repro of the original.
#
# --adamw-weight-trace-every 200 (finer than GTR-L's 500, since this run's
# budget is 4x smaller and the model is 10k params vs 46,701 -- cheap to
# trace more densely and still land on whatever step the mastery
# transition happens at with good resolution).
exec /home/user/.local/bin/uv run --with torch --index https://download.pytorch.org/whl/cpu -- python -u gator_cells_vectorized.py \
  --steps 20000 \
  --seed 0 \
  --checkpoint "$ROOT/checkpoint/arw_10k_seed0.pt" \
  --adamw-weight-trace-every 200 \
  --adamw-weight-trace-dir "$ROOT/checkpoint/adamw_weight_trace" \
  > "$ROOT/train.stdout.log" 2> "$ROOT/train.stderr.log"
