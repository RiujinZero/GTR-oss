#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/user/Documents/GTR
ROOT="/mnt/c/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/transformer_causal_10k_seed0_200k"
mkdir -p "$ROOT/checkpoint"

# Autoregressive (causal, teacher-forced) counterpart to
# transformer_10k_seed0_200k -- same 10k-param scale (d_model=14 heads=2
# ff_width=52 layers=4 -> 10,019 params, vs the one-shot model's 10,020),
# same task/seed/step budget, but decodes output slots one token at a
# time (teacher-forced during training, real autoregressive generation at
# eval) instead of predicting all 7 slots simultaneously in one pass.
#
# User's ask (2026-08-09): "The run we should queue up after the 10k
# transformer 200k steps run is 10k teacher forced single token decode
# 200k steps to see if it can output 7 slots autoregressively at least."
#
# New model (PrefixTransformerCausal) and script (transformer_prefix_causal_run.py)
# added this session -- smoke-tested for 20 steps first (clean run,
# parameters=10019, both teacher-forced and free-running generated metrics
# compute correctly) before queueing the real 200k-step run.
exec /home/user/.local/bin/uv run --with torch --index https://download.pytorch.org/whl/cpu -- python -u transformer_prefix_causal_run.py \
  --d-model 14 \
  --heads 2 \
  --ff-width 52 \
  --layers 4 \
  --steps 200000 \
  --batch-size 128 \
  --seed 0 \
  --portable-batches \
  --checkpoint-dir "$ROOT/checkpoint" \
  > "$ROOT/train.stdout.log" 2> "$ROOT/train.stderr.log"
