#!/usr/bin/env bash
# Break-in queue items 3 and 4 (Runbooks/RIG_BREAKIN_ABLATION_QUEUE.md), run
# overnight on the new RTX 5080 rig via WSL2, 2026-08-12.
#
# Base shape is the documented "current Gator-S/Arrow language experiments"
# config from GATOR_LANGUAGE_TRAINER_ARGUMENTS.md's Minimal launch section
# (Drive copy, since it's not checked into the repo): d_model=8, cells=62,
# nodes=4, relays=4, neighbors=4, vocab=2048, response=40, reference=32,
# communal lexical mode, feature_gated cell input, grouped_concat cell
# output, stewardship transport, hybrid_hemispheric node topology, monotonic
# formation objective, optimized_gpu_safe preset.
#
# REVISED 2026-08-12 after the first attempt: fp32 batch=32 already reserved
# 19.9GB at context=1024 on a 16GB 5080 (confirmed via nvidia-smi/runtime
# metrics vs the card's 16303MiB total) -- spilling into slow WDDM shared
# memory rather than crashing outright, which would have silently
# contaminated the steps/sec numbers instead of failing loudly. Switched to
# --mixed-precision bf16 and a fixed --batch-size 1 for the whole item-3
# sweep so context=4096 actually fits.
#
# --compile is applied ONLY to linear_scan runs, never convolution.
# Confirmed by direct test: convolution+compile+bf16 at ctx=4096/batch=1
# produced genuine NaN gradients (not just a norm overflow -- actual NaN
# elements, ~8-32% of each tensor) by optimizer step 3. linear_scan+compile
# is known-good from prior sessions. This matches this repo's own documented
# history of compiled-Blackwell runs eventually producing NaNs, just faster
# than expected here -- convolution without --compile (bf16 only) still
# runs correctly, just slower, which is itself the expected/interesting
# result for an O(context^2) backend.
#
# --routing-dim and --d-model earlier looked like they belonged to
# gator_cells_vectorized.py per the runbook text; --d-model does exist here
# on gator_language_train.py (confirmed via --help), but --routing-dim does
# not exist anywhere in the repo's git history on any branch. Items 1/2
# (routing-dim bisection, seed variance) are therefore NOT run by this
# script -- only items 3 and 4, which don't depend on that flag.
#
# One variable changed at a time per GATOR_LANGUAGE_TRAINER_ARGUMENTS.md's
# "Common experimental mistakes" #1 (compare routing/execution knobs one at
# a time, not combined).
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
source ~/gtr_venv/bin/activate

DATA=~/gtr_data/smoltalk_everyday.jsonl
TINYSTORIES=~/gtr_data/TinyStories-valid.txt
ROOT=~/gtr_ablation_20260812
mkdir -p "$ROOT"

BASE_ARGS=(
  --data "$DATA"
  --vocab-size 2048
  --response-tokens 40 --reference-tokens 32
  --embedding-dim 32 --d-model 8 --cells 62
  --lexical-mode communal
  --cell-input-mode feature_gated
  --cell-output-mode grouped_concat
  --cell-transport-mode stewardship
  --nodes 4 --relays 4 --neighbors 4
  --node-topology hybrid_hemispheric
  --formation-objective monotonic
  --alignment-warmup-fraction 0.05
  --alignment-max-shift 8
  --sequence-quality-penalty 0.01
  --execution-preset optimized_gpu_safe
  --mixed-precision bf16
  --logging-slim
  --seed 0
  --max-runtime-seconds 1200
)

run_one() {
  local name="$1"; shift
  local outdir="$ROOT/$name"
  mkdir -p "$outdir"
  echo "=== $(date -Iseconds) starting $name ===" | tee -a "$ROOT/sweep.log"
  python -u gator_language_train.py "${BASE_ARGS[@]}" "$@" \
    --checkpoint-dir "$outdir" \
    > "$outdir/stdout.log" 2> "$outdir/stderr.log"
  local rc=$?
  echo "=== $(date -Iseconds) finished $name rc=$rc ===" | tee -a "$ROOT/sweep.log"
  if [ -f "$outdir/runtime_metrics.json" ]; then
    echo "$name: $(cat "$outdir/runtime_metrics.json" | tr '\n' ' ')" >> "$ROOT/sweep_results.jsonl"
  else
    echo "$name: NO_RUNTIME_METRICS rc=$rc (see stderr.log)" >> "$ROOT/sweep_results.jsonl"
  fi
}

# --- Item 3: linear_scan vs convolution, context-length isolation ---
# batch fixed at 1 (the fixed value that still fits context=4096 with
# convolution's O(context^2) memory even under bf16); context swept;
# backend swept. --compile only for linear_scan (see header note).
for CONTEXT in 512 1024 2048 4096; do
  for BACKEND in convolution linear_scan; do
    EXTRA=()
    if [ "$BACKEND" = "linear_scan" ]; then
      EXTRA+=(--compile)
    fi
    run_one "item3_ctx${CONTEXT}_${BACKEND}" \
      --context-tokens "$CONTEXT" \
      --receiver-memory-backend "$BACKEND" \
      --batch-size 1 \
      --steps 100 \
      "${EXTRA[@]}"
  done
done

# --- Item 4: linear_scan at small-scale language modeling ---
# Deliberately smaller/faster shape than item 3 per the queue's own wording
# ("n_cells=4, d_model small, quick to iterate"), TinyStories mixed in to
# test the free-text/relative-position hypothesis, matched params across
# both backends, more steps since this is about learning signal, not just
# raw throughput. --compile only for linear_scan (see header note).
for BACKEND in convolution linear_scan; do
  EXTRA=()
  if [ "$BACKEND" = "linear_scan" ]; then
    EXTRA+=(--compile)
  fi
  run_one "item4_tinystories_${BACKEND}" \
    --cells 4 --d-model 8 --embedding-dim 32 \
    --context-tokens 256 --response-tokens 40 --reference-tokens 32 \
    --tinystories-data "$TINYSTORIES" \
    --tinystories-mix-fraction 0.40 \
    --receiver-memory-backend "$BACKEND" \
    --steps 500 \
    "${EXTRA[@]}"
done

echo "=== $(date -Iseconds) ALL DONE ===" | tee -a "$ROOT/sweep.log"
