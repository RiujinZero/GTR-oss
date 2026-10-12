#!/usr/bin/env bash
# Rerun of item 4 only, fixing a config bug from the first attempt:
# --neighbors 4 (inherited from the item-3 base args) violated
# "neighbors must be positive, even, and less than n_cells" once item 4
# overrode --cells to 4. Fixed to --neighbors 2.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
source ~/gtr_venv/bin/activate

DATA=~/gtr_data/smoltalk_everyday.jsonl
TINYSTORIES=~/gtr_data/TinyStories-valid.txt
ROOT=~/gtr_ablation_20260812

for BACKEND in convolution linear_scan; do
  EXTRA=()
  if [ "$BACKEND" = "linear_scan" ]; then
    EXTRA+=(--compile)
  fi
  outdir="$ROOT/item4_tinystories_${BACKEND}"
  rm -rf "$outdir"
  mkdir -p "$outdir"
  python -u gator_language_train.py \
    --data "$DATA" --vocab-size 2048 \
    --response-tokens 40 --reference-tokens 32 \
    --embedding-dim 32 --d-model 8 \
    --lexical-mode communal --cell-input-mode feature_gated \
    --cell-output-mode grouped_concat --cell-transport-mode stewardship \
    --nodes 4 --relays 4 --node-topology hybrid_hemispheric \
    --formation-objective monotonic --alignment-warmup-fraction 0.05 \
    --alignment-max-shift 8 --sequence-quality-penalty 0.01 \
    --execution-preset optimized_gpu_safe --mixed-precision bf16 \
    --logging-slim --seed 0 --max-runtime-seconds 1200 \
    --cells 4 --neighbors 2 \
    --context-tokens 256 --tinystories-data "$TINYSTORIES" \
    --tinystories-mix-fraction 0.40 --receiver-memory-backend "$BACKEND" \
    --steps 500 "${EXTRA[@]}" \
    --checkpoint-dir "$outdir" \
    > "$outdir/stdout.log" 2> "$outdir/stderr.log"
  rc=$?
  echo "item4_${BACKEND} rc=$rc"
  if [ -f "$outdir/runtime_metrics.json" ]; then
    echo "item4_tinystories_${BACKEND}: $(cat "$outdir/runtime_metrics.json" | tr '\n' ' ')" >> "$ROOT/sweep_results.jsonl"
  else
    echo "item4_tinystories_${BACKEND}: NO_RUNTIME_METRICS rc=$rc (see stderr.log)" >> "$ROOT/sweep_results.jsonl"
  fi
done
