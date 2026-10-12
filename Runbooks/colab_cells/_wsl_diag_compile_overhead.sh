#!/usr/bin/env bash
# Isolate --compile's fixed one-time overhead from steady-state per-step
# time, for linear_scan vs convolution at context=512. Diffing runtime at
# two step counts cancels the fixed overhead (compile happens once,
# regardless of step count):
#   per_step_time = (runtime_100 - runtime_10) / 90
#   fixed_overhead = runtime_10 - 10 * per_step_time
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
source ~/gtr_venv/bin/activate

DATA=~/gtr_data/smoltalk_everyday.jsonl

BASE_ARGS=(
  --data "$DATA" --vocab-size 2048 --response-tokens 40 --reference-tokens 32
  --embedding-dim 32 --d-model 8 --cells 62 --lexical-mode communal
  --cell-input-mode feature_gated --cell-output-mode grouped_concat
  --cell-transport-mode stewardship --nodes 4 --relays 4 --neighbors 4
  --node-topology hybrid_hemispheric --formation-objective monotonic
  --alignment-warmup-fraction 0.05 --alignment-max-shift 8
  --sequence-quality-penalty 0.01 --execution-preset optimized_gpu_safe
  --mixed-precision bf16 --logging-slim --seed 0
  --context-tokens 512 --batch-size 1
)

echo "=== linear_scan steps=10 (compile+bf16) ==="
rm -rf /tmp/diag_ls10
python -u gator_language_train.py "${BASE_ARGS[@]}" --receiver-memory-backend linear_scan --compile --steps 10 --checkpoint-dir /tmp/diag_ls10 > /tmp/diag_ls10.log 2>&1
tail -3 /tmp/diag_ls10.log
cat /tmp/diag_ls10/runtime_metrics.json 2>/dev/null

echo ""
echo "=== convolution steps=10 (no compile) ==="
rm -rf /tmp/diag_conv10
python -u gator_language_train.py "${BASE_ARGS[@]}" --receiver-memory-backend convolution --steps 10 --checkpoint-dir /tmp/diag_conv10 > /tmp/diag_conv10.log 2>&1
tail -3 /tmp/diag_conv10.log
cat /tmp/diag_conv10/runtime_metrics.json 2>/dev/null
