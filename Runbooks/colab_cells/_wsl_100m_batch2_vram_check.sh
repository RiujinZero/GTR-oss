#!/usr/bin/env bash
# Quick VRAM check at batch=2/accum=16 (same effective batch=32) before
# committing to a longer steady-state run, per user's recollection that
# batch=2 is close to or over 18GB on this architecture. Only linear_scan
# (the winning backend from the 15-minute comparison) -- no need to retest
# convolution given it already lost decisively at batch=1.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
source ~/gtr_venv/bin/activate

export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache

DATA=~/gtr_data/smoltalk_everyday.jsonl
OUT=~/100m_batch2_check
rm -rf "$OUT"
mkdir -p "$OUT"

python -u gator_language_train.py \
  --data "$DATA" \
  --cells 8 --nodes 4 --d-model 1312 --embedding-dim 2624 \
  --lexical-embedding-dim 256 --relays 2 --neighbors 4 \
  --node-topology asymmetric \
  --cell-input-mode shared_projection --cell-output-mode shared_pool \
  --cell-transport-mode broadcast_mix \
  --response-tokens 16 \
  --context-tokens 2048 \
  --batch-size 2 --gradient-accumulation-steps 16 \
  --mixed-precision bf16 \
  --learning-rate 8e-5 \
  --smollm3-peak-lr 8e-5 \
  --execution-preset optimized_gpu \
  --stewardship-execution dense \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --eval-batch-size 1 --eval-examples 2 \
  --receiver-memory-backend linear_scan \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --seed 0 --logging-slim --steps 64 \
  --checkpoint-dir "$OUT" > "$OUT.log" 2>&1
echo "rc=$?"
tail -15 "$OUT.log"
cat "$OUT/runtime_metrics.json" 2>/dev/null
