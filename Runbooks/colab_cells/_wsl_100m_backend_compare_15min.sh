#!/usr/bin/env bash
# 15-minute wall-clock comparison of the real ~100M SmolLM3-recipe GTR-ARW
# config at context=2048, convolution vs linear_scan.
#
# REVISED 2026-08-12: first attempt wrote checkpoints and torch.compile's
# Inductor kernel cache to /tmp, which is a RAM-backed tmpfs capped at
# 7.5GB in this WSL2 instance -- it filled completely (Inductor cache alone
# hit 5.6GB from today's earlier compile runs) and corrupted a checkpoint
# write mid-run (torch.save "unexpected pos" zip error) before the 15-minute
# window was up. Checkpoint dir and TORCHINDUCTOR_CACHE_DIR both moved to
# the real disk (~/, ext4, 946GB free, confirmed via `df -h /`) instead.
#
# Also noticed in the failed run's partial log: loss exploded to ~1322
# (vs. a sane ~11.8 floor) with degenerate repeated-token output by
# step=1000. This isn't using --training-recipe smollm3-26b (kept the
# simpler --data/local-tokenizer path instead of the full HF-streaming
# recipe), so --smollm3-peak-lr 8e-5 is inert outside that recipe --
# --learning-rate silently defaulted to 0.008, wildly too high for this
# scale, which is the real cause of the blowup, not a receiver-memory-
# backend issue. Added --learning-rate 8e-5 explicitly to both arms to
# actually apply the intended peak LR value.
#
# --block-causal requires --receiver-memory-backend linear_scan
# (gator_language_train.py:5009-5012) so it's dropped for the convolution
# arm only -- two variables differ for that arm (backend + block-causal),
# not a clean single-variable A/B, but it's the only way convolution runs
# at all under this architecture.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
source ~/gtr_venv/bin/activate

export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache

DATA=~/gtr_data/smoltalk_everyday.jsonl
DURATION=900
OUT=~/100m_15min_compare
mkdir -p "$OUT"

COMMON=(
  --data "$DATA"
  --cells 8 --nodes 4 --d-model 1312 --embedding-dim 2624
  --lexical-embedding-dim 256 --relays 2 --neighbors 4
  --node-topology asymmetric
  --cell-input-mode shared_projection --cell-output-mode shared_pool
  --cell-transport-mode broadcast_mix
  --response-tokens 16
  --context-tokens 2048
  --batch-size 1 --gradient-accumulation-steps 32
  --mixed-precision bf16
  --learning-rate 8e-5
  --smollm3-peak-lr 8e-5
  --execution-preset optimized_gpu
  --stewardship-execution dense
  --compile --compile-mode max-autotune-no-cudagraphs
  --dense-kernel-padding 8 --dense-node-backend folded_bmm
  --eval-batch-size 1 --eval-examples 2
  --seed 0 --logging-slim
  --steps 100000 --max-runtime-seconds "$DURATION"
)

echo "=== linear_scan (block-causal, as given) 15min ==="
rm -rf "$OUT/ls"
python -u gator_language_train.py "${COMMON[@]}" \
  --receiver-memory-backend linear_scan \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --checkpoint-dir "$OUT/ls" > "$OUT/ls.log" 2>&1
echo "linear_scan rc=$?"
tail -8 "$OUT/ls.log"
cat "$OUT/ls/runtime_metrics.json" 2>/dev/null

echo ""
echo "=== convolution (no block-causal) 15min ==="
rm -rf "$OUT/conv"
python -u gator_language_train.py "${COMMON[@]}" \
  --receiver-memory-backend convolution \
  --checkpoint-dir "$OUT/conv" > "$OUT/conv.log" 2>&1
echo "convolution rc=$?"
tail -8 "$OUT/conv.log"
cat "$OUT/conv/runtime_metrics.json" 2>/dev/null
