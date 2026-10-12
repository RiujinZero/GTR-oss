#!/usr/bin/env bash
# Quick validation (few steps) of the real ~100M SmolLM3-recipe GTR-ARW
# config (from Runbooks/GTR_RUN_REGISTRY.md's
# gtr_arw_100m_smollm3_cell8_26b row) at context=2048, comparing
# convolution vs linear_scan. --block-causal requires linear_scan
# (gator_language_train.py:5009-5012, confirmed 2026-08-12 after resync) so
# it's dropped for the convolution arm only -- user's call, two variables
# differ for that arm (backend + block-causal), not a clean single-variable
# A/B, but it's the only way convolution runs at all here.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
source ~/gtr_venv/bin/activate

DATA=~/gtr_data/smoltalk_everyday.jsonl

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
  --smollm3-peak-lr 8e-5
  --execution-preset optimized_gpu
  --stewardship-execution dense
  --compile --compile-mode max-autotune-no-cudagraphs
  --dense-kernel-padding 8 --dense-node-backend folded_bmm
  --eval-batch-size 1 --eval-examples 2
  --seed 0 --logging-slim --steps 2
)

echo "=== linear_scan (block-causal, as given) steps=2 ==="
rm -rf /tmp/diag_100m_ls
python -u gator_language_train.py "${COMMON[@]}" \
  --receiver-memory-backend linear_scan \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --checkpoint-dir /tmp/diag_100m_ls > /tmp/diag_100m_ls.log 2>&1
echo "rc=$?"
tail -30 /tmp/diag_100m_ls.log

echo ""
echo "=== convolution (no block-causal) steps=2 ==="
rm -rf /tmp/diag_100m_conv
python -u gator_language_train.py "${COMMON[@]}" \
  --receiver-memory-backend convolution \
  --checkpoint-dir /tmp/diag_100m_conv > /tmp/diag_100m_conv.log 2>&1
echo "rc=$?"
tail -30 /tmp/diag_100m_conv.log
