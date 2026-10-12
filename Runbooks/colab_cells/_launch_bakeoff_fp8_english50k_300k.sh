#!/usr/bin/env bash
# 24h-bake-off candidate #1: dense (folded_bmm) node-compute at fp8_te
# precision, d_model=1312, the new English-only 50k-vocab tokenizer,
# 300,000-update full run with resumable checkpoints every 500 updates
#
# Context length: start at --smollm3-sequence-length 512 (context_tokens
# = 512-16 = 496), not the eventual 2048 target. User's call: cheaper to
# train short and extend context later than to pay 2048-context cost
# the whole way through. This repo doesn't use literal RoPE/YaRN, but
# already has its own equivalent mechanism for exactly this staged
# growth: --position-norm-length stays at 4096 (>= the eventual 2048
# target, unchanged from before) so position encodings don't need
# rescaling later, and --smollm3-allow-context-growth + --resume-checkpoint
# is the existing, already-implemented path for resuming a checkpoint at
# a longer sequence length. gator/smollm3_recipe.py's stage_for_update
# already scales the data-mixture stage-transition update thresholds
# proportionally to sequence_length, so no separate correction needed
# there. At context=512 each update covers only 512*32=16,384 tokens
# vs. 2048's 65,536 (4x fewer) -- --steps raised 300,000 -> 1,200,000
# (exactly 4x) so this phase covers the same ~19.66B tokens (~76% of
# the 26B full-recipe budget) that 300k updates covered at context=2048,
# rather than a shorter initial phase.
#
# --batch-size 8 --gradient-accumulation-steps 4 (up from 2/16, same
# effective batch=32): batch*sequence_length is unchanged (8*512 ==
# 2*2048 == 4096 tokens/microstep), so this shouldn't meaningfully
# raise peak VRAM over the original context=2048 config -- and
# receiver_memory_backend=linear_scan is genuinely O(context), not
# quadratic, so if anything the shorter sequence length should leave
# more headroom, not less. Chosen to cut per-update dispatch overhead
# (16 micro-steps -> 4), same lever that ~doubled the toy-model sanity
# check's throughput once VRAM headroom made it viable.
# (tightened from the 5,000 default -- at ~0.15 u/s that's ~55min
# between checkpoints instead of ~9h, much safer worst-case restart
# loss for an unattended multi-day run) and the fixed 10-prompt eval
# also every 500 updates.
#
# Checkpoint disk footprint is now bounded, not unbounded: the training
# loop only ever writes latest.pt (every interval) and best.pt (only
# when that interval's average loss is the lowest seen so far, so a
# later degeneration/destabilization can be rolled back to), never
# per-update timestamped snapshots -- ~2 x ~967MB total regardless of
# how long the run goes, not ~1GB x (updates / 500).
#
# Exact fp8_te flag combination matches the earlier proven production
# run (Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md, "Real
# 300-update production run, fp8_te vs. bf16 baseline"): dense_node_backend
# MUST stay folded_bmm (fp8_te is incompatible with dense_node_backend
# monarch, which requires bf16 strictly).
#
# QUEUED, NOT AUTO-LAUNCHED -- run this script by hand (or via Monitor)
# only once given the explicit go-ahead.
#
# To resume after the initial 24h window:
#   add --resume-checkpoint "$CHECKPOINT_DIR/latest.pt" to the command
#   below and rerun this script (or a copy with that flag added).
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_bakeoff_fp8_english50k
export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache
: "${HF_TOKEN:?export HF_TOKEN before running}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="$RUN_DIR/huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

python -u gator_language_train.py \
  --training-recipe smollm3-26b \
  --architecture gator \
  --hf-tokenizer "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/tokenizers/gtr_arw_english_50k" \
  --cells 8 --nodes 4 --d-model 1312 --embedding-dim 2624 --lexical-embedding-dim 256 \
  --relays 2 --neighbors 4 --node-topology asymmetric \
  --cell-input-mode shared_projection --cell-output-mode shared_pool \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --cell-transport-mode broadcast_mix \
  --stewardship-execution dense \
  --receiver-memory-backend linear_scan \
  --receiver-memory-hgrn-scan \
  --response-tokens 16 \
  --steps 1200000 \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --mixed-precision bf16 \
  --node-compute-precision fp8_te \
  --readout-precision fp8_te \
  --broadcast-mix-cuda-graph \
  --stewardship-traffic-cuda-graph \
  --smollm3-peak-lr 8e-5 \
  --smollm3-fixed-eval-every-updates 500 \
  --execution-preset optimized_gpu \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-every-updates 500 \
  --log-every 10 \
  --log-every-seconds 900 \
  --smollm3-stream-cache-dir "$RUN_DIR/smollm3_stream_cache" \
  --seed 6 \
  --smollm3-max-host-ram-gb 24 \
  --smollm3-max-open-streams 10 \
  --smollm3-prefetch-workers 8 \
  --smollm3-token-chunk-tokens 262144 \
  --smollm3-prefetch-tokens 196608 \
  --smollm3-sequence-length 512 \
  > "$RUN_DIR/bakeoff_fp8_english50k.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/bakeoff_fp8_english50k.log"
