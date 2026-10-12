#!/usr/bin/env bash
# The real 100M-param production run, following the toy-scale
# investigation's conclusion (2026-08-17, Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md):
# "given enough capacity and a broad enough mix to prevent collapse it
# can learn language." Toy sweeps are concluded; this is the actual
# multi-day run.
#
# Supersedes the earlier _launch_bakeoff_fp8_english50k_300k.sh
# (RUN_DIR gtr_arw_100m_bakeoff_fp8_english50k), which only ever reached
# update=710/300,000 before being set aside for the toy detour -- not
# resumable from that checkpoint anyway, since both d_model and
# response_tokens change below (different model shape). Fresh RUN_DIR.
#
# --d-model 1504 --embedding-dim 3008 (up from 1312/2624): the earlier
# script's width predates the switch to the 50k-vocab English tokenizer
# and only reaches 79,678,064 total params at that vocab size, well
# short of the 100M target. Re-sized via
# Runbooks/colab_cells/_100m_width_sweep_dense_50k_relays2.py (same
# bisection approach as the earlier Monarch width sweep, adapted for
# dense folded_bmm backend + 50k vocab + relays=2): d_model=1504 ->
# 100,430,768 total params (+0.43% from 100M), the closest 32-aligned
# width found.
#
# --response-tokens 8 (down from 16): per the user's explicit
# instruction after the toy investigation ("bump width til it hits 100M
# total and then do slots=8"), not carried over from the earlier
# script's un-revisited value.
#
# --relays 2 (unchanged, already correct): per the toy investigation's
# relays=2-vs-4 comparison -- relays=4 only gave a small improvement,
# not worth its ~1.4-1.8x throughput cost at production scale.
#
# Everything else identical to the earlier proven script: context=512
# via --smollm3-sequence-length 512 (--position-norm-length 4096 stays
# >= the eventual 2048 target so position encodings don't need
# rescaling later; --smollm3-allow-context-growth + --resume-checkpoint
# is the existing path for extending context after this phase),
# --batch-size 8 --gradient-accumulation-steps 4 (effective batch=32),
# --steps 1200000 (4x the original 300,000 at context=2048, same
# ~19.66B-token budget), fp8_te node-compute/readout precision + both
# CUDA graphs (proven stable across many hours this session, unrelated
# to the toy detour's separate --training-recipe standard crashes),
# --compile --compile-mode max-autotune-no-cudagraphs, peak-lr=8e-5,
# checkpoint-every-updates=500 (tight, safe restart loss for an
# unattended multi-day run), bounded latest.pt/best.pt checkpointing,
# fixed-prompt eval every 500 updates.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_production_d1504_slots8_relays2
export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache
: "${HF_TOKEN:?export HF_TOKEN before running}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="$RUN_DIR/huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

RESUME_ARGS=()
if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
  RESUME_ARGS=(--resume-checkpoint "$CHECKPOINT_DIR/latest.pt")
fi

python -u gator_language_train.py \
  --training-recipe smollm3-26b \
  --architecture gator \
  --hf-tokenizer "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/tokenizers/gtr_arw_english_50k" \
  --cells 8 --nodes 4 --d-model 1504 --embedding-dim 3008 --lexical-embedding-dim 256 \
  --relays 2 --neighbors 4 --node-topology asymmetric \
  --cell-input-mode shared_projection --cell-output-mode shared_pool \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --cell-transport-mode broadcast_mix \
  --stewardship-execution dense \
  --receiver-memory-backend linear_scan \
  --receiver-memory-hgrn-scan \
  --response-tokens 8 \
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
  "${RESUME_ARGS[@]}" \
  >> "$RUN_DIR/production_d1504_slots8_relays2.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/production_d1504_slots8_relays2.log"
