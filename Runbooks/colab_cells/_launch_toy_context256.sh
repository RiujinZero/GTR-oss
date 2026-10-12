#!/usr/bin/env bash
# Follow-up to _launch_toy_degenerate_check.sh (context=2048 toy run,
# stopped deliberately at update=65080/loss=7.4624 to pivot here): same
# d_model=128 toy width, fp8_te, 50k-vocab tokenizer, architecture --
# only context/output_slots/steps/batch change, to isolate how much of
# the long plateau was context=2048 specifically being too much for a
# 660k-core-param model to route well, vs. width/capacity alone.
#
# --smollm3-sequence-length 256 --response-tokens 8 (down from 2048/16).
# --steps 2400000 (up from 300,000) chosen so total tokens seen at
# completion matches exactly: 256*32*2,400,000 == 2048*32*300,000 ==
# 19,660,800,000 tokens either way -- compare against the context=2048
# run at matched tokens-seen checkpoints, not matched update counts,
# since the two configs process very different tokens/update (8,192 vs
# 65,536).
#
# --batch-size 32 --gradient-accumulation-steps 1 (up from 8/4, still
# effective batch=32): eliminates the accumulation loop entirely (1
# microstep/update instead of 4). batch*sequence_length per microstep
# is now 32*256=8,192, actually *less* than the context=2048 run's
# 8*2048=16,384 that already ran comfortably at ~8.3GB VRAM, so this
# should fit with room to spare.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_context256
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
  --cells 8 --nodes 4 --d-model 128 --embedding-dim 256 --lexical-embedding-dim 256 \
  --relays 2 --neighbors 4 --node-topology asymmetric \
  --cell-input-mode shared_projection --cell-output-mode shared_pool \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --cell-transport-mode broadcast_mix \
  --stewardship-execution dense \
  --receiver-memory-backend linear_scan \
  --receiver-memory-hgrn-scan \
  --response-tokens 8 \
  --steps 2400000 \
  --batch-size 32 --gradient-accumulation-steps 1 \
  --mixed-precision bf16 \
  --node-compute-precision fp8_te \
  --readout-precision fp8_te \
  --broadcast-mix-cuda-graph \
  --stewardship-traffic-cuda-graph \
  --smollm3-peak-lr 8e-5 \
  --smollm3-fixed-eval-every-updates 8000 \
  --execution-preset optimized_gpu \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-every-updates 2000 \
  --log-every 10 \
  --log-every-seconds 300 \
  --smollm3-stream-cache-dir "$RUN_DIR/smollm3_stream_cache" \
  --seed 6 \
  --smollm3-max-host-ram-gb 24 \
  --smollm3-max-open-streams 10 \
  --smollm3-prefetch-workers 8 \
  --smollm3-token-chunk-tokens 262144 \
  --smollm3-prefetch-tokens 196608 \
  --smollm3-sequence-length 256 \
  --resume-checkpoint "$CHECKPOINT_DIR/latest.pt" \
  >> "$RUN_DIR/toy_context256.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_context256.log"
