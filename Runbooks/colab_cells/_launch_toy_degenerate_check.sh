#!/usr/bin/env bash
# Sanity check before committing days to the real 300k-update bake-off:
# is this exact pipeline (fp8_te, 50k-vocab tokenizer, full SmolLM3
# mixture, block_causal + linear_scan/hgrn) capable of escaping the
# early "predict the single most-frequent token" collapse ('the the
# the...') seen in the real run's own update=500 eval, or does it stay
# stuck? Identical config to _launch_bakeoff_fp8_english50k_300k.sh in
# every respect except d_model/embedding_dim shrunk so the non-vocab
# "core" network is ~660k params instead of ~66M (via
# _toy_width_sweep.py: d_model=128 -> core=696,512) -- much cheaper per
# step, so many more updates fit in a short wall-clock window.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_degenerate_check
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
  --response-tokens 16 \
  --steps 300000 \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --mixed-precision bf16 \
  --node-compute-precision fp8_te \
  --readout-precision fp8_te \
  --broadcast-mix-cuda-graph \
  --stewardship-traffic-cuda-graph \
  --smollm3-peak-lr 8e-5 \
  --smollm3-fixed-eval-every-updates 2000 \
  --execution-preset optimized_gpu \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-every-updates 500 \
  --log-every 10 \
  --log-every-seconds 300 \
  --smollm3-stream-cache-dir "$RUN_DIR/smollm3_stream_cache" \
  --seed 6 \
  --smollm3-max-host-ram-gb 24 \
  --smollm3-max-open-streams 10 \
  --smollm3-prefetch-workers 8 \
  --smollm3-token-chunk-tokens 262144 \
  --smollm3-prefetch-tokens 196608 \
  --smollm3-sequence-length 2048 \
  --resume-checkpoint "$CHECKPOINT_DIR/latest.pt" \
  >> "$RUN_DIR/toy_degenerate_check.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_degenerate_check.log"
