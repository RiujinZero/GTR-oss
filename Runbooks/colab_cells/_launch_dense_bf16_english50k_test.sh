#!/usr/bin/env bash
# Same shape as _launch_monarch_bf16_baseline.sh (dense folded_bmm,
# d_model=1312, 300 steps) except --hf-tokenizer points at the new
# English-only 50k-vocab tokenizer instead of the default 128,256-vocab
# meta-llama/Llama-3.2-1B. Direct A/B against that existing baseline run
# to see if loss trajectory/throughput hold up with the smaller vocab.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_dense_bf16_english50k_test
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
  --steps 300 \
  --batch-size 2 --gradient-accumulation-steps 16 \
  --mixed-precision bf16 \
  --node-compute-precision bf16 \
  --readout-precision bf16 \
  --broadcast-mix-cuda-graph \
  --stewardship-traffic-cuda-graph \
  --smollm3-peak-lr 8e-5 \
  --execution-preset optimized_gpu \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-every-updates 300 \
  --log-every 10 \
  --log-every-seconds 60 \
  --smollm3-stream-cache-dir "$RUN_DIR/smollm3_stream_cache" \
  --seed 6 \
  --smollm3-max-host-ram-gb 24 \
  --smollm3-max-open-streams 10 \
  --smollm3-prefetch-workers 8 \
  --smollm3-token-chunk-tokens 262144 \
  --smollm3-prefetch-tokens 196608 \
  --smollm3-sequence-length 2048 \
  > "$RUN_DIR/dense_bf16_english50k_test.log" 2>&1
echo "rc=$?"
tail -60 "$RUN_DIR/dense_bf16_english50k_test.log"
