#!/usr/bin/env bash
# Queued to run right after the in-flight d_model=2944 width run finishes.
# Tests the "keep original width, add nodes" alternative for hitting
# ~100M params: n_nodes=36, node_topology=symmetric (asymmetric is
# hardcoded to exactly 4 nodes), d_model back to the original 1312.
# total_params=100,257,056 per _monarch_nodes_sweep.py.
# Tries batch=2/accum=16 first; auto-falls back to batch=1/accum=32
# on CUDA OOM (same effective batch size either way), matching the
# width-run methodology.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

echo "[queue] waiting for any in-flight gator_language_train.py to finish..."
while pgrep -f gator_language_train.py > /dev/null 2>&1; do
  sleep 10
done
echo "[queue] GPU is free, starting n_nodes=36 run."

export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache
: "${HF_TOKEN:?export HF_TOKEN before running}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

run_attempt () {
  local run_dir="$1"
  local batch="$2"
  local accum="$3"
  local checkpoint_dir="$run_dir/checkpoints"
  export HF_HOME="$run_dir/huggingface_cache"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  mkdir -p "$checkpoint_dir" "$HF_HOME" "$HF_DATASETS_CACHE"

  echo "[queue] === attempt batch=$batch accum=$accum -> $run_dir ==="
  python -u gator_language_train.py \
    --training-recipe smollm3-26b \
    --architecture gator \
    --cells 8 --nodes 36 --d-model 1312 --embedding-dim 2624 --lexical-embedding-dim 256 \
    --relays 2 --neighbors 4 --node-topology symmetric \
    --cell-input-mode shared_projection --cell-output-mode shared_pool \
    --block-causal --block-size 16 --position-norm-length 4096 \
    --cell-transport-mode broadcast_mix \
    --stewardship-execution dense \
    --receiver-memory-backend linear_scan \
    --receiver-memory-hgrn-scan \
    --response-tokens 16 \
    --steps 300 \
    --batch-size "$batch" --gradient-accumulation-steps "$accum" \
    --mixed-precision bf16 \
    --node-compute-precision bf16 \
    --readout-precision bf16 \
    --broadcast-mix-cuda-graph \
    --stewardship-traffic-cuda-graph \
    --smollm3-peak-lr 8e-5 \
    --execution-preset optimized_gpu \
    --compile --compile-mode max-autotune-no-cudagraphs \
    --dense-kernel-padding 8 --dense-node-backend monarch --dense-node-monarch-blocks 32 \
    --checkpoint-dir "$checkpoint_dir" \
    --checkpoint-every-updates 300 \
    --log-every 10 \
    --log-every-seconds 60 \
    --smollm3-stream-cache-dir "$run_dir/smollm3_stream_cache" \
    --seed 6 \
    --smollm3-max-host-ram-gb 24 \
    --smollm3-max-open-streams 10 \
    --smollm3-prefetch-workers 8 \
    --smollm3-token-chunk-tokens 262144 \
    --smollm3-prefetch-tokens 196608 \
    --smollm3-sequence-length 2048 \
    2>&1 | tee -a ~/gtr_arw_100m_monarch_nodes36_queue/queue.log
  return "${PIPESTATUS[0]}"
}

mkdir -p ~/gtr_arw_100m_monarch_nodes36_queue

run_attempt ~/gtr_arw_100m_monarch_nodes36_b2a16 2 16
rc=$?

if [ "$rc" -ne 0 ] && grep -qi "out of memory\|OutOfMemoryError" ~/gtr_arw_100m_monarch_nodes36_queue/queue.log; then
  echo "[queue] b2a16 OOM'd (rc=$rc), falling back to batch=1 accum=32"
  run_attempt ~/gtr_arw_100m_monarch_nodes36_b1a32 1 32
  rc=$?
fi

echo "[queue] final rc=$rc"
