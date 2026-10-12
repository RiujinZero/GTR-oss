#!/usr/bin/env bash
# Direct test of Papers/GTR_internal_findings.md Section 9's documented
# finding: embedding_dim should scale with vocab_size (rule of thumb
# embedding_dim ~= 0.01 x vocab_size, validated up to vocab=20,000),
# not with d_model. At our vocab=50,000, that rule suggests
# lexical_embedding_dim~=500 -- our toy runs have been using 256, about
# half that. Identical to _launch_toy_context256_slots4.sh (paused at
# update=55,990, loss=7.3395) in every other respect --
# --lexical-embedding-dim 500 is the only change -- so the two loss
# curves are directly comparable at matched tokens-seen (both process
# 24,576 tokens/update, unaffected by this change, so update numbers
# line up 1:1 between the two runs).
#
# Caveat from the same doc: the 0.01x rule was only validated up to
# vocab=20,000 (a follow-up sweep extending to 50,000 was queued but
# never run), and a separate finding in the same doc warns that
# independently-"optimal" ratios do not always compose cleanly with the
# rest of a config -- this is a real, testable hypothesis, not a
# guaranteed fix.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_context256_slots4_emb500
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
  --cells 8 --nodes 4 --d-model 128 --embedding-dim 256 --lexical-embedding-dim 500 \
  --relays 2 --neighbors 4 --node-topology asymmetric \
  --cell-input-mode shared_projection --cell-output-mode shared_pool \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --cell-transport-mode broadcast_mix \
  --stewardship-execution dense \
  --receiver-memory-backend linear_scan \
  --receiver-memory-hgrn-scan \
  --response-tokens 4 \
  --smollm3-effective-batch-sequences 96 \
  --steps 800000 \
  --batch-size 96 --gradient-accumulation-steps 1 \
  --mixed-precision bf16 \
  --node-compute-precision fp8_te \
  --readout-precision fp8_te \
  --broadcast-mix-cuda-graph \
  --stewardship-traffic-cuda-graph \
  --smollm3-peak-lr 8e-5 \
  --smollm3-fixed-eval-every-updates 3000 \
  --execution-preset optimized_gpu \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-every-updates 1000 \
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
  >> "$RUN_DIR/toy_context256_slots4_emb500.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_context256_slots4_emb500.log"
