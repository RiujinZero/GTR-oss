#!/usr/bin/env bash
# relays=2 variant of _launch_toy_chatonly_smoltalk_slots8.sh
# (response_tokens=8 chat-only run, stopped by user instruction at
# step=10,000/200,000, loss=2.6345, exact=0.270, quality already sliding
# from its step~7,000 peak -- new "r r"/"s s s" garbled-repeat pattern in
# formed samples alongside the loss/coverage reversal). User: "Cancel
# slots 16, do relay = 2 instead at slots 4 and 8" -- relays=4 was only
# ever tested against the broad smollm3 mixture, never directly compared
# to the GTR-ARW default relays=2 in the chat-only setting. Only
# --relays 4->2 changed; everything else (response-tokens=8, chunk-size=8,
# vocab-size 2048, architecture flags, bf16-only/no-compile precision
# fixes) identical. Fresh RUN_DIR.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_chatonly_smoltalk_slots8_relays2
export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache
export PYTHONUNBUFFERED=1

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

RESUME_ARGS=()
if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
  RESUME_ARGS=(--resume-checkpoint "$CHECKPOINT_DIR/latest.pt")
fi

python -u gator_language_train.py \
  --training-recipe standard \
  --data /home/user/gtr_data/smoltalk_everyday.jsonl \
  --architecture gator \
  --vocab-size 2048 \
  --context-tokens 248 --response-tokens 8 --chunk-size 8 \
  --d-model 128 --embedding-dim 256 --lexical-embedding-dim 256 \
  --cells 8 --nodes 4 --relays 2 --neighbors 4 --node-topology asymmetric \
  --cell-input-mode shared_projection --cell-output-mode shared_pool \
  --block-causal --block-size 16 --position-norm-length 4096 \
  --cell-transport-mode broadcast_mix \
  --stewardship-execution dense \
  --receiver-memory-backend linear_scan \
  --receiver-memory-hgrn-scan \
  --batch-size 32 --gradient-accumulation-steps 1 \
  --mixed-precision bf16 \
  --learning-rate 0.003 \
  --scheduler cosine \
  --steps 200000 \
  --execution-preset optimized_gpu_safe \
  --dense-kernel-padding 8 --dense-node-backend folded_bmm \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-every-updates 1000 \
  --log-every 500 \
  --logging-slim \
  --seed 6 \
  "${RESUME_ARGS[@]}" \
  >> "$RUN_DIR/toy_chatonly_smoltalk_slots8_relays2.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_chatonly_smoltalk_slots8_relays2.log"
