#!/usr/bin/env bash
# Follow-up to _launch_toy_chatonly_smoltalk.sh, per the user: "Let it
# run to 15k then start slots 8." That run plateaued around loss~2.1-2.4/
# acc~0.58-0.60/exact~0.35-0.37 from step 7,000 on (checkpoint preserved
# at ~/gtr_arw_100m_toy_chatonly_smoltalk/checkpoints/, final state
# recorded in the ablation doc). This variant only changes
# --response-tokens 4->8 and the matching --chunk-size 4->8 (chunk-size
# cannot exceed response-tokens) -- mirrors the earlier
# response_tokens=8-vs-4 test done on the broad-mixture smollm3 toy runs,
# now on the recipe that's actually converging. context-tokens dropped
# to 248 (256 total, matching the toy's own context/response split
# convention) since response-tokens grew. Fresh RUN_DIR: response_tokens
# changes model shape, not resumable from the response_tokens=4
# checkpoint. Everything else (vocab-size, architecture flags, bf16-only
# precision, no --compile) identical to the response_tokens=4 run, same
# reasoning: --compile crashed with a torch._dynamo bug and fp8_te+
# CUDA-graphs produced a step-1 NaN on this recipe's code path, both only
# ever validated inside smollm3-26b this session.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_chatonly_smoltalk_slots8
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
  --cells 8 --nodes 4 --relays 4 --neighbors 4 --node-topology asymmetric \
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
  >> "$RUN_DIR/toy_chatonly_smoltalk_slots8.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_chatonly_smoltalk_slots8.log"
