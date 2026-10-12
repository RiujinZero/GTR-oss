#!/usr/bin/env bash
# Fix for task #35 (Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md):
# every chat-only smoltalk_everyday.jsonl run this session crept toward
# or past a documented corpus-repetition collapse (this exact 2,260-
# conversation corpus is the tiny/highly-repeated one flagged in
# Papers/GATOR_PLATEAU_INSTABILITY_STUDY.md sec 7.1). That study's own
# fix was a larger/fresher corpus, not a hyperparameter change --
# --lmsys-data adds ~1M-conversation LMSYS-Chat-1M
# (~/gtr_data/lmsys_chat_1m.jsonl, copied from Google Drive
# C:\path\to\gdrive\GTR\training_data\lmsys_chat_1m.jsonl, same shared
# USER:/ASSISTANT: text-JSONL schema as smoltalk_everyday.jsonl,
# confirmed by inspection) as a streamed mixture source alongside the
# primary smoltalk_everyday.jsonl corpus (still supplies the stable
# validation benchmark per the trainer's own convention).
#
# --lmsys-load-mode stream (not the default `memory`) -- a 2.25GB/~1M-
# conversation file loaded fully into RAM is unnecessary and wasteful
# when streaming reads only what training actually consumes;
# --smollm3-max-host-ram-gb-style caution, applied generically here.
# --streaming-weight left at its default (falls back to --lmsys-weight's
# default 0.9 -- 90% LMSYS / 10% primary corpus per training example),
# an established default in this codebase, not a new choice.
#
# Same 1,222,848-param GTR-ARW-shaped core as every other chat-only run
# this session (d-model=128, cells=8, nodes=4, relays=4,
# neighbors=4, broadcast_mix/shared_projection/shared_pool,
# block-causal+linear_scan+hgrn-scan, vocab-size=2048,
# context-tokens=252/response-tokens=4/chunk-size=4, bf16-only/
# no-compile precision -- same reasoning as every prior chat-only
# script this session: --compile hit a real torch._dynamo bug and
# fp8_te+CUDA-graphs produced a step-1 NaN on this recipe's code path).
# Fresh RUN_DIR: the from-scratch CompactTokenizer refits on the new,
# much larger text distribution, so this isn't resumable from any prior
# chat-only checkpoint regardless of matching architecture flags.
#
# --steps 200000 kept as-is (not reduced): at ~1M LMSYS conversations
# plus the primary corpus, the repetition-epoch math that caused the
# collapse on the tiny corpus no longer applies at any step count this
# toy width would plausibly reach.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_chatonly_smoltalk_lmsys
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
  --lmsys-data /home/user/gtr_data/lmsys_chat_1m.jsonl \
  --lmsys-load-mode stream \
  --architecture gator \
  --vocab-size 2048 \
  --context-tokens 252 --response-tokens 4 --chunk-size 4 \
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
  >> "$RUN_DIR/toy_chatonly_smoltalk_lmsys.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_chatonly_smoltalk_lmsys.log"
