#!/usr/bin/env bash
# relays=2 variant of _launch_toy_chatonly_smoltalk.sh (response_tokens=4
# chat-only run, stopped by user instruction at step=15,000/200,000,
# loss=2.7340, exact=0.370, mild overfitting after step~7,000). User:
# "Cancel slots 16, do relay = 2 instead at slots 4 and 8" -- relays=4 was
# only ever tested against the broad smollm3 mixture, never directly
# compared to the GTR-ARW default relays=2 in the chat-only setting, so
# this isolates relay count as its own variable here. Only --relays
# 4->2 changed; everything else (vocab-size 2048, architecture flags,
# bf16-only/no-compile precision fixes) identical. Fresh RUN_DIR --
# relay count changes model shape, not resumable from the relays=4
# checkpoint.
#
# This is NOT the same trainer machinery as smollm3-26b -- --training-
# recipe standard has no --hf-tokenizer path at all; it fits its own
# from-scratch CompactTokenizer on the corpus, sized by --vocab-size.
# --vocab-size 2048 matches the prior reference run rather than the 50k
# English tokenizer (a 50k BPE trained on a 1.9MB corpus would be mostly
# unused/degenerate). This means the comparison isolates task/data
# breadth AND swaps the tokenizer/vocab together -- an unavoidable
# confound given the codebase's two separate recipe code paths, not an
# oversight; noted in the ablation doc.
#
# --context-tokens 252 --response-tokens 4 mirrors the toy's 256-token
# sequence split. --batch-size 32 (not 96) and --steps 200000 (not
# 800000/19.66B-token parity) are deliberately NOT token-matched to the
# mixture runs: the corpus is tiny (2,260 conversations, ~1.9MB), so
# hundreds of thousands of steps at batch=96 would mean thousands of
# epochs -- straight memorization, not a meaningful breadth comparison.
# 200,000 steps is already ~8x the prior reference run's 24,000-step
# result (experimental_checkpoints/logs/
# language_smoltalk_62cell_gated_grouped_ctx64_seed0_through_step24000.txt),
# giving real headroom while staying an exploratory-scale run.
# --learning-rate 0.003 is the doc's own stated common override
# (GATOR_LANGUAGE_TRAINER_ARGUMENTS.md default is 0.008); smollm3's 8e-5
# peak-lr was tuned for its own token-equivalent warmup schedule and has
# no meaning on this recipe's plain --scheduler cosine.
#
# --log-every 500 --logging-slim replaces the smollm3-only fixed-prompt
# eval harness -- the standard recipe already runs a real validation
# pass plus context/target/formed samples at every --log-every interval,
# which is the built-in equivalent of the qualitative progress signal
# used throughout this session.
#
# --execution-preset optimized_gpu_safe (no --compile), unlike every
# smollm3-26b toy run this session: a first attempt with --compile
# --compile-mode max-autotune-no-cudagraphs crashed immediately with a
# real torch._dynamo internal error (InternalTorchDynamoError: NameError:
# name 'SYNTHETIC_LOCAL' is not defined) on this recipe's model/loss code
# path, confirming GATOR_LANGUAGE_TRAINER_ARGUMENTS.md's own stated
# caution that compiled runs on this Arrow/gator path aren't currently
# recommended.
#
# --mixed-precision bf16 only -- a second attempt adding
# --node-compute-precision fp8_te --readout-precision fp8_te
# --broadcast-mix-cuda-graph --stewardship-traffic-cuda-graph (the exact
# combination every smollm3-26b toy run used successfully all session)
# produced a hard non-finite gradient at optimizer step 1
# (neighbor_gates nan=8/16), immediately, every time -- those
# optimizations were only ever validated within the smollm3-26b recipe's
# own code path this session, not this one. Dropped entirely here in
# favor of plain bf16 autocast; this run doesn't need production
# throughput, just a correct, stable signal on task/data breadth.
set -uo pipefail

cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate

RUN_DIR=~/gtr_arw_100m_toy_chatonly_smoltalk_relays2
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
  --context-tokens 252 --response-tokens 4 --chunk-size 4 \
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
  >> "$RUN_DIR/toy_chatonly_smoltalk_relays2.log" 2>&1
echo "rc=$?"
tail -80 "$RUN_DIR/toy_chatonly_smoltalk_relays2.log"
