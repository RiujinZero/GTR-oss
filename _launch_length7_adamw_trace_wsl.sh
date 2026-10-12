#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/user/Documents/GTR
ROOT="/mnt/c/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/slot2_length7_adamw_trace_seed0"
mkdir -p "$ROOT/checkpoint"

# Dedicated AdamW-only (no Procoim) run on the standing 46,701-parameter GTR-L
# architecture (nodes=40/width=8/relays=4/streams=10, same as every Procoim run
# tonight), switched to --length 7 to chase the ~91%/~45k-param AdamW mastery
# result the user remembers but that was never written down anywhere in the
# repo (see Papers/GTRL_PROCOIM_TRACE.md §5.5 for the only AdamW-on-this-
# architecture datapoint on record: batch=64, 15,000 steps, flat at chance --
# too short and almost certainly the wrong learning rate).
#
# --learning-rate 3e-3, not the 8e-3 default: Papers/GATOR_LINEAR_DIAGNOSTIC.md
# section 3 found 8e-3 (inherited from the 1,106-parameter reference model)
# never escapes chance at GTR-L scale, while 3e-3 escapes at step 2500 and
# reaches 1.000 on length=2. Using the known-bad default here would just
# reproduce the old flat result for the wrong reason.
#
# --no-scan-normalise: section 3b of the same doc found the length-7 task is
# literally inexpressible with the normalised decayed scan (it flattens every
# accumulated sum to 1.0 regardless of position) -- this is a hard requirement
# for length=7, not a tuning choice.
#
# --hard-routing: matches every other run tonight, per user instruction.
#
# --adamw-weight-trace-every / --adamw-weight-trace-dir: turns on
# AdamWWeightTracer (gator/diagnostics.py) so weight/update/Adam-moment
# geometry is recorded every 500 steps alongside the normal training log --
# "tracing tags" to watch AdamW's own movements, not just loss/accuracy.
exec /home/user/.local/bin/uv run --with torch --index https://download.pytorch.org/whl/cpu -- python -u gator_linear_prefix_train.py \
  --optimizer adamw \
  --steps 100000 \
  --batch-size 144 \
  --eval-batch-size 512 \
  --prefix-episode-mode fresh \
  --length 7 \
  --streams 10 \
  --nodes 40 \
  --width 8 \
  --relays 4 \
  --degree 6 \
  --buses 4 \
  --mailbox-channels 4 \
  --mailbox-width 8 \
  --basis-rank 4 \
  --address-width 4 \
  --router-rank 8 \
  --conv-kernel 3 \
  --graph-layout small_world \
  --graph-seed 0 \
  --graph-rewire 0.5 \
  --ingress-temperature 1.0 \
  --device cpu \
  --amp-dtype none \
  --log-every 500 \
  --checkpoint-dir "$ROOT/checkpoint" \
  --hard-routing \
  --no-scan-normalise \
  --temperature-start 1.0 \
  --temperature-min 1.0 \
  --temperature-decay 1.0 \
  --balanced-prefix-batches \
  --prefix-loss-vocabulary valid \
  --learning-rate 3e-3 \
  --weight-decay 1e-4 \
  --gradient-clip 1.0 \
  --adamw-weight-trace-every 500 \
  --adamw-weight-trace-dir "$ROOT/checkpoint/adamw_weight_trace" \
  --seed 0 \
  > "$ROOT/train.stdout.log" 2> "$ROOT/train.stderr.log"
