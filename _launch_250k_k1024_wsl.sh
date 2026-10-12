#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/user/Documents/GTR
ROOT="/mnt/c/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/slot2_k1024_250k_seed0"
mkdir -p "$ROOT/checkpoint"

# Isolates the K-ceiling variable against the clean past-reference winner
# (slot2_adaptive_full_250k_seed0: 264 accepts, valid_exact=1.37%, no stall) --
# consolidation stays OFF here (unlike the concurrently-running
# slot2_procoim_s_utility_250k_seed0) so a result either way isn't confounded
# by two changed variables at once.
#
# --procoim-candidate-directions 256 -> 1024 ("give it more graph space",
# matching AdamW's full-parameter-space access). Adaptive K still starts at
# 16 and only grows on failure, so most groups stay cheap most of the time --
# this just raises how far a genuinely struggling group is allowed to widen.
#
# --procoim-directional-basis-rotation-forwards 500 -> replaced with
# --procoim-directional-basis-rotation-stall-banks 8. At K=1024 one bank
# costs 2*1024=2048 forwards, already 4x past the old 500-forward rotation
# threshold in a single attempt -- the same interaction diagnosed as
# crippling the reference-guard runs at K=256 (basis rotates almost every
# attempt for any group that widens, regardless of whether it's making
# progress). Stall-based rotation only rotates a group's basis after 8
# consecutive failed banks, decoupling "how wide a group searches" from
# "how often its subspace gets reshuffled."
exec /home/user/.local/bin/uv run --with torch --index https://download.pytorch.org/whl/cpu -- python -u gator_linear_prefix_train.py \
  --optimizer procoim \
  --steps 250000 \
  --batch-size 144 \
  --eval-batch-size 512 \
  --prefix-episode-mode fresh \
  --length 2 \
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
  --procoim-slot-objective-reduction batch_mean \
  --procoim-reduction-fraction 0.0001 \
  --procoim-adaptive-reduction-fraction \
  --procoim-reduction-fraction-decay-rate 0.1 \
  --procoim-minimum-reduction-fraction 1e-6 \
  --procoim-candidate-directions 1024 \
  --procoim-adaptive-candidate-directions \
  --procoim-minimum-candidate-directions 16 \
  --procoim-candidate-directions-growth 2.0 \
  --procoim-max-candidate-banks 16 \
  --procoim-max-accepted-updates 64 \
  --procoim-probe-scale 0.03 \
  --procoim-directional-adam \
  --procoim-directional-adam-learning-rate 0.03 \
  --procoim-directional-basis-mode rotating_factorized \
  --procoim-directional-factor-rank 1 \
  --procoim-directional-basis-rotation-stall-banks 8 \
  --procoim-past-reference-buffer-size 64 \
  --procoim-past-reference-policy improve \
  --procoim-past-reference-tolerance-fraction 0.0 \
  --procoim-anchor-buffer-size 0 \
  --procoim-anchor-sentinel-count 0 \
  --procoim-consolidation-interval-examples 0 \
  --procoim-parameter-groups routing receiver_memory stream_mix node_compute readout embedding \
  --seed 0 \
  > "$ROOT/train.stdout.log" 2> "$ROOT/train.stderr.log"
