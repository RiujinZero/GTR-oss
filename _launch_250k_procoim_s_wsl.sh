#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/user/Documents/GTR
ROOT="/mnt/c/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805/slot2_procoim_s_utility_250k_seed0"
mkdir -p "$ROOT/checkpoint"

# Same base config as the best-performing run tonight (slot2_adaptive_full_250k_seed0:
# valid_exact=1.37%, 264 accepts, no comprehensive stall), plus the new
# --procoim-consolidation-mode utility layer: ProcoimSRetentionWindow consolidates
# every 8 accepted-episode examples using Procoim-S's aggregate-utility scorer
# (candidate_utility.evaluate_candidate_utility) against a stratified replay
# reservoir sampled from recent episodes, instead of the anchor hard-audit that
# has been disabled (--procoim-consolidation-interval-examples 0) all night.
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
  --procoim-candidate-directions 256 \
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
  --procoim-directional-basis-rotation-forwards 500 \
  --procoim-past-reference-buffer-size 64 \
  --procoim-past-reference-policy improve \
  --procoim-past-reference-tolerance-fraction 0.0 \
  --procoim-anchor-buffer-size 0 \
  --procoim-anchor-sentinel-count 0 \
  --procoim-consolidation-interval-examples 8 \
  --procoim-consolidation-mode utility \
  --procoim-s-replay-capacity 256 \
  --procoim-s-replay-sample-size 4 \
  --procoim-s-replay-soft-tolerance 0.02 \
  --procoim-s-replay-catastrophic-tolerance 0.20 \
  --procoim-s-minimum-utility 0.0 \
  --procoim-parameter-groups routing receiver_memory stream_mix node_compute readout embedding \
  --seed 0 \
  > "$ROOT/train.stdout.log" 2> "$ROOT/train.stderr.log"
