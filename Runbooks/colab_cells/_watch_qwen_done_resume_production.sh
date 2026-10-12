#!/usr/bin/env bash
# Watches for ~/Documents/Qwen_Done.txt (user's "Qwen3.8 benchmarked, GPU
# released" signal), then before resuming the paused 100M production run
# (paused at update=35,420, loss=7.8025), confirms:
#   1. the GPU is genuinely idle (utilization near 0%)
#   2. no llama.cpp server process is still running
#   3. no gator_language_train.py process is already running (avoid a
#      double-launch if this watcher somehow fires twice)
# Only resumes if all three hold; otherwise prints what blocked it and
# exits without touching training.
set -uo pipefail

DONE_FILE="/mnt/c/Users/user/Documents/Qwen_Done.txt"
LAUNCH_WRAPPER="/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_bg_launch_production_d1504_slots8_relays2.sh"

echo "$(date -Is) watching for $DONE_FILE ..."
while [ ! -f "$DONE_FILE" ]; do
  sleep 15
done

echo "$(date -Is) === Qwen_Done.txt appeared ==="
cat "$DONE_FILE"
echo

GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
echo "GPU utilization: ${GPU_UTIL}%"

BLOCKED=0

if [ "${GPU_UTIL:-100}" -gt 5 ]; then
  echo "BLOCKED: GPU utilization is ${GPU_UTIL}%, not idle"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
  BLOCKED=1
fi

if pgrep -afi 'llama' > /dev/null 2>&1; then
  echo "BLOCKED: a llama-related process is still running"
  pgrep -afi 'llama'
  BLOCKED=1
fi

if pgrep -f gator_language_train.py > /dev/null 2>&1; then
  echo "BLOCKED: gator_language_train.py is already running"
  pgrep -af gator_language_train.py
  BLOCKED=1
fi

if [ "$BLOCKED" -ne 0 ]; then
  echo "$(date -Is) Not resuming automatically -- conditions not clear. Re-run this script by hand once resolved."
  exit 1
fi

echo "$(date -Is) GPU idle, no llama process, no existing training process -- resuming production run"
bash "$LAUNCH_WRAPPER"
