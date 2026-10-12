#!/usr/bin/env bash
# Runs both 300-step English-50k-vocab smoke tests sequentially: dense
# baseline first, then Monarch. Each writes its own log; this wrapper
# also tees combined progress into one file for easy monitoring.
set -uo pipefail

echo "[queue] === dense (english50k) ==="
bash "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_launch_dense_bf16_english50k_test.sh"
echo "[queue] dense done"

echo "[queue] === monarch (english50k) ==="
bash "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_launch_monarch_bf16_english50k_test.sh"
echo "[queue] monarch done"

echo "[queue] both smoke tests complete"
