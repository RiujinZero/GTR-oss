#!/usr/bin/env bash
set -uo pipefail

BASE="/mnt/c/Users/user/AppData/Local/Temp/gtrl_procoim_fix_20260805"
DRIVER_LOG="$BASE/transformer_scaling_ladder.driver.log"
QUEUE_LOG="$BASE/transformer_causal_queue.log"

# Waits for the 2M->4M->8M->16M->10k/200k scaling ladder driver
# (_launch_transformer_scaling_ladder_wsl.sh) to finish, then launches the
# autoregressive causal-decode 10k/200k run
# (_launch_transformer_causal_10k_200k_wsl.sh). Polls every 60s rather than
# editing the already-running ladder driver script in place (unsafe --
# bash may have already buffered the whole script into memory).
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for ladder completion" >> "$QUEUE_LOG"
while true; do
  if grep -q "=== ladder complete ===" "$DRIVER_LOG" 2>/dev/null; then
    break
  fi
  sleep 60
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ladder complete, launching causal run" >> "$QUEUE_LOG"
exec /mnt/c/Users/user/Documents/GTR/_launch_transformer_causal_10k_200k_wsl.sh
