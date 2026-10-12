#!/bin/bash
LOG=/home/user/production_memory_check3.log
for i in $(seq 1 55); do
  if grep -qE "^done$|OOM|Traceback" "$LOG" 2>/dev/null; then
    echo "POLL: finished after ${i}0s"
    break
  fi
  sleep 10
done
tail -20 "$LOG"
