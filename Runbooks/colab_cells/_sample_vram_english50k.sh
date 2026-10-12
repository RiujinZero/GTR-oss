#!/usr/bin/env bash
# Polls nvidia-smi every 15s until both english50k smoke tests finish
# (dense then monarch), so peak VRAM isn't missed like the earlier
# activation-checkpointing run.
OUT=/home/user/english50k_vram_samples.log
: > "$OUT"
while true; do
  ts=$(date +%s)
  sample=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)
  echo "$ts $sample" >> "$OUT"
  if [ -f /home/user/gtr_arw_100m_monarch_bf16_english50k_test/monarch_bf16_english50k_test.log ] \
     && grep -q "update=300/300" /home/user/gtr_arw_100m_monarch_bf16_english50k_test/monarch_bf16_english50k_test.log 2>/dev/null; then
    echo "monarch run reached update=300/300, stopping sampler" >> "$OUT"
    break
  fi
  if ! pgrep -f gator_language_train.py > /dev/null 2>&1 \
     && [ -f /home/user/gtr_arw_100m_monarch_bf16_english50k_test/monarch_bf16_english50k_test.log ]; then
    echo "no training process running and monarch log exists but no update=300/300, stopping sampler (possible crash)" >> "$OUT"
    break
  fi
  sleep 15
done
