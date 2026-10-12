#!/usr/bin/env bash
LOG=~/gtr_arw_100m_toy_context256_slots4_relays4/toy_context256_slots4_relays4.log
tail -F -n 0 "$LOG" | awk -F'loss=' '
/loss=/ {
  split($2, a, " ")
  if (a[1] + 0 < 7.2) { print; fflush(); exit }
  next
}
/Traceback|Error|FAILED|Killed|OOM|out of memory|device not ready/ { print; fflush(); exit }
'
