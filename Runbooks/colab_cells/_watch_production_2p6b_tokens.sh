#!/usr/bin/env bash
# One-shot watcher for the 2.6B-token milestone -- matches NanoWhale-100M's
# training token count, for a side-by-side comparison point (user, 2026-08-18).
LOG=~/gtr_arw_100m_production_d1504_slots8_relays2/production_d1504_slots8_relays2.log
tail -F -n 0 "$LOG" | awk -F'tokens=' '
/tokens=/ {
  split($2, a, " ")
  if (a[1] + 0 >= 2600000000) { print; fflush(); exit }
  next
}
/Traceback|Error|FAILED|assert|Killed|OOM|out of memory|device not ready|FloatingPointError/ { print; fflush(); exit }
'
