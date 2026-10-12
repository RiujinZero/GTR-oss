#!/usr/bin/env bash
RUN_DIR=~/gtr_arw_100m_toy_context256_slots4_relays4
LOG="$RUN_DIR/toy_context256_slots4_relays4.log"
nohup bash "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_launch_toy_context256_slots4_relays4.sh" > /home/user/toy_context256_slots4_relays4_launch.log 2>&1 &
echo "launched pid $!"
while [ ! -f "$LOG" ]; do sleep 1; done
tail -F "$LOG"
