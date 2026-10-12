#!/usr/bin/env bash
RUN_DIR=~/gtr_arw_100m_production_d1504_slots8_relays2
LOG="$RUN_DIR/production_d1504_slots8_relays2.log"
nohup bash "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_launch_production_d1504_slots8_relays2.sh" > /home/user/production_d1504_slots8_relays2_launch.log 2>&1 &
echo "launched pid $!"
while [ ! -f "$LOG" ]; do sleep 1; done
tail -F "$LOG"
