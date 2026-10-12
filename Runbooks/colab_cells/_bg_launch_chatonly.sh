#!/usr/bin/env bash
RUN_DIR=~/gtr_arw_100m_toy_chatonly_smoltalk
LOG="$RUN_DIR/toy_chatonly_smoltalk.log"
nohup bash "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_launch_toy_chatonly_smoltalk.sh" > /home/user/toy_chatonly_smoltalk_launch.log 2>&1 &
echo "launched pid $!"
while [ ! -f "$LOG" ]; do sleep 1; done
tail -F "$LOG"
