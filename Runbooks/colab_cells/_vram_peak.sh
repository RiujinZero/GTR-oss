#!/usr/bin/env bash
awk '{print $2}' /home/user/english50k_vram_samples.log | sort -n > /tmp/vram_sorted.txt
echo "min:"; head -3 /tmp/vram_sorted.txt
echo "max:"; tail -3 /tmp/vram_sorted.txt
