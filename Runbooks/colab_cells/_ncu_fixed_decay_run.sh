#!/usr/bin/env bash
set -uo pipefail
cd "/mnt/c/Users/user/Documents/projects/GTR"
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
PYBIN=/home/user/gtr_venv_mx310/bin/python

sudo env \
  "PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
  ncu --kernel-name regex:fixed_decay --launch-count 12 --set basic -f \
    -o /tmp/ncu_fixed_decay_report --target-processes all \
    "$PYBIN" Runbooks/colab_cells/_ncu_fixed_decay_target.py
