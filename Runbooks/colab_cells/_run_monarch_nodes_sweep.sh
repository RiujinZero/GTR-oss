#!/usr/bin/env bash
set -uo pipefail
cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv_mx310/bin/activate
python "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/colab_cells/_monarch_nodes_sweep.py"
