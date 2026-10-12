#!/usr/bin/env bash
set -uo pipefail
cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
export TORCHINDUCTOR_CACHE_DIR=~/torchinductor_cache
export PATH="/usr/lib/wsl/lib:$PATH"
source ~/gtr_venv/bin/activate

PYLOG=~/gpu_util_compute_loop.log
CSV=~/gpu_util_samples.csv
rm -f "$PYLOG" "$CSV"

python Runbooks/colab_cells/_gpu_util_compute_loop.py > "$PYLOG" 2>&1 &
PYPID=$!

echo "waiting for START_COMPUTE_LOOP..."
while ! grep -q "START_COMPUTE_LOOP" "$PYLOG" 2>/dev/null; do
  if ! kill -0 "$PYPID" 2>/dev/null; then
    echo "python process died before reaching compute loop"
    cat "$PYLOG"
    exit 1
  fi
  sleep 1
done
echo "compute loop started, sampling nvidia-smi for 20s..."

echo "timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm" > "$CSV"
for i in $(seq 1 40); do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm --format=csv,noheader >> "$CSV"
  sleep 0.5
done

wait "$PYPID"
echo "=== python output ==="
cat "$PYLOG"
echo "=== gpu samples ==="
cat "$CSV"
