#!/usr/bin/env bash
set -uo pipefail
cd "/mnt/c/Users/user/Documents/projects/GTR"
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/user/gtr_venv_mx310/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.8/lib64
source ~/gtr_venv/bin/activate
export HF_HOME=/home/user/gtr_arw_100m_smollm3_naked/huggingface_cache
: "${HF_TOKEN:?export HF_TOKEN before running}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTHONUNBUFFERED=1

python -u Runbooks/colab_cells/_build_english_tokenizer.py \
  --target-bytes 9663676416 \
  --vocab-size 50000 \
  --corpus-path /home/user/gtr_english_tokenizer_corpus.txt \
  --output-dir "/mnt/c/Users/user/Documents/projects/GTR/Runbooks/tokenizers/gtr_arw_english_50k" \
  --stage 1 \
  > /home/user/build_english_tokenizer.log 2>&1
echo "rc=$?"
tail -80 /home/user/build_english_tokenizer.log
