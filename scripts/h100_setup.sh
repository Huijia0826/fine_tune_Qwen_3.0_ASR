#!/usr/bin/env bash
# Build the x86_64 environment ON a compute node of the H100 cluster (login node is aarch64, compute nodes x86_64).
#   srun -p debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=32G -t 01:00:00 bash scripts/h100_setup.sh > setup.log 2>&1
set -euo pipefail
W=$HOME/whisperflow; cd "$W"
MIRROR=https://mirrors.aliyun.com/pypi/simple/
echo "node $(hostname) $(uname -m); python3 $(python3 --version 2>&1)"
# x86_64 uv binary, unpacked on the login node from the PyPI wheel (compute-node python3 has no pip):
#   pip download uv==0.13.6 --platform manylinux_2_17_x86_64 --only-binary=:all: --no-deps -d .uvx86/wheel
UV="$W/.uvx86/uv-0.13.6.data/scripts/uv"; chmod +x "$UV"; export UV_CACHE_DIR="$W/.uvcache" UV_PYTHON_INSTALL_DIR="$W/.uvpython"
"$UV" --version
"$UV" venv --python 3.12 .venv || "$UV" venv --python python3 .venv  # fall back to the system Python if 3.12 can't be fetched
UV_INDEX_URL=$MIRROR "$UV" pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match -r requirements.cloud.txt
"$UV" pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'peft', peft.__version__)"
echo SETUP_DONE
