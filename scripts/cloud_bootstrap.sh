#!/usr/bin/env bash
# Run ON a fresh AutoDL box after scripts/cloud_sync.sh: Python 3.12 venv with the pinned ML dependencies.
# requirements.cloud.txt = `uv export --frozen --extra ml --no-hashes --no-emit-project` (torch 2.14.1+cu130 on the
# RTX PRO 6000 / RTX 5090 boxes). PyPI via the Aliyun mirror; torch from the PyTorch CUDA 13.0 index.
set -euo pipefail
cd /root/autodl-tmp/wt
command -v uv >/dev/null || pip install -i https://mirrors.aliyun.com/pypi/simple/ uv
uv venv --python 3.12 .venv
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -r requirements.cloud.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)"
