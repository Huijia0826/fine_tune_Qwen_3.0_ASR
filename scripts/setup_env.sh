#!/usr/bin/env bash
# Create a Python environment with everything needed to train and decode (run on a Linux GPU machine).
#   bash scripts/setup_env.sh [ENV_DIR]          (default .venv)
# Uses uv when available, else python -m venv + pip. Set PIP_INDEX_URL for a mirror, e.g.
#   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash scripts/setup_env.sh
set -euo pipefail
ENV_DIR=${1:-.venv}
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 "$ENV_DIR"
  uv pip install --python "$ENV_DIR/bin/python" -e ".[train,zh,dev]"
else
  python3 -m venv "$ENV_DIR"
  "$ENV_DIR/bin/pip" install -U pip
  "$ENV_DIR/bin/pip" install -e ".[train,zh,dev]"
fi
"$ENV_DIR/bin/python" -c "import qwen_asr, transformers, torch; print('qwen_asr ok | transformers', transformers.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
