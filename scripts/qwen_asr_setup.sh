#!/usr/bin/env bash
# Qwen3-ASR zero-shot (2026-11-09): separate .venv-qwen, because qwen-asr 0.0.6 pins transformers==4.57.6 while .venv
# runs transformers 5.x. Run on a compute node (x86_64); weights are downloaded on the login node (has internet).
#   ssh h100-coding 'cd ~/whisperflow && srun -p debug --cpus-per-task=8 --mem=32G -t 01:00:00 bash scripts/qwen_asr_setup.sh'
set -euo pipefail
W=$HOME/whisperflow; cd "$W"
UV="$W/.uvx86/uv-0.13.6.data/scripts/uv"; export UV_CACHE_DIR="$W/.uvcache" UV_PYTHON_INSTALL_DIR="$W/.uvpython"
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
echo "node $(hostname) $(uname -m) $(date -Is)"
[ -x .venv-qwen/bin/python ] || "$UV" venv --python 3.12 .venv-qwen
"$UV" pip install --python .venv-qwen/bin/python "qwen-asr==0.0.6" -e . 2>&1 | tail -5
.venv-qwen/bin/python -c "import qwen_asr, transformers, torch; print('qwen_asr ok, transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
chmod 700 "$W"
echo SETUP_DONE
