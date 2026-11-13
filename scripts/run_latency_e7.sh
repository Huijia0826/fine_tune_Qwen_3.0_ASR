#!/usr/bin/env bash
# E7 latency benchmark on the H100 cluster (2026-11-02): batch 1 over users' val (125 clips), six variants on one GPU.
#   ssh h100-coding 'cd ~/whisperflow && nohup bash -lc "srun -p debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G \
#     -t 03:00:00 bash scripts/run_latency_e7.sh" > runs/latency_e7.log 2>&1 < /dev/null &'
# faster-whisper lives in its own .venv-ct2: CTranslate2 wheels need CUDA 12 cuBLAS/cuDNN 9, while .venv's torch
# ships CUDA 13 libraries. Steps already done are skipped, so the script can be rerun after a dropped job.
set -euo pipefail
W=$HOME/whisperflow; cd "$W"
UV="$W/.uvx86/uv-0.13.6.data/scripts/uv"; export UV_CACHE_DIR="$W/.uvcache" UV_PYTHON_INSTALL_DIR="$W/.uvpython"
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ HF_HOME=$W/hf_cache
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # should be empty: other processes skew latency

# 1. environments
"$UV" pip install --python .venv/bin/python --no-deps -e .   # registers the new wf-latency entry point
.venv/bin/python -c "import ctranslate2" 2>/dev/null || "$UV" pip install --python .venv/bin/python ctranslate2  # converter only
if [ ! -x .venv-ct2/bin/python ]; then
  "$UV" venv --python 3.12 .venv-ct2 || "$UV" venv --python python3 .venv-ct2
  "$UV" pip install --python .venv-ct2/bin/python -e . faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
fi
# namespace packages: no __file__, use __path__
LIBS=$(.venv-ct2/bin/python -c 'import nvidia.cublas.lib, nvidia.cudnn.lib; print(":".join(m.__path__[0] for m in (nvidia.cublas.lib, nvidia.cudnn.lib)))')
.venv-ct2/bin/python -c "import faster_whisper, ctranslate2; print('faster-whisper', faster_whisper.__version__, 'ct2', ctranslate2.__version__, 'cuda devices', ctranslate2.get_cuda_device_count())"

# 2. turbo weights (reference variant), pinned revision
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("openai/whisper-large-v3-turbo", revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("turbo", p)
PY
export HF_HUB_OFFLINE=1

# 3. merge C step_600 + convert to CTranslate2 (float16 weights; int8 is chosen at load time)
[ -e runs/ct2/C600_fp16/model.bin ] || .venv/bin/python scripts/export_ct2.py --base openai/whisper-large-v3 \
  --revision 06f233fe06e710322aca913c1bc4249a0d71fce1 --adapter runs/lora_large_v3_C/checkpoints/step_600 \
  --out runs/ct2/C600_fp16
ls -la runs/ct2/C600_fp16

# 4. variants (same clips, same GPU, one after another)
for v in hf_fp32_beam5 hf_bf16_beam5 hf_bf16_greedy ct2_fp16_beam5 ct2_int8f16_beam5 turbo_zs_bf16_beam5 ct2_fp16_greedy; do
  [ -e reports/runs/lat_$v/latency.json ] && { echo "skip $v"; continue; }
  # CUDA 12 libs only for the ct2 runs: exported globally they leaked into later torch runs (cuDNN load failure)
  case $v in ct2_*) PY=.venv-ct2/bin; LDP=$LIBS:${LD_LIBRARY_PATH:-};; *) PY=.venv/bin; LDP=${LD_LIBRARY_PATH:-};; esac
  LD_LIBRARY_PATH=$LDP $PY/wf-latency --config configs/experiments/latency/$v.yaml --run-id lat_$v > runs/lat_$v.out 2>&1 \
    || { echo "FAILED $v (runs/lat_$v.out)"; tail -5 runs/lat_$v.out; continue; }
  $PY/python -c "import json; s = json.load(open('reports/runs/lat_$v/latency.json')); l = s['latency']
print('$v', 'MER', s['official_mer_pct'], ' '.join(f\"{k}:n={b['n']},p50={b['p50_s']},p95={b['p95_s']}\" for k, b in l.items()))"
done
chmod 700 "$W"   # top dir only: nothing below is reachable without it (chmod -R takes minutes here)
echo E7_DONE
