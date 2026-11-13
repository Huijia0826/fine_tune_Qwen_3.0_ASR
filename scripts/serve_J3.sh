#!/usr/bin/env bash
# Trial server for qwen3asr_J3 checkpoint-600 on an H100 compute node (2026-11-12). Run from the login node:
#   srun --partition=debug --gres=gpu:h100:1 --cpus-per-task=22 --mem=64G --time=12:00:00 bash scripts/serve_J3.sh
# Binds 0.0.0.0:6006 on the compute node; from the Mac: ssh -N -L 7860:<node>:6006 h100-coding → http://localhost:7860
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WF_SERVE_CONFIG=configs/serving/serve_J3_600.yaml WF_SERVE_LOG=data/private/serve_logs/J3/requests.jsonl
echo "node $(hostname) $(date -Is) port ${PORT:-6006}"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
exec .venv-qwen/bin/uvicorn whisperflow.serving.server:app --host 0.0.0.0 --port "${PORT:-6006}"
