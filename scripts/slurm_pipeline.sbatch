#!/usr/bin/env bash
# Run the whole pipeline (prepare -> train -> select -> test) as one Slurm job.
#   sbatch scripts/slurm_pipeline.sbatch configs/example.yaml [--set key=value ...]
# Adjust the #SBATCH lines to your cluster. Environment variables you may want:
#   QASR_ENV   path of the Python environment with qwen-asr installed (default: .venv)
#   HF_HOME    Hugging Face cache (set HF_HUB_OFFLINE=1 on compute nodes without internet)
#SBATCH -J qasr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/slurm_pipeline.sbatch CONFIG [--set k=v ...]}; shift
QASR_ENV=${QASR_ENV:-.venv}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$QASR_ENV/bin/qasr" pipeline -c "$CONFIG" "$@"
