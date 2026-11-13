#!/usr/bin/env bash
# Qwen3 instruct post-processing of our ASR outputs with the production cleanup prompts (2026-11-10).
#   sbatch -J pp-qwen3 … --wrap "bash scripts/run_pp_qwen3.sh"
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_DISABLE_XET=1  # xet downloads kept failing with SSL EOF (2026-11-10)
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-4B-Instruct-2507', allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt','vocab.json']))"
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/postprocess/llm_cleanup.py \
  bv1_qwen3asr_J3_step_600_ctx bv1_qwen3asr_J3_step_600_noctx dict_qwen3asr_J3_step_600_ctx \
  sel_qwen3asr_J3_step_600_bczh sel_qwen3asr_J3_step_600_online bv1_lora_large_v3_I2_step_600 \
  ownertest_qwen3asr_J3_step_600_noctx v2test_qwen3asr_J3_step_600_ctx
chmod 700 "$HOME/whisperflow"
echo PP_DONE
