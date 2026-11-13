#!/usr/bin/env bash
# Qwen3-ASR-1.7B zero-shot with 25 s chunks (2026-11-09): users' val only (benchmark v1 stays untouched while
# the decoding setting is being chosen). sbatch -J qwen-chunk25 -p batch --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G
#   -t 01:00:00 -o runs/qwen3_asr_chunk25.log --wrap "bash scripts/run_qwen3_asr_chunk25.sh"
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1
PY=.venv-qwen/bin; cfg=configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml; tag=qwen3asr_17b_chunk25
echo "node $(hostname) $(date -Is)"
for set in bczh:benchmark_chinese online:online_phase1; do
  id=sel_${tag}_base_${set%%:*}
  $PY/wf-transcribe --config $cfg --manifest data/manifests/${set#*:}.jsonl --split val --run-id $id > runs/$id.out 2>&1
  echo "$id $($PY/wf-score reports/runs/$id 2>/dev/null | grep -o 'official MER [0-9.]*%')"
done
echo "$tag users' val pooled: $($PY/python scripts/pick_checkpoint.py $tag base 2>&1 | head -1)"
chmod 700 "$HOME/whisperflow"
echo CHUNK25_DONE
