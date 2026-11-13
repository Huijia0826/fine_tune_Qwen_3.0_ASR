#!/usr/bin/env bash
# Qwen3-ASR-1.7B zero-shot, 25 s chunks (setting chosen on users' val, 2026-11-09): benchmark v1 + owner tests ONCE.
# sbatch -J qwen-c25-test -p batch --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G -t 01:00:00
#   -o runs/qwen3_asr_chunk25_test.log --wrap "bash scripts/run_qwen3_asr_chunk25_test.sh"
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1
PY=.venv-qwen/bin; cfg=configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml; tag=qwen3asr_17b_chunk25
echo "node $(hostname) $(date -Is)"
for set in bv1:benchmark_v1 ownertest:recording_freeform_test v2test:recording_v2_test; do
  id=${set%%:*}_${tag}_zs
  [ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config $cfg --manifest data/manifests/${set#*:}.jsonl \
    --split test --allow-test --run-id $id > runs/$id.out 2>&1
  echo "$id $($PY/wf-score reports/runs/$id 2>/dev/null | grep -o 'official MER [0-9.]*%')"
done
chmod 700 "$HOME/whisperflow"
echo C25_TEST_DONE
