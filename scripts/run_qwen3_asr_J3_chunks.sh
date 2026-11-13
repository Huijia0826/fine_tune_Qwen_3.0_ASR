#!/usr/bin/env bash
# J3 long-audio decoding check (2026-11-10): J3 was trained on whole 30–180 s clips but decoded with 25 s chunks.
# Compare max_chunk_sec 25 / 60 / 1200 (= whole clip) on VALIDATION sets only (users' val, val_longform); the setting
# for a single benchmark v1 run is chosen from these numbers.
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-qwen/bin; CK=runs/qwen3asr_J3/checkpoint-600; CFG=configs/experiments/qwen3asr_ctxJ2_chunk25.yaml
for chunk in 60 1200; do
  cfg=/tmp/j3chunk_$chunk.yaml
  sed -e "s#checkpoint_path: null.*#checkpoint_path: $CK#" -e "s#max_chunk_sec: 25.*#max_chunk_sec: $chunk#" $CFG > $cfg
  for set in bczh:benchmark_chinese online:online_phase1 longform:val_longform; do
    id=sel_qwen3asr_J3_step_600_chunk${chunk}_${set%%:*}
    [ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config $cfg --manifest data/manifests/${set#*:}.jsonl \
      --split val --run-id $id > runs/$id.out 2>&1
    echo "$id $($PY/wf-score reports/runs/$id 2>/dev/null | grep -o 'official MER [0-9.]*%')"
  done
done
sed -e "s#checkpoint_path: null.*#checkpoint_path: $CK#" $CFG > /tmp/j3chunk_25.yaml
id=sel_qwen3asr_J3_step_600_chunk25_longform
[ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config /tmp/j3chunk_25.yaml \
  --manifest data/manifests/val_longform.jsonl --split val --run-id $id > runs/$id.out 2>&1
echo "$id $($PY/wf-score reports/runs/$id 2>/dev/null | grep -o 'official MER [0-9.]*%')"
chmod 700 "$HOME/whisperflow"
echo CHUNKS_DONE
