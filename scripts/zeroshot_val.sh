#!/usr/bin/env bash
# Zero-shot bigger Whisper models on the three val sets, reference decoding v2 settings (2026-10-31).
# Usage (on the GPU box): scripts/zeroshot_val.sh medium large_v3 large_v3_turbo
set -euo pipefail
PY=.venv/bin
for tag in "$@"; do
  for set in "ascend:data/manifests/ascend.jsonl" "bczh:data/manifests/benchmark_chinese.jsonl" "longform:data/manifests/val_longform.jsonl"; do
    s=${set%%:*}; man=${set#*:}; id=zs_${tag}_$s
    $PY/wf-transcribe --config configs/experiments/zeroshot_$tag.yaml --manifest "$man" --split val --run-id "$id" > /dev/null 2>&1
    echo "$tag $s $($PY/wf-score reports/runs/$id 2>/dev/null | grep -o 'official MER [0-9.]*%') $(grep -o '"batch_rtf_this_invocation": [0-9.]*' reports/runs/$id/run_meta.json)"
  done
done
