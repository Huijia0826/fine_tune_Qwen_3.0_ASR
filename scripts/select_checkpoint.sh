#!/usr/bin/env bash
# Decode val sets with the base model and each saved checkpoint, using reference decoding v2, then score them.
# Usage (on the GPU box): scripts/select_checkpoint.sh runs/lora_v2 [step_600 step_800 ...]
# With checkpoint names, only those are decoded (no base: reuse an earlier base run on unchanged val manifests).
# WITH_ONLINE=1 also decodes online_phase1 val. BASE_CFG selects the decoding config of the base model (default: whisper-small reference decoding v2).
set -euo pipefail
RUN=$1; shift; BASE_CFG=${BASE_CFG:-configs/experiments/reference_v2.yaml}; NAME=$(basename "$RUN"); PY=.venv/bin
CKS=${*:-base $(ls "$RUN/checkpoints" | sort -t_ -k2 -n)}
for ck in $CKS; do
  cfg=/tmp/sel_$ck.yaml
  if [ "$ck" = base ]; then cp "$BASE_CFG" "$cfg"
  else sed "s#adapter_path: null#adapter_path: $RUN/checkpoints/$ck#" "$BASE_CFG" > "$cfg"; fi
  SETS="ascend:data/manifests/ascend.jsonl bczh:data/manifests/benchmark_chinese.jsonl longform:data/manifests/val_longform.jsonl"
  [ -n "${WITH_ONLINE:-}" ] && SETS="$SETS online:data/manifests/online_phase1.jsonl"  # owner-reviewed users' val (2026-11-02)
  for set in $SETS; do
    tag=${set%%:*}; man=${set#*:}; id=sel_${NAME}_${ck}_$tag
    $PY/wf-transcribe --config "$cfg" --manifest "$man" --split val --run-id "$id" > /dev/null 2>&1
    echo "$ck $tag $($PY/wf-score reports/runs/$id 2>/dev/null | grep -o 'official MER [0-9.]*%')"
  done
done
