#!/usr/bin/env bash
# Re-score val runs against the current (calibrated, 2026-11-02) val manifests; predictions are not re-decoded.
# Usage: scripts/rescore_val.sh <run_id prefix> ...   e.g. scripts/rescore_val.sh zs_large_v3 sel_lora_large_v3_A_step_600
set -euo pipefail
for p in "$@"; do
  for set in "ascend:data/manifests/ascend.jsonl" "bczh:data/manifests/benchmark_chinese.jsonl" "longform:data/manifests/val_longform.jsonl"; do
    tag=${set%%:*}; man=${set#*:}; r=reports/runs/${p}_$tag
    [ -d "$r" ] && echo "$p $tag $(uv run wf-score "$r" --manifest "$man" 2>/dev/null | grep -o 'official MER [0-9.]*%')"
  done
done
