#!/usr/bin/env bash
# Final evals with reference decoding v2 (2026-10-30). Fallback only changes long-form (>30 s) decoding, so the
# baseline and lora_v1 re-decode just benchmark v1's 69 long clips and reuse their short-clip predictions.
set -euo pipefail
export HF_HUB_OFFLINE=1
T="uv run wf-transcribe --split test --allow-test"
merge() {  # merge <old run> <long-only run> <new run>: long-clip predictions replace the old ones
  mkdir -p reports/runs/$3
  python3 - "$@" <<'PY'
import json, sys
old, new_long, out = sys.argv[1:4]
L = {json.loads(l)["utterance_id"]: l for l in open(f"reports/runs/{new_long}/predictions.jsonl")}
with open(f"reports/runs/{out}/predictions.jsonl", "w") as f:
    for l in open(f"reports/runs/{old}/predictions.jsonl"):
        f.write(L.get(json.loads(l)["utterance_id"], l))
meta = json.load(open(f"reports/runs/{new_long}/run_meta.json"))
meta.update(run_id=out, manifest="data/manifests/benchmark_v1.jsonl",
            note=f"short clips from {old} (unchanged decoding path); clips > 30 s from {new_long}")
json.dump(meta, open(f"reports/runs/{out}/run_meta.json", "w"), indent=2)
PY
  uv run wf-score reports/runs/$3 --manifest data/manifests/benchmark_v1.jsonl | grep -i "official MER"
}
$T --config configs/experiments/reference_v2.yaml --manifest /tmp/benchmark_v1_long.jsonl --run-id bv1_long_base_ref2
merge benchmark_v1_fixed bv1_long_base_ref2 benchmark_v1_base_ref2
$T --config configs/experiments/reference_v2_lora_v1.yaml --manifest /tmp/benchmark_v1_long.jsonl --run-id bv1_long_lora_v1_ref2
merge benchmark_v1_lora_v1 bv1_long_lora_v1_ref2 benchmark_v1_lora_v1_ref2
$T --config configs/experiments/reference_v2_lora_v2.yaml --run-id benchmark_v1_lora_v2_ref2
uv run wf-score reports/runs/benchmark_v1_lora_v2_ref2 | grep -i "official MER"
$T --config configs/experiments/reference_v2_lora_v2.yaml --manifest data/manifests/ascend.jsonl --run-id ascend_test_lora_v2_ref2
uv run wf-score reports/runs/ascend_test_lora_v2_ref2 | grep -i "official MER"
echo ALL_DONE
