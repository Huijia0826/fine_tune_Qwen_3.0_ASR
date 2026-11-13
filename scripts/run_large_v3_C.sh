#!/usr/bin/env bash
# lora_large_v3_C (2026-11-02): train, decode val sets (incl. online_phase1 val) for C, zero-shot and B 800, pick the
# checkpoint by the predeclared rule (scripts/pick_checkpoint.py), then benchmark v1 once for that checkpoint.
#   nohup bash scripts/run_large_v3_C.sh > runs/large_v3_C.log 2>&1 < /dev/null &
set -euo pipefail
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 BASE_CFG=configs/experiments/zeroshot_large_v3.yaml WITH_ONLINE=1
PY=.venv/bin
[ -e runs/lora_large_v3_C/summary.json ] || $PY/wf-train --config ${C_CONFIG:-configs/train_lora_large_v3_C_cloud.yaml} \
  --run-id lora_large_v3_C > runs/lora_large_v3_C.out 2>&1
echo "trained C: $(tail -1 runs/lora_large_v3_C/log.jsonl)"
for pair in "zs_large_v3:null" "sel_lora_large_v3_B_step_800:runs/lora_large_v3_B/checkpoints/step_800"; do
  id=${pair%%:*}; ad=${pair#*:}
  sed "s#adapter_path: null#adapter_path: $ad#" $BASE_CFG > /tmp/online_$id.yaml
  $PY/wf-transcribe --config /tmp/online_$id.yaml --manifest data/manifests/online_phase1.jsonl --split val \
    --run-id ${id}_online > /dev/null 2>&1
  echo "$id online $($PY/wf-score reports/runs/${id}_online 2>/dev/null | grep -o 'official MER [0-9.]*%')"
done
cks=$(ls runs/lora_large_v3_C/checkpoints | sort -t_ -k2 -n | grep -v -E '^step_(200|400)$' | tr '\n' ' ')
echo "== select C: $cks"
bash scripts/select_checkpoint.sh runs/lora_large_v3_C $cks
best=$($PY/python scripts/pick_checkpoint.py lora_large_v3_C $cks)
echo "== picked C $best (lowest pooled users' val)"
sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_C/checkpoints/$best#" $BASE_CFG > configs/experiments/bv1_lora_large_v3_C_picked.yaml
$PY/wf-transcribe --config configs/experiments/bv1_lora_large_v3_C_picked.yaml --manifest data/manifests/benchmark_v1.jsonl \
  --split test --allow-test --run-id bv1_lora_large_v3_C_$best > runs/bv1_C.out 2>&1
echo "bv1 C $best $($PY/wf-score reports/runs/bv1_lora_large_v3_C_$best 2>/dev/null | grep -o 'official MER [0-9.]*%')"
echo ALL_DONE
