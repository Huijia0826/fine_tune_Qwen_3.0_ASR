#!/usr/bin/env bash
# benchmark v1 for the ASCEND-share ablation picks (owner request 2026-11-03): E step_400, F step_200. Same decoding
# config as C / D (reference decoding v2). Each run is logged in reports/test_access_log.jsonl by wf-transcribe.
set -euo pipefail
cd "$HOME/whisperflow"; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1; PY=.venv/bin
for pair in E:step_400 F:step_200; do
  v=${pair%%:*}; ck=${pair#*:}; cfg=configs/experiments/bv1_lora_large_v3_${v}_picked.yaml
  sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_$v/checkpoints/$ck#" configs/experiments/zeroshot_large_v3.yaml > $cfg
  $PY/wf-transcribe --config $cfg --manifest data/manifests/benchmark_v1.jsonl --split test --allow-test \
    --run-id bv1_lora_large_v3_${v}_$ck > runs/bv1_$v.out 2>&1
  echo "bv1 $v $ck $($PY/wf-score reports/runs/bv1_lora_large_v3_${v}_$ck 2>/dev/null | grep -o 'official MER [0-9.]*%')"
done
chmod 700 "$HOME/whisperflow"; echo ALL_DONE
