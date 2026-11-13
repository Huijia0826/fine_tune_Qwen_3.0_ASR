#!/usr/bin/env bash
# lora_large_v3 A / B (2026-11-01): train both, then pick checkpoints on val (base = zero-shot large-v3, already
# decoded as reports/runs/zs_large_v3_*). Run on the GPU box from the repo root, detached:
#   nohup bash scripts/run_large_v3_ab.sh > runs/large_v3_ab.log 2>&1 < /dev/null &
set -euo pipefail
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1
for v in A B; do
  [ -e runs/lora_large_v3_$v/summary.json ] || .venv/bin/wf-train --config configs/train_lora_large_v3_${v}_cloud.yaml \
    --run-id lora_large_v3_$v > runs/lora_large_v3_$v.out 2>&1
  echo "trained $v: $(grep -c val_loss runs/lora_large_v3_$v/log.jsonl) evals, $(tail -1 runs/lora_large_v3_$v/log.jsonl)"
done
for v in A B; do
  cks=$(ls runs/lora_large_v3_$v/checkpoints | sort -t_ -k2 -n | grep -v -E '^step_(200|400)$' | tr '\n' ' ')
  echo "== select $v: $cks"
  BASE_CFG=configs/experiments/zeroshot_large_v3.yaml bash scripts/select_checkpoint.sh runs/lora_large_v3_$v $cks
done
echo ALL_DONE
