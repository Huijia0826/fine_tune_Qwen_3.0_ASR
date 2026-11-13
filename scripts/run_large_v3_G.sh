#!/usr/bin/env bash
# lora_large_v3_G (2026-11-04) on the H100 cluster: D recipe, online_pool batch 1 full (451 clips). Train, decode val sets for
# every checkpoint after step 400 (same decoding config as C's selection: reference decoding v2), pick by the
# predeclared rule (lowest pooled users' val, scripts/pick_checkpoint.py), then benchmark v1 ONCE for that checkpoint.
#   nohup bash -lc "srun -p debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G -t 08:00:00 \
#     bash scripts/run_large_v3_G.sh" > runs/large_v3_G.log 2>&1 < /dev/null &
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 BASE_CFG=configs/experiments/zeroshot_large_v3.yaml WITH_ONLINE=1
PY=.venv/bin
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
[ -e runs/lora_large_v3_G/summary.json ] || $PY/wf-train --config configs/train_lora_large_v3_G_h100.yaml \
  --run-id lora_large_v3_G > runs/lora_large_v3_G.out 2>&1
echo "trained G $(date -Is): $(tail -1 runs/lora_large_v3_G/log.jsonl)"
cks=$(ls runs/lora_large_v3_G/checkpoints | sort -t_ -k2 -n | grep -v -E '^step_(200|400)$' | tr '\n' ' ')
echo "== select G: $cks"
bash scripts/select_checkpoint.sh runs/lora_large_v3_G $cks
best=$($PY/python scripts/pick_checkpoint.py lora_large_v3_G $cks)
echo "== picked G $best (lowest pooled users' val)"
sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_G/checkpoints/$best#" $BASE_CFG \
  > configs/experiments/bv1_lora_large_v3_G_picked.yaml
$PY/wf-transcribe --config configs/experiments/bv1_lora_large_v3_G_picked.yaml --manifest data/manifests/benchmark_v1.jsonl \
  --split test --allow-test --run-id bv1_lora_large_v3_G_$best > runs/bv1_G.out 2>&1
echo "bv1 G $best $($PY/wf-score reports/runs/bv1_lora_large_v3_G_$best 2>/dev/null | grep -o 'official MER [0-9.]*%')"
chmod 700 "$HOME/whisperflow"
echo ALL_DONE
