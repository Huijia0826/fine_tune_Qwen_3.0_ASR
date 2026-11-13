#!/usr/bin/env bash
# lora_large_v3_I (2026-11-05) on the H100 cluster (batch partition, 1 GPU): H4 recipe + recording_v2 train ×4. Train, decode val sets for
# every checkpoint after step 400 (same decoding config as C's selection: reference decoding v2), pick by the
# predeclared rule (lowest pooled users' val, scripts/pick_checkpoint.py), then benchmark v1 ONCE for that checkpoint.
#   nohup bash -lc "srun -p batch --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G -t 08:00:00 \
#     bash scripts/run_large_v3_I.sh" > runs/large_v3_I.log 2>&1 < /dev/null &
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 BASE_CFG=configs/experiments/zeroshot_large_v3.yaml WITH_ONLINE=1
PY=.venv/bin
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
[ -e runs/lora_large_v3_I/summary.json ] || $PY/wf-train --config configs/train_lora_large_v3_I_h100.yaml \
  --run-id lora_large_v3_I > runs/lora_large_v3_I.out 2>&1
echo "trained I $(date -Is): $(tail -1 runs/lora_large_v3_I/log.jsonl)"
cks=$(ls runs/lora_large_v3_I/checkpoints | sort -t_ -k2 -n | grep -v -E '^step_(200|400)$' | tr '\n' ' ')
echo "== select I: $cks"
bash scripts/select_checkpoint.sh runs/lora_large_v3_I $cks
best=$($PY/python scripts/pick_checkpoint.py lora_large_v3_I $cks)
echo "== picked I $best (lowest pooled users' val)"
sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_I/checkpoints/$best#" $BASE_CFG \
  > configs/experiments/bv1_lora_large_v3_I_picked.yaml
$PY/wf-transcribe --config configs/experiments/bv1_lora_large_v3_I_picked.yaml --manifest data/manifests/benchmark_v1.jsonl \
  --split test --allow-test --run-id bv1_lora_large_v3_I_$best > runs/bv1_I.out 2>&1
echo "bv1 I $best $($PY/wf-score reports/runs/bv1_lora_large_v3_I_$best 2>/dev/null | grep -o 'official MER [0-9.]*%')"
# owner tests (test only): freeform (21) and recording script v2 test part (41), for G / H4 / I
for ck in lora_large_v3_G/checkpoints/step_600 lora_large_v3_H4/checkpoints/step_800 lora_large_v3_I/checkpoints/$best; do
  tag=$(echo $ck | cut -d/ -f1 | sed 's/lora_large_v3_//')_$(basename $ck)
  sed "s#adapter_path: null#adapter_path: runs/$ck#" $BASE_CFG > /tmp/ownertest_$tag.yaml
  for set in ownertest:recording_freeform_test v2test:recording_v2_test; do
    name=${set%%:*}; man=${set#*:}
    [ -e reports/runs/${name}_$tag/predictions.jsonl ] && continue
    $PY/wf-transcribe --config /tmp/ownertest_$tag.yaml --manifest data/manifests/$man.jsonl \
      --split test --allow-test --run-id ${name}_$tag > runs/${name}_$tag.out 2>&1
    echo "$name $tag $($PY/wf-score reports/runs/${name}_$tag 2>/dev/null | grep -o 'official MER [0-9.]*%')"
  done
done
chmod 700 "$HOME/whisperflow"
echo ALL_DONE
