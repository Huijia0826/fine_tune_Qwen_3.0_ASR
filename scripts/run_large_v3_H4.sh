#!/usr/bin/env bash
# lora_large_v3_H4 (2026-11-04) on the H100 cluster: G recipe + recording_v1 (see the config). Train, decode val sets for
# every checkpoint after step 400 (same decoding config as C's selection: reference decoding v2), pick by the
# predeclared rule (lowest pooled users' val, scripts/pick_checkpoint.py), then benchmark v1 ONCE for that checkpoint.
#   nohup bash -lc "srun -p debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G -t 08:00:00 \
#     bash scripts/run_large_v3_H4.sh" > runs/large_v3_H4.log 2>&1 < /dev/null &
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 BASE_CFG=configs/experiments/zeroshot_large_v3.yaml WITH_ONLINE=1
PY=.venv/bin
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
[ -e runs/lora_large_v3_H4/summary.json ] || $PY/wf-train --config configs/train_lora_large_v3_H4_h100.yaml \
  --run-id lora_large_v3_H4 > runs/lora_large_v3_H4.out 2>&1
echo "trained H4 $(date -Is): $(tail -1 runs/lora_large_v3_H4/log.jsonl)"
cks=$(ls runs/lora_large_v3_H4/checkpoints | sort -t_ -k2 -n | grep -v -E '^step_(200|400)$' | tr '\n' ' ')
echo "== select H4: $cks"
bash scripts/select_checkpoint.sh runs/lora_large_v3_H4 $cks
best=$($PY/python scripts/pick_checkpoint.py lora_large_v3_H4 $cks)
echo "== picked H4 $best (lowest pooled users' val)"
sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_H4/checkpoints/$best#" $BASE_CFG \
  > configs/experiments/bv1_lora_large_v3_H4_picked.yaml
$PY/wf-transcribe --config configs/experiments/bv1_lora_large_v3_H4_picked.yaml --manifest data/manifests/benchmark_v1.jsonl \
  --split test --allow-test --run-id bv1_lora_large_v3_H4_$best > runs/bv1_H4.out 2>&1
echo "bv1 H4 $best $($PY/wf-score reports/runs/bv1_lora_large_v3_H4_$best 2>/dev/null | grep -o 'official MER [0-9.]*%')"
# owner's freeform test set (recording_freeform_test, 21 clips, test only; added 2026-11-04 before the job started)
for ck in lora_large_v3_H4/checkpoints/$best; do
  tag=$(echo $ck | cut -d/ -f1 | sed 's/lora_large_v3_//')_$(basename $ck)
  sed "s#adapter_path: null#adapter_path: runs/$ck#" $BASE_CFG > /tmp/ownertest_$tag.yaml
  $PY/wf-transcribe --config /tmp/ownertest_$tag.yaml --manifest data/manifests/recording_freeform_test.jsonl \
    --split test --allow-test --run-id ownertest_$tag > runs/ownertest_$tag.out 2>&1
  echo "ownertest $tag $($PY/wf-score reports/runs/ownertest_$tag 2>/dev/null | grep -o 'official MER [0-9.]*%')"
done
chmod 700 "$HOME/whisperflow"
echo ALL_DONE
