#!/usr/bin/env bash
# lora_large_v3_H (2026-11-04) on the H100 cluster: G recipe + recording_v1 (see the config). Train, decode val sets for
# every checkpoint after step 400 (same decoding config as C's selection: reference decoding v2), pick by the
# predeclared rule (lowest pooled users' val, scripts/pick_checkpoint.py), then benchmark v1 ONCE for that checkpoint.
# AutoDL RTX 5090 variant (2026-11-04): no srun; run H then H4 in one session: nohup bash -c 'bash scripts/run_large_v3_H_autodl.sh; bash scripts/run_large_v3_H4_autodl.sh' > runs/H_H4_autodl.log 2>&1 &
set -euo pipefail
cd /root/autodl-tmp/wt
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 BASE_CFG=configs/experiments/zeroshot_large_v3.yaml WITH_ONLINE=1
PY=.venv/bin
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
[ -e runs/lora_large_v3_H/summary.json ] || $PY/wf-train --config configs/train_lora_large_v3_H_5090.yaml \
  --run-id lora_large_v3_H > runs/lora_large_v3_H.out 2>&1
echo "trained H $(date -Is): $(tail -1 runs/lora_large_v3_H/log.jsonl)"
cks=$(ls runs/lora_large_v3_H/checkpoints | sort -t_ -k2 -n | grep -v -E '^step_(200|400)$' | tr '\n' ' ')
echo "== select H: $cks"
bash scripts/select_checkpoint.sh runs/lora_large_v3_H $cks
best=$($PY/python scripts/pick_checkpoint.py lora_large_v3_H $cks)
echo "== picked H $best (lowest pooled users' val)"
sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_H/checkpoints/$best#" $BASE_CFG \
  > configs/experiments/bv1_lora_large_v3_H_picked.yaml
$PY/wf-transcribe --config configs/experiments/bv1_lora_large_v3_H_picked.yaml --manifest data/manifests/benchmark_v1.jsonl \
  --split test --allow-test --run-id bv1_lora_large_v3_H_$best > runs/bv1_H.out 2>&1
echo "bv1 H $best $($PY/wf-score reports/runs/bv1_lora_large_v3_H_$best 2>/dev/null | grep -o 'official MER [0-9.]*%')"
# owner's freeform test set (recording_freeform_test, 21 clips, test only; added 2026-11-04 before the job started)
for ck in lora_large_v3_G/checkpoints/step_600 lora_large_v3_H/checkpoints/$best; do
  tag=$(echo $ck | cut -d/ -f1 | sed 's/lora_large_v3_//')_$(basename $ck)
  sed "s#adapter_path: null#adapter_path: runs/$ck#" $BASE_CFG > /tmp/ownertest_$tag.yaml
  $PY/wf-transcribe --config /tmp/ownertest_$tag.yaml --manifest data/manifests/recording_freeform_test.jsonl \
    --split test --allow-test --run-id ownertest_$tag > runs/ownertest_$tag.out 2>&1
  echo "ownertest $tag $($PY/wf-score reports/runs/ownertest_$tag 2>/dev/null | grep -o 'official MER [0-9.]*%')"
done
chmod 700 /root/autodl-tmp/wt
echo ALL_DONE
