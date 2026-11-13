#!/usr/bin/env bash
# ASCEND-share ablation runs (2026-11-03) on the H100 cluster: train, decode the val sets for EVERY saved checkpoint
# (E / F are short: ~530 / ~350 steps, so D's "after step 400" rule would leave one checkpoint), pick by the same
# predeclared rule as C / D (lowest pooled users' val). No benchmark v1 here: the locked test set is run once, only
# for a variant that wins on val.
#   bash scripts/run_large_v3_ablation.sh E configs/train_lora_large_v3_E_h100.yaml
set -euo pipefail
V=$1; CFG=$2; RUN=lora_large_v3_$V
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 BASE_CFG=configs/experiments/zeroshot_large_v3.yaml WITH_ONLINE=1
PY=.venv/bin
echo "node $(hostname) $(date -Is) $V $CFG"
[ -e runs/$RUN/summary.json ] || $PY/wf-train --config "$CFG" --run-id $RUN > runs/$RUN.out 2>&1
echo "trained $V $(date -Is): $(grep -m1 'optimizer steps' runs/$RUN.out) | $(tail -1 runs/$RUN/log.jsonl)"
cks=$(ls runs/$RUN/checkpoints | sort -t_ -k2 -n | tr '\n' ' ')
echo "== select $V: $cks"
bash scripts/select_checkpoint.sh runs/$RUN $cks
echo "== picked $V $($PY/python scripts/pick_checkpoint.py $RUN $cks) (lowest pooled users' val)"
chmod 700 "$HOME/whisperflow"
echo ALL_DONE
