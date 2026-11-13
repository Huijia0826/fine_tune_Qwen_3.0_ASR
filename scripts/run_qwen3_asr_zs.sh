#!/usr/bin/env bash
# Qwen3-ASR zero-shot (2026-11-09, owner approved): 0.6B and 1.7B on users' val (bczh val + online val), benchmark v1
# (owner asked for it; logged), the owner's recording tests, then batch-1 latency on users' val. Same scorer as Whisper.
# Needs .venv-qwen (scripts/qwen_asr_setup.sh). Steps already done are skipped, so it can be rerun.
#   ssh h100-coding 'cd ~/whisperflow && bash -lc "sbatch -J qwen3asr-zs -p batch --gres=gpu:h100:1 --cpus-per-task=8 \
#     --mem=64G -t 04:00:00 -o runs/qwen3_asr_zs.log --wrap \"bash scripts/run_qwen3_asr_zs.sh\""'
# (sbatch, not nohup srun: a pending srun was cancelled when the ssh session closed, 2026-11-09)
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache
PY=.venv-qwen/bin
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
$PY/python -c "import torch; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available()); torch.zeros(1).cuda()"

# 1. weights, pinned revisions
$PY/python - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev in [("Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
                  ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5")]:
    print(repo, snapshot_download(repo, revision=rev))
PY
export HF_HUB_OFFLINE=1

score() { echo "$1 $($PY/wf-score reports/runs/$1 2>/dev/null | grep -o 'official MER [0-9.]*%')"; }
for size in 0.6b 1.7b; do
  cfg=configs/experiments/zeroshot_qwen3_asr_$size.yaml; tag=qwen3asr_${size/./}
  # 2. smoke test (5 clips) before the full runs
  $PY/wf-transcribe --config $cfg --manifest data/manifests/benchmark_chinese.jsonl --split val --limit 5 \
    --run-id smoke_$tag > runs/smoke_$tag.out 2>&1 || { tail -20 runs/smoke_$tag.out; exit 1; }
  # 3. users' val (selection set) — same run names as select_checkpoint.sh, so pick_checkpoint.py can pool them
  for set in bczh:benchmark_chinese online:online_phase1; do
    id=sel_${tag}_base_${set%%:*}
    $PY/wf-transcribe --config $cfg --manifest data/manifests/${set#*:}.jsonl --split val --run-id $id > runs/$id.out 2>&1
    score $id
  done
  echo "$tag users' val pooled: $($PY/python scripts/pick_checkpoint.py $tag base 2>&1 | head -1)"
  # 4. locked test sets, once
  for set in bv1:benchmark_v1 ownertest:recording_freeform_test v2test:recording_v2_test; do
    id=${set%%:*}_${tag}_zs
    [ -e reports/runs/$id/run_meta.json ] && { score $id; continue; }
    $PY/wf-transcribe --config $cfg --manifest data/manifests/${set#*:}.jsonl --split test --allow-test \
      --run-id $id > runs/$id.out 2>&1
    score $id
  done
done
# 5. latency, batch 1, users' val (GPU otherwise idle)
for size in 0.6b 1.7b; do
  id=lat_qwen3asr_${size/./}_bf16
  [ -e reports/runs/$id/latency.json ] && continue
  $PY/wf-latency --config configs/experiments/zeroshot_qwen3_asr_$size.yaml --run-id $id > runs/$id.out 2>&1 \
    || { echo "FAILED $id"; tail -5 runs/$id.out; continue; }
  $PY/python -c "import json; s = json.load(open('reports/runs/$id/latency.json')); l = s['latency']
print('$id', 'MER', s['official_mer_pct'], ' '.join(f\"{k}:n={b['n']},p50={b['p50_s']},p95={b['p95_s']}\" for k, b in l.items()))"
done
chmod 700 "$HOME/whisperflow"
echo QWEN_ZS_DONE
