#!/usr/bin/env bash
# Prepare a cloud GPU box (AutoDL) with everything the current experiments need (owner approved uploads: training
# data 2026-10-30/31, benchmark v1 for evaluation 2026-11-02; delete cloud copies after the experiments).
# Usage: scripts/cloud_sync.sh <port> [user@host] [dest]     e.g. scripts/cloud_sync.sh 39642
#        scripts/cloud_sync.sh - h100-coding '~/whisperflow'   (SSH config alias, no port; H100 cluster 2026-11-02)
# Remote layout: /root/autodl-tmp/wt. A fresh box also needs the environment: run scripts/cloud_bootstrap.sh there first
# (or migrate an existing instance, which keeps the environment). Prints file counts at the end to verify.
set -euo pipefail
PORT=$1; HOST=${2:-root@connect.bjb1.seetacloud.com}; DEST=${3:-/root/autodl-tmp/wt}
SSHP=(); [ "$PORT" != "-" ] && SSHP=(-p "$PORT")
RS=(rsync -az -e "ssh ${SSHP[*]:-}")   # macOS ships openrsync: keep to basic flags
ssh() { command ssh ${SSHP[@]+"${SSHP[@]}"} "$@"; }  # bash 3.2: empty arrays trip set -u
ssh "$HOST" "mkdir -p $DEST/data/private $DEST/hf_cache/hub $DEST/runs"
"${RS[@]}" --exclude .venv --exclude runs --exclude data/private --exclude .git --exclude '__pycache__' \
  --exclude .ruff_cache --exclude .pytest_cache --exclude reports/runs --exclude reports/test_access_log.jsonl ./ "$HOST:$DEST/"
lock() { ssh "$HOST" "chmod 700 $DEST $DEST/data/private $DEST/runs 2>/dev/null; true"; }  # rsync ./ copies the local 755
lock
ssh "$HOST" "mkdir -p $DEST/data/private/ascend $DEST/data/private/online_phase1 $DEST/data/private/online_pool"
"${RS[@]}" data/private/ascend/wav "$HOST:$DEST/data/private/ascend/"
"${RS[@]}" data/private/benchmark_chinese data/private/val_longform data/private/segmented "$HOST:$DEST/data/private/"
for a in arena_019f8e51 arena_019f6442 benchmark_v1 online_phase1; do   # wavs only: no raw ogg, no metadata tables
  ssh "$HOST" "mkdir -p $DEST/data/private/$a"
  "${RS[@]}" --include='*.wav' --exclude='*' data/private/$a/ "$HOST:$DEST/data/private/$a/"
done
[ -d data/private/online_pool/audio ] && "${RS[@]}" data/private/online_pool/audio data/private/online_pool/pool.jsonl \
  "$HOST:$DEST/data/private/online_pool/"   # unreviewed training pool (for drafts)
MODELS="whisper-small whisper-large-v3"; [ -n "${SKIP_HF:-}" ] && MODELS=""   # SKIP_HF=1: the box downloads models
for m in $MODELS; do
  "${RS[@]}" ~/.cache/huggingface/hub/models--openai--$m "$HOST:$DEST/hf_cache/hub/"
done
for c in lora_large_v3_B/checkpoints/step_800 lora_large_v3_C/checkpoints/step_600; do   # current best adapters
  ssh "$HOST" "mkdir -p $DEST/runs/$(dirname $c)"
  "${RS[@]}" runs/$c "$HOST:$DEST/runs/$(dirname $c)/"
done
ssh "$HOST" "cd $DEST; for d in ascend/wav benchmark_chinese val_longform segmented arena_019f8e51 arena_019f6442 \
  benchmark_v1 online_phase1 online_pool/audio; do echo \"\$d \$(find data/private/\$d -type f | wc -l)\"; done"
lock
echo "synced to $HOST:$DEST (chmod 700)"
