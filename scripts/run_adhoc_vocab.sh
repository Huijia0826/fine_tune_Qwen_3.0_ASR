#!/usr/bin/env bash
# Does a vocabulary in the context fix spoken acronyms (RL, ARR)? J3 and Qwen zero-shot, two vocabularies.
set -euo pipefail
cd "$HOME/whisperflow"; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1; Q=.venv-qwen/bin
CFG=configs/experiments/qwen3asr_ctxJ2_chunk25.yaml
for v in vocab_only vocab_mixed; do
  for m in J3 zs; do
    ck=$([ $m = J3 ] && echo runs/qwen3asr_J3/checkpoint-600 || echo null)
    sed -e "s#checkpoint_path: null.*#checkpoint_path: $ck#" -e "s#context_file: .*#context_file: data/private/adhoc/mianmian/contexts_$v.jsonl#" $CFG > /tmp/av_${m}_${v}.yaml
    id=adhoc_mianmian_${m}_$v
    $Q/wf-transcribe --config /tmp/av_${m}_${v}.yaml --manifest data/manifests/adhoc_mianmian.jsonl --split val --run-id $id > runs/$id.out 2>&1
    echo "== $id"; python3 -c "import json; [print(' ',json.loads(l)['utterance_id'],'|',json.loads(l)['hypothesis']) for l in open('reports/runs/$id/predictions.jsonl')]"
  done
done
chmod 700 "$HOME/whisperflow"
