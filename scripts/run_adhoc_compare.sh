#!/usr/bin/env bash
# Ad-hoc comparison of our systems on a small manifest (owner request 2026-11-10: two production clips from one user).
#   sbatch … --wrap "bash scripts/run_adhoc_compare.sh <manifest name> <contexts.jsonl>"
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1
MAN=$1; CTX=$2; Q=.venv-qwen/bin; W=.venv/bin
CFG=configs/experiments/qwen3asr_ctxJ2_chunk25.yaml
run() {  # run <id> <venv> <cfg>
  $2/wf-transcribe --config $3 --manifest data/manifests/$MAN.jsonl --split val --run-id $1 > runs/$1.out 2>&1
  echo "== $1"; cat reports/runs/$1/predictions.jsonl | python3 -c "import json,sys; [print(' ',json.loads(l)['utterance_id'],'|',json.loads(l)['hypothesis']) for l in sys.stdin]"
}
sed -e "s#checkpoint_path: null.*#checkpoint_path: runs/qwen3asr_J3/checkpoint-600#" -e "s#context_file: .*#context_file: $CTX#" $CFG > /tmp/a_j3ctx.yaml
sed -e "s#checkpoint_path: null.*#checkpoint_path: runs/qwen3asr_J3/checkpoint-600#" -e "s#context_file: .*#context_file: null#" $CFG > /tmp/a_j3.yaml
sed -e "s#context_file: .*#context_file: null#" $CFG > /tmp/a_zs.yaml
sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_I2/checkpoints/step_600#" configs/experiments/zeroshot_large_v3_guard.yaml > /tmp/a_i2.yaml
run ${MAN}_J3_ctx $Q /tmp/a_j3ctx.yaml
run ${MAN}_J3_noctx $Q /tmp/a_j3.yaml
run ${MAN}_qwen_zs $Q /tmp/a_zs.yaml
run ${MAN}_whisper_I2 $W /tmp/a_i2.yaml
chmod 700 "$HOME/whisperflow"
