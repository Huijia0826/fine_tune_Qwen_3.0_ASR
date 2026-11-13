#!/usr/bin/env bash
# qwen3asr_J1 (2026-11-09, owner): Qwen3-ASR-1.7B full fine-tune on I2's data WITHOUT ASCEND, labels in annotation
# policy v1, production context as the system prompt (scripts/make_qwen_e2e_data.py). Same recipe and selection rule
# as qwen3asr_I2 (lowest pooled users' val, 25 s chunks; here decoded WITH each clip's context). Then, once, for the
# picked checkpoint: benchmark v1, owner tests and the dictionary benchmark, each with and without context
# (ablation), plus the comparison systems on the dictionary benchmark.
#   ssh h100-coding 'cd ~/whisperflow && bash -lc "sbatch -J qwen3asr-J1 -p batch --gres=gpu:h100:1 --cpus-per-task=16 \
#     --mem=128G -t 16:00:00 -o runs/qwen3asr_J1.log --wrap \"bash scripts/run_qwen3_asr_J1.sh\""'
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-qwen/bin; RUN=runs/qwen3asr_J1; DATA=runs/qwen_sft/J1
BASE=$(ls -d hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5)
CFG=configs/experiments/qwen3asr_ctx_chunk25.yaml
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# 1. data (paths are resolved on this machine)
$PY/python scripts/make_qwen_e2e_data.py --out $DATA --extra-contexts data/private/e2e/contexts_dictionary_bench.jsonl \
  > runs/qwen_sft_J1_data.out 2>&1 || { cat runs/qwen_sft_J1_data.out; exit 1; }
$PY/python -c "import json; s=json.load(open('$DATA/summary.json')); print('data', s['train'], s['val'], s['leaks'])"

# 2. training: same recipe as qwen3asr_I2, but batch 8 x grad_acc 2 (= 16): with context prompts, batch 16 ran out of
#    memory at step 168 (the fp32 logits over a 151k vocabulary for 16 longer sequences, +6.7 GB; 2026-11-09)
[ -e $RUN/done ] || { $PY/python scripts/qwen3_asr_sft.py --model_path $BASE --train_file $DATA/train.jsonl \
  --eval_file $DATA/val.jsonl --output_dir $RUN --lr 2e-5 --lr_scheduler_type linear --warmup_ratio 0.05 \
  --batch_size 8 --grad_acc 2 --epochs 3 --save_steps 200 --save_total_limit 50 --num_workers 8 --seed 1234 \
  --log_steps 10 --save_only_model 1 > $RUN.out 2>&1 || { tail -30 $RUN.out; exit 1; }; touch $RUN/done; }
grep -o "{'eval_loss'[^}]*}" $RUN.out | cut -c1-60
echo "trained $(date -Is)"

score() { echo "$1 $($PY/wf-score reports/runs/$1 2>/dev/null | grep -o 'official MER [0-9.]*%')"; }
decode() {  # decode <run id> <manifest> <split> <checkpoint dir or base> <ctx|noctx>
  local id=$1 man=$2 split=$3 ck=$4 mode=$5 cfg=/tmp/j1_$1.yaml
  [ -e reports/runs/$id/run_meta.json ] && { score $id; return; }
  sed -e "s#checkpoint_path: null.*#checkpoint_path: $([ $ck = base ] && echo null || echo $ck)#" $CFG > $cfg
  [ $mode = noctx ] && sed -i 's#context_file: .*#context_file: null#' $cfg
  $PY/wf-transcribe --config $cfg --manifest data/manifests/$man.jsonl --split $split \
    $([ $split = test ] && echo --allow-test) --run-id $id > runs/$id.out 2>&1
  score $id
}

# 3. selection on users' val, with context
cks=""
for ck in $(ls -d $RUN/checkpoint-* | sort -t- -k2 -n); do
  step=step_${ck##*-}; cks="$cks $step"
  for set in bczh:benchmark_chinese online:online_phase1; do
    decode sel_qwen3asr_J1_${step}_${set%%:*} ${set#*:} val $ck ctx
  done
done
best=$($PY/python scripts/pick_checkpoint.py qwen3asr_J1 $cks)
echo "== picked $best (lowest pooled users' val, with context)"
CK=$RUN/checkpoint-${best#step_}

# 4. picked checkpoint, once: with and without context
for set in bv1:benchmark_v1 ownertest:recording_freeform_test v2test:recording_v2_test dict:dictionary_bench; do
  for mode in ctx noctx; do decode ${set%%:*}_qwen3asr_J1_${best}_$mode ${set#*:} test $CK $mode; done
done
# users' val without context (same checkpoint): how much of the val gain is the context
for set in bczh:benchmark_chinese online:online_phase1; do decode sel_qwen3asr_J1_${best}_noctx_${set%%:*} ${set#*:} val $CK noctx; done

# 5. comparison systems on the dictionary benchmark (test, run once each)
decode dict_qwen3asr_zs_noctx dictionary_bench test base noctx
decode dict_qwen3asr_zs_ctx dictionary_bench test base ctx
decode dict_qwen3asr_I2_step_1000 dictionary_bench test runs/qwen3asr_I2/checkpoint-1000 noctx
if [ ! -e reports/runs/dict_I2_step_600/run_meta.json ]; then  # Whisper I2 (main .venv)
  sed "s#adapter_path: null#adapter_path: runs/lora_large_v3_I2/checkpoints/step_600#" \
    configs/experiments/zeroshot_large_v3.yaml > /tmp/dict_I2.yaml
  .venv/bin/wf-transcribe --config /tmp/dict_I2.yaml --manifest data/manifests/dictionary_bench.jsonl --split test \
    --allow-test --run-id dict_I2_step_600 > runs/dict_I2_step_600.out 2>&1
fi
score dict_I2_step_600
chmod 700 "$HOME/whisperflow"
echo QWEN_J1_DONE
