#!/usr/bin/env bash
# qwen3asr_J3 (2026-11-10, owner): J1 data (policy v1) + J2 cursor-context augmentation + LIGHT vocabulary
# augmentation (--vocab-mild) + 100 whole real long recordings (30–180 s, --long-real) + MiniMax synthetic clips
# (pilot 500 + long 100 + more 1,500, QC-filtered, 15% lightly augmented). Goal: beat production on benchmark v1
# (production final output 7.63%; fine-tuned Qwen loses on > 30 s). Batch 4 x grad_acc 4 (= 16) for the long clips.
# Selection predeclared: pooled users' val with context, v1 references; decoding = 25 s chunks + short-tail merge.
#   sbatch -J qwen3asr-J3 -p batch --gres=gpu:h100:1 --cpus-per-task=16 --mem=128G -t 16:00:00
#     -o runs/qwen3asr_J3.log --wrap "bash scripts/run_qwen3_asr_J3.sh"
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-qwen/bin; RUN=runs/qwen3asr_J3; DATA=runs/qwen_sft/J3
BASE=$(ls -d hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5)
CFG=configs/experiments/qwen3asr_ctxJ2_chunk25.yaml; J1=runs/qwen3asr_J1/checkpoint-400  # comparison
DIST=data/private/e2e/contexts_dictionary_bench_distract.jsonl
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# 0. QC of the new synthetic sets (round trip with Qwen zero-shot; TTS misreadings join the drop list)
for part in long more; do
  id=qc_synth_${part}_qwen_zs
  [ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml \
    --manifest data/manifests/synth_$part.jsonl --split train --run-id $id > runs/$id.out 2>&1
  $PY/python scripts/synth/qc.py data/manifests/synth_$part.jsonl $id --write-drop | head -3
done
SYN=""
for part in pilot:pilot_500:audio_aug long:long_100:audio_aug_long more:more_1500:audio_aug_more; do
  IFS=: read -r name texts aug <<< "$part"
  m=data/manifests/synth_$name.jsonl
  if [ -e $m ]; then SYN="$SYN $m:data/private/synth/$texts.jsonl:data/private/synth/$aug"; fi
done
$PY/python scripts/make_qwen_e2e_data.py --out $DATA --augment --vocab-mild --long-real --synth $SYN \
  --extra-contexts data/private/e2e/contexts_dictionary_bench.jsonl > runs/qwen_sft_J3_data.out 2>&1 \
  || { cat runs/qwen_sft_J3_data.out; exit 1; }
$PY/python -c "import json; s=json.load(open('$DATA/summary.json')); print('data', s['train'], s['augment'])"

[ -e $RUN/done ] || { $PY/python scripts/qwen3_asr_sft.py --model_path $BASE --train_file $DATA/train.jsonl \
  --eval_file $DATA/val.jsonl --output_dir $RUN --lr 2e-5 --lr_scheduler_type linear --warmup_ratio 0.05 \
  --batch_size 4 --grad_acc 4 --epochs 3 --save_steps 200 --save_total_limit 50 --num_workers 8 --seed 1234 \
  --log_steps 10 --save_only_model 1 > $RUN.out 2>&1 || { tail -30 $RUN.out; exit 1; }; touch $RUN/done; }
echo "trained $(date -Is)"

score() { echo "$1 $($PY/wf-score reports/runs/$1 2>/dev/null | grep -o 'official MER [0-9.]*%')"; }
decode() {  # decode <run id> <manifest> <split> <checkpoint dir | base> <ctx | noctx | guard | distract>
  local id=$1 man=$2 split=$3 ck=$4 mode=$5 cfg=/tmp/j2_$1.yaml
  [ -e reports/runs/$id/run_meta.json ] && { score $id; return; }
  sed -e "s#checkpoint_path: null.*#checkpoint_path: $([ $ck = base ] && echo null || echo $ck)#" $CFG > $cfg
  case $mode in
    noctx) sed -i 's#context_file: .*#context_file: null#' $cfg;;
    guard) sed -i 's#^decoding:#decoding:\n  copy_guard: true#' $cfg;;
    distract) sed -i "s#context_file: .*#context_file: $DIST#" $cfg;;
  esac
  $PY/wf-transcribe --config $cfg --manifest data/manifests/$man.jsonl --split $split \
    $([ $split = test ] && echo --allow-test) --run-id $id > runs/$id.out 2>&1
  score $id
}

cks=""
for ck in $(ls -d $RUN/checkpoint-* | sort -t- -k2 -n); do
  step=step_${ck##*-}; cks="$cks $step"
  for set in bczh:benchmark_chinese online:online_phase1; do decode sel_qwen3asr_J3_${step}_${set%%:*} ${set#*:} val $ck ctx; done
done
best=$($PY/python scripts/pick_checkpoint.py qwen3asr_J3 $cks --refs v1)
echo "== picked $best (lowest pooled users' val, with context, v1 references)"
CK=$RUN/checkpoint-${best#step_}

for set in bczh:benchmark_chinese online:online_phase1; do
  for mode in noctx guard; do decode sel_qwen3asr_J3_${best}_${mode}_${set%%:*} ${set#*:} val $CK $mode; done
  decode sel_qwen3asr_J1ctx2_step_400_${set%%:*} ${set#*:} val $J1 ctx        # J1 with J2-form contexts
done
for set in bv1:benchmark_v1 ownertest:recording_freeform_test v2test:recording_v2_test dict:dictionary_bench; do
  for mode in ctx noctx guard; do decode ${set%%:*}_qwen3asr_J3_${best}_$mode ${set#*:} test $CK $mode; done
done
for set in ownertest:recording_freeform_test v2test:recording_v2_test dict:dictionary_bench; do
  decode ${set%%:*}_qwen3asr_J1ctx2_step_400 ${set#*:} test $J1 ctx
done
decode dict_qwen3asr_J3_${best}_distract dictionary_bench test $CK distract
decode dict_qwen3asr_J1_step_400_distract dictionary_bench test $J1 distract
decode dict_qwen3asr_zs_distract dictionary_bench test base distract
decode dict_qwen3asr_zs_ctx2 dictionary_bench test base ctx
chmod 700 "$HOME/whisperflow"
echo QWEN_J2_DONE
