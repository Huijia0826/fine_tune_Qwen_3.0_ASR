#!/usr/bin/env bash
# qwen3asr_J4 (2026-11-10, owner): J3 recipe + 300 MiniMax sentences with spoken acronyms inside Chinese (RL, ARR,
# RLHF, …) + 150 long English dictations (~39 s) + 300 concatenated real long clips (30–60 s, 3.1 h) + LibTV label
# fix. Same selection rule and tests as J3, plus the two production clips with spoken acronyms (adhoc_mianmian).
#   sbatch -J qwen3asr-J4 -p batch --gres=gpu:h100:1 --cpus-per-task=16 --mem=128G -t 16:00:00
#     -o runs/qwen3asr_J4.log --wrap "bash scripts/run_qwen3_asr_J4.sh"
set -euo pipefail
cd "$HOME/whisperflow"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-qwen/bin; RUN=runs/qwen3asr_J4; DATA=runs/qwen_sft/J4
BASE=$(ls -d hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5)
CFG=configs/experiments/qwen3asr_ctxJ2_chunk25.yaml; J1=runs/qwen3asr_J1/checkpoint-400  # comparison
DIST=data/private/e2e/contexts_dictionary_bench_distract.jsonl
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# 0. QC of the new synthetic sets (round trip with Qwen zero-shot; TTS misreadings join the drop list)
for part in long more acro longen; do
  id=qc_synth_${part}_qwen_zs
  [ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml \
    --manifest data/manifests/synth_$part.jsonl --split train --run-id $id > runs/$id.out 2>&1
  $PY/python scripts/synth/qc.py data/manifests/synth_$part.jsonl $id --write-drop | head -3
done
SYN=""
for part in pilot:pilot_500:audio_aug long:long_100:audio_aug_long more:more_1500:audio_aug_more \
            acro:acronym_300:audio_aug_acro longen:long_en_150:audio_aug_longen; do
  IFS=: read -r name texts aug <<< "$part"
  m=data/manifests/synth_$name.jsonl
  if [ -e $m ]; then SYN="$SYN $m:data/private/synth/$texts.jsonl:data/private/synth/$aug"; fi
done
$PY/python scripts/make_qwen_e2e_data.py --out $DATA --augment --vocab-mild --long-real --concat data/manifests/concat_long.jsonl --synth $SYN \
  --extra-contexts data/private/e2e/contexts_dictionary_bench.jsonl > runs/qwen_sft_J4_data.out 2>&1 \
  || { cat runs/qwen_sft_J4_data.out; exit 1; }
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
  for set in bczh:benchmark_chinese online:online_phase1; do decode sel_qwen3asr_J4_${step}_${set%%:*} ${set#*:} val $ck ctx; done
done
best=$($PY/python scripts/pick_checkpoint.py qwen3asr_J4 $cks --refs v1)
echo "== picked $best (lowest pooled users' val, with context, v1 references)"
CK=$RUN/checkpoint-${best#step_}

for set in bczh:benchmark_chinese online:online_phase1; do
  for mode in noctx guard; do decode sel_qwen3asr_J4_${best}_${mode}_${set%%:*} ${set#*:} val $CK $mode; done
  decode sel_qwen3asr_J1ctx2_step_400_${set%%:*} ${set#*:} val $J1 ctx        # J1 with J2-form contexts
done
for set in bv1:benchmark_v1 ownertest:recording_freeform_test v2test:recording_v2_test dict:dictionary_bench; do
  for mode in ctx noctx guard; do decode ${set%%:*}_qwen3asr_J4_${best}_$mode ${set#*:} test $CK $mode; done
done
for set in ownertest:recording_freeform_test v2test:recording_v2_test dict:dictionary_bench; do
  decode ${set%%:*}_qwen3asr_J1ctx2_step_400 ${set#*:} test $J1 ctx
done
decode dict_qwen3asr_J4_${best}_distract dictionary_bench test $CK distract
decode dict_qwen3asr_J1_step_400_distract dictionary_bench test $J1 distract
decode dict_qwen3asr_zs_distract dictionary_bench test base distract
decode dict_qwen3asr_zs_ctx2 dictionary_bench test base ctx
# the two production clips with spoken acronyms (RL, ARR), with and without a vocabulary
for v in "" _vocab_only _vocab_mixed; do
  cf=data/private/adhoc/mianmian/contexts${v}.jsonl
  sed -e "s#checkpoint_path: null.*#checkpoint_path: $CK#" -e "s#context_file: .*#context_file: $cf#" $CFG > /tmp/j4_adhoc$v.yaml
  $PY/wf-transcribe --config /tmp/j4_adhoc$v.yaml --manifest data/manifests/adhoc_mianmian.jsonl --split val \
    --run-id adhoc_mianmian_J4$v > /dev/null 2>&1
  echo "adhoc$v: $(python3 -c "import json; print(' | '.join(json.loads(l)['hypothesis'] for l in open('reports/runs/adhoc_mianmian_J4$v/predictions.jsonl')))")"
done
chmod 700 "$HOME/whisperflow"
echo QWEN_J4_DONE
