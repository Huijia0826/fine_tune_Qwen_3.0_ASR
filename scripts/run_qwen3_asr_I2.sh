#!/usr/bin/env bash
# qwen3asr_I2 (2026-11-09, owner approved): full fine-tune of Qwen3-ASR-1.7B on I2's exact data (official recipe,
# scripts/qwen3_asr_sft.py), then the I2 selection protocol: every checkpoint on users' val (25 s chunks), pick the
# lowest pooled MER (scripts/pick_checkpoint.py), benchmark v1 + owner tests ONCE for the picked checkpoint.
#   ssh h100-coding 'cd ~/whisperflow && bash -lc "sbatch -J qwen3asr-I2 -p batch --gres=gpu:h100:1 --cpus-per-task=16 \
#     --mem=128G -t 16:00:00 -o runs/qwen3asr_I2.log --wrap \"bash scripts/run_qwen3_asr_I2.sh\""'
# Steps already done are skipped, so it can be resubmitted after a dropped job (training itself restarts from 0).
set -euo pipefail
cd "$HOME/whisperflow"
W=$PWD; UV="$W/.uvx86/uv-0.13.6.data/scripts/uv"; export UV_CACHE_DIR="$W/.uvcache" UV_PYTHON_INSTALL_DIR="$W/.uvpython"
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ HF_HOME=$W/hf_cache HF_HUB_OFFLINE=1
PY=.venv-qwen/bin; RUN=runs/qwen3asr_I2; DATA=runs/qwen_sft/I2
BASE=$(ls -d hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5)
CFG=configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml
echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# 1. data (Whisper tokenizer for the same usable_for_loss filter -> main .venv)
[ -e $DATA/summary.json ] || .venv/bin/python scripts/make_qwen_sft_data.py --config configs/train_lora_large_v3_I2_h100.yaml --out $DATA
# datasets for the SFT script; keep huggingface-hub < 1.0 (transformers 4.57.6; datasets alone pulled hub 2.4.0)
$PY/python -c "import datasets, transformers" 2>/dev/null || "$UV" pip install --python $PY/python datasets "huggingface-hub>=0.34,<1.0"
$PY/python -c "import datasets, transformers, huggingface_hub; print('datasets', datasets.__version__, 'transformers', transformers.__version__, 'hub', huggingface_hub.__version__)"

SFT="$PY/python scripts/qwen3_asr_sft.py --model_path $BASE --train_file $DATA/train.jsonl --lr 2e-5 \
  --lr_scheduler_type linear --warmup_ratio 0.05 --grad_acc 1 --num_workers 8 --seed 1234 --log_steps 10 --save_only_model 1"
# 2. smoke test: 10 steps, no eval
[ -e runs/qwen3asr_I2_smoke/done ] || { $SFT --output_dir runs/qwen3asr_I2_smoke --batch_size 16 --max_steps 10 \
  --save_steps 10 > runs/qwen3asr_I2_smoke.out 2>&1 || { tail -30 runs/qwen3asr_I2_smoke.out; exit 1; }
  touch runs/qwen3asr_I2_smoke/done; grep -o "{'loss'[^}]*}" runs/qwen3asr_I2_smoke.out | tail -2; }

# 3. training: batch 16 x 3 epochs (I2: 16 x 3), checkpoint + val loss every 200 steps, all checkpoints kept
[ -e $RUN/done ] || { $SFT --output_dir $RUN --eval_file $DATA/val.jsonl --batch_size 16 --epochs 3 --save_steps 200 \
  --save_total_limit 50 > $RUN.out 2>&1 || { tail -30 $RUN.out; exit 1; }; touch $RUN/done; }
grep -o "{'eval_loss'[^}]*}" $RUN.out | cut -c1-80
echo "trained $(date -Is)"

# 4. selection on users' val (decoding = Qwen zero-shot setting chosen on val: 25 s chunks)
score() { echo "$1 $($PY/wf-score reports/runs/$1 2>/dev/null | grep -o 'official MER [0-9.]*%')"; }
cks=""
for ck in $(ls -d $RUN/checkpoint-* | sort -t- -k2 -n); do
  step=step_${ck##*-}; cks="$cks $step"
  sed "s#checkpoint_path: null.*#checkpoint_path: $ck#" $CFG > /tmp/qwen_$step.yaml
  for set in bczh:benchmark_chinese online:online_phase1; do
    id=sel_qwen3asr_I2_${step}_${set%%:*}
    [ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config /tmp/qwen_$step.yaml \
      --manifest data/manifests/${set#*:}.jsonl --split val --run-id $id > runs/$id.out 2>&1
    score $id
  done
done
best=$($PY/python scripts/pick_checkpoint.py qwen3asr_I2 $cks)
echo "== picked $best (lowest pooled users' val)"
sed "s#checkpoint_path: null.*#checkpoint_path: $RUN/checkpoint-${best#step_}#" $CFG > configs/experiments/qwen3asr_I2_picked.yaml

# 5. locked test sets, once for the picked checkpoint
for set in bv1:benchmark_v1 ownertest:recording_freeform_test v2test:recording_v2_test; do
  id=${set%%:*}_qwen3asr_I2_$best
  [ -e reports/runs/$id/run_meta.json ] || $PY/wf-transcribe --config configs/experiments/qwen3asr_I2_picked.yaml \
    --manifest data/manifests/${set#*:}.jsonl --split test --allow-test --run-id $id > runs/$id.out 2>&1
  score $id
done
chmod 700 "$W"
echo QWEN_I2_DONE
