#!/usr/bin/env bash
# benchmark v1 (locked test) for large-v3 zero-shot, B step 800 (candidate, chosen on val) and A step 1400 (reference).
# Run once on the GPU box (owner approved the upload 2026-11-02). Each wf-transcribe appends to reports/test_access_log.jsonl.
set -euo pipefail
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1
T=".venv/bin/wf-transcribe --manifest data/manifests/benchmark_v1.jsonl --split test --allow-test"
$T --config configs/experiments/zeroshot_large_v3.yaml --run-id bv1_large_v3_zs > runs/bv1_zs.out 2>&1; echo "zs done"
$T --config configs/experiments/bv1_lora_large_v3_B_step_800.yaml --run-id bv1_lora_large_v3_B_step_800 > runs/bv1_B.out 2>&1; echo "B800 done"
$T --config configs/experiments/bv1_lora_large_v3_A_step_1400.yaml --run-id bv1_lora_large_v3_A_step_1400 > runs/bv1_A.out 2>&1; echo "A1400 done"
echo ALL_DONE
