#!/usr/bin/env bash
# lora_v3 final evals (2026-10-31), reference decoding v2, same local machine (MPS) as the base / lora_v2 numbers.
# Checkpoint step_1200 was chosen on val only (runs/lora_v3/select.log).
set -euo pipefail
export HF_HUB_OFFLINE=1
T="uv run wf-transcribe --split test --allow-test --config configs/experiments/reference_v2_lora_v3.yaml"
$T --run-id benchmark_v1_lora_v3_ref2
uv run wf-score reports/runs/benchmark_v1_lora_v3_ref2 | grep -i "official MER"
$T --manifest data/manifests/ascend.jsonl --run-id ascend_test_lora_v3_ref2
uv run wf-score reports/runs/ascend_test_lora_v3_ref2 | grep -i "official MER"
