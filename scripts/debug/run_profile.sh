#!/usr/bin/env bash
# E7 profiling (2026-11-02): stage breakdown + decoder-step bench, default SDPA vs cuDNN SDPA disabled.
set -euo pipefail
cd "$HOME/whisperflow"; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1
uptime
for v in default no_cudnn; do
  flag=""; [ $v = no_cudnn ] && flag="--no-cudnn-sdp"
  .venv/bin/python scripts/profile_decoding.py --config configs/experiments/latency/hf_bf16_beam5.yaml \
    --run-id profile_decoding_$v $flag > runs/profile_$v.out 2>&1 || { echo "FAILED $v"; tail -5 runs/profile_$v.out; }
  echo "done $v"
done
uptime
echo PROFILE_DONE
