# Fine Tune Qwen 3.0 ASR 

Fine-tune Fine Tune Qwen 3.0 ASR 1.7B (LoRA).

**Status (2026-10-27): Phase 0 complete, Phase 1 tooling built. No real data ingested, no
model weights downloaded, no metrics measured yet.** See `docs/decisions.md` → owner decisions.

## Setup
```bash
brew install uv ffmpeg
uv sync                 # core: data pipeline + scoring (no torch)
uv sync --extra ml      # + torch/transformers/peft for model inference & training
uv run pytest           # 77 tests, all synthetic fixtures
```

## Data pipeline
```bash
# 0. register + approve the source in configs/sources.yaml
# 1. ingest: metadata TSV (audio_path, transcript, speaker_id, ...) -> manifest
uv run wf-ingest --metadata meta.tsv --audio-dir /path/corpus --source-id ascend \
    --license-id ascend_ccbysa4 --convert-16k --out data/manifests/ascend.jsonl
# 2. validate: schema, provenance/license, hashes, decode, duration, silence, clipping
uv run wf-validate data/manifests/ascend.jsonl --report reports/dataset/validate.json
# 3. duplicates
uv run wf-dedup data/manifests/ascend.jsonl --out data/manifests/pool.jsonl
# 4. split (speaker/session-disjoint, stratified, deterministic) and lock the test set once
uv run wf-split data/manifests/pool.jsonl --out data/manifests/pilot.jsonl --lock-test
# 5. independent leakage audit + dataset report (markdown + json)
uv run wf-audit data/manifests/pilot.jsonl --out-dir reports/dataset/pilot
```

## Baseline evaluation (requires approved model download)
```bash
# set model.revision in configs/eval.yaml to the pinned commit first
uv run wf-transcribe --config configs/eval.yaml --split val           # -> reports/runs/<id>/
uv run wf-score reports/runs/<id>                                       # metrics.md/json, errors.csv
uv run wf-transcribe --split test --allow-test   # milestones only; logged to reports/test_access_log.jsonl
```

## Layout
```
configs/   dataset.yaml  eval.yaml  sources.yaml  train_lora.yaml (template)
docs/      decisions.md  dataset_spec.md  annotation_policy.md  data_sources.md  environment.md
src/whisperflow/
  text/normalize.py      versioned normalization + mixed zh/en tokenization (zh_en_v1)
  data/                  schema, ingest, validate, dedup, split, audit, audio utils
  eval/                  metrics (alignment, MER, bootstrap), transcribe (HF Whisper), score
  training/ decoding/ serving/ formatting/   (later phases)
tests/     synthetic-only unit + integration tests
```
