"""Assign train / val to benchmark_chinese by speaker (decision 2026-10-29, option A).

  * val speakers: chosen in a seeded, deterministic order from speakers NOT in benchmark v1, until val holds
    >= VAL_SHORT_HOURS of <=30 s audio. All their clips go to val, including clips > 30 s (val is decoded
    long-form).
  * all other speakers: clips <= 30 s -> train. Their clips > 30 s stay out (too long to train on, and they
    can't go to val because those speakers are already in train).
  * speakers shared with benchmark v1 stay in train (owner choice); evaluation reports seen / unseen speakers.
  * clips whose transcript cannot match the audio (wf-validate "implausibly fast token rate") are excluded.

Usage: uv run python scripts/assign_benchmark_chinese_splits.py
"""

import hashlib
import json
from collections import Counter, defaultdict

from whisperflow.data.schema import parse_record

SRC = "data/manifests/benchmark_chinese_ingested.jsonl"
OUT = "data/manifests/benchmark_chinese.jsonl"
BENCH_V1 = "data/manifests/benchmark_v1.jsonl"
MAX_TRAIN_SEC = 30.0
VAL_SHORT_HOURS = 0.12
SEED = "20261029"


def bare(speaker_id: str) -> str:
    return speaker_id.split("-", 1)[1]  # drop the "<source_id>-" prefix added by wf-ingest


recs = [json.loads(line) for line in open(SRC, encoding="utf-8")]
bench_v1_speakers = {bare(json.loads(line)["speaker_id"]) for line in open(BENCH_V1, encoding="utf-8")}

by_spk = defaultdict(list)
for r in recs:
    by_spk[r["speaker_id"]].append(r)

candidates = sorted((s for s in by_spk if bare(s) not in bench_v1_speakers),
                    key=lambda s: hashlib.sha256((SEED + s).encode()).hexdigest())
val_speakers, val_short_sec = set(), 0.0
for s in candidates:
    if val_short_sec >= VAL_SHORT_HOURS * 3600:
        break
    val_speakers.add(s)
    val_short_sec += sum(r["duration_sec"] for r in by_spk[s] if r["duration_sec"] <= MAX_TRAIN_SEC)

counts, hours = Counter(), Counter()
for r in recs:
    if any("implausibly fast" in w for w in parse_record(r)[0].soft_warnings(1e9)):
        r["split"] = "unassigned"
        r["quality_flags"].append("excluded_transcript_audio_mismatch")
    elif r["speaker_id"] in val_speakers:
        r["split"] = "val"
    elif r["duration_sec"] <= MAX_TRAIN_SEC:
        r["split"] = "train"
    else:
        r["split"] = "unassigned"
        r["quality_flags"].append("excluded_over_30s_train_speaker")
    counts[r["split"]] += 1
    hours[r["split"]] += r["duration_sec"] / 3600

with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

shared = {s for s in by_spk if bare(s) in bench_v1_speakers}
print(f"speakers: {len(by_spk)} total, {len(shared)} shared with benchmark v1, {len(val_speakers)} in val")
for split in ("train", "val", "unassigned"):
    print(f"  {split:10s} {counts[split]:4d} clips {hours[split]:.2f} h")
