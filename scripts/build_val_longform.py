"""Synthetic long-form validation set: consecutive val clips of one speaker joined into 35-60 s recordings.

Why: lora_v1 broke on clips > 30 s (long-form decoding needs timestamps), but val has only 8 real long clips and
benchmark v1 is the test set, so it must not drive decisions. Joining val clips gives a long-form check on val
speakers only. The joined clips have 0.3 s of silence between them, and their references are joined with a space.

Output: data/private/val_longform/*.wav + data/manifests/val_longform.jsonl (split "val", flag synthetic_concat).
Usage: uv run python scripts/build_val_longform.py
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SOURCES = ["data/manifests/ascend.jsonl", "data/manifests/benchmark_chinese.jsonl"]
DATA_ROOT = Path("data/private")
OUT_DIR = DATA_ROOT / "val_longform"
OUT = Path("data/manifests/val_longform.jsonl")
GAP = np.zeros(int(0.3 * 16000), dtype=np.float32)
MIN_SEC, MAX_SEC = 35.0, 60.0
MAX_PER_SOURCE = {"ascend": 40, "benchmark_chinese": 20}
rng = random.Random(20261030)

by_spk = defaultdict(list)
for path in SOURCES:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "val" and r["duration_sec"] <= 30:
            by_spk[(r["source_id"], r["speaker_id"])].append(r)

OUT_DIR.mkdir(parents=True, exist_ok=True)
made, per_source = [], defaultdict(int)
for (src, _spk), recs in sorted(by_spk.items()):
    recs.sort(key=lambda r: r["source_utterance_id"] or "")  # ASCEND names encode session + line order
    group, dur = [], 0.0
    target = rng.uniform(MIN_SEC, MAX_SEC)
    for r in recs:
        if per_source[src] >= MAX_PER_SOURCE[src]:
            break
        if dur + r["duration_sec"] + 0.3 > MAX_SEC:
            group, dur, target = [], 0.0, rng.uniform(MIN_SEC, MAX_SEC)  # too long; start over at this clip
        group.append(r)
        dur += r["duration_sec"] + 0.3
        if dur >= target:
            audio = np.concatenate([x for g in group
                                    for x in (sf.read(DATA_ROOT / g["audio_path"], dtype="float32")[0], GAP)])
            uid_src = "+".join(g["utterance_id"] for g in group)
            uid = "val_longform-" + hashlib.sha256(uid_src.encode()).hexdigest()[:16]
            wav = OUT_DIR / f"{uid}.wav"
            sf.write(wav, audio, 16000, subtype="PCM_16")
            made.append({
                **{k: group[0][k] for k in ("speaker_id", "session_id", "license_or_consent_id",
                                            "recording_device", "noise_tag", "label_status")},
                "utterance_id": uid, "audio_path": str(wav.relative_to(DATA_ROOT)),
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "duration_sec": round(len(audio) / 16000, 4), "source_id": "val_longform", "split": "val",
                "raw_transcript": " ".join(g["raw_transcript"] for g in group), "formatted_transcript": None,
                "primary_scenario": "zh_en_switch" if any(g["primary_scenario"] == "zh_en_switch" for g in group)
                else group[0]["primary_scenario"],
                "language_tags": sorted({t for g in group for t in g["language_tags"]}),
                "hotwords_present": [], "reviewer_id": group[0].get("reviewer_id"), "quality_flags": ["synthetic_concat"],
                "source_utterance_id": uid_src[:200], "parent_utterance_id": group[0]["utterance_id"],
                "sample_rate_hz": 16000, "num_channels": 1,
            })
            per_source[src] += 1
            group, dur, target = [], 0.0, rng.uniform(MIN_SEC, MAX_SEC)

with open(OUT, "w", encoding="utf-8") as f:
    for r in made:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(dict(per_source), f"{sum(r['duration_sec'] for r in made) / 3600:.2f} h ->", OUT)
