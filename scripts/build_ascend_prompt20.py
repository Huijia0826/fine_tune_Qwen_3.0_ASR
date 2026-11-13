"""Build the 20-clip ASCEND validation manifest for the prompt comparison (2026-10-28).

Input:  data/private/ascend/raw/validation-00000-of-00001.parquet
        (CAiRE/ASCEND, CC BY-SA 4.0, sha256 3bdec53d...cf7e2 verified against huggingface.co)
Output: data/private/ascend/prompt20/*.wav  +  data/manifests/ascend_prompt20.jsonl

Selection (deterministic, seed 1234): clips >= 2.0 s (very short fillers like "嗯" say nothing
about prompting) whose reference has no "[UNK]" (untranscribed speech -> the reference is
incomplete and the normalizer would score "unk" as a word), stratified by ASCEND's own
language label: 10 mixed, 6 zh, 4 en. Mixed is over-weighted because code-switching is what
the prompts target. This is a small pilot to get a direction, not a measurement — 20 clips
cannot separate small differences.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

PARQUET = Path("data/private/ascend/raw/validation-00000-of-00001.parquet")
OUT_AUDIO = Path("data/private/ascend/prompt20")
OUT_MANIFEST = Path("data/manifests/ascend_prompt20.jsonl")
QUOTA = {"mixed": 10, "zh": 6, "en": 4}
SCENARIO = {"mixed": "zh_en_switch", "zh": "zh_general", "en": "en_general"}
LANG_TAGS = {"mixed": ["zh", "en"], "zh": ["zh"], "en": ["en"]}
MIN_SEC, SEED = 2.0, 1234

df = pq.read_table(PARQUET).to_pandas()
df = df[(df["duration"] >= MIN_SEC) & ~df["transcription"].str.contains("[UNK]", regex=False)]
rng = np.random.default_rng(SEED)
picked = []
for lang, k in QUOTA.items():
    pool = df[df["language"] == lang].sort_values("id")
    picked.append(pool.iloc[sorted(rng.choice(len(pool), size=k, replace=False))])

OUT_AUDIO.mkdir(parents=True, exist_ok=True)
rows = []
for sub in picked:
    for r in sub.itertuples():
        audio, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, f"{r.id}: unexpected sample rate {sr}"
        wav = OUT_AUDIO / f"ascend_val_{r.id}.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        spk, ses = f"ascend-spk{r.original_speaker_id}", f"ascend-ses{r.session_id}"
        rows.append({
            "utterance_id": f"ascend_val_{r.id}",
            "audio_path": str(wav.relative_to("data/private")),
            "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "duration_sec": round(len(audio) / sr, 4),
            "speaker_id": spk,
            "session_id": ses,
            "source_id": "ascend",
            "license_or_consent_id": "ascend_cc_by_sa_4",
            "split": "val",
            "raw_transcript": r.transcription,
            "primary_scenario": SCENARIO[r.language],
            "language_tags": LANG_TAGS[r.language],
            "label_status": "source_provided",
            "quality_flags": [f"topic_{r.topic}"],
            "source_utterance_id": Path(r.path).name,
            "sample_rate_hz": sr,
            "num_channels": 1,
        })

OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{len(rows)} clips -> {OUT_MANIFEST}; "
      f"{sum(r['duration_sec'] for r in rows):.1f} s; speakers={sorted({r['speaker_id'] for r in rows})}")
