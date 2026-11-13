"""E3: split the train speakers' long clips (> 30 s) into <= 28 s pieces with slices of the human reference.

Candidates: benchmark_chinese (excluded_over_30s_train_speaker), arena_019f8e51 and arena_019f6442
(excluded_over_30s). Timing comes from base whisper-small long-form segments (reference decoding v2 settings,
language fixed per source); text comes from the reference (src/whisperflow/data/segment.py). Pieces failing
the quality check are dropped. Owner corrections in configs/label_corrections.yaml are applied last.

Output: data/private/segmented/*.wav, data/manifests/segmented_long.jsonl (split train, flag segmented_from_long),
reports/dataset/segmented_spotcheck.tsv (20 random pieces to listen to).
Usage: uv run python scripts/segment_long_clips.py
"""

import csv
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisperflow.data.corrections import apply_corrections
from whisperflow.data.segment import Segment, plan_pieces
from whisperflow.eval.transcribe import _disable_mps_deferred_stop

REV = "973afd24965f72e36ca33b3055d56a652f456b4d"
DATA_ROOT = Path("data/private")
OUT_DIR = DATA_ROOT / "segmented"
OUT = Path("data/manifests/segmented_long.jsonl")
SPOT = Path("reports/dataset/segmented_spotcheck.tsv")
SOURCES = [  # manifest, flag marking the long clips, decoding language
    ("data/manifests/benchmark_chinese.jsonl", "excluded_over_30s_train_speaker", "zh"),
    ("data/manifests/arena_019f8e51.jsonl", "excluded_over_30s", "en"),
    ("data/manifests/arena_019f6442.jsonl", "excluded_over_30s", "zh"),
]

_disable_mps_deferred_stop()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
proc = WhisperProcessor.from_pretrained("openai/whisper-small", revision=REV)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", revision=REV).to(device).eval()

OUT_DIR.mkdir(parents=True, exist_ok=True)
records, spot, stats = [], [], {"parents": 0, "pieces": 0, "dropped": 0, "parent_h": 0.0, "piece_h": 0.0}
for manifest, flag, lang in SOURCES:
    for line in open(manifest, encoding="utf-8"):
        parent = json.loads(line)
        if flag not in parent["quality_flags"]:
            continue
        audio, sr = sf.read(DATA_ROOT / parent["audio_path"], dtype="float32")
        feats = proc.feature_extractor([audio], sampling_rate=16000, return_tensors="pt", truncation=False,
                                       padding="longest", return_attention_mask=True)
        torch.manual_seed(1234)
        with torch.inference_mode():
            out = model.generate(feats.input_features.to(device), attention_mask=feats.attention_mask.to(device),
                                 return_timestamps=True, return_segments=True, num_beams=5, language=lang,
                                 task="transcribe", condition_on_prev_tokens=False,
                                 temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35,
                                 logprob_threshold=-1.0)
        segs = [Segment(float(s["start"]), float(s["end"]),
                        proc.tokenizer.decode(s["tokens"], skip_special_tokens=True)) for s in out["segments"][0]]
        pieces, dropped = plan_pieces(parent["raw_transcript"], segs, len(audio) / sr)
        stats["parents"] += 1
        stats["dropped"] += dropped
        stats["parent_h"] += parent["duration_sec"] / 3600
        for k, p in enumerate(pieces):
            chunk = audio[int(p.start * sr):int(p.end * sr)]
            wav = OUT_DIR / f"{parent['utterance_id']}_{k:02d}.wav"
            sf.write(wav, chunk, sr, subtype="PCM_16")
            digest = hashlib.sha256(wav.read_bytes()).hexdigest()
            records.append({
                **parent, "utterance_id": f"{parent['source_id']}-seg-{digest[:16]}",
                "audio_path": str(wav.relative_to(DATA_ROOT)), "sha256": digest,
                "duration_sec": round(len(chunk) / sr, 4), "split": "train", "raw_transcript": p.reference,
                "quality_flags": [f for f in parent["quality_flags"] if not f.startswith("excluded")]
                + ["segmented_from_long"],
                "parent_utterance_id": parent["utterance_id"],
                "source_utterance_id": f"{parent['source_utterance_id']}#{k:02d}",
            })
            spot.append((str(wav), p.token_error_rate, p.reference, p.hypothesis))
            stats["pieces"] += 1
            stats["piece_h"] += len(chunk) / sr / 3600
        print(f"{parent['utterance_id']}: {parent['duration_sec']:.0f}s -> {len(pieces)} pieces, "
              f"{dropped} dropped", flush=True)

records, applied, warnings = apply_corrections(records, "configs/label_corrections.yaml")
print(f"label corrections applied: {applied}", *warnings, sep="\n")
with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
SPOT.parent.mkdir(parents=True, exist_ok=True)
with open(SPOT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["audio", "token_error_rate", "reference_slice", "whisper_text"])
    w.writerows(random.Random(20261030).sample(spot, min(20, len(spot))))
print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}))
