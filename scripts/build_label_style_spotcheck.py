"""Label-style spot check: are benchmark_chinese / arena_019f8e51 references verbatim or cleaned up?

Samples 6 train clips (<= 30 s, not segmented) per source, decodes them with base whisper-small, and writes
reports/dataset/label_style_spotcheck.md for the owner to listen to.
Usage: uv run python scripts/build_label_style_spotcheck.py
"""

import json
import random
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisperflow.data.segment import edit_rate, tokens
from whisperflow.eval.transcribe import _disable_mps_deferred_stop
from whisperflow.text.normalize import _t2s

REV = "973afd24965f72e36ca33b3055d56a652f456b4d"
DATA_ROOT = Path("data/private")
OUT = Path("reports/dataset/label_style_spotcheck.md")
SOURCES = [("data/manifests/benchmark_chinese.jsonl", "zh"), ("data/manifests/arena_019f8e51.jsonl", "en")]
PER_SOURCE = 6

_disable_mps_deferred_stop()
device = "mps" if torch.backends.mps.is_available() else "cpu"
proc = WhisperProcessor.from_pretrained("openai/whisper-small", revision=REV)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", revision=REV).to(device).eval()
conv = _t2s()
rng = random.Random(20261031)

lines = ["# Label-style spot check (benchmark_chinese + arena_019f8e51)", "",
         "For each clip: listen, then say whether the **reference** is **verbatim** (matches what was said, word for",
         "word), **cleaned** (fillers/repeats removed or wording smoothed), or **wrong** (misheard words). Whisper's",
         "text is only a listening aid.", ""]
n = 0
for manifest, lang in SOURCES:
    recs = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    recs = [r for r in recs if r["split"] == "train" and r["duration_sec"] <= 30]
    for r in rng.sample(recs, PER_SOURCE):
        audio, sr = sf.read(DATA_ROOT / r["audio_path"], dtype="float32")
        feats = proc.feature_extractor([audio], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(feats.input_features.to(device), num_beams=5, language=lang, task="transcribe")
        hyp = conv.convert(proc.tokenizer.decode(ids[0], skip_special_tokens=True).strip())
        err = edit_rate([t.key for t in tokens(r["raw_transcript"])], [t.key for t in tokens(hyp)])
        n += 1
        flags = [f for f in r["quality_flags"] if f.startswith("label")]
        lines += [f"## {n}. {r['source_id']} — {r['duration_sec']:.0f} s, error {err:.0%}"
                  + (f" ({', '.join(flags)})" if flags else ""), "", "```",
                  f'open "data/private/{r["audio_path"]}"', "```", "",
                  f"- **Reference (used for training):** {r['raw_transcript']}", f"- Whisper heard: {hyp}",
                  "- Verdict: ", ""]
        print(n, r["utterance_id"], f"{err:.0%}", flush=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
