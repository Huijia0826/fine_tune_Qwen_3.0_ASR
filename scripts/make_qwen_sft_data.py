"""Convert a Whisper training config's data into the Qwen3-ASR SFT JSONL format (2026-11-09).

Same clips as the Whisper run: the config's manifests (repeats = oversampling), its train / eval splits and the
same usable_for_loss filter (audio <= 30 s, label <= max_label_tokens Whisper tokens). Label text = raw_transcript
(the strict transcript), prefixed the way Qwen3-ASR emits it:
    {"audio": "/abs/path.wav", "text": "language Chinese<asr_text>我们先跑一下"}
Language: pure-English clips -> English, everything else (zh, zh-en) -> Chinese (clip_language, as for Whisper;
Qwen's own zero-shot LID also calls our code-switch clips Chinese).

    .venv/bin/python scripts/make_qwen_sft_data.py --config configs/train_lora_large_v3_I2_h100.yaml \
        --out runs/qwen_sft/I2
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import WhisperTokenizer

from whisperflow.data.common import load_yaml
from whisperflow.training.data import clip_language, load_examples, usable_for_loss

QWEN_LANG = {"en": "English", "zh": "Chinese"}


def qwen_label(text: str, language: str) -> str:
    return f"language {QWEN_LANG[language]}<asr_text>{text}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="directory for train.jsonl / val.jsonl / summary.json")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = cfg["data"]
    base = cfg["base_model"]
    tok = WhisperTokenizer.from_pretrained(base["name_or_path"], revision=base["revision"])
    root = Path(d["data_root"]).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config": args.config}
    for name, split in (("train", d["train_split"]), ("val", d["eval_split"])):
        recs, dropped = usable_for_loss(load_examples(d["manifests"], split), tok, d["max_label_tokens"])
        langs = Counter()
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                lang = clip_language(r)
                langs[lang] += 1
                f.write(json.dumps({"audio": str(root / r.audio_path), "text": qwen_label(r.raw_transcript, lang),
                                    "utterance_id": r.utterance_id}, ensure_ascii=False) + "\n")
        summary[name] = {"clips": len(recs), "dropped_by_usable_for_loss": dropped,
                         "unique_clips": len({r.utterance_id for r in recs}),
                         "hours": round(sum(r.duration_sec for r in recs) / 3600, 2), "languages": dict(langs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
