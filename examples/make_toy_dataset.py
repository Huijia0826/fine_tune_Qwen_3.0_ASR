"""Download a tiny public dataset (English LibriSpeech dummy split, 73 clips, ~10 MB) and write manifests, so you
can check the whole pipeline end to end before using your own data.

    pip install datasets soundfile
    python examples/make_toy_dataset.py --out examples/toy
    qasr pipeline -c examples/toy.yaml          # 5 training steps, then select + test

The point is to verify the environment, not to get a good model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/toy")
    ap.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(a.dataset, a.config, split=a.split).cast_column("audio", Audio(sampling_rate=16000))
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        name = f"clip_{i:04d}.wav"
        sf.write(out / "audio" / name, ex["audio"]["array"], 16000)
        text = ex["text"].capitalize()
        rows.append({"id": f"toy-{i:04d}", "audio": f"audio/{name}", "text": text,
                     "speaker": str(ex.get("speaker_id", i)), "language": "en"})
    splits = {"train": rows[:50], "val": rows[50:60], "test": rows[60:]}
    for name, part in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in splits.items()}, "->", out)


if __name__ == "__main__":
    main()
