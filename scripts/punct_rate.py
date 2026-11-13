"""Punctuation in ASR output vs references (2026-11-09). MER ignores punctuation, but the product and the
post-processor see it: half of the training clips (ASCEND) have no punctuation in their labels, so a fine-tuned
model may learn to drop it on Chinese. Per run: share of clips with any punctuation, punctuation marks per 100
characters, and sentence-final mark rate, next to the same numbers for the references.

    uv run python scripts/punct_rate.py sel_lora_large_v3_I2_step_600_bczh sel_qwen3asr_17b_chunk25_base_bczh ...
"""

import json
import re
import sys
from pathlib import Path

MANIFESTS = {"bczh": "data/manifests/benchmark_chinese.jsonl", "online": "data/manifests/online_phase1.jsonl"}
PUNCT = re.compile(r"[，。？！、；：,.?!;:]")
FINAL = re.compile(r"[。？！.?!]\s*$")


def stats(texts: list[str]) -> dict:
    chars = sum(len(t) for t in texts) or 1
    n = len(texts)
    return {"clips": n, "with_punct_%": round(100 * sum(bool(PUNCT.search(t)) for t in texts) / n, 1),
            "marks_per_100_chars": round(100 * sum(len(PUNCT.findall(t)) for t in texts) / chars, 2),
            "final_mark_%": round(100 * sum(bool(FINAL.search(t)) for t in texts) / n, 1)}


def main() -> None:
    for run in sys.argv[1:]:
        tag = run.rsplit("_", 1)[-1]
        refs = {r["utterance_id"]: r["raw_transcript"]
                for r in map(json.loads, open(MANIFESTS[tag], encoding="utf-8"))}
        pred_path = Path("reports/runs") / run / "predictions.jsonl"
        preds = [json.loads(line) for line in open(pred_path, encoding="utf-8")]
        print(run, "hyp", stats([p["hypothesis"] for p in preds]),
              "ref", stats([refs[p["utterance_id"]] for p in preds]))


if __name__ == "__main__":
    main()
