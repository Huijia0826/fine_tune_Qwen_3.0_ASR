"""benchmark_chinese adapter: ground_truth.csv + index.csv + audio/*.ogg -> wf-ingest TSV.

Decisions (docs/decisions.md, 2026-10-29):
  * drop the recordings that are also in benchmark v1 (same S3 object = identical audio)
  * speaker ids are pseudonymised the same way as benchmark v1 ("u" + sha256(user)[:12]), so the
    two sources can be compared for shared speakers without storing raw user ids
  * screen_text / audio_context (PII) are not copied
  * labels kept as provided (punctuation, casing): label-style option B

Usage: uv run python scripts/benchmark_chinese_to_ingest.py
Then:  uv run wf-ingest --metadata data/private/benchmark_chinese/ingest_metadata.tsv \
         --audio-dir ~/Desktop/build_benchmark/benchmark_chinese/audio --convert-16k ...
"""

import csv
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

SRC = Path.home() / "Desktop/build_benchmark/benchmark_chinese"
OUT = Path("data/private/benchmark_chinese/ingest_metadata.tsv")
BENCH_V1_META = "data/private/benchmark_v1/audio/*.json"


def pseudo(uid: str) -> str:
    return "u" + hashlib.sha256(uid.encode()).hexdigest()[:12]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


gt = read_csv(SRC / "ground_truth.csv")
index = {r["audio_id"] + ".ogg": r for r in read_csv(SRC / "index.csv")}
in_bench_v1 = {json.loads(Path(p).read_text(encoding="utf-8"))["input"] for p in glob.glob(BENCH_V1_META)}

stats = Counter()
rows = []
for r in gt:
    stats["n_total"] += 1
    if not (SRC / "audio" / r["file"]).is_file():
        stats["n_no_audio"] += 1
        continue
    if not r["transcript"].strip():
        stats["n_empty_transcript"] += 1
        continue
    if r["audio_url"] in in_bench_v1:
        stats["n_in_benchmark_v1"] += 1
        continue
    spk = pseudo(index[r["file"]]["user"])
    rows.append({
        "audio_path": r["file"],
        "transcript": r["transcript"].strip(),
        "speaker_id": spk,
        "session_id": spk,  # no session info; one group per user
        "source_utterance_id": Path(r["file"]).stem,  # = audio_id
        "recording_device": index[r["file"]]["client_source"] or "unknown",
    })
    stats["n_written"] += 1

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(dict(stats), "->", OUT)
