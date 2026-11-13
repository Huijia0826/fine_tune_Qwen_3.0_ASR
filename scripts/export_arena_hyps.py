"""Export one system's outputs from the benchmark v1.0 arena xlsx as {utterance_id, hypothesis} JSONL,
keyed by our manifest utterance_id (mapped via source_utterance_id = arena file name).

Usage: python scripts/export_arena_hyps.py <xlsx> <column-prefix> <out.jsonl>
  e.g. column prefix "VoiceCursorE2E:voicecursor-e2e-transcript|lang:auto|raw_"
Failed rows are exported with an empty hypothesis (they count as full deletions, like in arena).
"""

import json
import sys

import openpyxl

from whisperflow.data.schema import load_manifest

xlsx, prefix, out = sys.argv[1:4]
rows = list(openpyxl.load_workbook(xlsx).active.iter_rows(values_only=True))
hdr = rows[0]
by_file = {r[0]: dict(zip(hdr, r, strict=False)) for r in rows[1:]}
n = 0
with open(out, "w", encoding="utf-8") as f:
    for rec in load_manifest("data/manifests/benchmark_v1.jsonl"):
        row = by_file[rec.source_utterance_id]
        ok = row[prefix + "status"] == "succeeded"
        f.write(json.dumps({"utterance_id": rec.utterance_id,
                            "hypothesis": (row[prefix + "text"] or "") if ok else ""},
                           ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} hypotheses -> {out}")
