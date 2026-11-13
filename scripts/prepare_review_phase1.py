"""Prepare the phase 1 review table written by trace-lookup bench.py draft-gt (owner choices 2026-10-31).

- final_transcript is pre-filled with the draft (Whisper large-v3 since 2026-11-01; Gemini before) minus
  嗯 / 呃 / um / uh (option B); the owner decides the
  context-dependent fillers (就是 / 那个 / 然后 …) by ear.
- Adds split / scenario / seen / duration and `draft_vs_online` (token error rate of the draft against the
  production ASR text), and sorts the most different rows first: likely hard cases.
- Drafts stuck in a repetition loop (compression ratio > 2.4, a known large-v3 short-clip failure) are replaced by
  the production ASR text, with a note.
- Rows the owner already reviewed (verdict filled) are never overwritten.
Usage: uv run python scripts/prepare_review_phase1.py
"""

import csv
import json
import zlib
from pathlib import Path

from whisperflow.eval.metrics import align_text
from whisperflow.text.fillers import strip_fillers

BENCH = Path("data/private/online_phase1/bench/online-zh-phase1")
CANDS = Path("data/private/online_phase1/candidates.jsonl")
FIRST = ["priority", "split", "scenario", "seen", "duration_s", "draft_vs_online", "audio_file", "final_transcript",
         "verdict", "note", "draft_1", "online_raw", "online_output"]


def is_loop(t: str) -> bool:
    b = t.encode()
    return len(b) > 20 and len(b) / len(zlib.compress(b)) > 2.4


def diff_rate(a: str, b: str) -> float:
    al = align_text(a, b)
    n = len(al.ref) or 1
    return round(sum(op != "eq" for op, _, _ in al.ops) / n, 3)


def main() -> None:
    meta = {r["audio_id"]: r for r in map(json.loads, open(CANDS, encoding="utf-8"))}
    path = BENCH / "review.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    for r in rows:
        m = meta[r["audio_id"]]
        r.update(split=m["split"], scenario=m["scenario"], seen="yes" if m["seen"] else "no",
                 duration_s=round((m["cleanup_audio_duration_ms"] or 0) / 1000, 1))
        draft = r.get("draft_1") or ""
        r["draft_vs_online"] = diff_rate(r.get("online_raw") or "", draft) if draft else ""
        if not r.get("verdict"):
            r["final_transcript"] = strip_fillers(draft)
            if draft and is_loop(draft):
                r["final_transcript"] = strip_fillers(r.get("online_raw") or "")
                r["note"] = "Whisper 草稿陷入重复循环，已改用线上识别预填"
            r["priority"] = "high" if not draft or r["draft_vs_online"] > 0.15 else "low"
    rows.sort(key=lambda r: (r["verdict"] != "", r["priority"] != "high", -(r["draft_vs_online"] or 9)))
    cols = FIRST + [c for c in rows[0] if c not in FIRST]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    hi = sum(r["priority"] == "high" for r in rows)
    print(f"{path}: {len(rows)} rows, {hi} high priority (draft vs production ASR differ > 15%, or no draft)")


if __name__ == "__main__":
    main()
