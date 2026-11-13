"""Build the wf-ingest metadata CSV for benchmark v1.0 (arena batch 01a17cb6...).

Inputs (all under data/private/benchmark_v1/, gitignored):
  arena_object_keys.csv   file_name, object_key, ground_truth   (exported read-only from arena)
  audio/*.json            per-file metadata written by the audio-fetch skill ("input" = object_key)

Speaker IDs: the online user id (PostHog distinct_id, or user_xxx in the S3 key) is
pseudonymised with sha256 so manifests never contain raw user ids. Files uploaded to arena
manually have no user id -> one pseudo-speaker per file, flagged in the output.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("data/private/benchmark_v1")
USER_RE = re.compile(r"/(user_[A-Za-z0-9]+)/")


def pseudo(uid: str) -> str:
    return "u" + hashlib.sha256(uid.encode()).hexdigest()[:12]


meta_by_key = {}
for p in (ROOT / "audio").glob("*.json"):
    m = json.loads(p.read_text(encoding="utf-8"))
    meta_by_key[m["input"]] = m

rows, unknown = [], 0
with open(ROOT / "arena_object_keys.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        m = meta_by_key[r["object_key"]]
        user = m.get("distinct_id") or (USER_RE.search(r["object_key"]) or [None, None])[1]
        if user:
            spk = pseudo(user)
        else:
            spk = "unknown-" + r["file_name"][:8]
            unknown += 1
        rows.append({
            "audio_path": Path(m["local_path"]).name,
            "transcript": r["ground_truth"],
            "speaker_id": spk,
            "session_id": spk,
            "source_utterance_id": r["file_name"],
            "recording_device": m.get("client_source") or "unknown",
        })

out = ROOT / "ingest_metadata.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}; speakers={len({r['speaker_id'] for r in rows})}, "
      f"unknown-speaker files={unknown}")
