"""Decode a manifest split through the production VoiceCursor HTTP API and write a run dir for wf-score.

Sends each clip to POST /api/transcription/transcript as an ordinary dictate request (server default model and
cleanup, languages auto, no context / dictionary, source=arena so no history rows are written). The scored
hypothesis is the final `text`; `original_text` (raw ASR) is kept as hypothesis_raw.

Usage:
  VOICECURSOR_REST_TOKEN=... python scripts/run_voicecursor_api.py \
      --manifest data/manifests/recording_v2_test.jsonl --split test --run-id v2test_voicecursor_prod
  .venv/bin/wf-score reports/runs/v2test_voicecursor_prod
Token: Clerk session JWT (env VOICECURSOR_REST_TOKEN, or --token-file). Clips already in predictions.jsonl
are skipped, so a run can be resumed with a fresh token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from whisperflow.data.schema import load_manifest


def _multipart(fields: dict[str, str], filename: str, data: bytes) -> tuple[bytes, str]:
    b = f"----wf{uuid.uuid4().hex}"
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def transcribe(base_url: str, token: str, path: Path, rec) -> tuple[int, dict | str, float]:
    fields = {"mode": "dictate", "audio_format": path.suffix.lstrip(".") or "wav",
              "sample_rate": str(rec.sample_rate_hz or 16000), "source": "arena",
              "audio_duration_seconds": f"{rec.duration_sec:.3f}"}
    body, ctype = _multipart(fields, path.name, path.read_bytes())
    req = Request(base_url.rstrip("/") + "/api/transcription/transcript", data=body, method="POST",
                  headers={"Authorization": f"Bearer {token}", "Content-Type": ctype, "Accept": "application/json"})
    t0 = time.monotonic()
    try:
        with urlopen(req, timeout=300) as r:
            code, raw = r.getcode(), r.read().decode("utf-8", "replace")
    except HTTPError as e:
        code, raw = e.code, (e.read().decode("utf-8", "replace") if e.fp else str(e))
    except URLError as e:
        code, raw = 0, str(e.reason)
    dt = time.monotonic() - t0
    try:
        return code, json.loads(raw), dt
    except json.JSONDecodeError:
        return code, raw, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-root", default="data/private")
    ap.add_argument("--base-url", default="https://api.voicecursor.ai")
    ap.add_argument("--token-file", type=Path)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    token = (a.token_file.read_text().strip() if a.token_file else os.environ.get("VOICECURSOR_REST_TOKEN", "")).strip()
    if not token:
        sys.exit("no token: set VOICECURSOR_REST_TOKEN or pass --token-file")

    recs = [r for r in load_manifest(a.manifest) if r.split == a.split][: a.limit]
    run_dir = Path("reports/runs") / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    done = {json.loads(line)["utterance_id"] for line in pred_path.open()} if pred_path.exists() else set()
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_id": a.run_id, "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {"model": {"name_or_path": "voicecursor-production-api"},
                   "decoding": {"endpoint": a.base_url + "/api/transcription/transcript", "mode": "dictate",
                                "languages": "auto (omitted)", "context": "none", "source": "arena",
                                "scored_field": "text (final, after cleanup)"},
                   "runtime": {"backend": "voicecursor_http"},
                   "data": {"data_root": a.data_root, "manifest": a.manifest, "split": a.split}},
        "transcriber": {"backend": "voicecursor_http", "base_url": a.base_url},
        "manifest": a.manifest, "split": a.split, "n_utterances": len(recs),
    }, indent=2, ensure_ascii=False))

    n_fail = 0
    with pred_path.open("a", encoding="utf-8") as out:
        for i, rec in enumerate(recs, 1):
            if rec.utterance_id in done:
                continue
            code, body, dt = transcribe(a.base_url, token, Path(a.data_root) / rec.audio_path, rec)
            if code in (401, 403):
                sys.exit(f"HTTP {code} (token expired or rejected) at {rec.utterance_id}; rerun with a new token to resume")
            ok = code == 200 and isinstance(body, dict) and not body.get("error")
            if not ok:
                n_fail += 1
                print(f"[{i}/{len(recs)}] FAIL {rec.utterance_id} HTTP {code}: {str(body)[:200]}", flush=True)
                continue  # not written, so a rerun retries it
            out.write(json.dumps({
                "utterance_id": rec.utterance_id, "hypothesis": body.get("text") or "",
                "hypothesis_raw": body.get("original_text") or "", "detected_language": body.get("language"),
                "vc_audio_id": body.get("audio_id"), "audio_sec": rec.duration_sec, "request_sec": round(dt, 3),
            }, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(recs)}] {dt:5.1f}s {rec.utterance_id}", flush=True)
    print(f"done {a.run_id}: {len(recs)} clips, {n_fail} failed (rerun to retry)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
