"""Fetch the owner's scripted recordings (recording script v1) and align them to the script (2026-11-04).

The owner dictated the script with VoiceCursor on the TEST environment: requests are not in production PostHog, but the
audio and a sidecar are in the voicecursor-test bucket, transcription/recordings/<user>/<date>/HHMMSS_<id>.{ogg|wav}
+ _transcript.json (audio_id, production ASR text, model). Everything in the time window is downloaded; each take is
matched to the script sentence its ASR text is closest to (character similarity after removing punctuation / case).
Takes scoring < MIN_SIM are not recording-script reads (mic checks, chat) and are left out; re-reads of one sentence
are all kept (each is a valid recording) and marked take = 1, 2, …
Output: data/private/recording_script/audio/<audio_id>.<ext> + takes.jsonl.
Usage: NO_PROXY="*" uv run --offline --with boto3 python scripts/fetch_recordings.py \
         --user user_3ILXK0ao0aXbsoDWGWHxONgE3AW --date 2026-11-04 --since 0725 --until 0950
"""

import argparse
import collections
import csv
import difflib
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH_PY = Path.home() / ".claude/skills/trace-lookup/scripts/bench.py"
OUT = Path("data/private/recording_script")
BUCKET = "voicecursor-test"
MIN_SIM = 0.4


def norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--since", default="0000", help="HHMM (UTC, as in the file names)")
    ap.add_argument("--until", default="2359")
    ap.add_argument("--freeform", action="store_true", help="not script reads: keep every take with ASR text, no matching;"
                    " written to <OUT>/freeform_<date>/ (labels are made from scratch in review)")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("bench", BENCH_PY)
    bench = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BENCH_PY.parent))
    spec.loader.exec_module(bench)
    bench.load_env(())
    s3 = bench.s3_client()
    prefix = f"transcription/recordings/{args.user}/{args.date}/"
    keys, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix, "MaxKeys": 1000, **({"ContinuationToken": token} if token else {})}
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    stems = {}
    for k in keys:
        name = k.rsplit("/", 1)[1]
        stem = name.split(".")[0].removesuffix("_transcript")
        if args.since <= stem[:4] <= args.until:
            stems.setdefault(stem, {})["json" if name.endswith("_transcript.json") else name.rsplit(".", 1)[1]] = k
    takes = []
    for stem, f in sorted(stems.items()):
        audio = f.get("ogg") or f.get("wav")
        if not audio or "json" not in f:
            continue
        side = json.loads(s3.get_object(Bucket=BUCKET, Key=f["json"])["Body"].read())
        takes.append({"stem": stem, "audio_key": audio, "ext": audio.rsplit(".", 1)[1], **side})
    if args.freeform:
        out = OUT / f"freeform_{args.date}"
        (out / "audio").mkdir(parents=True, exist_ok=True)
        kept = [t for t in takes if norm(t["transcript"])]
        for t in kept:
            dst = out / "audio" / f"{t['audio_id']}.{t['ext']}"
            if not dst.exists():
                s3.download_file(BUCKET, t["audio_key"], str(dst))
        with open(out / "items.jsonl", "w", encoding="utf-8") as f:
            for t in kept:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"{len(takes)} files in window, {len(kept)} with ASR text -> {out}")
        return
    script = list(csv.DictReader(open(OUT / "script_v1.csv", encoding="utf-8")))
    funnel = collections.Counter({"files in window": len(takes)})
    kept, per_id = [], collections.Counter()
    for t in takes:
        a = norm(t["transcript"])
        if not a:
            funnel["empty ASR text"] += 1
            continue
        best = max(script, key=lambda s: difflib.SequenceMatcher(None, a, norm(s["text"])).ratio())
        sim = difflib.SequenceMatcher(None, a, norm(best["text"])).ratio()
        if sim < MIN_SIM:
            funnel["not a script read"] += 1
            continue
        per_id[best["id"]] += 1
        kept.append({**t, "script_id": best["id"], "block": best["block"], "script_text": best["text"],
                     "targets": best["targets"], "sim": round(sim, 3), "take": per_id[best["id"]]})
    adir = OUT / "audio"
    adir.mkdir(parents=True, exist_ok=True)

    def get(t):
        dst = adir / f"{t['audio_id']}.{t['ext']}"
        if not dst.exists():
            s3.download_file(BUCKET, t["audio_key"], str(dst))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, kept))
    with open(OUT / "takes.jsonl", "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    funnel["kept takes"] = len(kept)
    print(dict(funnel))
    print(f"script sentences covered: {len(per_id)}/{len(script)}; missing: "
          f"{[s['id'] for s in script if s['id'] not in per_id]}; re-read: {sum(v > 1 for v in per_id.values())}")


if __name__ == "__main__":
    main()
