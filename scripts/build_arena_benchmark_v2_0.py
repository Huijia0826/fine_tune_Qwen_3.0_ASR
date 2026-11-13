"""Build the ASR Arena upload for benchmark v2.0 (owner, 2026-11-10): owner-reviewed clips from four sets.

  recording_freeform_test (test, 21) + recording_v2_test (test, 41) + online_phase1 (val 57 + train 230)
  + online_pool batch 1 (~/Downloads/chinese_tone_benchmark.zip, 412; TRAINING clips, owner chose to include them)
  + benchmark v2 English short candidates reviewed by the owner on the unified review page (2026-11-11,
    data/private/e2e/reviewed/bv2-en-short-review.csv; ok / fix kept, drop left out; transcript = box ① strict transcript)

Output (~/Desktop/build_benchmark/benchmark_v2.0/): audio/<audio_id>.<ext> (original production / test-env audio, not our
16 kHz wav) + benchmark_v2.0_upload.xlsx in the arena upload format (file, transcript, audio_context, audio_url, audio_id,
vocabulary, tone_app_id) + index.csv (which set each clip came from).
Transcript = manifest label (latest, includes the 2026-11-10 label fixes). audio_context = the request's raw context from
PostHog (Langfuse root input when PostHog has none); vocabulary = the dictionary production actually used
(effective_vocabulary); tone_app_id as logged. Originals missing locally are downloaded from S3. Every audio file is
real Ogg (arena upload requirement): production files named .ogg that are WebM / WAV are transcoded to Ogg Opus.
Usage: NO_PROXY="*" uv run --offline --with boto3 python scripts/build_arena_benchmark_v2_0.py
"""

import ast
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import openpyxl

PRIV = Path("data/private")
CTX = PRIV / "context_pull"
ZIP = Path.home() / "Downloads/chinese_tone_benchmark.zip"
OUT = Path.home() / "Desktop/build_benchmark/benchmark_v2.0"
SETS = [("recording_freeform_test", ("test",)), ("recording_v2_test", ("test",)), ("online_phase1", ("val", "train")),
        ("online_pool_b1full", ("train",))]
LOCAL_DIRS = [PRIV / "recording_script/freeform_2026-11-03/audio", PRIV / "recording_script/v2/audio",
              PRIV / "recording_script/audio"]
BV2_EN = (PRIV / "e2e/items_bv2_en_short.jsonl", PRIV / "e2e/reviewed/bv2-en-short-review.csv")
COLS = ["file", "transcript", "audio_context", "audio_url", "audio_id", "vocabulary", "tone_app_id"]


def as_dict(ctx):
    if isinstance(ctx, str):
        try:
            return json.loads(ctx)
        except json.JSONDecodeError:
            return ast.literal_eval(ctx)
    return ctx


def ensure_ogg(path: Path) -> None:
    if open(path, "rb").read(4) != b"OggS":
        tmp = path.with_suffix(".tmp.ogg")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-c:a", "libopus", "-b:a", "48k", str(tmp)],
                       check=True)
        tmp.replace(path)


def reviewed_en():
    """benchmark v2 English short candidates as manifest-like records (owner-reviewed, drops left out)."""
    review = {r["item_id"]: r for r in csv.DictReader(open(BV2_EN[1], encoding="utf-8-sig"))}
    for it in map(json.loads, open(BV2_EN[0], encoding="utf-8")):
        r = review[it["item_id"]]
        if r["verdict"] in ("ok", "fix"):
            yield {"utterance_id": f"benchmark_v2-{it['audio_id']}", "source_utterance_id": it["audio_id"],
                   "split": "test", "raw_transcript": r["transcript"].strip(), "language_tags": ["en"],
                   "duration_sec": it["duration_s"], "local_audio": it["audio_src"]}


def main() -> None:
    z = zipfile.ZipFile(ZIP)
    zip_audio = {re.search(r"audio/(.+)\.ogg$", n).group(1): n for n in z.namelist() if n.endswith(".ogg")}
    posthog = json.load(open(CTX / "_posthog.json"))["by_id"]
    freeform_keys = {json.loads(line)["audio_id"]: json.loads(line)["audio_key"]
                     for line in open(PRIV / "recording_script/freeform_2026-11-03/items.jsonl")}
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    s3 = None
    rows, index = [], []
    for name, splits in SETS + [("benchmark_v2", ("test",))]:
        ctx_rows = {json.loads(line)["utterance_id"]: json.loads(line) for line in open(CTX / f"{name}.jsonl")}
        recs = reviewed_en() if name == "benchmark_v2" else map(json.loads, open(f"data/manifests/{name}.jsonl"))
        for rec in recs:
            aid = rec["source_utterance_id"].split("#")[0]
            if rec["split"] not in splits or (name == "online_pool_b1full" and aid not in zip_audio):
                continue
            ph = [e for e in posthog.get(aid, []) if e.get("s3_url")]
            url = ph[0]["s3_url"] if ph else f"s3://voicecursor-test/{freeform_keys[aid]}"
            ext = url.rsplit(".", 1)[-1]
            dst = OUT / "audio" / f"{aid}.ogg"
            if not dst.exists():
                local = Path(rec["local_audio"]) if "local_audio" in rec else \
                    next((d / f"{aid}.{ext}" for d in LOCAL_DIRS if (d / f"{aid}.{ext}").exists()), None)
                if aid in zip_audio:
                    with z.open(zip_audio[aid]) as src, open(dst, "wb") as f:
                        shutil.copyfileobj(src, f)
                elif local:
                    shutil.copy(local, dst)
                else:
                    if s3 is None:
                        spec = importlib.util.spec_from_file_location(
                            "trace_lookup", Path.home() / ".claude/skills/trace-lookup/scripts/trace_lookup.py")
                        tl = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(tl)
                        s3 = (tl, tl.s3_client())
                    s3[0].s3_download(url, str(dst), s3[1])
            ensure_ogg(dst)
            ctx = next((e["audio_context"] for e in posthog.get(aid, []) if e.get("audio_context")), None)
            lf = CTX / "_langfuse" / f"{aid}.json"
            if ctx is None and lf.exists():
                ctx = json.load(open(lf))["root_input"].get("audio_context")
            c = ctx_rows[rec["utterance_id"]]
            rows.append([dst.name, rec["raw_transcript"],
                         json.dumps(as_dict(ctx), ensure_ascii=False, separators=(",", ":")) if ctx else None,
                         url, aid, "; ".join(c.get("effective_vocabulary") or []) or None, c.get("tone_app_id")])
            index.append({"file": dst.name, "set": name, "split_in_training": rec["split"], "utterance_id": rec["utterance_id"],
                          "language_tags": "+".join(rec["language_tags"]), "duration_sec": rec["duration_sec"]})
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(COLS)
    for r in rows:
        ws.append(r)
    wb.save(OUT / "benchmark_v2.0_upload.xlsx")
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"{len(rows)} rows -> {OUT}", file=sys.stderr)
    for i, col in enumerate(COLS):
        print(f"  {col}: {sum(1 for r in rows if r[i] not in (None, ''))} filled", file=sys.stderr)


if __name__ == "__main__":
    main()
