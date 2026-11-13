"""Dictionary benchmark (owner-reviewed) → ASR Arena upload sheet + ogg folder (2026-11-12).

Ground truth = data/manifests/dictionary_bench.jsonl `raw_transcript`, i.e. the owner's 2026-11-09 review (box ①,
strict transcript) plus the two D-column hearing fixes (data/private/e2e/reviewed/e2e-dict_D_fixes.json).
Excluded: the 6 `drop` items (already absent from the manifest) and the 4 items the owner never reviewed (empty
verdict in e2e-dict-review.csv) — the owner asked that every reference be one they reviewed.
audio_context / vocabulary are carried over unchanged from the first upload (vocab_benchmark_upload.csv); only the
transcript changes. Audio = original production ogg re-downloaded with the audio-fetch skill into --ogg-raw
(<ts>_<id8>.ogg + .json with audio_id), copied as <audio_id>.ogg.

    uv run --with openpyxl python scripts/dictionary_bench_upload.py
Outputs: data/private/dictionary_bench/upload/{dictionary_bench_reviewed_upload.xlsx,.csv, audio/*.ogg}
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())

COLS = ["file", "transcript", "audio_context", "audio_url", "audio_id", "vocabulary"]
MAX_UPLOAD = 4 * 1024 * 1024  # arena how_to: <= 4 MiB per file
MIN_BYTES = 1024  # production has header-only empty oggs (~100 bytes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/dictionary_bench.jsonl")
    ap.add_argument("--review", default="data/private/e2e/reviewed/e2e-dict-review.csv")
    ap.add_argument("--old-upload", default="/Users/huijia/Downloads/01a1bf4a-4136-760f-a549-67ed25b136c1/"
                    "source_ground_truth/vocab_benchmark_upload.csv")
    ap.add_argument("--items", default="data/private/e2e/items_dict.jsonl")
    ap.add_argument("--ogg-raw", default="data/private/dictionary_bench/ogg_raw")
    ap.add_argument("--out", default="data/private/dictionary_bench/upload")
    ap.add_argument("--recheck", default=None,
                    help="review.csv exported from the e2e-dict-recheck page (2026-11-12): the 4 never-reviewed "
                         "items + 2a176c9b; its verdict / transcript override the first review")
    a = ap.parse_args()

    review = {r["item_id"]: r for r in csv.DictReader(open(a.review, encoding="utf-8-sig"))}
    recheck = {}
    if a.recheck:
        recheck = {r["item_id"]: r for r in csv.DictReader(open(a.recheck, encoding="utf-8-sig")) if r["verdict"]}
        review.update(recheck)
    unreviewed = {k for k, r in review.items() if not r["verdict"]}
    manifest = [json.loads(line) for line in open(a.manifest, encoding="utf-8")]
    old = {r["file"].rsplit(".", 1)[0]: r for r in csv.DictReader(open(a.old_upload, encoding="utf-8-sig"))}
    items = {i["item_id"]: i for i in map(json.loads, open(a.items, encoding="utf-8"))}

    oggs = {}
    for meta in Path(a.ogg_raw).glob("*.json"):
        aid = json.load(open(meta, encoding="utf-8")).get("audio_id")
        ogg = meta.with_suffix(".ogg")
        if aid and ogg.exists():
            oggs[aid] = ogg

    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows, problems, reencoded = [], [], []
    for m in manifest:
        aid = m["source_utterance_id"]
        if review[aid]["verdict"] == "drop":
            if aid in recheck:
                continue
            raise SystemExit(f"dropped item in manifest: {aid}")
        if aid in unreviewed:
            continue
        if aid in recheck:
            m = {**m, "raw_transcript": recheck[aid]["transcript"].strip()}
        ogg = oggs.get(aid)
        if ogg is None:
            problems.append(f"{aid}: no ogg downloaded")
            continue
        size = ogg.stat().st_size
        if size < MIN_BYTES or size > MAX_UPLOAD:
            problems.append(f"{aid}: ogg size {size} bytes")
            continue
        dst = out / "audio" / f"{aid}.ogg"
        wav = Path(items[aid]["audio_src"])
        if abs(duration(ogg) - duration(wav)) > 0.6:
            # A few Android uploads are sped up 3x in production (sample-rate mislabel); the wav the owner
            # reviewed is the corrected speed, so encode the ogg from it instead.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "libopus", "-b:a", "32k",
                            str(dst)], check=True)
            reencoded.append(aid)
        else:
            shutil.copy2(ogg, dst)
        rows.append({"file": f"{aid}.ogg", "transcript": m["raw_transcript"],
                     "audio_context": old[aid]["audio_context"], "audio_url": "", "audio_id": "",
                     "vocabulary": "; ".join(items[aid]["target_vocab"])})

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ground_truth"
    ws.append(COLS)
    for r in rows:
        ws.append([r[c] or None for c in COLS])
    wb.save(out / "dictionary_bench_reviewed_upload.xlsx")
    with open(out / "dictionary_bench_reviewed_upload.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"rows {len(rows)} (manifest {len(manifest)}, unreviewed skipped {len(unreviewed)})")
    print("re-encoded from wav (production ogg sped up):", reencoded)
    print("problems:", len(problems))
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main()
