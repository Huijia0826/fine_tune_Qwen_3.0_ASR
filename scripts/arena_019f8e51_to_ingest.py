"""arena dataset 019f8e51 ("Refined V3 English E2E dataset") adapter -> wf-ingest TSV + leakage report.

Owner approved this dataset as TRAINING material on 2026-10-30. Two steps:

  1. export (system python3: needs psycopg2 + boto3 and ~/.config/vc_bench.env, same as the trace-lookup skill)
       python3 scripts/arena_019f8e51_to_ingest.py export
     Read-only SELECT of arena_dataset_audios for the dataset + S3 download of each .ogg into
     data/private/arena_019f8e51/. audio_context / screen_text are NOT exported (they are empty here anyway).

  2. build (project env)
       uv run python scripts/arena_019f8e51_to_ingest.py build
     Converts to 16 kHz WAV (same ffmpeg command as wf-ingest), runs the leakage checks against
     benchmark v1 (final test set) and benchmark_chinese, writes
       data/private/arena_019f8e51/ingest_metadata.tsv   (excluded rows removed)
       data/private/arena_019f8e51/leakage_report.json
     Then: uv run wf-ingest --metadata data/private/arena_019f8e51/ingest_metadata.tsv \
             --audio-dir data/private/arena_019f8e51/audio --source-id arena_019f8e51 \
             --license-id voicecursor_internal_training_arena_019f8e51 --label-status source_provided \
             --convert-16k --out data/manifests/arena_019f8e51_ingested.jsonl

Leakage rules (task brief 2026-10-30):
  * EXCLUDE: same S3 object key / file name as benchmark v1, identical decoded audio (PCM sha256),
    near-identical audio (same duration +-0.05 s and envelope correlation >= 0.98),
    identical or near-identical (>= 0.9 similarity) normalized transcript vs any benchmark v1 item.
  * benchmark_chinese overlap (audio_id / S3 url / audio sha) is excluded as well (it is already train/val).
  * shared speakers: reported only. This dataset has no user ids (arena uploads, not found in PostHog),
    so every file is its own pseudo-speaker "unknown-<file8>", the same convention as benchmark v1.
Labels are kept as provided (no punctuation / casing / number / filler rewriting).
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

DATASET_ID = "019f8e51-c658-75d0-9ada-5fc3f020e722"
ROOT = Path("data/private/arena_019f8e51")
ROWS_CSV = ROOT / "dataset_audios.csv"
OUT_TSV = ROOT / "ingest_metadata.tsv"
REPORT = ROOT / "leakage_report.json"
BV1 = Path("data/private/benchmark_v1")
BCH_SRC = Path.home() / "Desktop/build_benchmark/benchmark_chinese"
NEAR_TEXT = 0.9
NEAR_ENV = 0.98

EXPORT_COLS = ["id", "file_name", "object_key", "duration_seconds", "position", "size_bytes", "ground_truth",
               "initial_ground_truth", "edited", "ground_truth_edited_at", "labels", "notes", "verdict",
               "primary_tag", "revision"]


def export() -> None:
    sys.path.insert(0, str(Path.home() / ".claude/skills/trace-lookup/scripts"))
    import trace_lookup as T

    T.load_env(need=())
    cur = T.db().cursor()  # read-only session
    cur.execute("""SELECT id, file_name, object_key, duration_seconds, position, size_bytes, ground_truth,
                    initial_ground_truth, ground_truth_edited_by <> '' AS edited, ground_truth_edited_at, labels,
                    notes, verdict, primary_tag, revision
                    FROM arena_dataset_audios WHERE dataset_id = %s ORDER BY position""", (DATASET_ID,))
    cols = [c[0] for c in cur.description]
    rows = [{c: r[i] for i, c in enumerate(cols)} for r in cur.fetchall()]
    ROOT.joinpath("audio").mkdir(parents=True, exist_ok=True)
    with open(ROWS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLS)
        w.writeheader()
        w.writerows(rows)
    s3, bucket = T.s3_client(), T.env("AWS_STORAGE_BUCKET_NAME")
    for r in rows:
        dst = ROOT / "audio" / r["file_name"]
        if not (dst.is_file() and dst.stat().st_size == int(r["size_bytes"])):
            s3.download_file(bucket, r["object_key"], str(dst))
    print(f"{len(rows)} rows -> {ROWS_CSV}, audio -> {ROOT / 'audio'}")


def pcm(path: Path):
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="int16", always_2d=True)
    assert sr == 16000 and x.shape[1] == 1, path
    return np.ascontiguousarray(x[:, 0])


def envelope(x):
    import numpy as np

    hop = 160  # 10 ms
    n = x.size // hop
    e = np.log10(np.mean(x[: n * hop].astype("float64").reshape(n, hop) ** 2, axis=1) + 1.0)
    return (e - e.mean()) / (e.std() + 1e-9)


def env_corr(a, b) -> float:
    n = min(a.size, b.size)
    return float((a[:n] * b[:n]).mean()) if n > 50 else 0.0


def build() -> None:
    import numpy as np

    from whisperflow.data.audio import convert_to_16k_wav
    from whisperflow.text.normalize import normalize

    rows = list(csv.DictReader(open(ROWS_CSV, encoding="utf-8")))

    # ---- reference sets -------------------------------------------------------------------------
    bv1_manifest = [json.loads(line) for line in open("data/manifests/benchmark_v1.jsonl", encoding="utf-8")]
    bv1_test_src = {r["source_utterance_id"] for r in bv1_manifest if r["split"] == "test"}
    bv1_keys = list(csv.DictReader(open(BV1 / "arena_object_keys.csv", encoding="utf-8")))
    bv1_inputs = {json.loads(Path(p).read_text(encoding="utf-8"))["input"]
                  for p in glob.glob(str(BV1 / "audio/*.json"))}
    bv1_key_set = {r["object_key"] for r in bv1_keys} | bv1_inputs
    bv1_files = {r["file_name"] for r in bv1_keys} | {r["source_utterance_id"] for r in bv1_manifest}
    bv1_texts = {}  # normalized transcript -> (source_utterance_id, in_test)
    for r in bv1_manifest:
        bv1_texts.setdefault(normalize(r["raw_transcript"]), (r["source_utterance_id"], r["split"] == "test"))
    for r in bv1_keys:
        bv1_texts.setdefault(normalize(r["ground_truth"]), (r["file_name"], r["file_name"] in bv1_test_src))
    bv1_audio = {}  # pcm sha -> utterance; plus envelopes by duration
    bv1_env = []
    for r in bv1_manifest:
        x = pcm(Path("data/private") / r["audio_path"])
        bv1_audio[hashlib.sha256(x.tobytes()).hexdigest()] = (r["source_utterance_id"], r["split"])
        bv1_env.append((x.size / 16000, envelope(x), r["source_utterance_id"], r["split"]))
    bv1_file_sha = {r["sha256"] for r in bv1_manifest}
    bv1_ogg_sha = {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in glob.glob(str(BV1 / "audio/*.ogg"))}

    bch_manifest = [json.loads(line) for line in open("data/manifests/benchmark_chinese.jsonl", encoding="utf-8")]
    bch_ids = {r["source_utterance_id"] for r in bch_manifest}
    bch_gt = list(csv.DictReader(open(BCH_SRC / "ground_truth.csv", encoding="utf-8-sig")))
    bch_ids |= {Path(r["file"]).stem for r in bch_gt}
    bch_urls = {r["audio_url"] for r in bch_gt}
    bch_file_sha = {r["sha256"] for r in bch_manifest}
    bch_audio = {hashlib.sha256(pcm(Path("data/private") / r["audio_path"]).tobytes()).hexdigest()
                 for r in bch_manifest}
    bch_ogg_sha = {hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in glob.glob(str(BCH_SRC / "audio/*.ogg"))}
    bch_texts = {normalize(r["raw_transcript"]) for r in bch_manifest}

    # ---- per-item checks ------------------------------------------------------------------------
    hits: dict[str, list[dict]] = {k: [] for k in (
        "bv1_object_key", "bv1_file_name", "bv1_ogg_bytes", "bv1_pcm_sha256", "bv1_wav_sha256",
        "bv1_near_audio", "bv1_identical_text", "bv1_near_text",
        "bch_audio_id", "bch_s3_url", "bch_audio_sha256", "bch_identical_text", "self_duplicate_pcm")}
    seen_pcm: dict[str, str] = {}
    out, excluded, durs = [], {}, {}
    for r in rows:
        fn, stem = r["file_name"], Path(r["file_name"]).stem
        src = ROOT / "audio" / fn
        wav = ROOT / (stem + ".wav")  # the exact path wf-ingest --convert-16k writes
        convert_to_16k_wav(src, wav)
        x = pcm(wav)
        durs[fn] = x.size / 16000
        psha = hashlib.sha256(x.tobytes()).hexdigest()
        wsha = hashlib.sha256(wav.read_bytes()).hexdigest()
        osha = hashlib.sha256(src.read_bytes()).hexdigest()
        text = normalize(r["ground_truth"])
        reasons = []

        def hit(kind, _fn=fn, _reasons=reasons, **info):
            hits[kind].append({"file": _fn, **info})
            _reasons.append(kind)

        if r["object_key"] in bv1_key_set:
            hit("bv1_object_key")
        if fn in bv1_files:
            hit("bv1_file_name")
        if osha in bv1_ogg_sha:
            hit("bv1_ogg_bytes")
        if psha in bv1_audio:
            hit("bv1_pcm_sha256", ref=bv1_audio[psha])
        if wsha in bv1_file_sha:
            hit("bv1_wav_sha256")
        e = envelope(x)
        for d, re_, ref, split in bv1_env:
            if abs(d - durs[fn]) <= 0.05 and env_corr(e, re_) >= NEAR_ENV:
                hit("bv1_near_audio", ref=ref, split=split)
                break
        if text in bv1_texts:
            hit("bv1_identical_text", ref=bv1_texts[text][0], ref_in_test=bv1_texts[text][1],
                text=r["ground_truth"][:80])
        else:
            best = max(((SequenceMatcher(None, text, t).ratio(), t) for t in bv1_texts
                        if abs(len(t) - len(text)) <= 0.2 * max(len(t), len(text))), default=(0, ""))
            if best[0] >= NEAR_TEXT:
                hit("bv1_near_text", ratio=round(best[0], 3), ref=bv1_texts[best[1]][0],
                    ref_in_test=bv1_texts[best[1]][1])
        if stem in bch_ids:
            hit("bch_audio_id")
        if r["object_key"] in bch_urls or ("s3://voicecursor-production/" + r["object_key"]) in bch_urls:
            hit("bch_s3_url")
        if psha in bch_audio or wsha in bch_file_sha or osha in bch_ogg_sha:
            hit("bch_audio_sha256")
        if text in bch_texts:
            hit("bch_identical_text")
        if psha in seen_pcm:
            hit("self_duplicate_pcm", first=seen_pcm[psha])
        seen_pcm.setdefault(psha, fn)

        if reasons:
            excluded[fn] = reasons
            continue
        spk = "unknown-" + fn[:8]  # no user id available for this dataset (see module docstring)
        out.append({
            "audio_path": fn,
            "transcript": r["ground_truth"].strip(),
            "speaker_id": spk,
            "session_id": spk,
            "source_utterance_id": fn,  # arena file name, like benchmark v1
            "recording_device": "unknown",
        })

    # ground-truth provenance per file (wf-ingest takes one label_status, so this is kept on the side)
    def provenance(r: dict) -> str:
        if r["edited"] == "True":
            return "human_edited"
        if r["notes"].startswith("auto-passed"):
            return "auto_passed_no_human_review"  # 3-model consensus (strict WER < 5-7 %), nobody listened
        return "reviewed_unedited" if "reviewed" in r["labels"].split(",") else "not_reviewed"

    kept_files = {o["audio_path"] for o in out}
    with open(ROOT / "gt_provenance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "kept", "provenance", "labels"])
        for r in rows:
            w.writerow([r["file_name"], r["file_name"] in kept_files, provenance(r), r["labels"]])

    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)

    d = np.array([durs[r["file_name"]] for r in rows])
    kept = np.array([durs[o["audio_path"]] for o in out])
    stats = {
        "n_items": len(rows),
        "n_with_audio": sum((ROOT / "audio" / r["file_name"]).is_file() for r in rows),
        "n_with_transcript": sum(bool(r["ground_truth"].strip()) for r in rows),
        "hours_all": round(d.sum() / 3600, 3),
        "n_gt30": int((d > 30).sum()), "hours_gt30": round(d[d > 30].sum() / 3600, 3),
        "n_le30": int((d <= 30).sum()), "hours_le30": round(d[d <= 30].sum() / 3600, 3),
        "n_kept": len(out), "hours_kept": round(kept.sum() / 3600, 3),
        "n_kept_gt30": int((kept > 30).sum()), "n_excluded": len(excluded),
        "hit_counts": {k: len(v) for k, v in hits.items()},
        "exclusion_reason_combos": Counter("+".join(v) for v in excluded.values()),
        "provenance_all": Counter(provenance(r) for r in rows),
        "provenance_kept": Counter(provenance(r) for r in rows if r["file_name"] in kept_files),
    }
    REPORT.write_text(json.dumps({"stats": stats, "hits": hits, "excluded": excluded}, ensure_ascii=False,
                                 indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("->", OUT_TSV, REPORT)


if __name__ == "__main__":
    {"export": export, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
