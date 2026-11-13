"""Shared adapter for the 2026-10-30 arena TRAINING datasets -> wf-ingest TSV + leakage report.

Same flow as scripts/arena_019f8e51_to_ingest.py (which stays as is), parameterised by short id:

  01a0e713  arena dataset 01a0e713-7a81-7b82-ae4e-b390ca9c69cd  "纯中文_for_qwen"
  019f6442  arena dataset 019f6442-7412-7eac-a499-d007d1cd06b9  "E2E Intent ZH Pure (v7)"

Owner approved both as TRAINING material on 2026-10-30 (AutoDL upload approved).

  1. export (system python3: psycopg2 + boto3 + requests and ~/.config/vc_bench.env, trace-lookup skill)
       python3 scripts/arena_dataset_to_ingest.py export <short_id>
     Read-only SELECT of arena_dataset_audios + S3 download into data/private/arena_<short_id>/audio/.
     audio_context / screen_text / product text are NOT exported. Raw user ids and raw S3 keys (which
     contain user ids) are NOT written: speakers are pseudonymised like benchmark v1 ("u" +
     sha256(user)[:12]) and keys are stored as sha256("bucket/key") plus the key's base name.
     Speaker source: user_xxx in the S3 key, else the PostHog asr_metrics distinct_id of the audio_id,
     else one pseudo-speaker per file ("unknown-<file8>", same convention as benchmark v1).

  2. build (project env; needs BOTH datasets exported, they are checked against each other)
       uv run python scripts/arena_dataset_to_ingest.py build <short_id>
     Converts to 16 kHz WAV (same ffmpeg call as wf-ingest), runs the leakage checks, writes
       data/private/arena_<short_id>/ingest_metadata.tsv  (excluded rows removed)
       data/private/arena_<short_id>/gt_provenance.csv
       data/private/arena_<short_id>/leakage_report.json
     Then: uv run wf-ingest --metadata data/private/arena_<id>/ingest_metadata.tsv \
             --audio-dir data/private/arena_<id>/audio --source-id arena_<id> \
             --license-id voicecursor_internal_training_arena_<id> --label-status source_provided \
             --convert-16k --out data/manifests/arena_<id>_ingested.jsonl

Leakage checks per reference set (bv1 = benchmark v1, bch = benchmark_chinese, a8e51 = arena_019f8e51,
ascend = ASCEND, other = the other new arena dataset): same S3 key, same file name / audio id, identical
original bytes, identical decoded PCM (sha256), identical 16 kHz wav, near-identical audio (duration
+-0.05 s and energy-envelope correlation >= 0.98), identical normalized transcript, near-identical
transcript (SequenceMatcher >= 0.9), shared pseudonymised speaker.
  * bv1: every hit except shared speaker EXCLUDES the item (as for 019f8e51).
  * training sets (bch, a8e51, ascend, other): audio / key / id hits and identical transcripts exclude the
    item from the NEW dataset; for "other" only the dataset listed in DROP_CROSS_DUPES drops it, so the
    clip stays exactly once. Near-identical transcripts with different audio are reported only.
  * shared speakers: reported only.
Labels are kept as provided (no punctuation / casing / number / filler rewriting).
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

DATASETS = {
    "01a0e713": "01a0e713-7a81-7b82-ae4e-b390ca9c69cd",
    "019f6442": "019f6442-7412-7eac-a499-d007d1cd06b9",
}
DROP_CROSS_DUPES = {"019f6442"}  # overlaps between the two new sets are dropped from this one
DEFAULT_BUCKET = "voicecursor-production"
BV1 = Path("data/private/benchmark_v1")
BCH_SRC = Path.home() / "Desktop/build_benchmark/benchmark_chinese"
A8E51 = Path("data/private/arena_019f8e51")
NEAR_TEXT = 0.9
NEAR_ENV = 0.98
USER_RE = re.compile(r"/(user_[A-Za-z0-9]+)/")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")

EXPORT_COLS = ["id", "file_name", "audio_id", "object_key_sha256", "key_basename", "speaker", "speaker_source",
               "duration_seconds", "position", "size_bytes", "ground_truth", "initial_ground_truth", "edited",
               "ground_truth_edited_at", "labels", "notes", "verdict", "primary_tag", "revision"]


def root(short: str) -> Path:
    return Path(f"data/private/arena_{short}")


def pseudo(uid: str) -> str:
    return "u" + hashlib.sha256(uid.encode()).hexdigest()[:12]  # = build_benchmark_v1_metadata.pseudo


def norm_key(ref: str) -> str:
    """'s3://bucket/key' or bare key -> 'bucket/key' (bare keys live in the default arena bucket)."""
    m = re.match(r"^s3://([^/]+)/(.+)$", ref.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else f"{DEFAULT_BUCKET}/{ref.strip().lstrip('/')}"


def key_sha(ref: str) -> str:
    return hashlib.sha256(norm_key(ref).encode()).hexdigest()


def export(short: str) -> None:
    sys.path.insert(0, str(Path.home() / ".claude/skills/trace-lookup/scripts"))
    import trace_lookup as T

    T.load_env(need=())
    cur = T.db().cursor()  # read-only session
    cur.execute("""SELECT DISTINCT ON (a.position, a.id) a.id, a.file_name, a.object_key, a.duration_seconds,
                    a.position, a.size_bytes, a.ground_truth, a.initial_ground_truth,
                    a.ground_truth_edited_by <> '' AS edited, a.ground_truth_edited_at, a.labels, a.notes,
                    a.verdict, a.primary_tag, a.revision, b.gt_provenance->>'audio_id' AS pool_audio_id
                    FROM arena_dataset_audios a
                    LEFT JOIN arena_audio_benchmark_datasets b ON b.audio_url = a.object_key
                    WHERE a.dataset_id = %s ORDER BY a.position, a.id""", (DATASETS[short],))
    cols = [c[0] for c in cur.description]
    rows = [{c: r[i] for i, c in enumerate(cols)} for r in cur.fetchall()]
    for r in rows:
        stem = Path(r["file_name"]).stem
        r["audio_id"] = r.pop("pool_audio_id") or (stem if UUID_RE.match(stem) else "")
    # speakers: S3 key user_xxx, else PostHog distinct_id of the production request with this audio_id
    users = {}
    for r in rows:
        m = USER_RE.search(r["object_key"])
        if m:
            users[r["id"]] = (m.group(1), "s3_key")
    missing = {r["audio_id"]: r["id"] for r in rows if r["id"] not in users and r["audio_id"]}
    if missing:
        T.load_env()  # PostHog credentials
        ids = list(missing)
        for i in range(0, len(ids), 50):  # HogQL returns only 100 rows unless LIMIT is given
            inl = ",".join(f"'{x}'" for x in ids[i:i + 50])
            for aid, did, s3url in T.q(
                    "SELECT properties.audio_id, distinct_id, properties.s3_url FROM events "
                    f"WHERE event='asr_metrics' AND properties.audio_id IN ({inl}) "
                    "AND properties.client_source != 'arena' ORDER BY timestamp LIMIT 10000"):
                user = did if (did or "").startswith("user_") else (USER_RE.search(s3url or "") or [0, None])[1]
                if user:
                    users.setdefault(missing[aid], (user, "posthog_distinct_id"))
    root(short).joinpath("audio").mkdir(parents=True, exist_ok=True)
    s3 = T.s3_client()
    for r in rows:
        dst = root(short) / "audio" / r["file_name"]
        if not (dst.is_file() and dst.stat().st_size == int(r["size_bytes"])):
            bucket, key = norm_key(r["object_key"]).split("/", 1)
            s3.download_file(bucket, key, str(dst))
        user, src = users.get(r["id"], (None, "none"))
        r["speaker"], r["speaker_source"] = (pseudo(user) if user else ""), src
        r["object_key_sha256"] = key_sha(r["object_key"])
        r["key_basename"] = r.pop("object_key").rsplit("/", 1)[1]
    with open(root(short) / "dataset_audios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows, speakers by source {Counter(r['speaker_source'] for r in rows)}, "
          f"distinct speakers {len({r['speaker'] for r in rows if r['speaker']})} -> {root(short)}")


# ---- audio helpers (same definitions as the 019f8e51 adapter) -------------------------------------
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


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Ref:
    """Fingerprints of one reference set. Every value maps to (ref_id, split)."""

    def __init__(self, name: str):
        self.name = name
        self.keys: dict = {}
        self.names: dict = {}
        self.ogg: dict = {}
        self.pcm: dict = {}
        self.wav: dict = {}
        self.env: list = []
        self.texts: dict = {}
        self.speakers: dict = {}

    def add_audio(self, wav: Path, ref: str, split: str, file_sha: str | None = None) -> None:
        x = pcm(wav)
        self.pcm.setdefault(sha(x.tobytes()), (ref, split))
        self.wav.setdefault(file_sha or sha(wav.read_bytes()), (ref, split))
        self.env.append((x.size / 16000, envelope(x), ref, split))

    def add_text(self, text: str, ref: str, split: str) -> None:
        from whisperflow.text.normalize import normalize

        t = normalize(text or "")
        if t:
            self.texts.setdefault(t, (ref, split))


def strip_prefix(spk: str) -> str:
    return spk.split("-", 1)[1] if "-" in spk else spk


def load_manifest(name: str) -> list[dict]:
    return [json.loads(line) for line in open(f"data/manifests/{name}.jsonl", encoding="utf-8")]


def ref_bv1() -> Ref:
    R = Ref("bv1")
    man = load_manifest("benchmark_v1")
    split = {r["source_utterance_id"]: r["split"] for r in man}
    split_by_stem = {Path(r["audio_path"]).stem: r["split"] for r in man}
    for r in man:
        R.add_audio(Path("data/private") / r["audio_path"], r["source_utterance_id"], r["split"], r["sha256"])
        R.add_text(r["raw_transcript"], r["source_utterance_id"], r["split"])
        R.names.setdefault(Path(r["source_utterance_id"]).stem, (r["source_utterance_id"], r["split"]))
        if not strip_prefix(r["speaker_id"]).startswith("unknown"):
            R.speakers.setdefault(strip_prefix(r["speaker_id"]), (r["source_utterance_id"], r["split"]))
    for r in csv.DictReader(open(BV1 / "arena_object_keys.csv", encoding="utf-8")):
        s = split.get(r["file_name"], "?")
        R.keys.setdefault(key_sha(r["object_key"]), (r["file_name"], s))
        R.names.setdefault(Path(r["file_name"]).stem, (r["file_name"], s))
        R.add_text(r["ground_truth"], r["file_name"], s)
    for p in glob.glob(str(BV1 / "audio/*.json")):
        m = json.loads(Path(p).read_text(encoding="utf-8"))
        ref = Path(p).stem
        s = split_by_stem.get(ref, "?")
        R.keys.setdefault(key_sha(m["input"]), (ref, s))
        R.names.setdefault(m["input"].rsplit("/", 1)[1].rsplit(".", 1)[0], (ref, s))
        if m.get("audio_id"):
            R.names.setdefault(m["audio_id"], (ref, s))
        if (m.get("distinct_id") or "").startswith("user_"):
            R.speakers.setdefault(pseudo(m["distinct_id"]), (ref, s))
    for p in glob.glob(str(BV1 / "audio/*.ogg")):
        R.ogg.setdefault(sha(Path(p).read_bytes()), (Path(p).name, "?"))
    return R


def ref_bch() -> Ref:
    R = Ref("bch")
    for r in load_manifest("benchmark_chinese"):
        u = r["source_utterance_id"]
        R.add_audio(Path("data/private") / r["audio_path"], u, r["split"], r["sha256"])
        R.add_text(r["raw_transcript"], u, r["split"])
        R.names.setdefault(u, (u, r["split"]))
        R.speakers.setdefault(strip_prefix(r["speaker_id"]), (u, r["split"]))
    for r in csv.DictReader(open(BCH_SRC / "ground_truth.csv", encoding="utf-8-sig")):
        R.names.setdefault(Path(r["file"]).stem, (r["file"], "?"))
        if r.get("audio_url"):
            R.keys.setdefault(key_sha(r["audio_url"]), (r["file"], "?"))
            R.names.setdefault(r["audio_url"].rsplit("/", 1)[1].rsplit(".", 1)[0], (r["file"], "?"))
    for p in glob.glob(str(BCH_SRC / "audio/*.ogg")):
        R.ogg.setdefault(sha(Path(p).read_bytes()), (Path(p).name, "?"))
    return R


def ref_a8e51() -> Ref:
    R = Ref("a8e51")
    for r in load_manifest("arena_019f8e51"):
        u = r["source_utterance_id"]
        R.add_audio(Path("data/private") / r["audio_path"], u, r["split"], r["sha256"])
        R.add_text(r["raw_transcript"], u, r["split"])
    for r in csv.DictReader(open(A8E51 / "dataset_audios.csv", encoding="utf-8")):  # incl. excluded rows
        R.keys.setdefault(key_sha(r["object_key"]), (r["file_name"], "?"))
        R.names.setdefault(Path(r["file_name"]).stem, (r["file_name"], "?"))
        R.add_text(r["ground_truth"], r["file_name"], "?")
        src = A8E51 / "audio" / r["file_name"]
        if src.is_file():
            R.ogg.setdefault(sha(src.read_bytes()), (r["file_name"], "?"))
    return R


def ref_ascend() -> Ref:
    R = Ref("ascend")
    for r in load_manifest("ascend"):
        u = r["utterance_id"]
        R.add_audio(Path("data/private") / r["audio_path"], u, r["split"], r["sha256"])
        R.add_text(r["raw_transcript"], u, r["split"])
    return R


def ref_arena_new(short: str) -> Ref:
    from whisperflow.data.audio import convert_to_16k_wav

    R = Ref("other")
    for r in csv.DictReader(open(root(short) / "dataset_audios.csv", encoding="utf-8")):
        fn = r["file_name"]
        src = root(short) / "audio" / fn
        wav = root(short) / (Path(fn).stem + ".wav")
        if not wav.is_file():
            convert_to_16k_wav(src, wav)
        R.add_audio(wav, f"{short}:{fn}", "new")
        R.ogg.setdefault(sha(src.read_bytes()), (f"{short}:{fn}", "new"))
        R.keys.setdefault(r["object_key_sha256"], (f"{short}:{fn}", "new"))
        for n in (Path(fn).stem, Path(r["key_basename"]).stem, r["audio_id"]):
            if n:
                R.names.setdefault(n, (f"{short}:{fn}", "new"))
        R.add_text(r["ground_truth"], f"{short}:{fn}", "new")
        if r["speaker"]:
            R.speakers.setdefault(r["speaker"], (f"{short}:{fn}", "new"))
    return R


def best_near_text(text: str, texts: dict) -> tuple[float, str]:
    best, arg = 0.0, ""
    for t in texts:
        if abs(len(t) - len(text)) > 0.2 * max(len(t), len(text)):
            continue
        sm = SequenceMatcher(None, text, t)
        if sm.real_quick_ratio() < max(NEAR_TEXT, best) or sm.quick_ratio() < max(NEAR_TEXT, best):
            continue
        r = sm.ratio()
        if r > best:
            best, arg = r, t
    return best, arg


def lang_of(text: str) -> str:
    zh, en = bool(CJK.search(text)), bool(LATIN.search(text))
    return "zh-en" if zh and en else "zh" if zh else "en" if en else "none"


def provenance(r: dict) -> str:  # same rule as the 019f8e51 adapter
    if r["edited"] == "True":
        return "human_edited"
    if r["notes"].startswith("auto-passed"):
        return "auto_passed_no_human_review"
    return "reviewed_unedited" if "reviewed" in r["labels"].split(",") else "not_reviewed"


def build(short: str) -> None:
    import numpy as np

    from whisperflow.data.audio import convert_to_16k_wav
    from whisperflow.text.normalize import normalize

    ROOT = root(short)
    rows = list(csv.DictReader(open(ROOT / "dataset_audios.csv", encoding="utf-8")))
    (other,) = [s for s in DATASETS if s != short]
    refs = [ref_bv1(), ref_bch(), ref_a8e51(), ref_ascend(), ref_arena_new(other)]
    print("reference sets loaded:", {R.name: len(R.env) for R in refs}, file=sys.stderr)

    checks = ["object_key", "file_name_or_audio_id", "orig_bytes", "pcm_sha256", "wav_sha256", "near_audio",
              "identical_text", "near_text", "shared_speaker"]
    hits = {f"{R.name}_{c}": [] for R in refs for c in checks}
    hits["self_duplicate_pcm"] = []
    seen_pcm: dict[str, str] = {}
    out, excluded, durs, info = [], {}, {}, {}
    for r in rows:
        fn, stem = r["file_name"], Path(r["file_name"]).stem
        src = ROOT / "audio" / fn
        wav = ROOT / (stem + ".wav")  # the exact path wf-ingest --convert-16k writes
        if not wav.is_file():
            convert_to_16k_wav(src, wav)
        x = pcm(wav)
        durs[fn] = x.size / 16000
        psha, wsha, osha = sha(x.tobytes()), sha(wav.read_bytes()), sha(src.read_bytes())
        e = envelope(x)
        text = normalize(r["ground_truth"])
        names = {n for n in (stem, Path(r["key_basename"]).stem, r["audio_id"]) if n}
        reasons = []
        for R in refs:
            found = {}
            if r["object_key_sha256"] in R.keys:
                found["object_key"] = R.keys[r["object_key_sha256"]]
            for n in names:
                if n in R.names:
                    found["file_name_or_audio_id"] = R.names[n]
            if osha in R.ogg:
                found["orig_bytes"] = R.ogg[osha]
            if psha in R.pcm:
                found["pcm_sha256"] = R.pcm[psha]
            if wsha in R.wav:
                found["wav_sha256"] = R.wav[wsha]
            for d, re_, ref, split in R.env:
                if abs(d - durs[fn]) <= 0.05 and env_corr(e, re_) >= NEAR_ENV:
                    found["near_audio"] = (ref, split)
                    break
            if text in R.texts:
                found["identical_text"] = R.texts[text]
            elif text:
                ratio, t = best_near_text(text, R.texts)
                if ratio >= NEAR_TEXT:
                    found["near_text"] = (*R.texts[t], round(ratio, 3))
            if r["speaker"] and r["speaker"] in R.speakers:
                found["shared_speaker"] = R.speakers[r["speaker"]]
            for c, ref in found.items():
                hits[f"{R.name}_{c}"].append({"file": fn, "ref": ref[0], "ref_split": ref[1], **(
                    {"ratio": ref[2]} if len(ref) > 2 else {})})
                if c == "shared_speaker":
                    continue  # reported only
                if R.name == "bv1" or (c != "near_text" and (R.name != "other" or short in DROP_CROSS_DUPES)):
                    reasons.append(f"{R.name}_{c}")
        if psha in seen_pcm:
            hits["self_duplicate_pcm"].append({"file": fn, "first": seen_pcm[psha]})
            reasons.append("self_duplicate_pcm")
        seen_pcm.setdefault(psha, fn)
        info[fn] = {"lang": lang_of(r["ground_truth"])}
        if reasons:
            excluded[fn] = reasons
            continue
        spk = r["speaker"] or "unknown-" + stem[:8]
        out.append({
            "audio_path": fn,
            "transcript": r["ground_truth"].strip(),
            "speaker_id": spk,
            "session_id": spk,
            "source_utterance_id": fn,  # arena file name, like benchmark v1 / 019f8e51
            "recording_device": "unknown",
        })

    kept_files = {o["audio_path"] for o in out}
    with open(ROOT / "gt_provenance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "kept", "provenance", "labels"])
        for r in rows:
            w.writerow([r["file_name"], r["file_name"] in kept_files, provenance(r), r["labels"]])

    def hsum(files) -> float:
        return round(sum(durs[f] for f in files) / 3600, 3)

    d = np.array([durs[r["file_name"]] for r in rows])
    prov_h: Counter = Counter()
    for r in rows:
        prov_h[provenance(r)] += durs[r["file_name"]] / 3600
    stats = {
        "n_items": len(rows),
        "n_with_audio": sum((ROOT / "audio" / r["file_name"]).is_file() for r in rows),
        "n_with_transcript": sum(bool(r["ground_truth"].strip()) for r in rows),
        "hours_all": round(d.sum() / 3600, 3),
        "n_le30": int((d <= 30).sum()), "hours_le30": round(d[d <= 30].sum() / 3600, 3),
        "n_gt30": int((d > 30).sum()), "hours_gt30": round(d[d > 30].sum() / 3600, 3),
        "max_dur": round(float(d.max()), 2),
        "lang": Counter(v["lang"] for v in info.values()),
        "speakers_by_source": Counter(r["speaker_source"] for r in rows),
        "n_speakers_real": len({r["speaker"] for r in rows if r["speaker"]}),
        "n_kept": len(out), "hours_kept": hsum(kept_files), "n_excluded": len(excluded),
        "hit_counts": {k: len(v) for k, v in hits.items() if v},
        "shared_speaker_files": {R.name: len({h["file"] for h in hits[f"{R.name}_shared_speaker"]}) for R in refs},
        "shared_speaker_count": {R.name: len({r["speaker"] for r in rows if r["speaker"] in R.speakers})
                                 for R in refs},
        "exclusion_reason_combos": Counter("+".join(v) for v in excluded.values()),
        "provenance_all": Counter(provenance(r) for r in rows),
        "provenance_hours_all": {k: round(v, 3) for k, v in prov_h.items()},
        "provenance_kept": Counter(provenance(r) for r in rows if r["file_name"] in kept_files),
        "labels": Counter(lab for r in rows for lab in r["labels"].split(",") if lab),
    }
    (ROOT / "leakage_report.json").write_text(json.dumps(
        {"stats": stats, "hits": hits, "excluded": excluded}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    with open(ROOT / "ingest_metadata.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "speaker_id", "session_id",
                                           "source_utterance_id", "recording_device"], delimiter="\t")
        w.writeheader()
        w.writerows(out)
    print("->", ROOT / "ingest_metadata.tsv", ROOT / "leakage_report.json")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("export", "build") or sys.argv[2] not in DATASETS:
        sys.exit(f"usage: arena_dataset_to_ingest.py export|build {'|'.join(DATASETS)}")
    {"export": export, "build": build}[sys.argv[1]](sys.argv[2])
