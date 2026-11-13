"""Owner-reviewed scripted recordings (recording script v1) → wf-ingest TSV, then a train-only manifest (2026-11-04).

Keeps verdict ok / fix. Labels = the owner's final_transcript, with two corrections where a misrecognised spelling of
the very term the sentence targets was kept from the production ASR text (R031 "Typeus" → Typeless, R053 "PostHop" →
PostHog; owner informed). One speaker (the owner), so every clip goes to TRAIN; speaker id = "u"+sha256(user)[:12].
primary_scenario is inferred by wf-ingest from the text (zh / zh-en), except block C (self-correction / spelling).
Usage: python3 scripts/recordings_to_ingest.py build data/private/recording_script/review/recording-v1/review_2026-11-04.csv
       uv run wf-ingest --metadata data/private/recording_script/ingest_metadata.tsv \
         --audio-dir data/private/recording_script/audio --source-id recording_v1 \
         --license-id voicecursor_owner_recording --label-status human_verified --device "MacBook Pro Microphone" \
         --convert-16k --out data/manifests/recording_v1_ingested.jsonl
       python3 scripts/recordings_to_ingest.py assign   (writes data/manifests/recording_v1.jsonl, split=train)
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("data/private/recording_script")
OUT = ROOT / "ingest_metadata.tsv"
USER = "user_3ILXK0ao0aXbsoDWGWHxONgE3AW"
FIX = {"Typeus": "Typeless", "PostHop": "PostHog"}


def build(review: str) -> None:
    takes = {t["audio_id"]: t for t in map(json.loads, open(ROOT / "takes.jsonl", encoding="utf-8"))}
    spk = "u" + hashlib.sha256(USER.encode()).hexdigest()[:12]
    n = fixed = 0
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["audio_path", "transcript", "speaker_id", "session_id", "primary_scenario", "source_utterance_id",
                    "reviewer_id", "hotwords_present"])
        for r in csv.DictReader(open(review, encoding="utf-8-sig")):
            if r["verdict"] not in ("ok", "fix"):
                continue
            t = takes[r["audio_id"]]
            text = r["final_transcript"].strip()
            for bad, good in FIX.items():
                if bad in text:
                    text, fixed = text.replace(bad, good), fixed + 1
            scen = "self_correction" if t["block"] == "C_correction_spelling" else ""
            hot = "|".join(w for w in t["targets"].split() if w not in ("self_correction", "spelling") and w in text)
            w.writerow([f"{t['audio_id']}.{t['ext']}", text, spk, spk, scen, f"{t['audio_id']}#{t['script_id']}",
                        "owner", hot])
            n += 1
    print(f"{OUT}: {n} clips ({fixed} target-term spellings corrected)")


def assign() -> None:
    out = [{**json.loads(line), "split": "train"}
           for line in open("data/manifests/recording_v1_ingested.jsonl", encoding="utf-8")]
    with open("data/manifests/recording_v1.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"data/manifests/recording_v1.jsonl: {len(out)} records, all train, "
          f"{sum(r['duration_sec'] for r in out) / 60:.1f} min")


if __name__ == "__main__":
    assign() if sys.argv[1] == "assign" else build(sys.argv[2])
