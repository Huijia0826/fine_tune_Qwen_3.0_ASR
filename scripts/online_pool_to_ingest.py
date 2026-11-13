"""Owner-reviewed online training-pool clips → wf-ingest TSV, then a manifest with split=train (2026-11-03).

Keeps verdict ok / fix (drop = unusable). Labels = the owner's final_transcript, with one mechanical fix: doubled
commas ("，，" / ",,") collapsed (2 typos in batch 1 part 1). Speaker ids use the same pseudonym as online_phase1
(u + sha256(user_id)[:12]) so seen / unseen users can be matched across sources. Every clip goes to TRAIN: the pool
was selected as the training pool, val stays users' val.
Usage: python3 scripts/online_pool_to_ingest.py build <review.csv> [<review.csv> …]
       uv run wf-ingest --metadata data/private/online_pool/ingest_metadata.tsv --audio-dir data/private/online_pool/audio \
         --source-id online_pool --license-id voicecursor_internal_training --label-status human_verified \
         --convert-16k --out data/manifests/online_pool_ingested.jsonl
       python3 scripts/online_pool_to_ingest.py assign   (writes data/manifests/online_pool.jsonl, split=train)
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path("data/private/online_pool/ingest_metadata.tsv")
SCEN = {"CS": "zh_en_switch", "ZH": "zh_general", "S4": "zh_general", "S5": "self_correction"}


def pseudo(uid: str) -> str:
    return "u" + hashlib.sha256(uid.encode()).hexdigest()[:12]


def build(reviews: list[str]) -> None:
    pool = {r["audio_id"]: r for r in map(json.loads, open("data/private/online_pool/pool.jsonl", encoding="utf-8"))}
    rows = {}
    for path in reviews:  # later files win for the same audio_id
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            if r["verdict"] in ("ok", "fix"):
                rows[r["audio_id"]] = r
            elif r["verdict"] == "drop":
                rows.pop(r["audio_id"], None)
    fixed = 0
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["audio_path", "transcript", "speaker_id", "session_id", "primary_scenario", "source_utterance_id",
                    "reviewer_id"])
        for aid, r in rows.items():
            text = re.sub(r"([，,])\1+", r"\1", r["final_transcript"].strip())
            fixed += text != r["final_transcript"].strip()
            p = pool[aid]
            scen = SCEN[p["group"]]
            if scen == "zh_en_switch" and not re.search(r"[A-Za-z]", text):
                scen = "zh_general"
            spk = pseudo(p["user_id"])
            w.writerow([f"{aid}.ogg", text, spk, spk, scen, aid, "owner"])
    print(f"{OUT}: {len(rows)} clips ({fixed} with doubled commas collapsed)")


def assign() -> None:
    out = [{**json.loads(line), "split": "train"}
           for line in open("data/manifests/online_pool_ingested.jsonl", encoding="utf-8")]
    with open("data/manifests/online_pool.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"data/manifests/online_pool.jsonl: {len(out)} records, all train")


if __name__ == "__main__":
    assign() if sys.argv[1] == "assign" else build(sys.argv[2:])
