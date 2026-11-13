"""Owner-reviewed online phase 1 clips → wf-ingest TSV + split assignment (owner decision 2026-11-02).

Keeps verdict ok / fix from the exported review.csv (drops = low quality). The 230 clips first meant for the locked
test set go to TRAIN, the 57 val clips to VAL (joining users' val). Labels = the owner's final_transcript.
Usage: python3 scripts/online_phase1_to_ingest.py <exported review.csv>
Then:  uv run wf-ingest --metadata data/private/online_phase1/ingest_metadata.tsv \
         --audio-dir data/private/online_phase1/bench/online-zh-phase1/audio --source-id online_phase1 \
         --license-id voicecursor_internal_training --label-status human_verified --convert-16k \
         --out data/manifests/online_phase1_ingested.jsonl
       python3 scripts/online_phase1_to_ingest.py --assign   (writes data/manifests/online_phase1.jsonl)
"""

import csv
import hashlib
import re
import json
import sys
from pathlib import Path

OUT = Path("data/private/online_phase1/ingest_metadata.tsv")
SPLITS = Path("data/private/online_phase1/split_by_audio_id.json")
SCENARIO = {"S1_zh_en": "zh_en_switch", "S3_terms": "zh_en_switch", "S2_zh": "zh_general", "S4_short": "zh_general",
            "S5_selfcorr": "self_correction"}


def pseudo(uid: str) -> str:
    return "u" + hashlib.sha256(uid.encode()).hexdigest()[:12]


def build(review: str) -> None:
    cands = {json.loads(l)["audio_id"]: json.loads(l) for l in open("data/private/online_phase1/candidates.jsonl")}
    rows = [r for r in csv.DictReader(open(review, encoding="utf-8-sig")) if r["verdict"] in ("ok", "fix")]
    splits = {}
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["audio_path", "transcript", "speaker_id", "session_id", "primary_scenario", "source_utterance_id",
                    "reviewer_id"])
        for r in rows:
            c = cands[r["audio_id"]]
            spk = pseudo(c["user_id"])
            scen = SCENARIO[c["scenario"]]
            if scen == "zh_en_switch" and not re.search(r"[A-Za-z]", r["final_transcript"]):
                scen = "zh_general"  # the reviewed text has no English left (scenario came from the production ASR)
            w.writerow([f"{r['audio_id']}.ogg", r["final_transcript"].strip(), spk, spk, scen,
                        r["audio_id"], "owner"])
            splits[r["audio_id"]] = "train" if r["split"] == "test" else "val"
    SPLITS.write_text(json.dumps(splits))
    print(f"{OUT}: {len(rows)} rows; train {sum(v == 'train' for v in splits.values())}, "
          f"val {sum(v == 'val' for v in splits.values())}")


def assign() -> None:
    splits = json.loads(SPLITS.read_text())
    out = []
    for line in open("data/manifests/online_phase1_ingested.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out.append({**r, "split": splits[r["source_utterance_id"]]})
    with open("data/manifests/online_phase1.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"data/manifests/online_phase1.jsonl: {len(out)} records")


if __name__ == "__main__":
    assign() if sys.argv[1] == "--assign" else build(sys.argv[1])
