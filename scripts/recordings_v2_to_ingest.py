"""Recording script v2 (owner-reviewed) → wf-ingest TSVs for a train manifest and a test manifest (2026-11-05).

Source: data/private/recording_script/review/recording-v2/review_2026-11-05.csv (owner review of the 170 takes in
recording_script/v2/takes.jsonl; recorded 2026-11-05 on the owner's PRODUCTION account user_3Fq6…, mostly by the
owner, so the speaker id is the owner's — same as recording_v1).
Rules (owner, 2026-11-05):
  * verdict ok / fix kept; "前端部署在分校。" dropped (misrecognised Vercel left in the label).
  * Split = the script's split (script_v2.csv): train or test. Unmatched takes (not a script sentence) → train.
  * 22 spelling-block labels had the spelled letters removed / were formatted (Kafka, lisa@gmail.com, AB3729), against
    the 2026-11-04 policy (letters as spoken, "K-A-F-K-A"; "at" / "dot" spoken). The owner asked for the expected labels
    instead of re-listening: for the 12 TRAIN ones the label below is written from the script and the owner's wording
    (reviewer_id owner+assistant_expected); the 10 TEST ones are dropped, because test labels must be human-verified and
    moving them to train would leak the test-only words (Replit, Lovable).
One test clip ("把这个发给小李。") is identical to a training label (v1 shares this self-correction scene) and was removed
from recording_v2_test.jsonl after assign (2026-11-05).
Usage: python3 scripts/recordings_v2_to_ingest.py   → recording_script/v2/ingest_{train,test}.tsv
       then wf-ingest each (source recording_v2 / recording_v2_test) and assign the split (see the bottom of this file).
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

D = Path("data/private/recording_script/v2")
REVIEW = Path("data/private/recording_script/review/recording-v2/review_2026-11-05.csv")
OWNER = "user_3ILXK0ao0aXbsoDWGWHxONgE3AW"  # same pseudonym as recording_v1
DROP_TEXT = {"前端部署在分校。"}
EXPECTED = {  # audio_id → policy-conform label (train only)
    "2ad03307-006f-44b2-8b1e-77b3552618ec": "他的邮箱是 W-A-N-G dot hao at QQ dot com。",
    "5f4534a7-1284-4d39-94de-e99cca0795bd": "航班号是 C-A-9-8-1，明天早上8点半的。",
    "1021ec79-7382-4403-a0f0-a66f5232c698": "代码让 Codex 先写一版，C-O-D-E-X。",
    "554ea017-c35d-4e4a-a827-784d3522cb20": "账号是 C-H-R-I-S dot liu at company dot com。",
    "0d2a1213-5a41-487f-974a-4151b42c656e": "发票寄到 G-R-A-C-E 99 at gmail dot com。",
    "61aba911-72cc-443e-b052-88cd987adb3c": "我们先写一个 PRD，P-R-D，再排期。",
    "b4203c02-0851-4de6-adbf-96187eab345d": "邀请码是 X-7-Q-2，注册的时候填一下。",
    "dc89fa87-a416-4d6a-913d-6ddb299806aa": "我的邮箱是 L-I-S-A at gmail dot com。",
    "b826883f-d80b-486a-897d-0daab8d79ea5": "你发到 K-E-V-I-N dot chen at gmail dot com。",
    "2d3aae57-a6d6-494f-9646-0663cf4fbc83": "简历发到 J-E-S-S-I-C-A at gmail dot com 就行。",
    "321554da-4a02-43ac-a920-a3ed0f854ed1": "订单号是 A-B-3-7-2-9，你帮我查一下。",
    "2421773d-bdbd-4de8-aed2-6292edddd009": "会议号 M-T-6-6-0，密码是 1234。",
}
DROP_TEST = {"b8b30b92-e7a9-4bce-9c1b-a039777c2978", "adfdeea0-fee3-4416-b5ce-7f424f2bd673",
             "3d31fed9-e869-49e6-b9d1-bf7cb65ab1f2", "d2916ebb-c6f7-4460-b11c-0546cf6a86fe",
             "5a66a6f2-fff9-439f-9750-38869dc3c359", "7cb4d8c8-88af-4dda-a842-7e3f9e3bbed1",
             "24579e30-23e3-4a2e-bf3f-a3b586e9d847", "0900870f-698c-4780-bbb5-9fad230a9046",
             "28f676d1-175b-4144-860b-7c87881df0de", "0e984fd9-f47d-4e93-9e33-7476d81dfd54"}


def build() -> None:
    takes = {t["audio_id"]: t for t in map(json.loads, open(D / "takes.jsonl", encoding="utf-8"))}
    spk = "u" + hashlib.sha256(OWNER.encode()).hexdigest()[:12]
    out = {"train": [], "test": []}
    for r in csv.DictReader(open(REVIEW, encoding="utf-8-sig")):
        if r["verdict"] not in ("ok", "fix") or r["final_transcript"].strip() in DROP_TEXT or r["audio_id"] in DROP_TEST:
            continue
        t = takes[r["audio_id"]]
        split = t["split"] if t.get("matched", True) else "train"
        text, reviewer = r["final_transcript"].strip(), "owner"
        if EXPECTED.get(r["audio_id"]):
            assert split == "train", r["audio_id"]
            text, reviewer = EXPECTED[r["audio_id"]], "owner+assistant_expected"
        scen = "self_correction" if t["block"] == "C" else ""
        out[split].append([f"{t['audio_id']}.{t['ext']}", text, spk, spk, scen, f"{t['audio_id']}#{t['script_id']}", reviewer])
    for split, rows in out.items():
        with open(D / f"ingest_{split}.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["audio_path", "transcript", "speaker_id", "session_id", "primary_scenario", "source_utterance_id",
                        "reviewer_id"])
            w.writerows(rows)
        print(f"{D / f'ingest_{split}.tsv'}: {len(rows)} clips "
              f"({sum(r[6] != 'owner' for r in rows)} with assistant-written expected labels)")


def assign(src: str, dst: str, split: str) -> None:
    rows = [{**json.loads(line), "split": split} for line in open(src, encoding="utf-8")]
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{dst}: {len(rows)} records, split={split}, {sum(r['duration_sec'] for r in rows) / 60:.1f} min")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assign":
        assign("data/manifests/recording_v2_ingested.jsonl", "data/manifests/recording_v2.jsonl", "train")
        assign("data/manifests/recording_v2_test_ingested.jsonl", "data/manifests/recording_v2_test.jsonl", "test")
    else:
        build()
