"""Assign a 2026-10-30 arena training dataset to train, same owner policy as arena_019f8e51 (option B).

  * train: clips <= 30 s whose label is human-edited, human-reviewed, or auto-passed (3-model consensus,
    nobody listened; flagged `label_auto_passed_3model`).
  * excluded (`unassigned` + flag): labels tagged rw.self_correction, labels never reviewed, clips > 30 s
    (`excluded_over_30s`, kept in the manifest because they may be segmented later).
  * no val yet. If the dataset has real (pseudonymised) speaker ids the speaker count is printed: a
    speaker-disjoint val slice would be possible, but the owner has not decided, so everything stays
    train / unassigned for now.

Usage: uv run python scripts/assign_arena_splits.py <short_id>      (e.g. 019f6442)
The 019f8e51 dataset keeps its own script (scripts/assign_arena_019f8e51_splits.py).
"""

import csv
import json
import sys
from collections import Counter

MAX_TRAIN_SEC = 30.0
# Datasets whose review is recorded at dataset level ("reviewed by default"), not per item: there, a missing
# per-item `reviewed` tag is not evidence of no review. Owner confirmed for 019f6442 on 2026-10-30.
REVIEWED_BY_DEFAULT = {"019f6442"}


def main(short: str) -> None:
    src = f"data/manifests/arena_{short}_ingested.jsonl"
    out = f"data/manifests/arena_{short}.jsonl"
    prov = {r["file_name"]: r for r in csv.DictReader(open(f"data/private/arena_{short}/gt_provenance.csv",
                                                           encoding="utf-8"))}
    counts, hours, train_spk = Counter(), Counter(), set()
    recs = [json.loads(line) for line in open(src, encoding="utf-8")]
    for r in recs:
        p = prov[r["source_utterance_id"]]
        if "rw.self_correction" in p["labels"].split(","):
            r["quality_flags"].append("excluded_label_self_correction")
            reason = "excluded_self_correction"
        elif p["provenance"] == "not_reviewed" and short not in REVIEWED_BY_DEFAULT:
            r["quality_flags"].append("excluded_label_not_reviewed")
            reason = "excluded_not_reviewed"
        elif r["duration_sec"] > MAX_TRAIN_SEC:
            r["quality_flags"].append("excluded_over_30s")
            reason = "excluded_over_30s"
        else:
            r["split"] = "train"
            train_spk.add(r["speaker_id"])
            if p["provenance"] == "auto_passed_no_human_review":
                r["quality_flags"].append("label_auto_passed_3model")
                reason = "train_auto_passed"
            elif p["provenance"] == "not_reviewed":
                r["quality_flags"].append("label_reviewed_by_dataset_policy")
                reason = "train_reviewed_by_dataset_policy"
            else:
                reason = "train_human_checked"
        counts[reason] += 1
        hours[reason] += r["duration_sec"] / 3600

    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k in sorted(counts):
        print(f"  {k:26s} {counts[k]:4d} clips {hours[k]:.3f} h")
    real = {r["speaker_id"] for r in recs if "-unknown-" not in r["speaker_id"]}
    if real:
        print(f"  real speaker ids: {len(real)} in dataset, {len(train_spk)} in train "
              "-> a speaker-disjoint val slice is possible (not assigned; owner decision pending)")
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1])
