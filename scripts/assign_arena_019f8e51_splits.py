"""Assign arena_019f8e51 (English, refined E2E labels) to train — owner decision 2026-10-30, option B.

  * train: clips <= 30 s whose label is human-edited, human-reviewed, or auto-passed (3-model consensus,
    nobody listened). Auto-passed clips get the quality flag `label_auto_passed_3model` so they can be
    removed in an ablation (A vs B).
  * excluded (`unassigned` + flag): labels tagged rw.self_correction (the label drops spoken content),
    labels never reviewed, clips > 30 s (too long to train on; see experiment backlog E3).
  * no val: there are no speaker ids, so a speaker-disjoint val split is impossible.

Usage: uv run python scripts/assign_arena_019f8e51_splits.py
"""

import csv
import json
from collections import Counter

SRC = "data/manifests/arena_019f8e51_ingested.jsonl"
OUT = "data/manifests/arena_019f8e51.jsonl"
PROV = "data/private/arena_019f8e51/gt_provenance.csv"
MAX_TRAIN_SEC = 30.0

prov = {r["file_name"]: r for r in csv.DictReader(open(PROV, encoding="utf-8"))}
counts, hours = Counter(), Counter()
recs = [json.loads(line) for line in open(SRC, encoding="utf-8")]
for r in recs:
    p = prov[r["source_utterance_id"]]
    if "rw.self_correction" in p["labels"].split(","):
        r["quality_flags"].append("excluded_label_self_correction")
        reason = "excluded_self_correction"
    elif p["provenance"] == "not_reviewed":
        r["quality_flags"].append("excluded_label_not_reviewed")
        reason = "excluded_not_reviewed"
    elif r["duration_sec"] > MAX_TRAIN_SEC:
        r["quality_flags"].append("excluded_over_30s")
        reason = "excluded_over_30s"
    else:
        r["split"] = "train"
        if p["provenance"] == "auto_passed_no_human_review":
            r["quality_flags"].append("label_auto_passed_3model")
            reason = "train_auto_passed"
        else:
            reason = "train_human_checked"
    counts[reason] += 1
    hours[reason] += r["duration_sec"] / 3600

with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for k in sorted(counts):
    print(f"  {k:26s} {counts[k]:4d} clips {hours[k]:.2f} h")
