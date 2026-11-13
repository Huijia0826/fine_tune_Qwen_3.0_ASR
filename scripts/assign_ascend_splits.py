"""Merge the three ASCEND manifests into data/manifests/ascend.jsonl with the official split set.

Decision 2026-10-29: keep ASCEND's speaker-disjoint splits (train / val / test-as-diagnostic).
Inputs come from wf-ingest, which leaves every record `unassigned`.
Clips shorter than 0.3 s (a wf-validate error) are left out of the merged manifest; they remain in the
per-split ingest manifests (ascend_{train,validation,test}.jsonl), so they can still be traced.
"""

import json
from collections import Counter

SPLITS = {"train": "train", "validation": "val", "test": "test"}  # manifest name -> split value
MIN_DURATION_SEC = 0.3  # configs/dataset.yaml validation.min_duration_sec

counts = Counter()
with open("data/manifests/ascend.jsonl", "w", encoding="utf-8") as out:
    for name, split in SPLITS.items():
        with open(f"data/manifests/ascend_{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["duration_sec"] < MIN_DURATION_SEC:
                    counts[f"dropped_too_short_{split}"] += 1
                    continue
                rec["split"] = split
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[split] += 1
print(dict(counts))
