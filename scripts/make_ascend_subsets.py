"""ASCEND training-share ablation (2026-11-03): ascend_ninth (1/3 of ascend_third's train, i.e. ~1/9 of ASCEND train)
and ascend_none (no ASCEND train). Same method as ascend_third: train clips not kept become `unassigned`; ASCEND
val / test are unchanged, so the val loss and ASCEND val MER stay comparable across C, D, E and F.
Usage: python3 scripts/make_ascend_subsets.py
"""

import json
import random

src = [json.loads(line) for line in open("data/manifests/ascend_third.jsonl", encoding="utf-8")]
train = sorted(r["utterance_id"] for r in src if r["split"] == "train")
keep = set(random.Random(1234).sample(train, round(len(train) / 3)))
for name, kept in (("ascend_ninth", keep), ("ascend_none", set())):
    with open(f"data/manifests/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in src:
            if r["split"] == "train" and r["utterance_id"] not in kept:
                r = {**r, "split": "unassigned"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"data/manifests/{name}.jsonl: {len(kept)} train clips "
          f"({sum(r['duration_sec'] for r in src if r['utterance_id'] in kept) / 3600:.2f} h)")
