"""Exclude the 7 Spanish clips from benchmark v1 (owner decision 2026-10-29).

They are tagged `en` in benchmark v1.0 but are Spanish speech; the product targets zh/en only.
Same mechanism as the 6 Japanese clips: split -> "unassigned" + quality flag, edited in place.
The previous manifest is kept as benchmark_v1.pre_es_exclusion.jsonl (sha256 673b32e0...).
"""

import json
from pathlib import Path

PATH = Path("data/manifests/benchmark_v1.jsonl")
FLAG = "excluded_non_target_language_es"
SPANISH = {  # utterance_id suffixes; identified from the reference transcripts
    "6d7f728b", "3811a730", "d44ea828", "ba2a446d", "1c76d885", "8065a12d", "13875622",
}

recs = [json.loads(ln) for ln in PATH.read_text(encoding="utf-8").splitlines()]
hit = 0
for r in recs:
    if r["utterance_id"][-8:] in SPANISH:
        hit += 1
        r["split"] = "unassigned"
        if FLAG not in r["quality_flags"]:
            r["quality_flags"].append(FLAG)
assert hit == len(SPANISH), "exclusion list did not match exactly"
PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
print(f"flagged {hit}; test clips now {sum(r['split'] == 'test' for r in recs)}")
