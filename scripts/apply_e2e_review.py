"""Apply an owner review exported from the unified review page (scripts/build_e2e_review.py) (2026-11-09).

Writes data/private/e2e/reviewed/<pack>.jsonl: one row per item with the reviewed strict transcript, the
reviewed final output, verdict, note and review seconds (unreviewed rows keep the prefill, verdict "").
For the dictionary benchmark it also rebuilds data/manifests/dictionary_bench.jsonl: reference = reviewed
transcript, dropped clips removed, unreviewed clips kept with the policy-v1 prefill (flagged
`unreviewed_prefill`); the previous manifest is kept as dictionary_bench.pre_review.jsonl.

    uv run python scripts/apply_e2e_review.py ~/Downloads/e2e-dict-review.csv --pack e2e-dict
"""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--pack", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    out_dir = Path("data/private/e2e/reviewed")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.csv, out_dir / f"{args.pack}-review.csv")  # keep the owner's export as delivered
    with open(out_dir / f"{args.pack}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            keys = ("item_id", "source", "split", "transcript", "final_output", "verdict", "note",
                    "review_seconds", "prefill_transcript", "old_label")
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
    print(args.pack, Counter(r["verdict"] or "unreviewed" for r in rows))

    if args.pack == "e2e-dict":
        man_path = Path("data/manifests/dictionary_bench.jsonl")
        backup = man_path.with_name("dictionary_bench.pre_review.jsonl")
        if not backup.exists():
            shutil.copy2(man_path, backup)
        by_item = {r["item_id"]: r for r in rows}
        kept = Counter()
        with open(man_path, "w", encoding="utf-8") as f:
            for m in map(json.loads, open(backup, encoding="utf-8")):
                r = by_item[m["source_utterance_id"]]
                if r["verdict"] == "drop":
                    kept["dropped"] += 1
                    continue
                if r["verdict"]:
                    m["raw_transcript"] = r["transcript"].strip()
                    m["formatted_transcript"] = r["final_output"].strip()
                    m["reviewer_id"] = "owner"
                    kept["reviewed"] += 1
                else:
                    m["quality_flags"] = sorted(set(m["quality_flags"]) | {"unreviewed_prefill"})
                    kept["unreviewed"] += 1
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print("dictionary_bench manifest:", dict(kept))


if __name__ == "__main__":
    main()
