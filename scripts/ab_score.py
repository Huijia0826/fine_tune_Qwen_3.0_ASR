"""Score a blind A/B review (scripts/build_ab_review.py): un-blind with key.json, report win / tie / loss per
system overall and per stratum, a sign-test p-value (ties dropped), and the "added content / answered" flags.

    uv run python scripts/ab_score.py data/private/ab_review/ab-bv1 ~/Downloads/ab-bv1-review.csv
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    folder, review = Path(sys.argv[1]), sys.argv[2]
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    systems = list(key["systems"])
    rows = [r for r in csv.DictReader(open(review, encoding="utf-8-sig")) if r["choice"]]
    by = defaultdict(Counter)
    added = Counter()
    for r in rows:
        k = key["items"][r["item"]]
        winner = {"A": k["A"], "B": k["B"]}.get(r["choice"], r["choice"])  # system name, "tie" or "bad"
        for group in ("all", k["stratum"]):
            by[group][winner] += 1
        if r["addA"] == "1":
            added[k["A"]] += 1
        if r["addB"] == "1":
            added[k["B"]] += 1
    a, b = systems
    print(f"{len(rows)} judged items · systems: {key['systems']}")
    print(f"| group | n | {a} better | {b} better | about the same | both bad | sign test p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for g in ["all"] + sorted(k for k in by if k != "all"):
        c = by[g]
        n = sum(c.values())
        print(f"| {g} | {n} | {c[a]} ({100 * c[a] / n:.0f}%) | {c[b]} ({100 * c[b] / n:.0f}%) | {c['tie']} | "
              f"{c['bad']} | {sign_test(c[a], c[b]):.3f} |")
    print("added content / answered the question:", dict(added))


if __name__ == "__main__":
    main()
