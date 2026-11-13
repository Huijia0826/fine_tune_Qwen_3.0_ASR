"""Pick anonymised before/after examples (dictation output -> edited result) for the report.

Scrubs emails, phones, URLs, @handles and long digit runs; names must still be checked by hand.
Usage: python3 scripts/edit_prefs/examples.py [--per-pref 1] > data/private/edit_prefs/examples_candidates.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "private", "edit_prefs")
SCRUB = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"), (re.compile(r"https?://\S+|www\.\S+"), "[url]"),
         (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[phone]"), (re.compile(r"@\w+"), "@[handle]")]


def scrub(t: str, n: int = 160) -> str:
    t = (t or "").replace("\n", " ⏎ ")
    for rx, rep in SCRUB:
        t = rx.sub(rep, t)
    return t if len(t) <= n else t[:n] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pref", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]
    prefs = [json.loads(l) for l in open(os.path.join(D, "pref_pairs.jsonl"), encoding="utf-8")]
    want = {(p["user"], p["pref"]) for p in prefs}
    print("## Learned-preference users: dictation output -> edited result\n")
    for user, pref in sorted(want, key=lambda x: x[1]):
        c = [p for p in pairs if p["user"] == user and p["category"] == pref and p["how"] == "exact"
             and 3 <= len(p["dictate_output"] or "") <= 260]
        for p in random.sample(c, min(a.per_pref, len(c))):
            print(f"- [{pref} | {p['lang_group']} | {p['app']} | user {user} | +{p['gap_min']} min]\n"
                  f"  - instruction: {scrub(p['instruction'], 100)}\n  - before: {scrub(p['dictate_output'])}\n"
                  f"  - after: {scrub(p['edited_result'])}")


if __name__ == "__main__":
    main()
