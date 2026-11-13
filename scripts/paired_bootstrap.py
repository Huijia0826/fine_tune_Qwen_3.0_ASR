"""Paired speaker bootstrap of the official-MER difference (benchmark scorer, as in the reports) between two runs
on the same manifest.

Resamples speakers with replacement (utterances of one speaker stay together), recomputes pooled MER for both
systems on the same resample, and reports the difference B − A with a 95% interval, overall and per language slice.
Reads reference/hypothesis from errors.csv written by wf-score and re-scores each utterance with official_mer.
Usage: uv run python scripts/paired_bootstrap.py reports/runs/<A> reports/runs/<B> [--n 2000] [--seed 1234]
"""

import argparse
import csv
import random
from collections import defaultdict

from whisperflow.eval.official import official_mer

SLICES = {"all": None, "en": "en_only", "zh": "zh_only", "zh-en": "zh_en_mixed"}


def load(run: str) -> dict[str, dict]:
    rows = {}
    for r in csv.DictReader(open(f"{run}/errors.csv", encoding="utf-8")):
        o = official_mer(r["reference"], r["hypothesis"])
        if o.included:  # the official scorer leaves empty references out of pooled rates
            rows[r["utterance_id"]] = {**r, "err": o.errors, "n": o.n_ref}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = sorted(A.keys() & B.keys())
    print(f"A = {args.a}\nB = {args.b}\n{len(ids)} shared utterances; difference = B − A, 95% speaker bootstrap\n")
    print(f"{'slice':<7}{'N':>5}{'spk':>5}{'A':>9}{'B':>9}{'B − A':>9}   95% interval")
    for name, mix in SLICES.items():
        sel = [i for i in ids if mix is None or A[i]["language_mix"] == mix]
        by_spk = defaultdict(list)
        for i in sel:
            by_spk[A[i]["speaker_id"]].append(i)
        spks = sorted(by_spk)

        def mer(utts: list[str], R: dict) -> float:
            return 100 * sum(R[i]["err"] for i in utts) / sum(R[i]["n"] for i in utts)

        rng = random.Random(args.seed)
        diffs = []
        for _ in range(args.n):
            utts = [i for s in rng.choices(spks, k=len(spks)) for i in by_spk[s]]
            diffs.append(mer(utts, B) - mer(utts, A))
        diffs.sort()
        lo, hi = diffs[int(0.025 * args.n)], diffs[int(0.975 * args.n) - 1]
        a, b = mer(sel, A), mer(sel, B)
        print(f"{name:<7}{len(sel):>5}{len(spks):>5}{a:>8.2f}%{b:>8.2f}%{b - a:>+8.2f}   [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
