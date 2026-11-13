"""Score the online service against the owner's labels from the lid-probe review pack (run after the review).

Reads the exported review.csv (verdict ok / fix = labelled, drop = excluded) and bench/selection.jsonl, and prints
official MER of the production raw ASR text and final output vs the owner's transcript, per group:
  en-only Chinese speech (re-weighted by 1/sel_prob back to the probe sample), its parts (hidden / H), and control.
Usage: uv run python scripts/lid_probe/score_review.py <exported review.csv> [--name lid-probe-zh]
"""

import argparse
import collections
import csv
import json
from pathlib import Path

from whisperflow.eval.official import official_mer_num

OUT = Path(__file__).resolve().parents[2] / "data/private/lid_probe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("review_csv")
    ap.add_argument("--name", default="lid-probe-zh")
    a = ap.parse_args()
    sel = {s["audio_id"]: s for s in map(json.loads, open(OUT / "review" / a.name / "bench/selection.jsonl"))}
    rows = [r for r in csv.DictReader(open(a.review_csv, encoding="utf-8-sig")) if r["verdict"] in ("ok", "fix")]
    drops = sum(1 for r in csv.DictReader(open(a.review_csv, encoding="utf-8-sig")) if r["verdict"] == "drop")
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # err_raw, err_out, n_ref, clips (weighted by 1/sel_prob)
    for r in rows:
        s = sel[r["audio_id"]]
        w = 1 / s["sel_prob"]
        raw = official_mer_num(r["final_transcript"], r["online_raw"])
        out = official_mer_num(r["final_transcript"], r["online_output"])
        top = "en_only" if s["group"].startswith("en_only") else "control"
        for g in (top, s["group"], f"{top}|{s['model']}"):
            agg[g][0] += w * raw.errors
            agg[g][1] += w * out.errors
            agg[g][2] += w * raw.n_ref
            agg[g][3] += 1
    print(f"labelled {len(rows)}, dropped {drops}")
    print("| group | clips | raw MER | final MER |\n|---|---|---|---|")
    for g in sorted(agg):
        e1, e2, n, k = agg[g]
        if n:
            print(f"| {g} | {k} | {e1 / n:.1%} | {e2 / n:.1%} |")


if __name__ == "__main__":
    main()
