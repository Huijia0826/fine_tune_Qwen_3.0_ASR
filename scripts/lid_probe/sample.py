"""Step 3a: stratified random sample of en-only requests (+ a zh-setting control group) for audio-based LID.

Population: data/private/lid_probe/population.jsonl (landscape.py), 1 s <= duration <= 60 s.
en-only strata (all known from metadata, so the sample can be re-weighted to the full population):
  H  production raw or final text is zh / zh-en (Chinese characters)            n=100
  Z  text is not Chinese, but the user produced Chinese text in some request     n=100
     in the 30 days (any setting): where "Chinese speech came out as English" would hide
  R  everything else (users who never produced Chinese text)                     n=200
control C: setting includes zh (zh only / zh+en / zh+other)                       n=150
Within a stratum: platform cells with proportional allocation (at least 8 per platform when available), random
order, at most 3 clips per user per stratum (a Chinese-text user can appear in H and Z: 3 + 3).
Inclusion probabilities: requests of one user in one cell are exchangeable, so pi_i = E[clips of user u in cell] /
N_{u,cell}; E[...] comes from re-running the exact sampler 4000 times. weight = 1 / pi.
Writes data/private/lid_probe/sample.jsonl
Usage: uv run python scripts/lid_probe/sample.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT  # noqa: E402

SEED, MC, CAP, MIN_CELL = 20261109, 4000, 3, 8
SIZES = {"H": 100, "Z": 100, "R": 200, "C": 150}
PLATFORMS = ["android", "ios", "macos", "windows", "web"]
ZH_TEXT = ("zh", "zh-en")


def stratum(p, zh_users):
    if p["setting"] in ("zh_only", "zh_en", "zh_plus_other"):
        return "C"
    if p["setting"] != "en_only":
        return None
    if p["raw_class"] in ZH_TEXT or p["out_class"] in ZH_TEXT:
        return "H"
    return "Z" if p["user_id"] in zh_users else "R"


def allocate(cells: dict, n: int) -> dict:
    """Proportional allocation with a floor of MIN_CELL (or the whole cell if smaller)."""
    tot = sum(len(v) for v in cells.values())
    alloc = {k: min(len(v), max(MIN_CELL, round(n * len(v) / tot))) for k, v in cells.items()}
    big = max(cells, key=lambda k: len(cells[k]))
    alloc[big] += n - sum(alloc.values())  # keep the stratum total exact
    return alloc


def draw(cells, alloc, rng):
    picked, per_user = [], collections.Counter()
    for k in sorted(cells):
        rows, need, got = cells[k], alloc[k], 0
        for i in rng.permutation(len(rows)):
            if got >= need:
                break
            u = rows[i]["user_id"]
            if per_user[u] < CAP:
                per_user[u] += 1
                picked.append((k, i))
                got += 1
    return picked


def main() -> None:
    pop = [json.loads(line) for line in open(OUT / "population.jsonl", encoding="utf-8")]
    zh_users = {p["user_id"] for p in pop if p["raw_class"] in ZH_TEXT or p["out_class"] in ZH_TEXT}
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(pop, key=lambda x: x["audio_id"]):
        if not 1 <= p["dur"] <= 60:
            continue
        s = stratum(p, zh_users)
        if s:
            cell = p["setting"] if s == "C" else p["platform"]  # control: stratify by setting class instead
            strata[s][cell].append(p)
    rng = np.random.default_rng(SEED)
    out, summary = [], {}
    for s, n in SIZES.items():
        cells = strata[s]
        alloc = allocate(cells, n)
        sample = draw(cells, alloc, np.random.default_rng([SEED, ord(s)]))
        # expected clips per (cell, user) by re-running the sampler
        exp = collections.Counter()
        for _ in range(MC):
            for k, i in draw(cells, alloc, rng):
                exp[(k, cells[k][i]["user_id"])] += 1
        n_user_cell = collections.Counter((k, r["user_id"]) for k, rows in cells.items() for r in rows)
        for k, i in sample:
            r = dict(cells[k][i])
            key = (k, r["user_id"])
            pi = exp[key] / MC / n_user_cell[key]
            r.update({"stratum": s, "cell": k, "pi": pi, "weight": 1 / pi,
                      "N_stratum": sum(len(v) for v in cells.values())})
            out.append(r)
        summary[s] = {"N": sum(len(v) for v in cells.values()), "n": len(sample),
                      "alloc": alloc, "cells_N": {k: len(v) for k, v in cells.items()},
                      "users_in_sample": len({cells[k][i]["user_id"] for k, i in sample})}
        print(s, summary[s])
    with open(OUT / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(summary, open(OUT / "sample_design.json", "w"), indent=1)
    print(f"{len(out)} clips -> {OUT / 'sample.jsonl'}")


if __name__ == "__main__":
    main()
