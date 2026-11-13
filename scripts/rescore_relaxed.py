"""Rescore saved runs with relaxed diagnostics (2026-11-02): how much of each system's MER is English spacing /
compounding ("TopstepX" vs "top step X") and numeral style? Official MER stays the headline; nothing is re-decoded.

    uv run python scripts/rescore_relaxed.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

from whisperflow.eval.official import merge_compounds, official_mer, official_mer_num

sys.path.insert(0, str(Path(__file__).parent))
from build_c_comparison import BV1, bootstrap, load_bv1  # noqa: E402

VAL = {"zero-shot + loop guard": "guard_zs_{}", "B step 800": "sel_lora_large_v3_B_step_800_{}",
       "C step 600": "sel_lora_large_v3_C_step_600_{}"}
VAL_SETS = {"bczh": "data/manifests/benchmark_chinese.jsonl", "online": "data/manifests/online_phase1.jsonl"}


def score_all(ref: str, hyp: str) -> dict:
    merged, n = merge_compounds(ref, hyp)
    o = official_mer(ref, hyp)
    return {"n": o.n_ref, "included": o.included, "official": o.errors,
            "compound": official_mer(ref, merged).errors, "num": official_mer_num(ref, hyp).errors,
            "compound_num": official_mer_num(ref, merged).errors, "merges": len(n), "pairs": n}


def pooled(rows: list[dict], key: str) -> float:
    rows = [r for r in rows if r["included"]]
    return 100 * sum(r[key] for r in rows) / sum(r["n"] for r in rows)


def main() -> None:
    keys = ["official", "compound", "num", "compound_num"]
    print("## benchmark v1 (412)\n")
    print("| system | official | spacing relaxed | numerals relaxed | both | clips with a merge |")
    print("|---|---:|---:|---:|---:|---:|")
    per_sys, pairs = {}, Counter()
    for name, run in BV1.items():
        rows = {}
        for uid, r in load_bv1(run).items():
            s = score_all(r["reference"], r["hypothesis"])
            rows[uid] = {**r, **s}
            if name in ("C step 600", "VoiceCursor"):
                for a, b in s["pairs"]:
                    pairs[(name, a, b)] += 1
        per_sys[name] = rows
        rs = list(rows.values())
        print(f"| {name} | " + " | ".join(f"{pooled(rs, k):.2f}%" for k in keys)
              + f" | {sum(r['merges'] > 0 for r in rs)} |")

    print("\nslices (C step 600 vs VoiceCursor):\n")
    print("| slice | C official | VC official | C relaxed both | VC relaxed both |")
    print("|---|---:|---:|---:|---:|")
    for sl, mix in [("all", None), ("en", "en_only"), ("zh", "zh_only"), ("zh-en", "zh_en_mixed")]:
        vals = []
        for sysname in ("C step 600", "VoiceCursor"):
            vals.append([r for r in per_sys[sysname].values() if mix is None or r["language_mix"] == mix])
        print(f"| {sl} | {pooled(vals[0], 'official'):.2f}% | {pooled(vals[1], 'official'):.2f}% | "
              f"{pooled(vals[0], 'compound_num'):.2f}% | {pooled(vals[1], 'compound_num'):.2f}% |")

    print("\npaired speaker bootstrap (difference, 95% interval):\n")
    for a, b in [("VoiceCursor", "C step 600"), ("C step 600", "B step 800")]:
        for key in ("official", "compound", "compound_num"):
            A = {u: {**r, "err": r[key]} for u, r in per_sys[a].items()}
            B = {u: {**r, "err": r[key]} for u, r in per_sys[b].items()}
            ids = sorted(A.keys() & B.keys())
            d, lo, hi = bootstrap(A, B, ids)
            print(f"- {a} − {b}, {key}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    print("\n## users' val pooled (125)\n")
    print("| system | official | spacing relaxed | numerals relaxed | both |")
    print("|---|---:|---:|---:|---:|")
    for name, pattern in VAL.items():
        rs = []
        for tag, man in VAL_SETS.items():
            refs = {r["utterance_id"]: r["raw_transcript"] for r in map(json.loads, open(man, encoding="utf-8"))
                    if r["split"] == "val"}
            preds = Path(f"reports/runs/{pattern.format(tag)}/predictions.jsonl")
            for p in map(json.loads, open(preds, encoding="utf-8")):
                rs.append(score_all(refs[p["utterance_id"]], p["hypothesis"]))
        print(f"| {name} | " + " | ".join(f"{pooled(rs, k):.2f}%" for k in keys) + " |")

    print("\nall rewrites on benchmark v1 (system: hypothesis span -> reference span):")
    for (sysname, a, b), n in sorted(pairs.items()):
        print(f"  {sysname}: {n}× {a!r} -> {b!r}")


if __name__ == "__main__":
    main()
