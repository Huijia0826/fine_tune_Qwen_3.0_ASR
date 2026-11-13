"""Phase 1 of docs/data_collection_plan.md: pick locked-test and val candidates from online.py scans.

Per scenario (scripts/scan_scenarios.py rules) and user, in a seeded random order: up to 3 clips → test, the next 1 →
val, the rest stay in the train pool (reviewed later). Test / mic-check recordings and tester accounts are excluded
(scripts/test_utterances.py, owner 2026-11-01). Users may appear in several splits (owner, 2026-10-31); each
test/val clip is tagged `seen` when its user also has train-pool clips. Clips already in our datasets are excluded, and
train-pool clips whose text duplicates a test/val clip are dropped.

Output (data/private/online_phase1/): candidates.jsonl (test + val + train pool, with split / scenario / seen),
bench/online-zh-phase1/{spec.yaml, items.jsonl} for trace-lookup bench.py draft-gt (test + val only), summary.md.
Usage: python3 scripts/select_online_phase1.py data/private/online_scan/zh_dictate_2026-06-01_10-31.jsonl
"""

import argparse
import collections
import importlib.util
import json
import random
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location("scan_scenarios", Path(__file__).with_name("scan_scenarios.py"))
scan_scenarios = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan_scenarios)
spec = importlib.util.spec_from_file_location("test_utterances", Path(__file__).with_name("test_utterances.py"))
test_utterances = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_utterances)

GROUP = {"S1_zh_en": "CS", "S3_terms": "CS", "S2_zh": "ZH", "S4_short": "S4", "S5_selfcorr": "S5"}
TEST_TARGET = {"CS": None, "ZH": 200, "S4": 150, "S5": None}  # None = take every candidate
TEST_PER_USER, VAL_PER_USER = 3, 1
EXISTING = ["benchmark_chinese", "benchmark_v1", "arena_019f8e51", "arena_019f6442", "arena_019f8e51_ingested",
            "arena_019f6442_ingested", "benchmark_chinese_ingested"]
OUT = Path("data/private/online_phase1")
BENCH = OUT / "bench" / "online-zh-phase1"
SEED = 20261031


def dur(r: dict) -> float:
    return (r.get("cleanup_audio_duration_ms") or 0) / 1000


def norm_text(t: str) -> str:
    return re.sub(r"[\W_]+", "", (t or "").lower())


def existing_ids() -> set[str]:
    ids = set()
    for name in EXISTING:
        p = Path(f"data/manifests/{name}.jsonl")
        if p.exists():
            for line in open(p, encoding="utf-8"):
                sid = (json.loads(line).get("source_utterance_id") or "").split("#")[0]
                ids.add(re.sub(r"\.\w+$", "", sid))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.scan, encoding="utf-8")][1:]
    known = existing_ids()
    tests = test_utterances.TestDetector(rows)
    funnel = collections.Counter({"scanned": len(rows)})
    pool, seen_text = [], set()
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        if r.get("error_stage") or not r.get("s3_key") or r.get("app") == "com.voicecursor.android.type":
            funnel["error / no audio / onboarding"] += 1
        elif tests.why(r):
            funnel[f"test recording ({tests.why(r)})"] += 1
        elif not 1 <= dur(r) <= 30:
            funnel["outside 1–30 s"] += 1
        elif r["audio_id"] in known:
            funnel["already in our datasets"] += 1
        elif GROUP.get(scan_scenarios.scenario(r)) is None:
            funnel["no zh scenario (empty / no Chinese)"] += 1
        elif norm_text(r.get("transcription")) in seen_text:
            funnel["duplicate text"] += 1
        else:
            seen_text.add(norm_text(r.get("transcription")))
            r["group"], r["scenario"] = GROUP[scan_scenarios.scenario(r)], scan_scenarios.scenario(r)
            pool.append(r)
    funnel["pool"] = len(pool)

    rng = random.Random(SEED)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in pool:
        by[r["group"]][r["user_id"]].append(r)
    for g, users in by.items():
        test_cands = []
        for u in sorted(users):
            rs = users[u]
            rng.shuffle(rs)
            for k, r in enumerate(rs):
                r["split"] = "test" if k < TEST_PER_USER else "val" if k < TEST_PER_USER + VAL_PER_USER else "train"
            test_cands += [r for r in rs if r["split"] == "test"]
        target = TEST_TARGET[g]
        if target and len(test_cands) > target:  # keep one clip per user first, then fill: as many speakers as possible
            rng.shuffle(test_cands)
            first, rest, used = [], [], set()
            for r in test_cands:
                (rest if r["user_id"] in used else first).append(r)
                used.add(r["user_id"])
            keep = {id(r) for r in (first + rest)[:target]}
            for r in test_cands:
                if id(r) not in keep:
                    r["split"] = "train"
    eval_text = {norm_text(r["transcription"]) for r in pool if r["split"] != "train"}
    pool = [r for r in pool if r["split"] != "train" or norm_text(r["transcription"]) not in eval_text]
    train_users = {r["user_id"] for r in pool if r["split"] == "train"}
    for r in pool:
        r["seen"] = r["split"] != "train" and r["user_id"] in train_users

    OUT.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    keys = ("audio_id", "user_id", "split", "group", "scenario", "seen", "timestamp", "client_source", "model", "app",
            "s3_key", "langfuse_trace_id", "langfuse", "flags", "transcription", "processed",
            "cleanup_audio_duration_ms", "mode", "lang")
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False) + "\n")
    with open(BENCH / "items.jsonl", "w", encoding="utf-8") as f:
        for r in pool:
            if r["split"] != "train":
                f.write(json.dumps({**{k: r.get(k) for k in keys}, "lang_detected": "zh"}, ensure_ascii=False) + "\n")
    (BENCH / "spec.yaml").write_text(
        "# Items chosen by scripts/select_online_phase1.py (not by bench.py build); used for draft-gt only.\n"
        "name: online-zh-phase1\npurpose: targeted\nwindow: {since: 2026-08-31, until: 2026-11-01}\n"
        "filter: {mode: dictate, lang: zh}\n"
        "drafters: ['whisper:large-v3']   # owner 2026-11-01: Whisper, not Gemini\nseed: 0\n",
        encoding="utf-8")

    lines = ["# Online phase 1 candidates (private: contains user text)", "", "## Funnel", ""]
    lines += [f"- {k}: {v}" for k, v in funnel.items()]
    lines += ["", "## Split × scenario", "", "| split | group | clips | users | seen | hours |", "|---|---|---:|---:|---:|---:|"]
    for sp in ("test", "val", "train"):
        for g in ("CS", "ZH", "S4", "S5"):
            sel = [r for r in pool if r["split"] == sp and r["group"] == g]
            lines.append(f"| {sp} | {g} | {len(sel)} | {len({r['user_id'] for r in sel})} | "
                         f"{sum(r['seen'] for r in sel)} | {sum(dur(r) for r in sel) / 3600:.2f} |")
    lines += ["", "## Random test examples (production ASR text)", ""]
    rng2 = random.Random(1)
    for g in ("CS", "ZH", "S4", "S5"):
        sel = [r for r in pool if r["split"] == "test" and r["group"] == g]
        lines.append(f"### {g}")
        lines += [f"- ({dur(r):.1f} s, {r['scenario']}) {r['transcription']}" for r in rng2.sample(sel, min(6, len(sel)))]
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:lines.index("## Random test examples (production ASR text)")]))


if __name__ == "__main__":
    main()
