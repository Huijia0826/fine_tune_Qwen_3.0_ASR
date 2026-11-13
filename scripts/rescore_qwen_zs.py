"""Qwen3-ASR zero-shot vs our Whisper systems (2026-11-09), from saved runs; nothing is re-decoded.

Qwen3-ASR writes hesitation fillers (呃 / 嗯 / um / uh) that our references and Whisper leave out, so every system
is scored three ways: official MER (headline rule), fillers removed from the hypothesis (strip_fillers, the
label rule; the post-processor removes them anyway), and fillers + numeral style relaxed (official_mer_num).

    uv run python scripts/rescore_qwen_zs.py > reports/qwen3_asr_zs_tables.md
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from whisperflow.eval.official import official_mer, official_mer_num
from whisperflow.text.fillers import strip_fillers

RUNS = Path("reports/runs")
SYSTEMS = {  # display name -> {set: run dir}
    "large-v3 zero-shot": {"bczh": "guard_zs_bczh", "online": "guard_zs_online", "bv1": "bv1_large_v3_zs"},
    "I2 step 600": {"bczh": "sel_lora_large_v3_I2_step_600_bczh", "online": "sel_lora_large_v3_I2_step_600_online",
                    "bv1": "bv1_lora_large_v3_I2_step_600", "ownertest": "ownertest_I2_step_600",
                    "v2test": "v2test_I2_step_600"},
    "Qwen3-ASR-0.6B": {"bczh": "sel_qwen3asr_06b_base_bczh", "online": "sel_qwen3asr_06b_base_online",
                       "bv1": "bv1_qwen3asr_06b_zs", "ownertest": "ownertest_qwen3asr_06b_zs",
                       "v2test": "v2test_qwen3asr_06b_zs"},
    "Qwen3-ASR-1.7B": {"bczh": "sel_qwen3asr_17b_base_bczh", "online": "sel_qwen3asr_17b_base_online",
                       "bv1": "bv1_qwen3asr_17b_zs", "ownertest": "ownertest_qwen3asr_17b_zs",
                       "v2test": "v2test_qwen3asr_17b_zs"},
    "Qwen3-ASR-1.7B, 25 s chunks": {"bczh": "sel_qwen3asr_17b_chunk25_base_bczh",
                                    "online": "sel_qwen3asr_17b_chunk25_base_online",
                                    "bv1": "bv1_qwen3asr_17b_chunk25_zs",
                                    "ownertest": "ownertest_qwen3asr_17b_chunk25_zs",
                                    "v2test": "v2test_qwen3asr_17b_chunk25_zs"},
    "Qwen3-ASR-1.7B fine-tuned (I2 data) step 1000": {"bczh": "sel_qwen3asr_I2_step_1000_bczh",
                                                       "online": "sel_qwen3asr_I2_step_1000_online",
                                                       "bv1": "bv1_qwen3asr_I2_step_1000",
                                                       "ownertest": "ownertest_qwen3asr_I2_step_1000",
                                                       "v2test": "v2test_qwen3asr_I2_step_1000"},
    "VoiceCursor": {"bv1": "voicecursor_e2e_benchmark_v1"},
}
MANIFESTS = {"bczh": ("data/manifests/benchmark_chinese.jsonl", "val"),
             "online": ("data/manifests/online_phase1.jsonl", "val"),
             "ownertest": ("data/manifests/recording_freeform_test.jsonl", "test"),
             "v2test": ("data/manifests/recording_v2_test.jsonl", "test")}
RULES = {"official": lambda r, h: official_mer(r, h),
         "no fillers": lambda r, h: official_mer(r, strip_fillers(h)),
         "no fillers + numerals": lambda r, h: official_mer_num(r, strip_fillers(h))}


def load(set_name: str, run: str) -> dict[str, dict] | None:
    """utterance_id -> {reference, hypothesis, speaker_id, language_mix?}; None if the run is missing."""
    if set_name == "bv1":
        path = RUNS / run / "errors.csv"
        if not path.exists():
            return None
        return {r["utterance_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    path = RUNS / run / "predictions.jsonl"
    if not path.exists():
        return None
    man, split = MANIFESTS[set_name]
    recs = {r["utterance_id"]: r for r in map(json.loads, open(man, encoding="utf-8")) if r["split"] == split}
    return {p["utterance_id"]: {"reference": recs[p["utterance_id"]]["raw_transcript"], "hypothesis": p["hypothesis"],
                                "speaker_id": recs[p["utterance_id"]]["speaker_id"]}
            for p in map(json.loads, open(path, encoding="utf-8"))}


def scored(rows: dict, rule: str) -> dict[str, dict]:
    out = {}
    for uid, r in rows.items():
        o = RULES[rule](r["reference"], r["hypothesis"])
        if o.included:
            out[uid] = {**r, "err": o.errors, "n": o.n_ref}
    return out


def mer(rows: dict, ids) -> float:
    return 100 * sum(rows[i]["err"] for i in ids) / sum(rows[i]["n"] for i in ids)


def bootstrap(a: dict, b: dict, ids: list[str], n: int = 2000, seed: int = 1234) -> tuple[float, float, float]:
    """Paired speaker bootstrap of MER(a) − MER(b) (same resampling as scripts/paired_bootstrap.py)."""
    by_spk = defaultdict(list)
    for i in ids:
        by_spk[b[i]["speaker_id"]].append(i)
    spks = sorted(by_spk)
    rng = random.Random(seed)
    diffs = sorted(mer(a, u) - mer(b, u) for u in
                   ([i for s in rng.choices(spks, k=len(spks)) for i in by_spk[s]] for _ in range(n)))
    return mer(a, ids) - mer(b, ids), diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def main() -> None:
    data = {name: {s: load(s, run) for s, run in sets.items()} for name, sets in SYSTEMS.items()}
    for rule in RULES:
        print(f"\n## {rule}\n")
        print("| system | bczh val (68) | online val (57) | users' val pooled | bv1 all | bv1 en | bv1 zh | bv1 zh-en "
              "| owner freeform | recording_v2 test |")
        print("|---|" + "---:|" * 9)
        for name, sets in data.items():
            cells = []
            sc = {s: scored(rows, rule) if rows else None for s, rows in sets.items()}
            for s in ("bczh", "online"):
                cells.append(f"{mer(sc[s], sc[s]):.2f}%" if sc.get(s) else "")
            if sc.get("bczh") and sc.get("online"):
                pooled = {**{f"b{k}": v for k, v in sc["bczh"].items()}, **{f"o{k}": v for k, v in sc["online"].items()}}
                cells.append(f"**{mer(pooled, pooled):.2f}%**")
            else:
                cells.append("")
            bv = sc.get("bv1")
            for mix in (None, "en_only", "zh_only", "zh_en_mixed"):
                ids = [i for i, r in (bv or {}).items() if mix is None or r["language_mix"] == mix]
                cells.append(f"{mer(bv, ids):.2f}%" if ids else "")
            for s in ("ownertest", "v2test"):
                cells.append(f"{mer(sc[s], sc[s]):.2f}%" if sc.get(s) else "")
            print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## paired speaker bootstrap on benchmark v1 (A − B, 95% CI)\n")
    print("| rule | A | B | all | en | zh | zh-en |")
    print("|---|---|---|---:|---:|---:|---:|")
    pairs = [("Qwen3-ASR-1.7B fine-tuned (I2 data) step 1000", "I2 step 600"),
             ("Qwen3-ASR-1.7B fine-tuned (I2 data) step 1000", "Qwen3-ASR-1.7B, 25 s chunks"),
             ("Qwen3-ASR-1.7B, 25 s chunks", "I2 step 600"), ("Qwen3-ASR-1.7B, 25 s chunks", "VoiceCursor"),
             ("Qwen3-ASR-1.7B, 25 s chunks", "large-v3 zero-shot"),
             ("Qwen3-ASR-1.7B", "I2 step 600"), ("Qwen3-ASR-1.7B", "VoiceCursor"),
             ("Qwen3-ASR-1.7B", "large-v3 zero-shot"), ("Qwen3-ASR-0.6B", "I2 step 600")]
    for rule in ("official", "no fillers"):
        for a, b in pairs:
            if not (data[a].get("bv1") and data[b].get("bv1")):
                continue
            sa, sb = scored(data[a]["bv1"], rule), scored(data[b]["bv1"], rule)
            cells = []
            for mix in (None, "en_only", "zh_only", "zh_en_mixed"):
                ids = sorted(i for i in sa.keys() & sb.keys() if mix is None or sb[i]["language_mix"] == mix)
                d, lo, hi = bootstrap(sa, sb, ids)
                cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            print(f"| {rule} | {a} | {b} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
