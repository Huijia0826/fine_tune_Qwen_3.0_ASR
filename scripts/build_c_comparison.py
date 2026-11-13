"""Comparison page for lora_large_v3_C step_600 (2026-11-02); numbers recomputed from saved runs (official MER).

    uv run python scripts/build_c_comparison.py   # -> data/private/c_comparison.html (contains transcripts)

Sections: benchmark v1 by slice (+ paired speaker bootstrap), val sets (users' val, ASCEND, long-form), where
and on which clips VoiceCursor beats C, and E7 latency. No decoding happens here.
"""

import csv
import difflib
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from whisperflow.eval.official import official_mer

RUNS = Path("reports/runs")
OUT = Path("data/private/c_comparison.html")

BV1 = {  # display name -> run dir; order = categorical slot order in the charts
    "small base": "benchmark_v1_base_ref2",
    "large-v3 zero-shot": "bv1_large_v3_zs",
    "B step 800": "bv1_lora_large_v3_B_step_800",
    "C step 600": "bv1_lora_large_v3_C_step_600",
    "VoiceCursor": "voicecursor_e2e_benchmark_v1",
}
SLICES = [("all", None, None), ("en", "language_mix", "en_only"), ("zh", "language_mix", "zh_only"),
          ("zh-en", "language_mix", "zh_en_mixed"), ("<5 s", "duration", "<5s"), ("5-15 s", "duration", "5-15s"),
          ("15-30 s", "duration", "15-30s"), (">30 s", "duration", ">30s")]
PAIRS = [("C step 600", "B step 800"), ("C step 600", "large-v3 zero-shot"), ("VoiceCursor", "C step 600")]

VAL_SETS = {  # name -> (manifest, run-dir suffix)
    "users' val: benchmark_chinese (68)": ("data/manifests/benchmark_chinese.jsonl", "bczh"),
    "users' val: online (57)": ("data/manifests/online_phase1.jsonl", "online"),
    "ASCEND val (1000)": ("data/manifests/ascend.jsonl", "ascend"),
    "long-form val (43)": ("data/manifests/val_longform.jsonl", "longform"),
}
VAL_SYSTEMS = {"large-v3 zero-shot": "zs_large_v3_{}", "zero-shot + loop guard": "guard_zs_{}",
               "B step 800": "sel_lora_large_v3_B_step_800_{}", "C step 600": "sel_lora_large_v3_C_step_600_{}"}

LATENCY = [("hf_fp32_beam5", "HF fp32 beam5 (current)"), ("hf_bf16_beam5", "HF bf16 beam5"),
           ("hf_bf16_greedy", "HF bf16 greedy"), ("ct2_fp16_beam5", "faster-whisper fp16 beam5"),
           ("ct2_fp16_greedy", "faster-whisper fp16 greedy"), ("ct2_int8f16_beam5", "faster-whisper int8 beam5"),
           ("turbo_zs_bf16_beam5", "turbo zero-shot bf16 beam5")]


def load_bv1(run: str) -> dict[str, dict]:
    rows = {}
    for r in csv.DictReader(open(RUNS / run / "errors.csv", encoding="utf-8")):
        o = official_mer(r["reference"], r["hypothesis"])
        if o.included:
            rows[r["utterance_id"]] = {**r, "err": o.errors, "n": o.n_ref}
    return rows


def mer(rows: dict, ids) -> float:
    return 100 * sum(rows[i]["err"] for i in ids) / sum(rows[i]["n"] for i in ids)


def bootstrap(a: dict, b: dict, ids: list[str], n: int = 2000, seed: int = 1234) -> tuple[float, float, float]:
    """Paired speaker bootstrap of MER(a) − MER(b), same resampling as scripts/paired_bootstrap.py."""
    by_spk = defaultdict(list)
    for i in ids:
        by_spk[b[i]["speaker_id"]].append(i)
    spks = sorted(by_spk)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        utts = [i for s in rng.choices(spks, k=len(spks)) for i in by_spk[s]]
        diffs.append(mer(a, utts) - mer(b, utts))
    diffs.sort()
    return mer(a, ids) - mer(b, ids), diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def val_mer(manifest: str, run: str) -> dict | None:
    path = RUNS / run / "predictions.jsonl"
    if not path.exists():
        return None
    refs = {r["utterance_id"]: r["raw_transcript"] for r in map(json.loads, open(manifest, encoding="utf-8"))
            if r["split"] == "val"}
    err = tok = 0
    for p in map(json.loads, open(path, encoding="utf-8")):
        o = official_mer(refs[p["utterance_id"]], p["hypothesis"])
        if o.included:
            err, tok = err + o.errors, tok + o.n_ref
    return {"err": err, "n": tok}


# Display tokens: one CJK character, one Latin/number word, whitespace, or one other character (punctuation).
_DISPLAY_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*|\s+|.", re.S)
CONTEXT = 6  # words kept on each side of a difference when unchanged runs are folded


def _is_word(tok: str) -> bool:
    return bool(re.match(r"[\u4e00-\u9fff]|[A-Za-z0-9]", tok))


def diff_html(ref: str, hyp: str) -> str:
    """Hypothesis text with its differences from the reference marked, for reading (not scoring).

    Words are aligned with difflib on lower-cased word tokens; punctuation and spacing are shown but never marked
    (the official scorer ignores them too). <del> = reference words the hypothesis missed or replaced, <mark> =
    hypothesis words that are wrong or extra. Unchanged runs longer than 2 × CONTEXT words get their middle wrapped
    in <span class="fold">, which the page can hide."""
    r_words = [t for t in _DISPLAY_TOKEN.findall(ref) if _is_word(t)]
    h_toks = _DISPLAY_TOKEN.findall(hyp)
    h_word_pos = [i for i, t in enumerate(h_toks) if _is_word(t)]
    sm = difflib.SequenceMatcher(None, [w.lower() for w in r_words], [h_toks[i].lower() for i in h_word_pos],
                                 autojunk=False)
    status = ["eq"] * len(h_word_pos)  # per hypothesis word
    deleted_before: dict[int, list[str]] = {}  # hypothesis word index -> missed reference words shown before it
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete"):
            deleted_before.setdefault(j1, []).extend(r_words[i1:i2])
        if op in ("replace", "insert"):
            for j in range(j1, j2):
                status[j] = "bad"

    def joined(words: list[str]) -> str:
        sep = " " if any(re.match(r"[A-Za-z0-9]", w) for w in words) else ""
        return sep.join(words)

    if not deleted_before and "bad" not in status:
        return html.escape(hyp) + ' <span class="same">（词与参考一致）</span>'
    # fold the middle of long unchanged runs
    keep = [st != "eq" or j in deleted_before for j, st in enumerate(status)]
    if len(status) in deleted_before:
        keep.append(True)
    near = [any(keep[max(0, j - CONTEXT):j + CONTEXT + 1]) for j in range(len(status))]
    j = 0
    while j < len(near):  # only fold runs of at least CONTEXT words; shorter gaps stay visible
        if near[j]:
            j += 1
            continue
        k = j
        while k < len(near) and not near[k]:
            k += 1
        if k - j < CONTEXT:
            near[j:k] = [True] * (k - j)
        j = k
    out: list[str] = []
    pending: list[str] = []  # punctuation / spaces seen since the last word
    word_idx, folding, marking = 0, False, False
    for t in h_toks:
        if not _is_word(t):
            pending.append(html.escape(t))
            continue
        j = word_idx
        bad = status[j] == "bad"
        if marking and (not bad or j in deleted_before):  # a run of wrong words ends
            out.append("</mark>")
            marking = False
        out.extend(pending)  # inside the mark only if the run continues across it
        pending = []
        if not near[j] and not folding:
            out.append('<span class="fold">')
            folding = True
        elif near[j] and folding:
            out.append("</span>")
            folding = False
        if j in deleted_before:
            out.append(f"<del>{html.escape(joined(deleted_before[j]))}</del>")
        if bad and not marking:
            out.append("<mark>")
            marking = True
        out.append(html.escape(t))
        word_idx += 1
    if marking:
        out.append("</mark>")
    out.extend(pending)
    if folding:
        out.append("</span>")
    if len(status) in deleted_before:  # reference words missing at the very end
        out.append(f"<del>{html.escape(joined(deleted_before[len(status)]))}</del>")
    return "".join(out)


def main() -> None:
    data = {k: load_bv1(v) for k, v in BV1.items()}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    ref_rows = data["C step 600"]
    bv1 = []
    for name, col, val in SLICES:
        sel = [i for i in ids if col is None or ref_rows[i][col] == val]
        bv1.append({"slice": name, "n": len(sel), **{s: round(mer(d, sel), 2) for s, d in data.items()}})
    boot = []
    for a, b in PAIRS:
        for name, col, val in SLICES[:4]:
            sel = [i for i in ids if col is None or ref_rows[i][col] == val]
            d, lo, hi = bootstrap(data[a], data[b], sel)
            boot.append({"pair": f"{a} − {b}", "slice": name, "diff": round(d, 2), "lo": round(lo, 2),
                         "hi": round(hi, 2)})

    val = []
    for vname, (manifest, suffix) in VAL_SETS.items():
        row = {"set": vname}
        for sname, pattern in VAL_SYSTEMS.items():
            m = val_mer(manifest, pattern.format(suffix))
            row[sname] = round(100 * m["err"] / m["n"], 2) if m else None
            row[sname + "__raw"] = m
        val.append(row)
    pooled = {"set": "users' val pooled (125)"}
    for sname in VAL_SYSTEMS:
        parts = [r[sname + "__raw"] for r in val[:2]]
        pooled[sname] = round(100 * sum(p["err"] for p in parts) / sum(p["n"] for p in parts), 2) \
            if all(parts) else None
    val.insert(0, pooled)
    for r in val:
        for k in [k for k in r if k.endswith("__raw")]:
            del r[k]

    gap = []  # clips where VoiceCursor beats C by the most errors
    vc, c = data["VoiceCursor"], data["C step 600"]
    for i in sorted(ids, key=lambda i: vc[i]["err"] - c[i]["err"])[:12]:
        gap.append({"id": i[-8:], "lang": c[i]["language_mix"], "dur": round(float(c[i]["duration_sec"]), 1),
                    "c_err": c[i]["err"], "vc_err": vc[i]["err"], "n": c[i]["n"], "ref": c[i]["reference"],
                    "c": c[i]["hypothesis"], "vc": vc[i]["hypothesis"],
                    "c_html": diff_html(c[i]["reference"], c[i]["hypothesis"]),
                    "vc_html": diff_html(vc[i]["reference"], vc[i]["hypothesis"])})
    LANG = {"en_only": "en", "zh_only": "zh", "zh_en_mixed": "zh-en"}
    cells = defaultdict(lambda: {"c": 0, "vc": 0, "n": 0, "clips": 0})
    for i in ids:  # error gap C − VoiceCursor by length bucket × language (positive = VoiceCursor ahead)
        k = (c[i]["duration"], LANG[c[i]["language_mix"]])
        cells[k]["c"] += c[i]["err"]
        cells[k]["vc"] += vc[i]["err"]
        cells[k]["n"] += c[i]["n"]
        cells[k]["clips"] += 1
    order = sorted(cells.items(), key=lambda kv: -(kv[1]["c"] - kv[1]["vc"]))
    gap_cells = [{"bucket": f"{d} · {lang}", **v, "gap": v["c"] - v["vc"]} for (d, lang), v in order]
    listed = [i for i in ids if "\n" in c[i]["reference"] or "1." in c[i]["reference"]]
    list_gap = {"clips": len(listed), "gap": sum(c[i]["err"] - vc[i]["err"] for i in listed)}
    wins = {"C better": sum(c[i]["err"] < vc[i]["err"] for i in ids),
            "VoiceCursor better": sum(vc[i]["err"] < c[i]["err"] for i in ids),
            "tie": sum(vc[i]["err"] == c[i]["err"] for i in ids)}

    lat = []
    for key, name in LATENCY:
        p = RUNS / f"lat_{key}" / "latency.json"
        if p.exists():
            s = json.loads(p.read_text())
            lat.append({"variant": name, "mer": s["official_mer_pct"],
                        **{b: s["latency"][b]["p50_s"] for b in ("<5 s", "5-15 s", "15-30 s", ">30 s", "all")},
                        "p95_short": s["latency"]["<5 s"]["p95_s"]})

    payload = {"systems": list(BV1), "val_systems": list(VAL_SYSTEMS), "bv1": bv1, "boot": boot, "val": val,
               "gap": gap, "gap_cells": gap_cells, "list_gap": list_gap, "wins": wins, "n_bv1": len(ids),
               "latency": lat}
    html = Path("scripts/templates/c_comparison.html").read_text(encoding="utf-8")
    OUT.write_text(html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"-> {OUT}  ({len(ids)} benchmark v1 clips, {len(lat)} latency variants)")


if __name__ == "__main__":
    main()
