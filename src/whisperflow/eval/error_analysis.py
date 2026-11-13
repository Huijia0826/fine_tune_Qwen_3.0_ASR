"""Error-cause analysis: decompose official MER into script / formatting / recognition errors.

Usage:
    wf-analyze reports/runs/<run_id> [--compare name=hyps.jsonl ...]

`--compare` takes JSONL files of {"utterance_id", "hypothesis"} for other systems scored on the
same manifest records (e.g. VoiceCursor outputs from the arena export), so the gap can be
attributed layer by layer.

Layers (each is the drop in pooled MER when one more normalization is applied):
  1. script      official MER  -> official MER with full OpenCC t2s on both sides
  2. formatting  -> additionally: English number words -> digits, contractions expanded
  3. recognition what remains, broken down by token language, op type, bursts, language-ID
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from whisperflow.data.schema import Utterance, load_manifest
from whisperflow.eval.metrics import align
from whisperflow.eval.vendor import benchmark_metrics as _bm
from whisperflow.text.normalize import _t2s, token_lang

LAYERS = ("official", "script_t2s", "formatting")
BURST = 4  # >= this many consecutive deletions/insertions counts as a dropped/hallucinated span

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "first": "1st", "second": "2nd", "third": "3rd",
}
_CONTRACTIONS = {
    "i'm": "i am", "it's": "it is", "that's": "that is", "don't": "do not", "can't": "can not",
    "won't": "will not", "i'll": "i will", "we're": "we are", "you're": "you are",
    "they're": "they are", "isn't": "is not", "doesn't": "does not", "didn't": "did not",
    "i've": "i have", "we've": "we have", "let's": "let us", "there's": "there is",
    "what's": "what is", "i'd": "i would", "wasn't": "was not", "aren't": "are not",
    "cannot": "can not",
}
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b", re.I)


def tokens_official(text: str) -> list[str]:
    return _bm.tokenize(text, "mer")


def tokens_t2s(text: str) -> list[str]:
    return _bm.tokenize(_t2s().convert(text), "mer")


def tokens_formatted(text: str) -> list[str]:
    text = _t2s().convert(text).replace("’", "'")
    text = _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)
    return [_NUM_WORDS.get(t, t) for t in _bm.tokenize(text, "mer")]


def _lang_share(tokens: list[str]) -> float:
    langs = [token_lang(t) for t in tokens if token_lang(t) != "num"]
    return sum(lang == "zh" for lang in langs) / len(langs) if langs else 0.0


def analyze_system(recs: list[Utterance], hyps: dict[str, str]) -> dict:
    layers = {"official": [0, 0], "script_t2s": [0, 0], "formatting": [0, 0]}
    rec_err: Counter = Counter()
    n_ref_lang: Counter = Counter()
    subs_zh: Counter = Counter()
    subs_en: Counter = Counter()
    burst_del = burst_ins = lid_fail = 0
    burst_del_tokens = burst_ins_tokens = 0
    per_utt = []
    for r in recs:
        hyp = hyps.get(r.utterance_id, "")
        for name, fn in (("official", tokens_official), ("script_t2s", tokens_t2s),
                         ("formatting", tokens_formatted)):
            ref_t, hyp_t = fn(r.raw_transcript), fn(hyp)
            if not ref_t:
                continue
            a = align(ref_t, hyp_t)
            layers[name][0] += a.errors
            layers[name][1] += len(ref_t)
        # recognition-level breakdown on the most-normalized tokens
        ref_t, hyp_t = tokens_formatted(r.raw_transcript), tokens_formatted(hyp)
        if not ref_t:
            continue
        a = align(ref_t, hyp_t)
        for t in ref_t:
            n_ref_lang[token_lang(t)] += 1
        run_op, run_len = None, 0
        for op, rt, ht in [*a.ops, ("eq", None, None)]:
            if op in ("del", "ins") and op == run_op:
                run_len += 1
            else:
                if run_op in ("del", "ins") and run_len >= BURST:
                    if run_op == "del":
                        burst_del += 1
                        burst_del_tokens += run_len
                    else:
                        burst_ins += 1
                        burst_ins_tokens += run_len
                run_op, run_len = op, 1
            if op == "eq" or rt is None and ht is None:
                continue
            lang = token_lang(rt if rt is not None else ht)
            rec_err[(lang, op)] += 1
            if op == "sub":
                (subs_zh if token_lang(rt) == "zh" else subs_en)[(rt, ht)] += 1
        ref_zh, hyp_zh = _lang_share(ref_t), _lang_share(hyp_t)
        lid = (ref_zh >= 0.5 and hyp_zh < 0.1) or (ref_zh <= 0.1 and hyp_zh >= 0.5 and len(hyp_t) > 3)
        lid_fail += lid
        per_utt.append({"utterance_id": r.utterance_id, "errors": a.errors, "n_ref": len(ref_t),
                        "lid_fail": lid})
    rate = {k: (v[0] / v[1] if v[1] else None) for k, v in layers.items()}
    n_total = layers["formatting"][1]
    return {
        "mer_layers": rate,
        "layer_contrib": {
            "script": rate["official"] - rate["script_t2s"],
            "formatting": rate["script_t2s"] - rate["formatting"],
            "recognition": rate["formatting"],
        },
        "recognition_errors": {f"{lang}_{op}": n for (lang, op), n in sorted(rec_err.items())},
        "recognition_rate_of_total_ref": {f"{lang}_{op}": n / n_total for (lang, op), n in rec_err.items()},
        "n_ref_by_lang": dict(n_ref_lang),
        "burst_deletions": {"count": burst_del, "tokens": burst_del_tokens},
        "burst_insertions": {"count": burst_ins, "tokens": burst_ins_tokens},
        "language_id_failures": lid_fail,
        "top_zh_substitutions": [[f"{a}→{b}", n] for (a, b), n in subs_zh.most_common(15)],
        "top_en_substitutions": [[f"{a}→{b}", n] for (a, b), n in subs_en.most_common(15)],
        "n_utterances": len(per_utt),
        "per_utterance": per_utt,
    }


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.2f}%"


def render(results: dict[str, dict], slices: dict[str, list[str]]) -> str:
    names = list(results)
    out = ["# Error-cause analysis", "",
           "MER after each additional normalization layer (pooled over the same utterances).", "",
           "| layer | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for layer, label in (("official", "official MER"), ("script_t2s", "+ full Traditional→Simplified"),
                         ("formatting", "+ number words / contractions")):
        vals = [_pct(results[n]["all"]["mer_layers"][layer]) for n in names]
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    out += ["", "## Where the errors come from (percentage points of official MER)", "",
            "| cause | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k in ("script", "formatting", "recognition"):
        out.append(f"| {k} | " + " | ".join(_pct(results[n]["all"]["layer_contrib"][k]) for n in names) + " |")
    keys = sorted({k for n in names for k in results[n]["all"]["recognition_rate_of_total_ref"]})
    out += ["", "## Recognition errors by token language × operation (share of all ref tokens)", "",
            "| type | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k in keys:
        out.append(f"| {k} | " + " | ".join(_pct(results[n]["all"]["recognition_rate_of_total_ref"].get(k, 0))
                                          for n in names) + " |")
    out += ["", "## Structural failures", "", "| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for label, fn in (("dropped spans (≥4 consecutive deletions)",
                       lambda r: f"{r['burst_deletions']['count']} ({r['burst_deletions']['tokens']} tok)"),
                      ("inserted spans (≥4 consecutive insertions)",
                       lambda r: f"{r['burst_insertions']['count']} ({r['burst_insertions']['tokens']} tok)"),
                      ("language-ID failures (output in wrong language)", lambda r: r["language_id_failures"])):
        out.append(f"| {label} | " + " | ".join(str(fn(results[n]["all"])) for n in names) + " |")
    out += ["", "## By slice: official → +t2s → +formatting", ""]
    for sl in slices:
        out.append(f"- **{sl}**: " + "; ".join(
            f"{n}: " + " → ".join(_pct(results[n][sl]["mer_layers"][k]) for k in LAYERS)
            for n in names))
    for n in names:
        out += ["", f"## Most frequent substitutions — {n}", "",
                "zh: " + ", ".join(f"{p}×{c}" for p, c in results[n]["all"]["top_zh_substitutions"]),
                "", "en: " + ", ".join(f"{p}×{c}" for p, c in results[n]["all"]["top_en_substitutions"])]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--compare", action="append", default=[], help="name=path.jsonl")
    args = ap.parse_args(argv)
    meta = json.loads((args.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    preds = {json.loads(x)["utterance_id"]: json.loads(x)["hypothesis"]
             for x in (args.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if x}
    recs = [r for r in load_manifest(meta["manifest"]) if r.utterance_id in preds]
    systems = {"this_run": preds}
    for spec in args.compare:
        name, path = spec.split("=", 1)
        systems[name] = {json.loads(x)["utterance_id"]: json.loads(x)["hypothesis"]
                         for x in Path(path).read_text(encoding="utf-8").splitlines() if x}
    by_slice: dict[str, list[Utterance]] = defaultdict(list)
    for r in recs:
        by_slice["all"].append(r)
        by_slice[r.primary_scenario].append(r)
        by_slice["long>30s" if r.duration_sec > 30 else "short<=30s"].append(r)
    results = {n: {sl: analyze_system(rs, h) for sl, rs in by_slice.items()} for n, h in systems.items()}
    slices = [s for s in by_slice if s != "all"]
    (args.run_dir / "error_analysis.json").write_text(json.dumps(
        {n: {sl: {k: v for k, v in d.items() if k != "per_utterance"} for sl, d in res.items()}
         for n, res in results.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(results, slices)
    (args.run_dir / "error_analysis.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
