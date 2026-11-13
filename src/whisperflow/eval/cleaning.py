"""Cleaning diagnostic: how much of what the cleaned references remove does a system's output still contain?

We have no verbatim transcripts, only cleaned references. Aligning a system's output to the cleaned reference
shows where it did NOT clean: words only in the output (fillers, repeats) or English words in a different form
(grammar the reference fixed). Each maximal run of non-matching alignment ops is one Span; `classify_span` puts
it in a category (exercise 10, docs/learning/exercises.md); `summarize` counts categories per 100 reference
tokens. Deletions are reported too, so a system that drops words does not look "clean".

Usage: wf-cleaning reports/runs/<run_id> [reports/runs/<run_id> ...] [--out reports/cleaning/<name>.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from whisperflow.data.schema import load_manifest
from whisperflow.eval.metrics import align_text
from whisperflow.text.normalize import detect_languages, is_cjk, tokenize

CATEGORIES = ("filler", "repeat", "grammar", "other")

# Spoken fillers / discourse markers that the cleaned references delete. Matched against the whole span text,
# see exercise 10 step 1. Chinese entries are joined characters (tokens are single characters).
FILLERS_ZH = {"嗯", "呃", "啊", "哦", "那个", "这个", "就是", "就是说", "然后", "对", "对吧", "就", "这", "那", "反正", "其实"}
FILLERS_EN = {"um", "uh", "like", "so", "yeah", "okay", "ok", "well", "basically", "actually", "just",
              "you know", "i mean", "kind of", "sort of"}

# English word pairs that are the "same word, different grammar" beyond plain suffixes (exercise 10 step 3).
IRREGULAR = {frozenset(p) for p in [("has", "have"), ("is", "are"), ("was", "were"), ("does", "do"),
                                     ("did", "do"), ("am", "is"), ("a", "an")]}


@dataclass
class Span:
    """One maximal run of non-matching alignment ops.

    kind:   "ins" (only in the output), "del" (only in the reference) or "sub" (anything containing a substitution)
    ref:    reference tokens in the span (empty for "ins")
    hyp:    output tokens in the span (empty for "del")
    before: output tokens before the span (whole utterance prefix); after: output tokens after it
    """
    kind: str
    ref: list[str]
    hyp: list[str]
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)


def join_tokens(tokens: list[str]) -> str:
    """Tokens back to text: no space between Chinese characters, one space between English words."""
    out = ""
    for t in tokens:
        if out and not (is_cjk(out[-1]) or is_cjk(t[0])):
            out += " "
        out += t
    return out


def extract_spans(ref_text: str, hyp_text: str) -> list[Span]:
    """Align output to reference and group consecutive non-"eq" ops into Spans."""
    ops = align_text(ref_text, hyp_text).ops
    spans, hyp_seen, run = [], [], []

    def close() -> None:
        if not run:
            return
        kinds = {op for op, _, _ in run}
        kind = "sub" if "sub" in kinds or kinds == {"ins", "del"} else kinds.pop()
        hyp = [h for _, _, h in run if h is not None]
        start = len(hyp_seen) - len(hyp)
        spans.append(Span(kind, [r for _, r, _ in run if r is not None], hyp, hyp_seen[:start], []))
        run.clear()

    for op, r, h in ops:
        if op == "eq":
            close()
        else:
            run.append((op, r, h))
        if h is not None:
            hyp_seen.append(h)
    close()
    for s in spans:  # fill `after` now that the whole output is known
        s.after = hyp_seen[len(s.before) + len(s.hyp):]
    return spans

FILLER_WORDS = sorted(FILLERS_ZH | FILLERS_EN, key=len, reverse=True)  # 所有口头禅，长的排前面


def is_all_fillers(text: str) -> bool:
    while text != "":                      # 还没剥完，就继续
        found = False                      # 这一轮有没有剥掉一个词？先假设没有
        for w in FILLER_WORDS:             # 一个词一个词地试
            if text.startswith(w):         # 开头是这个词吗？
                text = text[len(w):].strip()   # 是：剥掉它（strip 去掉英文词之间的空格）
                found = True
                break                      # 剥掉一个就够了，回到 while 再来一轮
        if not found:                      # 所有词都试过，一个都没剥掉
            return False                   # 卡住了 → 不是全由口头禅组成
    return True                            # while 结束 = text 剥空了 → 全是口头禅


SUFFIXES = ("s", "es", "ed", "ing")


def is_same_word(a: str, b: str) -> bool:
    """a 和 b 是同一个英文词的不同形式吗？（has/have、intentions/intention、used/use）"""
    if not (a.isascii() and a.isalpha() and b.isascii() and b.isalpha()):
        return False                                  # 不是英文词（比如中文"然"），直接不算
    if frozenset((a, b)) in IRREGULAR:
        return True                                   # 不规则变化，查表
    for long, short in ((a, b), (b, a)):              # 两个方向都试
        for suf in SUFFIXES:
            if long == short + suf:
                return True                           # 长的 = 短的 + 后缀
        if short.endswith("e") and long == short + "d":
            return True                               # 以 e 结尾的词只加 d：use → used
    return False                                      # 都不是 → 不同的词


def classify_span(span: Span) -> str:
    """Exercise 10: return one of CATEGORIES ("filler", "repeat", "grammar", "other").

    Rules (apply in this order; spec and tests in docs/learning/exercises.md, tests/test_cleaning.py):
      step 1  "ins" span whose text is made only of FILLERS_ZH / FILLERS_EN entries      -> "filler"
      step 2  "ins" span equal to the tokens right before or right after it             -> "repeat"
      step 3  "sub" of one English word by another form of the same word                -> "grammar"
      step 4  anything else (including every "del")                                      -> "other"
    """
    if span.kind == "ins":
        if is_all_fillers(join_tokens(span.hyp)):
            return "filler"
        if span.hyp == span.before[-len(span.hyp):] or span.hyp == span.after[:len(span.hyp)]:
            return "repeat"
    if span.kind == "sub" and len(span.ref) == 1 and len(span.hyp) == 1 and is_same_word(span.ref[0], span.hyp[0]):
        return "grammar"
    return "other"


def summarize(pairs: list[tuple[str, str]]) -> dict:
    """Count span categories over (reference, output) pairs. Rates are per 100 reference tokens.

    filler / repeat / grammar: places the output did not clean; other_ins / other_sub: unexplained extra or changed
    words (mostly recognition errors); deleted: reference tokens missing from the output (guards against a system
    that looks clean because it drops words).
    """
    counts = Counter()
    examples: dict[str, Counter] = {c: Counter() for c in ("filler", "repeat", "grammar")}
    n_ref = 0
    for ref, hyp in pairs:
        n_ref += len(tokenize(ref))
        for s in extract_spans(ref, hyp):
            cat = classify_span(s)
            if cat == "other":
                if s.kind == "del":
                    counts["deleted"] += len(s.ref)
                else:
                    counts[f"other_{s.kind}"] += 1
                continue
            counts[cat] += 1
            examples[cat][join_tokens(s.hyp) if cat != "grammar" else f"{s.ref[0]}→{s.hyp[0]}"] += 1
    keys = ("filler", "repeat", "grammar", "other_ins", "other_sub", "deleted")
    return {"n_utt": len(pairs), "n_ref_tokens": n_ref,
            "per_100": {k: round(100 * counts[k] / max(n_ref, 1), 2) for k in keys},
            "counts": {k: counts[k] for k in keys},
            "top": {c: ex.most_common(8) for c, ex in examples.items()}}


def _load_run(run_dir: Path) -> tuple[dict, dict[str, str]]:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    hyps = {}
    for line in open(run_dir / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        hyps[p["utterance_id"]] = p.get("hypothesis") or ""
    return meta, hyps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="reports/runs/<run_id> directories, same manifest + split")
    ap.add_argument("--manifest", help="default: the first run's run_meta manifest")
    ap.add_argument("--split", help="default: the first run's run_meta split")
    ap.add_argument("--purpose", default="", help="logged to reports/test_access_log.jsonl when split=test")
    ap.add_argument("--out", type=Path, help="write the summary JSON here")
    args = ap.parse_args(argv)

    loaded = [(d.name, *_load_run(d)) for d in args.runs]
    manifest = args.manifest or loaded[0][1].get("manifest")
    split = args.split or loaded[0][1].get("split")
    records = [r for r in load_manifest(manifest) if r.split == split]
    if split == "test":
        log = Path("reports/test_access_log.jsonl")
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                "run_id": "wf-cleaning:" + ",".join(n for n, _, _ in loaded),
                                "purpose": args.purpose or "cleaning diagnostic"}, ensure_ascii=False) + "\n")
    # Same utterances for every system: present and non-empty in all runs (failed VoiceCursor rows are empty).
    ids = [r.utterance_id for r in records if all(h.get(r.utterance_id) for _, _, h in loaded)]
    refs = {r.utterance_id: r.raw_transcript for r in records}
    langs = {r.utterance_id: "en" if detect_languages(r.raw_transcript) == {"en"} else "zh/mixed" for r in records}
    print(f"{manifest} split={split}: {len(ids)} of {len(records)} utterances decoded by all {len(loaded)} systems\n")

    result = {"manifest": manifest, "split": split, "n_common": len(ids), "systems": {}}
    for name, _, hyps in loaded:
        result["systems"][name] = {
            sl: summarize([(refs[i], hyps[i]) for i in ids if sl == "all" or langs[i] == sl])
            for sl in ("all", "zh/mixed", "en")}

    cols = ("filler", "repeat", "grammar", "other_ins", "other_sub", "deleted")
    for sl in ("all", "zh/mixed", "en"):
        n = result["systems"][loaded[0][0]][sl]
        print(f"[{sl}] {n['n_utt']} utterances, {n['n_ref_tokens']} reference tokens; per 100 reference tokens")
        print(f"  {'system':<34}" + "".join(f"{c:>10}" for c in cols))
        for name, _, _ in loaded:
            row = result["systems"][name][sl]["per_100"]
            print(f"  {name:<34}" + "".join(f"{row[c]:>10.2f}" for c in cols))
        print()
    for name, _, _ in loaded:
        top = result["systems"][name]["all"]["top"]
        print(f"{name} most common: " + "; ".join(f"{c}: " + ", ".join(f"{t}×{k}" for t, k in top[c][:5])
                                                     for c in top))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
