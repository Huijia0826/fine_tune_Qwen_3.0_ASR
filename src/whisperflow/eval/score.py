"""Score a transcription run: corpus + per-slice metrics, bootstrap CIs, error-case CSV.

Usage:
    wf-score reports/runs/<run_id> [--manifest M] [--config configs/eval.yaml]

Writes into the run directory:
    metrics.json   all numbers (machine-readable)
    metrics.md     human-readable report with slice table and sample-size warnings
    errors.csv     one row per utterance, sorted by error count, with aligned diff
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from whisperflow.data.audit import language_mix
from whisperflow.data.common import load_yaml
from whisperflow.data.schema import Utterance, load_manifest
from whisperflow.eval.metrics import (
    UttScore,
    align_text,
    cluster_bootstrap_ci,
    corpus_rate,
    score_alignment,
)
from whisperflow.eval.official import OFFICIAL_SCORER_VERSION, official_mer, official_mer_num, official_mer_t2s
from whisperflow.text.normalize import (
    POLICY_VERSION,
    _t2s,
    contains_term,
    normalize,
    token_lang,
    tokenize,
)


def _duration_bucket(d: float) -> str:
    if d < 5:
        return "<5s"
    if d < 15:
        return "5-15s"
    if d <= 30:
        return "15-30s"
    return ">30s"


def _has_traditional(text: str) -> bool:
    return _t2s().convert(text) != text


def slice_keys(r: Utterance, hyp: str, train_speakers: set[str]) -> dict[str, str]:
    has_num = any(token_lang(t) == "num" for t in tokenize(r.raw_transcript) + tokenize(hyp))
    return {
        "primary_scenario": r.primary_scenario,
        "language_mix": language_mix(r),
        "hotword": "present" if r.hotwords_present else "absent",
        "noise_tag": r.noise_tag,
        "speaker": "seen_in_train" if r.speaker_id in train_speakers else "unseen",
        "digits_in_ref_or_hyp": "yes" if has_num else "no",
        "duration": _duration_bucket(r.duration_sec),
        "source_id": r.source_id,
    }


def summarize(rows: list[dict], *, n_boot: int, seed: int, min_n: int) -> dict:
    scores: list[UttScore] = [x["score"] for x in rows]
    n_ref = sum(s.n_ref for s in scores)
    zh_n = sum(s.n_ref_by_lang["zh"] for s in scores)
    en_n = sum(s.n_ref_by_lang["en"] for s in scores)
    hw_total = sum(len(x["rec"].hotwords_present) for x in rows)
    hw_hit = sum(x["hotword_hits"] for x in rows)
    ci = cluster_bootstrap_ci(scores, [x["rec"].speaker_id for x in rows],
                              n_boot=n_boot, seed=seed)

    def rate(num, den):
        return round(num / den, 4) if den else None

    def pooled(key: str) -> tuple[float | None, list | None]:
        inc = [x for x in rows if x[key].included]
        n = sum(x[key].n_ref for x in inc)
        if not n:
            return None, None
        ci = cluster_bootstrap_ci([x[key] for x in inc], [x["rec"].speaker_id for x in inc],
                                  n_boot=n_boot, seed=seed)
        return round(sum(x[key].errors for x in inc) / n, 4), ([round(ci[0], 4), round(ci[1], 4)] if ci else None)

    off, off_ci = pooled("official")
    off_t2s, off_t2s_ci = pooled("official_t2s")
    off_num, off_num_ci = pooled("official_num")
    return {
        "official_mer": off,
        "official_mer_ci95_speaker_bootstrap": off_ci,
        "official_t2s_mer": off_t2s,
        "official_t2s_mer_ci95_speaker_bootstrap": off_t2s_ci,
        "official_num_mer": off_num,
        "official_num_mer_ci95_speaker_bootstrap": off_num_ci,
        "n": len(rows),
        "hours": round(sum(x["rec"].duration_sec for x in rows) / 3600, 4),
        "n_speakers": len({x["rec"].speaker_id for x in rows}),
        "n_ref_tokens": n_ref,
        "mer": round(corpus_rate(scores), 4) if n_ref else None,
        "mer_ci95_speaker_bootstrap": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "sub_rate": rate(sum(s.sub for s in scores), n_ref),
        "del_rate": rate(sum(s.dele for s in scores), n_ref),
        "ins_rate": rate(sum(s.ins for s in scores), n_ref),
        # errors attributed to the language of the token involved (ins: hyp token's language)
        "zh_char_error_rate": rate(sum(s.err_by_lang["zh"] for s in scores), zh_n),
        "en_word_error_rate": rate(sum(s.err_by_lang["en"] for s in scores), en_n),
        "en_word_accuracy": rate(sum(s.en_correct for s in scores), en_n),
        "n_zh_ref_tokens": zh_n,
        "n_en_ref_tokens": en_n,
        "hotword_recall": rate(hw_hit, hw_total),
        "n_hotwords": hw_total,
        "n_empty_hyp": sum(1 for x in rows if not x["hyp_norm"]),
        "n_possible_hallucination": sum(1 for x in rows if x["hallucination"]),
        "n_hyp_traditional_script": sum(1 for x in rows if x["traditional"]),
        "insufficient_n": len(rows) < min_n,
    }


def score_run(run_dir: Path, manifest: str, scfg: dict) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if scfg.get("normalization_policy", POLICY_VERSION) != POLICY_VERSION:
        raise ValueError(f"config requests policy {scfg['normalization_policy']} but code "
                         f"implements {POLICY_VERSION}")
    recs = load_manifest(manifest)
    by_id = {r.utterance_id: r for r in recs}
    train_speakers = {r.speaker_id for r in recs if r.split == "train"}
    preds = [json.loads(line) for line in
             (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]

    rows, n_out_of_split = [], 0
    for p in preds:
        r = by_id.get(p["utterance_id"])
        if r is None:
            raise KeyError(f"prediction {p['utterance_id']} not in manifest {manifest}")
        if r.split != meta["split"]:  # e.g. excluded from the benchmark after this run decoded it
            n_out_of_split += 1
            continue
        hyp = p["hypothesis"]
        a = align_text(r.raw_transcript, hyp)
        s = score_alignment(a)
        rows.append({
            "rec": r, "hyp": hyp, "hyp_norm": " ".join(a.hyp), "align": a, "score": s,
            "official": official_mer(r.raw_transcript, hyp),
            "official_t2s": official_mer_t2s(r.raw_transcript, hyp),
            "official_num": official_mer_num(r.raw_transcript, hyp),
            "hotword_hits": sum(contains_term(hyp, h) for h in r.hotwords_present),
            "hallucination": s.ins > max(5, s.n_ref) or len(a.hyp) > 2 * s.n_ref + 10,
            # model's own output, before any output_script conversion
            "traditional": _has_traditional(p.get("hypothesis_raw", hyp)),
            "slices": slice_keys(r, hyp, train_speakers),
        })
    missing = {r.utterance_id for r in recs if r.split == meta["split"]} - {
        x["rec"].utterance_id for x in rows}

    kw = dict(n_boot=int(scfg.get("bootstrap_iters", 1000)),
              seed=int(meta["config"]["runtime"].get("seed", 0)),
              min_n=int(scfg.get("min_slice_n", 30)))
    result = {
        "run_id": meta["run_id"],
        "split": meta["split"],
        "normalization_policy": POLICY_VERSION,
        "official_scorer": OFFICIAL_SCORER_VERSION,
        "model": meta["config"]["model"],
        "transcriber": meta["transcriber"],
        "decoding": meta["config"]["decoding"],
        "n_manifest_utterances_not_decoded": len(missing),
        "n_predictions_out_of_split_skipped": n_out_of_split,
        "overall": summarize(rows, **kw),
        "slices": {},
    }
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for x in rows:
        for dim, val in x["slices"].items():
            groups[dim][val].append(x)
    for dim in sorted(groups):
        result["slices"][dim] = {v: summarize(g, **kw) for v, g in sorted(groups[dim].items())}

    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    _write_errors_csv(rows, run_dir / "errors.csv")
    (run_dir / "metrics.md").write_text(render_md(result), encoding="utf-8")
    return result


def _write_errors_csv(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda x: (-x["score"].errors, x["rec"].utterance_id))
    slice_dims = list(rows[0]["slices"]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "speaker_id", "duration_sec", *slice_dims, "n_ref", "sub",
                    "del", "ins", "mer", "hotwords", "hotword_hits", "possible_hallucination",
                    "reference", "hypothesis", "ref_norm", "hyp_norm", "alignment"])
        for x in rows:
            r, s = x["rec"], x["score"]
            w.writerow([r.utterance_id, r.speaker_id, r.duration_sec,
                        *[x["slices"][d] for d in slice_dims], s.n_ref, s.sub, s.dele, s.ins,
                        round(s.mer, 4) if not math.isnan(s.mer) else "",
                        "|".join(r.hotwords_present), x["hotword_hits"], x["hallucination"],
                        r.raw_transcript, x["hyp"], normalize(r.raw_transcript), x["hyp_norm"],
                        x["align"].render()])


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.2f}%"


def _ci(c) -> str:
    return f"{_pct(c[0])}–{_pct(c[1])}" if c else "n/a"


def render_md(res: dict) -> str:
    o = res["overall"]
    ci = o["mer_ci95_speaker_bootstrap"]
    lines = [
        f"# Eval run `{res['run_id']}` — split `{res['split']}`", "",
        f"- model: `{res['model']['name_or_path']}` revision `{res['model'].get('revision')}` "
        f"(resolved commit `{res['transcriber'].get('model_commit')}`), "
        f"adapter `{res['model'].get('adapter_path')}`",
        f"- decoding: `{json.dumps(res['decoding'], ensure_ascii=False)}`",
        f"- backend: `{json.dumps(res['transcriber'])}`",
        f"- normalization: `{res['normalization_policy']}`",
        f"- manifest utterances in split not decoded: {res['n_manifest_utterances_not_decoded']}",
        f"- predictions skipped (utterance no longer in split): "
        f"{res['n_predictions_out_of_split_skipped']}",
        "",
        f"**Official MER ({res['official_scorer']}): {_pct(o['official_mer'])}** "
        f"(95% CI {_ci(o['official_mer_ci95_speaker_bootstrap'])})",
        "",
        f"**Official MER + full t2s (script-insensitive): {_pct(o['official_t2s_mer'])}** "
        f"(95% CI {_ci(o['official_t2s_mer_ci95_speaker_bootstrap'])})",
        f"**Official MER + t2s + numeral style (三点五十 = 3:50): {_pct(o['official_num_mer'])}** "
        f"(95% CI {_ci(o['official_num_mer_ci95_speaker_bootstrap'])})",
        "",
        f"Diagnostic MER (zh_en_v1): {_pct(o['mer'])} "
        f"(95% speaker-bootstrap CI {(_pct(ci[0]) + ' – ' + _pct(ci[1])) if ci else 'n/a'}), "
        f"N={o['n']}, {o['hours']} h, {o['n_speakers']} speakers",
        "",
        "| slice | value | N | spk | official MER | 95% CI | official+t2s | diag. MER | zh CER | en WER "
        "| hotword recall | halluc. | trad. script |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(dim, val, s):
        warn = " ⚠" if s["insufficient_n"] else ""
        return (f"| {dim} | {val} | {s['n']}{warn} | {s['n_speakers']} | {_pct(s['official_mer'])} | "
                f"{_ci(s['official_mer_ci95_speaker_bootstrap'])} | {_pct(s['official_t2s_mer'])} | "
                f"{_pct(s['mer'])} | "
                f"{_pct(s['zh_char_error_rate'])} | {_pct(s['en_word_error_rate'])} | "
                f"{_pct(s['hotword_recall'])} | {s['n_possible_hallucination']} | "
                f"{s['n_hyp_traditional_script']} |")

    lines.append(row("overall", "all", o))
    for dim, vals in res["slices"].items():
        for val, s in vals.items():
            lines.append(row(dim, val, s))
    lines += ["", "⚠ = N below the configured minimum; treat the number as anecdotal.",
              "CI is n/a with fewer than 2 speakers. Traditional-script hypotheses are "
              "converted to Simplified before scoring (policy zh_en_v1).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--manifest", help="defaults to the manifest recorded in run_meta.json")
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args(argv)
    meta = json.loads((args.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    scfg = load_yaml(args.config).get("scoring", {})
    res = score_run(args.run_dir, args.manifest or meta["manifest"], scfg)
    o = res["overall"]
    print(f"official MER {_pct(o['official_mer'])}  CI={o['official_mer_ci95_speaker_bootstrap']}  "
          f"official+t2s {_pct(o['official_t2s_mer'])}  official+num {_pct(o['official_num_mer'])}  "
          f"(diagnostic MER {_pct(o['mer'])})  N={o['n']}")
    print(f"Wrote {args.run_dir}/metrics.json, metrics.md, errors.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
