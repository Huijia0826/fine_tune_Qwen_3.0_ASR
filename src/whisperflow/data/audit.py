"""Split audit: leakage checks + dataset statistics report.

The audit re-checks leakage independently of split.py (defence in depth: a bug in the
splitter, a hand-edited manifest, or a merge of two manifests is caught here).

Usage:
    wf-audit data/manifests/pilot.jsonl --out-dir reports/dataset/pilot
Exit code 1 if any leakage error is found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from whisperflow.data.common import IssueLog, load_yaml
from whisperflow.data.schema import Utterance, load_manifest
from whisperflow.data.split import SPLITS, check_lock_integrity, load_lock
from whisperflow.text.normalize import detect_languages, normalize


def language_mix(r: Utterance) -> str:
    langs = detect_languages(r.raw_transcript)
    if langs == {"zh"}:
        return "zh_only"
    if {"zh", "en"} <= langs:
        return "zh_en_mixed"
    if langs == {"en"}:
        return "en_only"
    return "other"


def check_leakage(records: list[Utterance], log: IssueLog) -> None:
    assigned = [r for r in records if r.split in SPLITS]
    checks = {
        "speaker_id": ("leak_speaker", "error"),
        "session_id": ("leak_session", "error"),
        "sha256": ("leak_audio", "error"),
        "lineage": ("leak_augmentation", "error"),
        "text": ("shared_transcript", "warning"),
    }
    for key, (code, level) in checks.items():
        where: dict[str, set[str]] = defaultdict(set)
        for r in assigned:
            if key == "lineage":
                val = r.parent_utterance_id or r.utterance_id
            elif key == "text":
                val = normalize(r.raw_transcript)
            else:
                val = getattr(r, key)
            where[val].add(r.split)
        leaks = {v: s for v, s in where.items() if len(s) > 1}
        for val, splits in sorted(leaks.items())[:20]:
            msg = f"{key} {val[:40]!r} appears in {sorted(splits)}"
            (log.error if level == "error" else log.warn)(code, msg)
        if len(leaks) > 20:
            (log.error if level == "error" else log.warn)(code, f"... {len(leaks) - 20} more")


def _dur_stats(durs: list[float]) -> dict:
    if not durs:
        return {}
    a = np.asarray(durs)
    return {"min": round(float(a.min()), 2), "p50": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


def dataset_stats(records: list[Utterance]) -> dict:
    stats: dict = {}
    for split in (*SPLITS, "unassigned"):
        rs = [r for r in records if r.split == split]
        if not rs:
            continue

        def breakdown(fn, rs=rs):
            n, h = Counter(), Counter()
            for r in rs:
                k = fn(r)
                n[k] += 1
                h[k] += r.duration_sec / 3600
            return {k: {"n": n[k], "hours": round(h[k], 4)} for k in sorted(n)}

        stats[split] = {
            "n": len(rs),
            "hours": round(sum(r.duration_sec for r in rs) / 3600, 4),
            "n_speakers": len({r.speaker_id for r in rs}),
            "n_sessions": len({r.session_id for r in rs}),
            "duration_sec": _dur_stats([r.duration_sec for r in rs]),
            "primary_scenario": breakdown(lambda r: r.primary_scenario),
            "language_mix": breakdown(language_mix),
            "hotword_present": breakdown(lambda r: "yes" if r.hotwords_present else "no"),
            "noise_tag": breakdown(lambda r: r.noise_tag),
            "label_status": breakdown(lambda r: r.label_status),
            "source_id": breakdown(lambda r: r.source_id),
            "recording_device": breakdown(lambda r: r.recording_device),
        }
    return stats


def render_markdown(stats: dict, log: IssueLog, min_slice_n: int, manifest: str) -> str:
    lines = [f"# Dataset audit: `{manifest}`", ""]
    s = log.summary()
    lines += [f"**Leakage/integrity:** {s['n_errors']} error(s), {s['n_warnings']} warning(s)", ""]
    for i in log.issues:
        lines.append(f"- [{i.level}] `{i.code}` {i.message}")
    lines += ["", "## Split overview", "",
              "| split | N | hours | speakers | sessions | dur p50 | dur p95 | dur max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, st in stats.items():
        d = st["duration_sec"]
        lines.append(f"| {split} | {st['n']} | {st['hours']:.3f} | {st['n_speakers']} | "
                     f"{st['n_sessions']} | {d['p50']} | {d['p95']} | {d['max']} |")
    insufficient = []
    for dim in ("primary_scenario", "language_mix", "hotword_present", "noise_tag",
                "label_status", "source_id"):
        lines += ["", f"## {dim}", "", "| split | value | N | hours |", "|---|---|---:|---:|"]
        for split, st in stats.items():
            for k, v in st[dim].items():
                flag = ""
                if split in ("val", "test") and v["n"] < min_slice_n:
                    flag = " ⚠"
                    insufficient.append(f"{split}/{dim}={k} (N={v['n']})")
                lines.append(f"| {split} | {k} | {v['n']}{flag} | {v['hours']:.3f} |")
    lines += ["", f"## Slices below minimum N={min_slice_n} (val/test)", ""]
    lines += [f"- {x}" for x in insufficient] or ["- none"]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    records = load_manifest(args.manifest)
    log = IssueLog()
    check_leakage(records, log)
    lock = load_lock(Path(cfg["split"]["locked_test_file"]))
    if lock:
        for p in check_lock_integrity(records, lock):
            log.error("lock_violation", p)
    stats = dataset_stats(records)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(
        {"manifest": args.manifest, "issues": log.summary(), "stats": stats},
        ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(stats, log, cfg.get("audit", {}).get("min_slice_n", 30), args.manifest)
    (out / "audit.md").write_text(md, encoding="utf-8")
    log.print()
    print(f"Report: {out / 'audit.md'}")
    return 1 if log.errors else 0


if __name__ == "__main__":
    sys.exit(main())
