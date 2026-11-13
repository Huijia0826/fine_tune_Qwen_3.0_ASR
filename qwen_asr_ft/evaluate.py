"""Score predictions against a manifest, pick checkpoints, compare systems."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .data import read_manifests
from .metrics import Score, paired_bootstrap, pooled_mer, score, speaker_bootstrap_ci
from .text import language_mix


def load_predictions(path: str | Path) -> dict[str, str]:
    preds = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["id"]] = r.get("hypothesis", "")
    return preds


def _slice_value(row: dict, key: str) -> str:
    if key == "language_mix":
        return row.get("language_mix") or language_mix(row["text"])
    if key == "duration_bucket":
        d = row.get("duration")
        return "unknown" if d is None else ("<=30s" if float(d) <= 30 else ">30s")
    return str(row.get(key, "unknown"))


def evaluate(manifests, preds_path: str | Path, cfg: dict, out_dir: str | Path | None = None) -> dict:
    """Pooled MER overall and per slice (config eval.slice_by), speaker-bootstrap CI, per-clip CSV."""
    rows = read_manifests(manifests, cfg["data"].get("audio_root"))
    preds = load_predictions(preds_path)
    missing = [r["id"] for r in rows if r["id"] not in preds]
    scored = [(r, score(r["text"], preds[r["id"]])) for r in rows if r["id"] in preds]
    scored = [(r, s) for r, s in scored if s.n_ref > 0]
    nb = int(cfg["eval"].get("bootstrap", 1000))

    def summarize(items):
        scs = [s for _, s in items]
        ci = speaker_bootstrap_ci(scs, [r["speaker"] for r, _ in items], n_boot=nb) if nb else None
        return {"n": len(items), "speakers": len({r["speaker"] for r, _ in items}), "mer": round(pooled_mer(scs), 2),
                "ci95": [round(ci[0], 2), round(ci[1], 2)] if ci else None,
                "sub": sum(s.sub for s in scs), "del": sum(s.dele for s in scs), "ins": sum(s.ins for s in scs)}

    report = {"predictions": str(preds_path), "missing_predictions": len(missing), "overall": summarize(scored),
              "slices": {}}
    for key in cfg["eval"].get("slice_by", []):
        groups = defaultdict(list)
        for r, s in scored:
            groups[_slice_value(r, key)].append((r, s))
        report["slices"][key] = {v: summarize(g) for v, g in sorted(groups.items())}

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out / "per_clip.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "speaker", "n_ref", "sub", "del", "ins", "mer", "reference", "hypothesis"])
            for r, s in sorted(scored, key=lambda x: -x[1].errors):
                w.writerow([r["id"], r["speaker"], s.n_ref, s.sub, s.dele, s.ins, round(s.mer, 4), r["text"],
                            preds[r["id"]]])
        (out / "metrics.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    def line(name, m):
        ci = f"[{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]" if m["ci95"] else "–"
        return f"| {name} | {m['n']} | {m['speakers']} | {m['mer']:.2f}% | {ci} | {m['sub']} / {m['del']} / {m['ins']} |"

    out = [f"# Evaluation: `{report['predictions']}`", "",
           "| slice | clips | speakers | MER | 95% CI (speaker bootstrap) | sub / del / ins |",
           "| --- | ---: | ---: | ---: | --- | --- |", line("overall", report["overall"])]
    for key, groups in report["slices"].items():
        out += [line(f"{key} = {v}", m) for v, m in groups.items()]
    if report["missing_predictions"]:
        out += ["", f"⚠ {report['missing_predictions']} clips have no prediction and are not scored."]
    return "\n".join(out) + "\n"


def compare(manifests, preds_a: str | Path, preds_b: str | Path, cfg: dict, n_boot: int = 2000) -> dict:
    """B − A pooled MER with a paired speaker-bootstrap 95% interval, on clips both systems decoded."""
    rows = read_manifests(manifests, cfg["data"].get("audio_root"))
    A, B = load_predictions(preds_a), load_predictions(preds_b)
    rows = [r for r in rows if r["id"] in A and r["id"] in B]
    sa = [score(r["text"], A[r["id"]]) for r in rows]
    sb = [score(r["text"], B[r["id"]]) for r in rows]
    keep = [i for i, s in enumerate(sa) if s.n_ref > 0]
    sa, sb, spk = [sa[i] for i in keep], [sb[i] for i in keep], [rows[i]["speaker"] for i in keep]
    diff, lo, hi = paired_bootstrap(sa, sb, spk, n_boot=n_boot)
    wins = sum(b.errors < a.errors for a, b in zip(sa, sb))
    losses = sum(b.errors > a.errors for a, b in zip(sa, sb))
    return {"n": len(sa), "mer_a": round(pooled_mer(sa), 2), "mer_b": round(pooled_mer(sb), 2),
            "b_minus_a": round(diff, 2), "ci95": [round(lo, 2), round(hi, 2)],
            "speakers": len(set(spk)),
            "significant": (not (lo <= 0 <= hi)) if len(set(spk)) >= 2 else None,  # None: too few speakers
            "clips_b_better": wins, "clips_a_better": losses,
            "ties": len(sa) - wins - losses}


def pick_best(scores_by_ckpt: dict[str, float]) -> str:
    """Lowest pooled MER; ties go to the earlier checkpoint (smaller step)."""
    def step(name: str) -> int:
        tail = name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 0
    return min(scores_by_ckpt, key=lambda k: (round(scores_by_ckpt[k], 4), step(k)))


def mer_of(manifests, preds_path, cfg) -> float:
    rows = read_manifests(manifests, cfg["data"].get("audio_root"))
    preds = load_predictions(preds_path)
    scs: list[Score] = [score(r["text"], preds[r["id"]]) for r in rows if r["id"] in preds]
    return pooled_mer([s for s in scs if s.n_ref > 0])
