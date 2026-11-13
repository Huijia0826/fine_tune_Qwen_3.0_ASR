"""Preview what annotation policy v1 (text/label_style.py) would change in our labels (2026-11-09). Nothing is
written to the manifests: the owner spot-checks the change list first.

Sources = I2's training data without ASCEND (owner 2026-11-09). Each change is attributed to the stage(s) that made
it (fillers / 啊 / numerals / spacing), so the spot check can go rule by rule.

    uv run python scripts/label_v1_changes.py
Outputs: data/private/label_v1/changes.csv (every changed label), reports/label_v1_changes.md (counts + examples).
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

from whisperflow.text.fillers import strip_fillers
from whisperflow.text.label_style import fix_spacing, strip_a_filler, to_label_v1, zh_numbers_to_digits

SOURCES = ["benchmark_chinese", "online_phase1", "online_pool_b1full", "recording_v1", "recording_v2",
           "segmented_long", "arena_019f8e51", "arena_019f6442"]
STAGES = [("fillers", strip_fillers), ("啊", strip_a_filler), ("numerals", zh_numbers_to_digits),
          ("spacing", fix_spacing)]


def stages_changed(text: str) -> list[str]:
    out, cur = [], text
    for name, fn in STAGES:
        nxt = fn(cur)
        if nxt != cur:
            out.append(name)
        cur = nxt
    return out


def main() -> None:
    out_dir = Path("data/private/label_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, totals = [], Counter()
    for src in SOURCES:
        for r in map(json.loads, open(f"data/manifests/{src}.jsonl", encoding="utf-8")):
            if r["split"] not in ("train", "val", "test"):
                continue
            totals[(src, r["split"])] += 1
            new = to_label_v1(r["raw_transcript"])
            if new != r["raw_transcript"]:
                rows.append({"source": src, "split": r["split"], "utterance_id": r["utterance_id"],
                             "stages": "+".join(stages_changed(r["raw_transcript"])),
                             "old": r["raw_transcript"], "new": new})
    with open(out_dir / "changes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    changed = Counter((r["source"], r["split"]) for r in rows)
    by_stage = Counter(s for r in rows for s in r["stages"].split("+"))
    lines = ["# Label policy v1: preview of changes (2026-11-09)", "",
             "Nothing applied yet. Full list: data/private/label_v1/changes.csv. Rules: docs/annotation_policy.md",
             "v1; code: src/whisperflow/text/label_style.py. Spacing-only changes do not affect MER (spaces are",
             "ignored).",
             "", "| source | split | labels | changed | % |", "|---|---|---:|---:|---:|"]
    for (src, split), n in sorted(totals.items()):
        c = changed[(src, split)]
        lines.append(f"| {src} | {split} | {n} | {c} | {100 * c / n:.0f}% |")
    lines += ["", "Changes by rule (a label can hit several): "
              + ", ".join(f"{k} {v}" for k, v in by_stage.items()), ""]
    rng = random.Random(1234)
    for stage in ("啊", "numerals", "fillers"):
        pool = [r for r in rows if stage in r["stages"].split("+")]
        lines += [f"## {stage}: {len(pool)} labels, {min(25, len(pool))} random examples", "",
                  "| source | old | new |", "|---|---|---|"]
        for r in rng.sample(pool, min(25, len(pool))):
            cut = lambda t: t if len(t) <= 90 else t[:87] + "…"  # noqa: E731
            lines.append(f"| {r['source']} | {cut(r['old'])} | {cut(r['new'])} |")
        lines.append("")
    Path("reports/label_v1_changes.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
