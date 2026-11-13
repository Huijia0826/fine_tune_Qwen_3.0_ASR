"""Score an ASR Arena run export on benchmark v2.0 with our policy-v1 MER (2026-11-12).

The arena export (columns audio_url, ground_truth, <system>|raw_text, |raw_wer, |raw_latency, |raw_status, |raw_error)
is re-scored against the arena's ground_truth column, which is the source of truth (owner, 2026-11-12: the owner edited
grammar choices on arena after uploading, and clips missing from arena were deleted on purpose). Our upload sheet
(~/Desktop/Build_Benchmark/benchmark_v2.0/benchmark_v2.0_upload.xlsx) is only used for the source-set / language
slices (index.csv) and reported as a secondary number (39 references differ: 36 bv2-en-short + 3 recording_v2_test).
Pooled MER = errors / reference tokens (zh characters + en words, src/whisperflow/eval/metrics.py);
95% CI by bootstrap over clips. Slices: source set and language tags from index.csv; note 642 / 912 clips are in our
training split, so this set does not compare our fine-tuned models fairly (production systems are fine).

    uv run --with openpyxl python scripts/score_arena_export.py ~/Downloads/benchmark2.0_voicecursorstreaming.xlsx
Outputs: reports/arena/<export stem>.md + <export stem>_per_clip.csv
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import openpyxl

from whisperflow.eval.metrics import align_text, score_alignment

BV2 = Path.home() / "Desktop/Build_Benchmark/benchmark_v2.0"


def pooled(rows, key="err"):
    n = sum(r["n_ref"] for r in rows)
    return sum(r[key] for r in rows) / n if n else float("nan")


def boot_ci(rows, key="err", iters=1000, seed=0):
    rng = random.Random(seed)
    vals = sorted(pooled([rng.choice(rows) for _ in rows], key) for _ in range(iters))
    return vals[int(0.025 * iters)], vals[int(0.975 * iters)]


def pct(x):
    return f"{100 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--upload", default=str(BV2 / "benchmark_v2.0_upload.xlsx"))
    ap.add_argument("--index", default=str(BV2 / "index.csv"))
    ap.add_argument("--out-dir", default="reports/arena")
    a = ap.parse_args()

    ws = openpyxl.load_workbook(a.export).active  # not read_only: arena writes a wrong <dimension>
    hdr, *data = list(ws.iter_rows(values_only=True))
    system = hdr[2].rsplit("|", 1)[0]
    up = {str(r[0]): r[1] for r in list(openpyxl.load_workbook(a.upload).active.iter_rows(values_only=True))[1:]}
    idx = {r["file"]: r for r in csv.DictReader(open(a.index, encoding="utf-8-sig"))}

    rows = []
    for f, arena_gt, hyp, arena_wer, lat, status, err in (r[:7] for r in data):
        hyp = hyp or ""
        s = score_alignment(align_text(arena_gt or "", hyp))
        s_up = score_alignment(align_text(up[f], hyp))
        rows.append({"file": f, "set": idx[f]["set"], "lang": idx[f]["language_tags"],
                     "split": idx[f]["split_in_training"], "dur": float(idx[f]["duration_sec"]),
                     "status": status, "latency": lat, "arena_wer": arena_wer,
                     "n_ref": s.n_ref, "err": s.errors, "sub": s.sub, "del": s.dele, "ins": s.ins,
                     "n_ref_upload": s_up.n_ref, "err_upload": s_up.errors,
                     "gt_differs": (arena_gt or "") != up[f], "ref": arena_gt or "", "upload_gt": up[f], "hyp": hyp})
    missing = sorted(set(up) - {r["file"] for r in rows})  # deleted on arena by the owner
    ok = [r for r in rows if r["status"] == "succeeded"]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.export).stem
    with open(out / f"{stem}_per_clip.csv", "w", encoding="utf-8-sig", newline="") as fo:
        cols = ["file", "set", "lang", "split", "dur", "latency", "n_ref", "err", "sub", "del", "ins", "mer",
                "arena_wer", "gt_differs", "ref", "upload_gt", "hyp"]
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["err"]):
            w.writerow({**r, "mer": round(r["err"] / r["n_ref"], 4) if r["n_ref"] else ""})

    lat = sorted(r["latency"] for r in ok if isinstance(r["latency"], (int, float)))
    lo, hi = boot_ci(ok)
    upload_rows = [{**r, "n_ref": r["n_ref_upload"]} for r in ok]
    # our own models' exports (scripts/bv2_0_eval.py export) have no arena-side WER
    arena_wers = [r["arena_wer"] for r in ok if r["arena_wer"] is not None]
    arena_wer_mean = pct(sum(arena_wers) / len(arena_wers)) if arena_wers else "n/a (not scored on arena)"
    L = [f"# Arena export `{stem}` — benchmark v2.0, system `{system}`", "",
         f"Clips: {len(rows)} in export ({len(ok)} succeeded). Reference = arena ground_truth (source of truth; "
         f"{len(missing)} upload-sheet clip(s) deleted on arena by the owner). Scorer: policy v1 MER "
         "(scripts/score_arena_export.py).", "",
         "| reference | pooled MER | 95% CI |", "|---|---|---|",
         f"| **arena ground_truth (source of truth)** | **{pct(pooled(ok))}** | [{pct(lo)}, {pct(hi)}] |",
         f"| original upload sheet ({sum(r['gt_differs'] for r in ok)} rows differ; for reference only) | "
         f"{pct(pooled(upload_rows, 'err_upload'))} | |",
         f"| arena's own raw_wer (mean per clip, arena normaliser) | "
         f"{arena_wer_mean} | |", "",
         f"Latency (raw_latency, s): p50 {lat[len(lat) // 2]:.2f} · p90 {lat[int(0.9 * len(lat))]:.2f} · "
         f"max {lat[-1]:.2f}", "",
         f"Errors: sub {sum(r['sub'] for r in ok)} · del {sum(r['del'] for r in ok)} · ins {sum(r['ins'] for r in ok)}"
         f" over {sum(r['n_ref'] for r in ok)} reference tokens", ""]
    for name, keyf in [("split in our training data (train / val clips are seen by our fine-tuned models)",
                        lambda r: r["split"]), ("source set", lambda r: r["set"]), ("language", lambda r: r["lang"]),
                       ("duration", lambda r: "≤ 10 s" if r["dur"] <= 10 else "10–30 s" if r["dur"] <= 30 else "> 30 s")]:
        g = defaultdict(list)
        for r in ok:
            g[keyf(r)].append(r)
        L += [f"## By {name}", "", "| slice | clips | ref tokens | MER | 95% CI |", "|---|---|---|---|---|"]
        for k, rs in sorted(g.items(), key=lambda kv: -len(kv[1])):
            lo, hi = boot_ci(rs)
            L.append(f"| {k} | {len(rs)} | {sum(r['n_ref'] for r in rs)} | {pct(pooled(rs))} | [{pct(lo)}, {pct(hi)}] |")
        L.append("")
    L += ["## Worst 15 clips (by error count)", "", "| file | set | errors / ref | reference | hypothesis |",
          "|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: -r["err"])[:15]:
        cut = lambda s: s.replace("|", "/").replace("\n", " ")[:90]  # noqa: E731
        L.append(f"| {r['file'][:8]} | {r['set']} | {r['err']} / {r['n_ref']} | {cut(r['ref'])} | {cut(r['hyp'])} |")
    (out / f"{stem}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()
