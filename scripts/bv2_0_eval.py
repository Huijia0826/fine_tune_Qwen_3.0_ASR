"""Run our own models on benchmark v2.0 (owner, 2026-11-12) and score them exactly like the arena systems.

  prepare  arena export xlsx (clip list + ground_truth = source of truth) + index.csv → 16 kHz mono wavs
           data/private/benchmark_v2_0/<audio id>.wav (ffmpeg, from the arena upload's original audio) and
           data/manifests/benchmark_v2_0.jsonl. utterance_id = index.csv utterance_id, so the per-clip contexts in
           runs/qwen_sft_J3_contexts.jsonl apply as they are. Every row is split "test" for decoding only (index.csv's
           split_in_training says which clips our models trained on; the scorer slices by it).
  export   wf-transcribe predictions → an xlsx in the arena export layout (audio_url, ground_truth, <system>|raw_text,
           |raw_wer, |raw_latency, |raw_status, |raw_error) for scripts/score_arena_export.py. raw_latency = the
           clip's GPU decode time when decoded with batch_size 1 (no network, not comparable to arena latencies).
  compare  two score_arena_export per-clip csvs (A, B) → paired bootstrap over clips of the pooled-MER difference
           B − A, overall and per split_in_training / source set / language (same clips, same references).

    uv run --with openpyxl python scripts/bv2_0_eval.py prepare ~/Downloads/benchmark2.0_voicecursorstreaming.xlsx
    uv run --with openpyxl python scripts/bv2_0_eval.py export ~/Downloads/benchmark2.0_voicecursorstreaming.xlsx \
        reports/runs/bv2_0_qwen3asr_J3_step_600_ctx "Qwen3ASR-J3:step600-ctx" reports/arena/bv2_0_J3_step600_ctx.xlsx
    uv run python scripts/bv2_0_eval.py compare reports/arena/benchmark2.0_voicecursorstreaming_per_clip.csv \
        reports/arena/bv2_0_J3_step600_ctx_per_clip.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import openpyxl
import soundfile as sf

from whisperflow.data.schema import Utterance

BV2 = Path.home() / "Desktop/Build_Benchmark/benchmark_v2.0"
AUDIO_DIR = "benchmark_v2_0"


def arena_rows(export: str) -> list[tuple]:
    ws = openpyxl.load_workbook(export).active  # not read_only: arena writes a wrong <dimension>
    return list(ws.iter_rows(values_only=True))[1:]


def prepare(a) -> None:
    idx = {r["file"]: r for r in csv.DictReader(open(BV2 / "index.csv", encoding="utf-8-sig"))}
    out_dir = Path("data/private") / AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for f, gt, *_ in arena_rows(a.export):
        meta = idx[f]
        wav = out_dir / (Path(f).stem + ".wav")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(BV2 / "audio" / f), "-ac", "1",
                            "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        langs = meta["language_tags"].split("+")
        info = sf.info(str(wav))
        recs.append(Utterance(
            utterance_id=meta["utterance_id"], audio_path=f"{AUDIO_DIR}/{wav.name}",
            sha256=hashlib.sha256(wav.read_bytes()).hexdigest(), duration_sec=info.frames / info.samplerate,
            speaker_id=f"bv2_0-{meta['set']}", session_id=f"bv2_0-{meta['set']}", source_id="benchmark_v2_0",
            license_or_consent_id="voicecursor_internal_training", split="test", raw_transcript=gt or "",
            primary_scenario="en_general" if langs == ["en"] else "zh_en_switch" if len(langs) > 1 else "zh_general",
            language_tags=langs, label_status="human_verified", reviewer_id="owner_arena", source_utterance_id=f,
            sample_rate_hz=16000, num_channels=1))
    man = Path("data/manifests/benchmark_v2_0.jsonl")
    with open(man, "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(r.model_dump_json(exclude_none=True) + "\n")
    print(f"{len(recs)} clips → {man}, audio {out_dir} ({sum(r.duration_sec for r in recs) / 60:.1f} min)")


def export(a) -> None:
    man = {json.loads(line)["utterance_id"]: json.loads(line)
           for line in open("data/manifests/benchmark_v2_0.jsonl", encoding="utf-8")}
    by_file = {}
    for line in open(Path(a.run_dir) / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        by_file[man[p["utterance_id"]]["source_utterance_id"]] = p
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["audio_url", "ground_truth"] + [f"{a.system}|raw_{k}" for k in
                                              ("text", "wer", "latency", "status", "error")])
    missing = 0
    for f, gt, *_ in arena_rows(a.export):
        p = by_file.get(f)
        missing += p is None
        ws.append([f, gt, p["hypothesis"] if p else "", None, round(p["batch_decode_sec"], 3) if p else None,
                   "succeeded" if p else "failed", None if p else "no prediction"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print(f"{ws.max_row - 1} rows ({missing} without a prediction) → {a.out}")


def compare(a) -> None:
    load = lambda f: {r["file"]: r for r in csv.DictReader(open(f, encoding="utf-8-sig"))}  # noqa: E731
    A, B = load(a.a), load(a.b)
    files = sorted(set(A) & set(B))
    rows = [(int(A[f]["n_ref"]), int(A[f]["err"]), int(B[f]["err"]), A[f]) for f in files]
    rng = random.Random(1234)

    def diff(rs):
        n = sum(r[0] for r in rs)
        return sum(r[1] for r in rs) / n, sum(r[2] for r in rs) / n

    def line(name, rs):
        ma, mb = diff(rs)
        ds = sorted((lambda x: x[1] - x[0])(diff([rng.choice(rs) for _ in rs])) for _ in range(2000))
        lo, hi = ds[50], ds[1949]
        sig = "" if lo <= 0 <= hi else " *"
        return f"| {name} | {len(rs)} | {100 * ma:.2f}% | {100 * mb:.2f}% | {100 * (mb - ma):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{sig} |"

    print(f"A = {a.a}\nB = {a.b}\n{len(files)} shared clips; * = 95% interval excludes 0\n")
    print("| slice | clips | A | B | B − A [95% CI] |\n|---|---|---|---|---|")
    print(line("all", rows))
    for key in ("split", "set", "lang"):
        g = defaultdict(list)
        for r in rows:
            g[r[3][key]].append(r)
        for k, rs in sorted(g.items(), key=lambda kv: -len(kv[1])):
            print(line(f"{key}: {k}", rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("export")
    e = sub.add_parser("export")
    for k in ("export", "run_dir", "system", "out"):
        e.add_argument(k)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    {"prepare": prepare, "export": export, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
