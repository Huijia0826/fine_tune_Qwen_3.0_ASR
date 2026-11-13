"""Per-clip latency benchmark (E7): batch 1, the way a user waits for one dictation after they stop talking.

Usage:
    wf-latency --config configs/experiments/latency/hf_fp32_beam5.yaml --run-id lat_hf_fp32_beam5

Decodes users' val (benchmark_chinese val + online_phase1 val) one clip at a time. Timed span = audio array in
memory -> final text (feature extraction, language ID, decoding, t2s); file loading is excluded because
production audio is already in memory when the user stops. The first --warmup clips are decoded once untimed
(CUDA context, kernel selection, allocator). Outputs reports/runs/<run_id>/{predictions.jsonl, latency.json}:
p50/p95 per length bucket, RTF, and pooled official MER so speed and accuracy are read together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from whisperflow.data.audio import load_16k_mono
from whisperflow.data.common import load_yaml
from whisperflow.data.schema import load_manifest
from whisperflow.eval.official import official_mer
from whisperflow.eval.transcribe import _hardware, _versions, config_hash, make_transcriber

USERS_VAL = ["data/manifests/benchmark_chinese.jsonl", "data/manifests/online_phase1.jsonl"]

BUCKETS = ["<5 s", "5-15 s", "15-30 s", ">30 s"]


def bucket_of(duration_s: float) -> str:
    """Lower bounds inclusive; exactly 30 s is still short-form (one encoder window)."""
    if duration_s < 5:
        return "<5 s"
    if duration_s < 15:
        return "5-15 s"
    if duration_s <= 30:
        return "15-30 s"
    return ">30 s"


def summarize_latency(records: list[dict]) -> dict:
    """records: [{"duration_s": float, "latency_s": float}, ...] -> {bucket: stats} plus "all".

    stats = n, p50_s, p95_s, mean_s, rtf (total latency / total audio). Empty buckets get n=0 and None values.
    Percentiles use numpy's default linear interpolation; with small n (e.g. 8 clips) p95 is close to the max,
    so always read it together with n.
    """
    groups: dict[str, list[dict]] = {label: [] for label in BUCKETS}
    for r in records:
        groups[bucket_of(r["duration_s"])].append(r)
    groups["all"] = list(records)

    out = {}
    for label, rs in groups.items():
        if not rs:
            out[label] = {"n": 0, "p50_s": None, "p95_s": None, "mean_s": None, "rtf": None}
            continue
        lat = np.array([r["latency_s"] for r in rs])
        dur = sum(r["duration_s"] for r in rs)
        out[label] = {"n": len(rs),
                      "p50_s": round(float(np.percentile(lat, 50)), 4),
                      "p95_s": round(float(np.percentile(lat, 95)), 4),
                      "mean_s": round(float(lat.mean()), 4),
                      "rtf": round(float(lat.sum() / dur), 4)}
    return out


def _sync() -> None:
    """GPU work is asynchronous: without a sync the clock stops before the kernels finish."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", action="append", help=f"repeatable; default users' val: {USERS_VAL}")
    ap.add_argument("--split", default="val")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first N clips (smoke tests)")
    args = ap.parse_args(argv)
    if args.split == "test":
        print("Latency runs use val only.")
        return 2

    cfg = load_yaml(args.config)
    data_root = Path(cfg["data"]["data_root"])
    records = [r for m in (args.manifest or USERS_VAL) for r in load_manifest(m) if r.split == args.split]
    records.sort(key=lambda r: r.utterance_id)
    if args.limit:
        records = records[:args.limit]
    audios = [load_16k_mono(data_root / r.audio_path) for r in records]

    print(f"Loading {cfg['model'].get('ct2_path') or cfg['model']['name_or_path']} ...", flush=True)
    t_load = time.perf_counter()
    tr = make_transcriber(cfg)
    load_s = time.perf_counter() - t_load
    for a in audios[:args.warmup]:
        tr.transcribe([a])
    _sync()

    out_dir = Path("reports/runs") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, err, tok = [], 0, 0
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, (r, a) in enumerate(zip(records, audios, strict=True)):
            _sync()
            t0 = time.perf_counter()
            hyp = tr.transcribe([a])[0]
            _sync()
            dt = time.perf_counter() - t0
            info = (getattr(tr, "last_info", None) or [{}])[0]
            o = official_mer(r.raw_transcript, hyp)
            if o.included:
                err, tok = err + o.errors, tok + o.n_ref
            row = {"utterance_id": r.utterance_id, "hypothesis": hyp, **info,
                   "duration_s": round(len(a) / 16000, 3), "latency_s": round(dt, 4)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(records)}", flush=True)

    summary = {"run_id": args.run_id, "config": cfg, "config_hash": config_hash(cfg),
               "manifests": args.manifest or USERS_VAL, "split": args.split, "warmup": args.warmup,
               "transcriber": tr.describe(), "model_load_s": round(load_s, 2),
               "official_mer_pct": round(100 * err / tok, 2) if tok else None,
               "latency": summarize_latency(rows), "versions": _versions(), "hardware": _hardware()}
    try:
        import torch

        if torch.cuda.is_available():
            summary["hardware"]["gpu"] = torch.cuda.get_device_name(0)
            summary["torch_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except ImportError:
        pass
    (out_dir / "latency.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("official_mer_pct", "latency")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
