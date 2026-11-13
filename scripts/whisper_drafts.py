"""Whisper draft transcripts for a trace-lookup bench dir (owner 2026-11-01: Whisper large-v3 instead of Gemini).

1. Downloads missing audio from S3 (bench.py's helpers, ~/.config/vc_bench.env).
2. Measures audio quality (scripts/audio_quality.py) for every item → <bench>/audio_quality.jsonl.
3. Decodes the high-quality items (or all with --all) with zero-shot large-v3, reference decoding v2 settings
   (configs/experiments/zeroshot_large_v3.yaml), locally: no audio leaves the machine.
4. Writes <bench>/drafts.jsonl and review.csv in bench.py's format (drafter "whisper:large-v3").
--no-decode stops after steps 1–2 (e.g. while the model is still downloading).
Usage: uv run python scripts/whisper_drafts.py data/private/online_phase1/bench/online-zh-phase1 [--all] [--no-decode] [--limit N]
"""

import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from audio_quality import SR, decode, quality  # noqa: E402

BENCH_PY = Path.home() / ".claude/skills/trace-lookup/scripts/bench.py"
CONFIG = "configs/experiments/drafts_large_v3_local.yaml"  # fp16, batch 1: fp32 / batch 8 swapped on the 24 GB Mac
DRAFTER = "whisper:large-v3"


def is_high_quality(q: dict) -> bool:
    """Quality thresholds (owner 2026-10-31) plus the plan's 1–30 s rule on the *decoded* duration: the production
    metadata duration is unreliable (clips "≤ 30 s" were up to 382 s long)."""
    return (not q.get("too_short") and q["speech_db"] >= -35 and q["snr_db"] >= 20 and q["clip_pct"] <= 0.1
            and q["hf_db"] >= -40 and 1 <= q["duration_s"] <= 30)


def main() -> None:
    d = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("bench", BENCH_PY)
    bench = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BENCH_PY.parent))
    spec.loader.exec_module(bench)
    s = bench.load_spec(d / "spec.yaml")
    items = bench.jsonl(str(d / "items.jsonl"))
    adir = d / "audio"
    adir.mkdir(exist_ok=True)

    missing = [r for r in items if not (adir / f"{r['audio_id']}.ogg").exists()]
    if missing:
        bench.load_env(())
        s3 = bench.s3_client()
        print(f"downloading {len(missing)} clips from S3 …", flush=True)
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(lambda r: bench.s3_download(r["s3_key"], str(adir / f"{r['audio_id']}.ogg"), s3), missing))

    qpath = d / "audio_quality.jsonl"
    with ThreadPoolExecutor(8) as ex:
        qs = list(ex.map(lambda r: quality(adir / f"{r['audio_id']}.ogg"), items))
    with open(qpath, "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    hq = {q["audio_id"] for q in qs if is_high_quality(q)}
    print(f"quality: {len(hq)} of {len(items)} high quality → {qpath}", flush=True)

    if "--no-decode" in sys.argv:
        return
    dpath = d / "drafts.jsonl"
    done = {x["audio_id"]: x for x in bench.jsonl(str(dpath))} if dpath.exists() else {}
    todo = [r for r in items if r["audio_id"] not in done and ("--all" in sys.argv or r["audio_id"] in hq)]
    if "--limit" in sys.argv:
        todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]
    if todo:
        from whisperflow.eval.transcribe import HFWhisperTranscriber

        cfg = yaml.safe_load(open(CONFIG))
        tr = HFWhisperTranscriber.from_config(cfg["model"], cfg["decoding"], cfg["runtime"])
        bs = int(cfg["runtime"]["batch_size"])
        print(f"decoding {len(todo)} clips with {cfg['model']['name_or_path']} …", flush=True)
        with open(dpath, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), bs):
                batch = todo[i:i + bs]
                audios = [decode(adir / f"{r['audio_id']}.ogg") for r in batch]
                try:
                    results = [(t, None) for t in tr.transcribe(audios)]
                except Exception:  # one bad clip must not stop the run: retry one by one, record failures
                    results = []
                    for a in audios:
                        try:
                            results.append((tr.transcribe([a])[0], None))
                        except Exception as e:
                            results.append(("", f"{type(e).__name__}: {str(e)[:200]}"))
                for r, (t, err) in zip(batch, results):
                    out = {"audio_id": r["audio_id"], "drafts": {DRAFTER: t.strip()} if not err else {},
                           "errors": {DRAFTER: err} if err else {}, "disagreement": None}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    done[r["audio_id"]] = out
                f.flush()
                print(f"  {min(i + bs, len(todo))}/{len(todo)}", flush=True)
    bench.write_review_csv(str(d), s, items, done)
    assert SR == 16000 and os.path.exists(d / "review.csv")


if __name__ == "__main__":
    main()
