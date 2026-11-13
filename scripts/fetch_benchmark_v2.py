"""Download benchmark v2 candidates, measure audio quality, and fix the final candidate list (2026-11-04).

A test set should look like real traffic, so the quality gate is looser than the training pool's high_quality rule: drop
only clips that cannot be labelled (too short / silent, speech < −45 dBFS, SNR < 10 dB, > 1% clipped). The training rule
is still recorded per clip (`high_quality`) so results can be sliced by it. zh / zh-en keep every usable clip; en is
trimmed back to TARGET["en"] in the selection order (one clip per user).
Output: data/private/benchmark_v2/audio/*.ogg + items.jsonl (+ summary printed).
Usage: NO_PROXY="*" uv run --offline --with boto3 python scripts/fetch_benchmark_v2.py
"""

import collections
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dump_online_pool as dop  # noqa: E402  (quality / traits / high_quality / bench helpers)
from select_benchmark_v2 import TARGET  # noqa: E402

OUT = Path("data/private/benchmark_v2")


def usable(q: dict) -> bool:
    return (not q.get("too_short") and q.get("speech_db", -99) >= -45 and q.get("snr_db", 0) >= 10
            and q.get("clip_pct", 100) <= 1.0)


def main() -> None:
    spec = importlib.util.spec_from_file_location("bench", dop.BENCH_PY)
    bench = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(dop.BENCH_PY.parent))
    spec.loader.exec_module(bench)
    cands = [json.loads(l) for l in open(OUT / "candidates.jsonl", encoding="utf-8")]
    adir = OUT / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in cands if not (adir / f"{r['audio_id']}.ogg").exists()]
    if todo:
        bench.load_env(())
        s3 = bench.s3_client()
        print(f"downloading {len(todo)} of {len(cands)} clips from S3 …", flush=True)

        def get(r):
            try:
                bench.s3_download(r["s3_key"], str(adir / f"{r['audio_id']}.ogg"), s3)
            except Exception as e:  # keep going; the clip counts as missing
                print(f"  {r['audio_id']}: {type(e).__name__}", flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, todo))
    have = [r for r in cands if (adir / f"{r['audio_id']}.ogg").exists()]
    with ThreadPoolExecutor(8) as ex:
        qs = list(ex.map(lambda r: dop.quality(adir / f"{r['audio_id']}.ogg"), have))
    funnel = collections.Counter({"candidates": len(cands), "downloaded": len(have)})
    items, n_en = [], 0
    for r, q in zip(have, qs):
        if not usable(q):
            funnel[f"unusable audio ({r['lang_class']})"] += 1
            continue
        if r["lang_class"] == "en":
            if n_en >= TARGET["en"]:
                continue
            n_en += 1
        items.append({**r, "quality": q, "high_quality": dop.high_quality(q), "traits": dop.traits(r, q.get("duration_s", 0))})
    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(funnel))
    h = lambda R: sum(r["quality"].get("duration_s", 0) for r in R) / 3600  # noqa: E731
    for c in TARGET:
        sel = [r for r in items if r["lang_class"] == c]
        print(f"  {c:<6} {len(sel):>4} clips {h(sel):.2f} h  users {len({r['user_id'] for r in sel}):>3}  seen {sum(r['seen'] for r in sel):>3}"
              f"  high_quality {sum(r['high_quality'] for r in sel):>3}  proper_noun {sum('proper_noun' in r['traits'] for r in sel):>3}")
    print(f"  total  {len(items)} clips {h(items):.2f} h")


if __name__ == "__main__":
    main()
