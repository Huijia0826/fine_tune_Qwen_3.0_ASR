"""Download the online phase 1 TRAINING POOL (candidates.jsonl split=train) and tag each clip for selection.

Per clip: audio from S3 (trace-lookup bench.py helpers, ~/.config/vc_bench.env), audio quality
(scripts/audio_quality.py, on the decoded duration), and value traits from the production ASR text (owner,
2026-11-02: prefer clips that teach proper nouns, intonation, self-correction, spelled-out letters).
Output: data/private/online_pool/audio/*.ogg + data/private/online_pool/pool.jsonl (+ summary printed).
Usage: NO_PROXY="*" uv run --offline --with boto3 python scripts/dump_online_pool.py
(NO_PROXY: the local HTTPS proxy 127.0.0.1:8118 drops S3 connections; direct access works, 2026-11-02)
"""

import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio_quality import quality  # noqa: E402

BENCH_PY = Path.home() / ".claude/skills/trace-lookup/scripts/bench.py"
OUT = Path("data/private/online_pool")
TERMS = set("cursor claude codex voicecursor gemini whisper openai notion chatgpt langfuse github anthropic posthog figma "
            "vercel supabase seedance slack".split())
SELF_CORR = re.compile(r"不对|我是说|我的意思是|改成|不是.{1,8}是|算了|重新说|sorry|i mean|actually|wait|scratch that", re.I)
SPELL = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][\s\-·]){2,}[A-Za-z](?![A-Za-z])|大写|小写|字母|拼写|spell", re.I)


def traits(r: dict, dur: float) -> list[str]:
    t, p, out = r.get("transcription") or "", r.get("processed") or "", []
    words = re.findall(r"[A-Za-z][A-Za-z0-9.+-]*", t)
    if any(w.lower() in TERMS or re.match(r"^[A-Z]{2,}$|^[A-Z][a-z]+[A-Z]\w*$", w) or (w[0].isupper() and len(w) > 2)
           for w in words):
        out.append("proper_noun")
    if SELF_CORR.search(t) or "pp_shrink" in (r.get("flags") or []):
        out.append("self_correction?")
    if SPELL.search(t):
        out.append("spelling?")
    if re.search(r"[？?！!]", p):
        out.append("question_or_exclamation")
    n = len(re.sub(r"\W", "", t))
    if dur >= 3 and (n / dur > 7 or n / dur < 2):
        out.append("extreme_rate")
    return out


def high_quality(q: dict) -> bool:
    return (not q.get("too_short") and q["speech_db"] >= -35 and q["snr_db"] >= 20 and q["clip_pct"] <= 0.1
            and q["hf_db"] >= -40 and 1 <= q["duration_s"] <= 30)


def main() -> None:
    spec = importlib.util.spec_from_file_location("bench", BENCH_PY)
    bench = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BENCH_PY.parent))
    spec.loader.exec_module(bench)
    pool = [json.loads(l) for l in open("data/private/online_phase1/candidates.jsonl", encoding="utf-8")]
    pool = [r for r in pool if r["split"] == "train"]
    adir = OUT / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in pool if not (adir / f"{r['audio_id']}.ogg").exists()]
    if todo:
        bench.load_env(())
        s3 = bench.s3_client()
        print(f"downloading {len(todo)} of {len(pool)} clips from S3 …", flush=True)

        def get(r):
            try:
                bench.s3_download(r["s3_key"], str(adir / f"{r['audio_id']}.ogg"), s3)
            except Exception as e:  # keep going; the clip is marked missing
                print(f"  {r['audio_id']}: {type(e).__name__}", flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, todo))
    have = [r for r in pool if (adir / f"{r['audio_id']}.ogg").exists()]
    with ThreadPoolExecutor(8) as ex:
        qs = list(ex.map(lambda r: quality(adir / f"{r['audio_id']}.ogg"), have))
    rows = []
    for r, q in zip(have, qs):
        dur = q.get("duration_s", 0)
        rows.append({**r, "quality": q, "high_quality": high_quality(q), "traits": traits(r, dur)})
    with open(OUT / "pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hq = [r for r in rows if r["high_quality"]]
    h = lambda R: sum(r["quality"].get("duration_s", 0) for r in R) / 3600  # noqa: E731
    print(f"pool {len(pool)} | downloaded {len(have)} | high quality {len(hq)} ({h(hq):.2f} h)")
    for t in ("proper_noun", "question_or_exclamation", "extreme_rate", "self_correction?", "spelling?"):
        sel = [r for r in hq if t in r["traits"]]
        print(f"  HQ with {t:<24} {len(sel):>5} clips {h(sel):5.2f} h {len({r['user_id'] for r in sel}):>3} users")
    print(f"  HQ with any trait {sum(1 for r in hq if r['traits'])}")


if __name__ == "__main__":
    main()
