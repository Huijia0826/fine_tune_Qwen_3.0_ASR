"""Synthesise audio for synthetic dictation texts with MiniMax stock voices (owner, 2026-11-10; no voice cloning).

One random voice per sample (seeded): English stock voices for English text, Chinese stock voices otherwise (they
also read the English words, which gives Chinese-accented English). The TTS reads `spoken_text`; the labels stay
`strict_transcript` / `final_output`. Parallel requests, resumable (existing wavs are skipped), 16 kHz mono wav.

    MINIMAX_API_KEY=… MINIMAX_API_BASE=https://api.minimax.io \
        uv run python scripts/synth/synthesize.py data/private/synth/pilot_500.jsonl --out data/private/synth/audio
Writes <out>/<id>.wav and <out>/manifest.jsonl (id, voice, duration_s, error).
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tts_compare"))
from synth import TARGET_SR, VENDORS, call_minimax  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.texts, encoding="utf-8")][:args.limit]
    voices = VENDORS["minimax"]["voices"]
    rng = random.Random(args.seed)
    plan = []
    for r in rows:  # voice drawn for every row, so adding rows later does not change earlier choices
        lang = "en" if r["language"] == "en" else "zh"
        plan.append((r, rng.choice([v for v in voices if v.lang == lang]), lang))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.jsonl"
    done = {}
    if mpath.exists():
        done = {m["id"]: m for m in map(json.loads, open(mpath, encoding="utf-8")) if not m.get("error")}
    lock = threading.Lock()

    def work(r, v, lang):
        wav = out / f"{r['id']}.wav"
        if wav.exists() and r["id"] in done:
            return done[r["id"]]
        for attempt in range(4):
            try:
                audio = call_minimax(r["spoken_text"], v, lang)
                sf.write(str(wav), audio, TARGET_SR, subtype="PCM_16")
                return {"id": r["id"], "voice": v.id, "voice_label": v.label,
                        "duration_s": round(len(audio) / TARGET_SR, 2)}
            except Exception as e:  # noqa: BLE001 - retry, then record the failure and continue
                err = f"{type(e).__name__}: {e}"[:300]
                time.sleep(3 * (attempt + 1))
        return {"id": r["id"], "voice": v.id, "error": err}

    results, t0 = [], time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, *p) for p in plan]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            with lock:
                results.append(res)
            if i % 25 == 0 or i == len(futs):
                errs = sum(1 for x in results if x.get("error"))
                print(f"{i}/{len(futs)} done, {errs} errors, {time.time() - t0:.0f}s", flush=True)
    order = {r["id"]: k for k, r in enumerate(rows)}
    results.sort(key=lambda x: order[x["id"]])
    with open(mpath, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    ok = [x for x in results if not x.get("error")]
    print(f"{len(ok)} ok, {len(results) - len(ok)} errors, {sum(x['duration_s'] for x in ok) / 3600:.2f} h audio")


if __name__ == "__main__":
    main()
