"""Decode a slice of the online training pool with a chosen model, as review drafts / triage (2026-11-02).

    python scripts/decode_pool.py --config configs/experiments/latency/hf_fp32_beam5.yaml --group CS --hq \
        --out data/private/online_pool/drafts_C600_batch1.jsonl

Pool clips are not in any manifest yet (unreviewed), so this reads data/private/online_pool/pool.jsonl directly.
Output per clip: audio_id, hypothesis (simplified), detected_language. Resumes: decoded audio_ids are skipped.
"""

import argparse
import json
from pathlib import Path

from whisperflow.data.audio import load_16k_mono
from whisperflow.data.common import load_yaml
from whisperflow.eval.transcribe import HFWhisperTranscriber

POOL = Path("data/private/online_pool")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--group", help="pool group, e.g. CS / ZH / S4 (pool.jsonl only)")
ap.add_argument("--items", help="decode this jsonl (audio_id per line) instead of the pool, e.g. benchmark_v2/items.jsonl;"
                " audio is read from <its dir>/audio")
ap.add_argument("--hq", action="store_true", help="only high_quality clips")
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg = load_yaml(args.config)
if args.items:
    POOL = Path(args.items).parent
    clips = [json.loads(line) for line in open(args.items, encoding="utf-8")]
else:
    clips = [r for r in map(json.loads, open(POOL / "pool.jsonl", encoding="utf-8"))
             if r["group"] == args.group and (r["high_quality"] or not args.hq)]
out = Path(args.out)
done = {json.loads(line)["audio_id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
todo = [r for r in clips if r["audio_id"] not in done]
print(f"{len(clips)} clips, {len(todo)} to decode", flush=True)
tr = HFWhisperTranscriber.from_config(cfg["model"], cfg["decoding"], cfg["runtime"])
with open(out, "a", encoding="utf-8") as f:
    for i, r in enumerate(todo):
        hyp = tr.transcribe([load_16k_mono(POOL / "audio" / f"{r['audio_id']}.ogg")])[0]
        f.write(json.dumps({"audio_id": r["audio_id"], "hypothesis": hyp,
                            "detected_language": tr.last_info[0]["detected_language"]}, ensure_ascii=False) + "\n")
        f.flush()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
print("DECODE_DONE")
