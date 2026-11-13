"""Step 3b: download the sample's production audio from S3 (read-only) and make 16 kHz mono FLAC copies for the cluster.

data/private/lid_probe/audio/<audio_id>.ogg (original) and audio16k/<audio_id>.flac (decoded with ffmpeg).
The local proxy can block S3, so NO_PROXY="*" is set before boto3 starts.
Usage: python3 scripts/lid_probe/download_audio.py
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
sys.path.insert(0, os.path.expanduser("~/.claude/skills/trace-lookup/scripts"))
from trace_lookup import s3_client, s3_download  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data/private/lid_probe"


def main() -> None:
    rows = [json.loads(line) for line in open(OUT / "sample.jsonl", encoding="utf-8")]
    (OUT / "audio").mkdir(exist_ok=True)
    (OUT / "audio16k").mkdir(exist_ok=True)
    client = s3_client()

    def one(r):
        ogg, flac = OUT / "audio" / f"{r['audio_id']}.ogg", OUT / "audio16k" / f"{r['audio_id']}.flac"
        try:
            if not ogg.exists() or ogg.stat().st_size < 200:
                s3_download(r["s3_key"], str(ogg), client)
            if not flac.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ogg), "-ac", "1", "-ar", "16000", str(flac)],
                               check=True)
            return r["audio_id"], "ok", ogg.stat().st_size
        except Exception as e:  # noqa: BLE001
            return r["audio_id"], f"fail: {type(e).__name__}: {str(e)[:120]}", 0

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, rows))
    bad = [x for x in res if x[1] != "ok" or x[2] < 200]
    print(f"{len(res) - len(bad)} ok, {len(bad)} problems")
    for b in bad:
        print("  ", b)
    json.dump({a: s for a, s, _ in res}, open(OUT / "download_status.json", "w"), indent=0)


if __name__ == "__main__":
    main()
