"""Step 3c (cluster, .venv-qwen): Qwen3-ASR-1.7B zero-shot, language = auto (no forcing), on the probe clips.

Uses the project wrapper (whisperflow.eval.qwen_asr_backend.QwenASRTranscriber) with
configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml unchanged (auto language, 25 s chunks, simplified output).
Writes one JSON line per clip: audio_id, detected_language (Qwen's own tag, "Chinese,English" when chunks differ),
hypothesis (t2s simplified), hypothesis_raw.
Run (on the cluster):
  sbatch -J lid-probe -p batch --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G -t 01:00:00 -o runs/lid_probe.log \
    --wrap "cd ~/whisperflow && HF_HOME=\$PWD/hf_cache HF_HUB_OFFLINE=1 .venv-qwen/bin/python scripts/lid_probe/run_qwen_lid.py"
"""

import json
import sys
import time
from pathlib import Path

import soundfile as sf
import yaml

from whisperflow.eval.qwen_asr_backend import QwenASRTranscriber

CFG = "configs/experiments/zeroshot_qwen3_asr_1.7b_chunk25.yaml"
AUDIO = Path("data/private/lid_probe/audio16k")
OUT = Path("data/private/lid_probe/qwen_lid.jsonl")
BATCH = 16


def main() -> None:
    cfg = yaml.safe_load(open(CFG))
    assert cfg["decoding"]["language"] == "auto"
    files = sorted(AUDIO.glob("*.flac"))
    done = set()
    if OUT.exists():
        done = {json.loads(line)["audio_id"] for line in open(OUT)}
    files = [f for f in files if f.stem not in done]
    print(f"{len(files)} clips to decode ({len(done)} already done)", flush=True)
    tr = QwenASRTranscriber(cfg["model"], cfg["decoding"], cfg["runtime"])
    print(json.dumps(tr.describe()), flush=True)
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(0, len(files), BATCH):
            chunk = files[i:i + BATCH]
            audios = []
            for p in chunk:
                a, sr = sf.read(p, dtype="float32")
                assert sr == 16000, (p, sr)
                audios.append(a if a.ndim == 1 else a.mean(axis=1))
            hyps = tr.transcribe(audios)
            for p, h, info in zip(chunk, hyps, tr.last_info):
                f.write(json.dumps({"audio_id": p.stem, "detected_language": info["detected_language"],
                                    "hypothesis": h, "hypothesis_raw": info["hypothesis_raw"],
                                    "duration_s": round(len(audios[chunk.index(p)]) / 16000, 2)},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(chunk)}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    print("LID_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
