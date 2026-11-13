"""Objective per-clip audio quality for a folder of recordings (any format ffmpeg reads).

Per clip, on 16 kHz mono, 30 ms frames:
  speech_db  loudness of the loudest 20% frames (dBFS): how loud the speech is
  noise_db   loudness of the quietest 20% frames (dBFS): background noise floor
  snr_db     speech_db − noise_db: how far speech stands above the background
  clip_pct   % of samples at |x| ≥ 0.99: clipping / distortion
  hf_db      energy 4–8 kHz vs 0–4 kHz in the loudest frames (dB): ≲ −40 = narrowband / muffled (e.g. Bluetooth
             headset call mode); wideband speech is roughly −15 to −25
Usage: uv run python scripts/audio_quality.py <audio dir> <out.jsonl>
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SR, FRAME = 16000, 480


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def quality(path: Path) -> dict:
    x = decode(path)
    n = len(x) // FRAME
    if n < 5:
        return {"audio_id": path.stem, "too_short": True}
    frames = x[: n * FRAME].reshape(n, FRAME)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    order = np.sort(db)
    k = max(1, n // 5)
    speech, noise = float(np.mean(order[-k:])), float(np.mean(order[:k]))
    loud = frames[db >= order[-k]]
    spec = np.abs(np.fft.rfft(loud * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    hf_db = float(10 * np.log10(spec[:, freqs >= 4000].sum() / (spec[:, freqs < 4000].sum() + 1e-12) + 1e-12))
    return {"audio_id": path.stem, "duration_s": round(len(x) / SR, 2), "speech_db": round(speech, 1),
            "noise_db": round(noise, 1), "snr_db": round(speech - noise, 1),
            "clip_pct": round(100 * float(np.mean(np.abs(x) >= 0.99)), 3), "hf_db": round(hf_db, 1)}


def main() -> None:
    files = sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_file())
    with ThreadPoolExecutor(8) as ex, open(sys.argv[2], "w") as f:
        for q in ex.map(quality, files):
            f.write(json.dumps(q) + "\n")
    print(f"{len(files)} clips → {sys.argv[2]}")


if __name__ == "__main__":
    main()
