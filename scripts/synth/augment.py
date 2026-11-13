"""Make some clean TTS clips sound more like production dictation (2026-11-10). The owner found the first,
heavy version noisy and odd: now only --rate of the clips (default 15%) are augmented, lightly; the rest stay
clean. For an augmented clip (seeded random chain):
  noise   (60%)  real background noise cut from low-energy stretches of production recordings (data/private/
                 online_pool/audio, our own users' audio) when available, else pink noise; SNR 20–35 dB
  reverb  (20%)  synthetic room impulse response (exponential decay, RT60 0.15–0.35 s), 15–35% wet
  level   (100%) gain −6…+2 dB, soft clip
  codec   (70%)  Ogg/Opus round trip at 16–32 kbit/s through ffmpeg (production audio is Ogg/Opus)
Only augmented clips are written to <out>/<id>.wav (chain in <out>/augment.jsonl); training uses the clean wav
for every other clip.

    uv run python scripts/synth/augment.py data/private/synth/audio --out data/private/synth/audio_aug
"""

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def noise_bank(n_files: int, rng: random.Random) -> list[np.ndarray]:
    """Low-energy stretches (the quietest 20% of 50 ms frames) of production recordings: room / device noise."""
    files = sorted(Path("data/private/online_pool/audio").glob("*.ogg"))
    bank = []
    for f in rng.sample(files, min(n_files, len(files))):
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", str(f), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
            pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            continue
        x = np.frombuffer(pcm, dtype=np.float32)
        frames = x[: len(x) // 800 * 800].reshape(-1, 800)
        if len(frames) < 20:
            continue
        energy = (frames ** 2).mean(axis=1)
        quiet = frames[energy <= np.quantile(energy, 0.2)].reshape(-1)
        if len(quiet) > SR and quiet.std() > 1e-5:
            bank.append(quiet)
    return bank


def colored_noise(n: int, rng: np.random.Generator, beta: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
    return np.fft.irfft(spec / f ** (beta / 2), n).astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, len(x) // len(noise) + 1)
    start = random.randrange(0, len(noise) - len(x) + 1)
    n = noise[start:start + len(x)]
    ps, pn = (x ** 2).mean() + 1e-12, (n ** 2).mean() + 1e-12
    return x + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def reverb(x: np.ndarray, rng: np.random.Generator, rt60: float, wet: float) -> np.ndarray:
    t = np.arange(int(rt60 * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    y = np.convolve(x, ir / np.abs(ir).sum() * 8)[: len(x)]
    return (1 - wet) * x + wet * y * (np.abs(x).max() / (np.abs(y).max() + 1e-9))


def opus_roundtrip(x: np.ndarray, kbps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        src, enc = Path(tmp) / "a.wav", Path(tmp) / "a.ogg"
        sf.write(src, x, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "libopus", "-b:a", f"{kbps}k",
                        str(enc)], check=True)
        cmd = ["ffmpeg", "-v", "error", "-i", str(enc), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
        pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, dtype=np.float32)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--rate", type=float, default=0.15)
    args = ap.parse_args()
    rng, nrng = random.Random(args.seed), np.random.default_rng(args.seed)
    random.seed(args.seed)
    bank = noise_bank(200, rng)
    print(f"noise bank: {len(bank)} production noise stretches")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for wav in sorted(Path(args.audio_dir).glob("*.wav")):
        x, sr = sf.read(str(wav), dtype="float32")
        assert sr == SR, wav
        if rng.random() >= args.rate:
            continue
        chain = {"id": wav.stem}
        if rng.random() < 0.6:
            snr = rng.uniform(20, 35)
            if bank and rng.random() < 0.7:
                x, chain["noise"] = add_noise(x, rng.choice(bank), snr), f"production@{snr:.0f}dB"
            else:
                x, chain["noise"] = add_noise(x, colored_noise(len(x), nrng, 1.0), snr), f"pink@{snr:.0f}dB"
        if rng.random() < 0.2:
            rt60, wet = rng.uniform(0.15, 0.35), rng.uniform(0.15, 0.35)
            x, chain["reverb"] = reverb(x, nrng, rt60, wet), f"rt60={rt60:.2f},wet={wet:.2f}"
        gain = rng.uniform(-6, 2)
        x = np.tanh(x * 10 ** (gain / 20) * 1.2) / 1.2
        chain["gain_db"] = round(gain, 1)
        if rng.random() < 0.7:
            kbps = rng.choice([16, 24, 32])
            x, chain["codec"] = opus_roundtrip(x, kbps), f"opus{kbps}k"
        sf.write(str(out / wav.name), x, SR, subtype="PCM_16")
        log.append(chain)
    with open(out / "augment.jsonl", "w", encoding="utf-8") as f:
        for c in log:
            f.write(json.dumps(c) + "\n")
    print(f"{len(log)} clips → {out}")


if __name__ == "__main__":
    main()
