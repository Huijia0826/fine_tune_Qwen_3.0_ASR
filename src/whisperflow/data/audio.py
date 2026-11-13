"""Audio I/O helpers: hashing, header probing, signal-quality checks."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000  # Whisper's feature extractor expects 16 kHz mono


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class AudioInfo:
    duration_sec: float
    sample_rate_hz: int
    num_channels: int
    frames: int


def probe(path: str | Path) -> AudioInfo:
    """Header-only probe (fast). Raises on unreadable files."""
    info = sf.info(str(path))
    return AudioInfo(
        duration_sec=info.frames / info.samplerate,
        sample_rate_hz=info.samplerate,
        num_channels=info.channels,
        frames=info.frames,
    )


@dataclass
class SignalStats:
    peak: float
    rms_dbfs: float
    clipped_fraction: float
    silent: bool
    leading_silence_sec: float
    trailing_silence_sec: float


def signal_stats(path: str | Path, silence_dbfs: float = -50.0) -> SignalStats:
    """Full decode; detects clipping, near-silence and long leading/trailing silence."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if x.size == 0:
        return SignalStats(0.0, -np.inf, 0.0, True, 0.0, 0.0)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
    clipped = float(np.mean(np.abs(x) >= 0.999))

    # 20 ms frame energy for silence trimming estimates
    hop = max(1, int(0.02 * sr))
    n_frames = x.size // hop
    if n_frames == 0:
        lead = trail = 0.0
    else:
        frames = x[: n_frames * hop].reshape(n_frames, hop)
        e = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-12)
        voiced = np.where(e > silence_dbfs)[0]
        if voiced.size == 0:
            lead = trail = n_frames * hop / sr
        else:
            lead = voiced[0] * hop / sr
            trail = (n_frames - 1 - voiced[-1]) * hop / sr
    return SignalStats(
        peak=peak,
        rms_dbfs=float(rms_dbfs),
        clipped_fraction=clipped,
        silent=bool(rms_dbfs < silence_dbfs),
        leading_silence_sec=float(lead),
        trailing_silence_sec=float(trail),
    )


def load_16k_mono(path: str | Path) -> np.ndarray:
    """Load audio as float32 mono at 16 kHz (resampling via ffmpeg if needed)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr == TARGET_SR:
        return x
    return _ffmpeg_decode(path)


def _ffmpeg_decode(path: str | Path) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def convert_to_16k_wav(src: str | Path, dst: str | Path) -> None:
    """Canonicalize to 16 kHz mono 16-bit PCM WAV. Never converts a file onto itself."""
    if Path(src).resolve() == Path(dst).resolve():
        raise ValueError(f"refusing in-place conversion of {src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
           "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
