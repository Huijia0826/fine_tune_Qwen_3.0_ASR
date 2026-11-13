"""Shared fixtures. Everything here is SYNTHETIC test data (tones / silence generated in a
temp dir, source_id=synthetic_test). None of it is speech and none of it yields metrics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from whisperflow.data.audio import sha256_file

REPO = Path(__file__).resolve().parents[1]


def write_wav(path: Path, seconds: float = 2.0, sr: int = 16000, kind: str = "tone",
              channels: int = 1, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    if kind == "tone":
        x = 0.2 * np.sin(2 * np.pi * (220 + 20 * seed) * t) + 0.01 * rng.standard_normal(n)
    elif kind == "silence":
        x = np.zeros(n)
    elif kind == "clipped":
        x = np.clip(2.0 * np.sin(2 * np.pi * 220 * t), -1, 1)
    else:
        raise ValueError(kind)
    if channels > 1:
        x = np.stack([x] * channels, axis=1)
    sf.write(path, x.astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A temp repo root with configs copied in and an empty data/private."""
    (tmp_path / "configs").mkdir()
    for name in ("dataset.yaml", "sources.yaml", "eval.yaml"):
        shutil.copy(REPO / "configs" / name, tmp_path / "configs" / name)
    reg = yaml.safe_load((tmp_path / "configs/sources.yaml").read_text())
    reg["sources"]["pending_corpus"] = {
        "status": "pending_review", "allowed_splits": ["train", "val", "test", "unassigned"],
        "license_or_consent_ids": ["pending_lic"]}
    reg["sources"]["train_only_corpus"] = {
        "status": "approved", "allowed_splits": ["train", "unassigned"],
        "license_or_consent_ids": ["to_lic"]}
    (tmp_path / "configs/sources.yaml").write_text(yaml.safe_dump(reg))
    (tmp_path / "data/private").mkdir(parents=True)
    (tmp_path / "data/manifests").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cfg(repo) -> dict:
    return yaml.safe_load((repo / "configs/dataset.yaml").read_text())


def make_record(repo: Path | None = None, uid: str = "u000001", *, audio: str | None = None,
                seconds: float = 2.0, kind: str = "tone", seed: int = 0, **over) -> dict:
    """Build a manifest dict. If `repo` is given, a real synthetic wav is written and hashed."""
    audio = audio or f"synthetic/{uid}.wav"
    rec = {
        "utterance_id": uid,
        "audio_path": audio,
        "sha256": "0" * 64,
        "duration_sec": seconds,
        "speaker_id": "s001",
        "session_id": "sess001",
        "source_id": "synthetic_test",
        "license_or_consent_id": "synthetic",
        "split": "train",
        "raw_transcript": "我们先跑一下 benchmark 然后看 CER",
        "formatted_transcript": "我们先跑一下 Benchmark，然后看 CER。",
        "primary_scenario": "zh_en_switch",
        "language_tags": ["zh", "en"],
        "hotwords_present": ["benchmark", "CER"],
        "recording_device": "synthetic",
        "noise_tag": "quiet",
        "label_status": "human_verified",
        "reviewer_id": "reviewer_test",
        "quality_flags": [],
    }
    if repo is not None:
        p = write_wav(repo / "data/private" / audio, seconds=seconds, kind=kind, seed=seed)
        rec["sha256"] = sha256_file(p)
    rec.update(over)
    return rec


def write_jsonl(path: Path, rows: list[dict | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")
    return path
