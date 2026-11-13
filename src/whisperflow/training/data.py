"""Training data for Whisper LoRA: manifest records -> (log-mel features, label ids) -> padded batches.

Practice targets (docs/learning/training_walkthrough.md): `make_labels` and `WhisperCollator`.

Label layout (what the decoder must learn to produce, one token at a time):

    <|startoftranscript|> <|zh|> <|transcribe|> <|notimestamps|>  我 们 先 跑 ...  <|endoftext|>
    \\____ dropped ____/  \\______________ labels (loss is computed on these) ______________/

The model builds its decoder input by shifting the labels right and putting <|startoftranscript|> in
front (`shift_tokens_right`), so the labels must NOT start with <|startoftranscript|> themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from whisperflow.data.schema import Utterance, load_manifest

IGNORE_INDEX = -100  # CrossEntropyLoss skips label positions with this value
TIMESTAMP_STEP = 0.02  # Whisper timestamp tokens: <|0.00|>, <|0.02|>, ..., <|30.00|>


def clip_language(rec: Utterance) -> str:
    """Language token for a clip: pure-English clips get <|en|>, everything else (zh, zh+en) <|zh|>.

    Code-switched speech is mostly Mandarin with English words; Whisper's own language ID calls it zh.
    Tagging pure-English clips as zh would teach the model "English audio -> zh mode".
    """
    return "en" if set(rec.language_tags) == {"en"} else "zh"


MAX_AUDIO_SEC = 30.0  # Whisper's encoder window


def load_examples(manifests: list[str], split: str) -> list[Utterance]:
    out: list[Utterance] = []
    for m in manifests:
        out += [r for r in load_manifest(m) if r.split == split]
    return out


def usable_for_loss(records: list[Utterance], tokenizer, max_tokens: int = 448) -> tuple[list[Utterance], int]:
    """Keep clips that fit Whisper's training window: audio <= 30 s and label <= max_tokens.

    Checked once before training so a bad clip fails at second 0, not at step 200. Longer val clips are
    still used later by wf-transcribe (long-form decoding), just not for the in-training val loss.
    """
    keep = []
    for r in records:
        if r.duration_sec > MAX_AUDIO_SEC:
            continue
        try:  # check the longer (timestamped) label form
            make_labels(tokenizer, r.raw_transcript, clip_language(r), max_tokens, end_sec=r.duration_sec)
        except ValueError:
            continue
        keep.append(r)
    return keep, len(records) - len(keep)


def timestamp_token(tokenizer, sec: float) -> int:
    sec = min(max(sec, 0.0), 30.0)
    return tokenizer.convert_tokens_to_ids("<|0.00|>") + round(sec / TIMESTAMP_STEP)


def make_labels(tokenizer, text: str, language: str, max_tokens: int = 448,
                end_sec: float | None = None) -> list[int]:
    """Token ids the decoder is trained to predict, WITHOUT the leading <|startoftranscript|>.

    end_sec=None -> no-timestamp form:  <|zh|><|transcribe|><|notimestamps|> text <|endoftext|>
    end_sec=2.3  -> timestamp form:     <|zh|><|transcribe|><|0.00|> text <|2.30|> <|endoftext|>
    Training on both keeps Whisper's timestamp prediction alive; long-form decoding depends on it
    (lora_v1 trained without timestamps and broke on clips > 30 s).
    """
    head = tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{language}|>", "<|transcribe|>"])
    body = tokenizer.encode(" " + text.strip() if language == "en" else text.strip(), add_special_tokens=False)
    if end_sec is None:
        ids = head + [tokenizer.convert_tokens_to_ids("<|notimestamps|>")] + body + [tokenizer.eos_token_id]
    else:
        ids = (head + [timestamp_token(tokenizer, 0.0)] + body
               + [timestamp_token(tokenizer, end_sec), tokenizer.eos_token_id])
    if len(ids) > max_tokens:
        raise ValueError(f"label has {len(ids)} tokens > {max_tokens}: {text[:40]}...")
    return ids[1:]


class WhisperDataset(torch.utils.data.Dataset):
    """One item = {"input_features": float32 [n_mels, 3000], "labels": list[int]}; audio is read lazily.

    n_mels comes from the processor: 80 for small / medium, 128 for large-v3."""

    def __init__(self, records: list[Utterance], processor, data_root: str | Path,
                 max_label_tokens: int = 448, timestamp_ratio: float = 0.0):
        self.records, self.processor = records, processor
        self.data_root, self.max_label_tokens = Path(data_root), max_label_tokens
        self.timestamp_ratio = timestamp_ratio

    def uses_timestamps(self, rec: Utterance) -> bool:
        """Deterministic per clip (hash of its id), so the same clip always gets the same label form."""
        return int(rec.sha256[:8], 16) / 0xFFFFFFFF < self.timestamp_ratio

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        audio, sr = sf.read(self.data_root / rec.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:  # manifests are 16 kHz after ingestion; fail loudly instead of resampling silently
            raise ValueError(f"{rec.audio_path}: {sr} Hz, expected 16000")
        feats = self.processor.feature_extractor(audio, sampling_rate=16000,
                                                 return_tensors="np").input_features[0]
        end = rec.duration_sec if self.uses_timestamps(rec) else None
        labels = make_labels(self.processor.tokenizer, rec.raw_transcript, clip_language(rec),
                             self.max_label_tokens, end_sec=end)
        return {"input_features": feats.astype(np.float32), "labels": labels}


@dataclass
class WhisperCollator:
    """Stack features; right-pad labels with IGNORE_INDEX so padding never contributes to the loss."""

    def __call__(self, batch: list[dict]) -> dict:
        feats = torch.from_numpy(np.stack([b["input_features"] for b in batch]))  # always [n_mels, 3000]
        longest = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
        for row, b in enumerate(batch):
            labels[row, : len(b["labels"])] = torch.tensor(b["labels"], dtype=torch.long)
        return {"input_features": feats, "labels": labels}
