"""Manifest schema: one JSON object per utterance (JSONL).

Two classes of problems are distinguished everywhere in the data pipeline:
  - errors:   the record is unusable / untraceable / would corrupt evaluation. Block.
  - warnings: suspicious but possibly legitimate; must be reviewed, not silently dropped.

Record-level checks live here. Checks that need other records (uniqueness, leakage) or
the filesystem (audio decode, hashes) live in validate.py / audit.py.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from whisperflow.text.normalize import contains_term, detect_languages, normalize

SCHEMA_VERSION = "1.0"

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Split(StrEnum):
    train = "train"
    val = "val"
    test = "test"
    unassigned = "unassigned"


class Scenario(StrEnum):
    """Mutually exclusive primary label (plan.md §3.1)."""

    zh_general = "zh_general"
    zh_en_switch = "zh_en_switch"
    technical_terms = "technical_terms"
    self_correction = "self_correction"
    dictation_formatting = "dictation_formatting"
    en_general = "en_general"  # English-only speech (benchmark v1.0 is ~65% English)


class LabelStatus(StrEnum):
    # Transcript came with a public corpus; not reviewed by us. Usable for training,
    # but test-set records should be human_verified (or explicitly accepted).
    source_provided = "source_provided"
    machine_draft = "machine_draft"  # pseudo-label; never trusted for train/val/test
    human_verified = "human_verified"
    rejected = "rejected"


class NoiseTag(StrEnum):
    quiet = "quiet"
    moderate = "moderate"
    noisy = "noisy"
    unknown = "unknown"


class LangTag(StrEnum):
    zh = "zh"
    en = "en"
    other = "other"


class Utterance(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    utterance_id: str
    audio_path: str
    sha256: str
    duration_sec: float = Field(gt=0)
    speaker_id: str
    session_id: str
    source_id: str
    license_or_consent_id: str
    split: Split
    raw_transcript: str
    formatted_transcript: str | None = None
    primary_scenario: Scenario
    language_tags: list[LangTag] = Field(min_length=1)
    hotwords_present: list[str] = Field(default_factory=list)
    recording_device: str = "unknown"
    noise_tag: NoiseTag = NoiseTag.unknown
    label_status: LabelStatus
    reviewer_id: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    # Optional traceability fields (not in the plan's minimum list).
    source_utterance_id: str | None = None  # ID in the original corpus
    parent_utterance_id: str | None = None  # set for augmented/derived variants
    sample_rate_hz: int | None = None
    num_channels: int | None = None

    @field_validator("utterance_id", "speaker_id", "session_id", "source_id",
                     "license_or_consent_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError(f"invalid identifier {v!r} (allowed: [A-Za-z0-9_.-], max 128)")
        return v

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, v: str) -> str:
        if not SHA256_RE.match(v):
            raise ValueError("sha256 must be 64 lowercase hex chars")
        return v

    @field_validator("raw_transcript")
    @classmethod
    def _check_transcript(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_transcript is empty")
        if not normalize(v):
            raise ValueError("raw_transcript has no scoreable tokens after normalization")
        return v

    @field_validator("audio_path")
    @classmethod
    def _check_path(cls, v: str) -> str:
        if Path(v).is_absolute():
            raise ValueError("audio_path must be relative to the data root (portability)")
        if ".." in Path(v).parts:
            raise ValueError("audio_path must not contain '..'")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> Utterance:
        if self.label_status == LabelStatus.human_verified and not self.reviewer_id:
            raise ValueError("label_status=human_verified requires reviewer_id")
        if self.primary_scenario == Scenario.zh_en_switch and LangTag.en not in self.language_tags:
            raise ValueError("primary_scenario=zh_en_switch requires 'en' in language_tags")
        if len(set(self.language_tags)) != len(self.language_tags):
            raise ValueError("duplicate language_tags")
        return self

    def soft_warnings(self, max_duration_sec: float) -> list[str]:
        """Suspicious-but-maybe-legitimate conditions for human review."""
        w: list[str] = []
        observed = detect_languages(self.raw_transcript)
        declared = set(self.language_tags)
        for lang in observed - declared:
            w.append(f"transcript contains '{lang}' tokens not in language_tags")
        for lang in (declared & {"zh", "en"}) - observed:
            w.append(f"language_tags declares '{lang}' but transcript has none")
        for hw in self.hotwords_present:
            if not contains_term(self.raw_transcript, hw):
                w.append(f"hotword {hw!r} not found in raw_transcript")
        if self.duration_sec > max_duration_sec:
            w.append(f"duration {self.duration_sec:.1f}s exceeds {max_duration_sec}s Whisper window")
        if self.duration_sec < 0.5:
            w.append(f"very short clip ({self.duration_sec:.2f}s)")
        # Characters per second sanity: Mandarin read speech is ~3-6 chars/s.
        n_tok = len(normalize(self.raw_transcript).split())
        rate = n_tok / self.duration_sec
        if rate > 12:
            w.append(f"implausibly fast token rate {rate:.1f}/s (misaligned transcript?)")
        elif self.duration_sec > 3 and rate < 0.5:
            w.append(f"implausibly slow token rate {rate:.2f}/s (truncated transcript / long silence?)")
        if self.label_status == LabelStatus.machine_draft and self.split != Split.unassigned:
            w.append("machine_draft label assigned to a split; pseudo-labels are not trusted")
        if self.split == Split.test and self.label_status != LabelStatus.human_verified:
            w.append("test record is not human_verified")
        return w


class ManifestRecordError(Exception):
    pass


def iter_manifest(path: str | Path) -> Iterator[tuple[int, dict | None, str | None]]:
    """Yield (line_no, parsed_json_or_None, json_error_or_None). Blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                yield i, None, f"invalid JSON: {e}"
                continue
            if not isinstance(obj, dict):
                yield i, None, "line is not a JSON object"
                continue
            yield i, obj, None


def parse_record(obj: dict) -> tuple[Utterance | None, list[str]]:
    try:
        return Utterance.model_validate(obj), []
    except ValidationError as e:
        errs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "<record>"
            errs.append(f"{loc}: {err['msg']}")
        return None, errs


def load_manifest(path: str | Path) -> list[Utterance]:
    """Strict loader for downstream stages: raises on the first invalid record."""
    out = []
    for line_no, obj, jerr in iter_manifest(path):
        if jerr:
            raise ManifestRecordError(f"{path}:{line_no}: {jerr}")
        rec, errs = parse_record(obj)
        if errs:
            raise ManifestRecordError(f"{path}:{line_no}: {'; '.join(errs)}")
        out.append(rec)
    return out


def write_manifest(records: list[Utterance], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
