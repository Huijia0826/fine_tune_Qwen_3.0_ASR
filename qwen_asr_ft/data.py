"""Manifests → Qwen3-ASR SFT jsonl.

Input manifest: one JSON object per line (see docs/DATA_FORMAT.md). Required: `audio`, `text`.
Optional: `id`, `speaker`, `language` (zh / en / Chinese / English / auto), `weight` (integer repeats),
`context` (free text) or structured context fields (`app`, `tone`, `vocabulary`, `before_cursor`, `after_cursor`;
names configurable), and any other fields you want to slice results by.

Output row (what the trainer reads):
    {"audio": "/abs/path.wav", "prompt": "<context or empty>", "text": "language Chinese<asr_text><label>", "id": "…"}
"""

from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path

from .context import Augmenter, build_context, fields, label_terms
from .text import has_cjk


def read_manifest(path: str | Path, audio_root: str | None = None) -> list[dict]:
    """Rows with `audio` resolved to an absolute path (relative paths: against audio_root, else the manifest's
    folder) and `id` filled in (manifest name + line number) when missing."""
    path = Path(path)
    root = Path(audio_root) if audio_root else path.parent
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if "audio" not in r or "text" not in r:
                raise ValueError(f"{path}:{i + 1}: every row needs 'audio' and 'text'")
            a = Path(r["audio"]).expanduser()
            r["audio"] = str(a if a.is_absolute() else (root / a).resolve())
            r.setdefault("id", f"{path.stem}-{i:06d}")
            r.setdefault("speaker", r["id"])
            rows.append(r)
    return rows


def read_manifests(paths, audio_root=None) -> list[dict]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = [r for p in paths for r in read_manifest(p, audio_root)]
    seen, dup = set(), set()
    for r in rows:
        (dup if r["id"] in seen else seen).add(r["id"])
    if dup:
        raise ValueError(f"duplicate ids across {list(map(str, paths))}: {sorted(dup)[:5]} … — ids must be unique")
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def language_name(row: dict, dcfg: dict) -> str:
    names = dcfg.get("language_names", {"zh": "Chinese", "en": "English"})
    lang = str(row.get("language") or dcfg.get("default_language", "auto"))
    if lang in names:
        return names[lang]
    if lang != "auto":
        return lang  # already a Qwen language name, e.g. "Cantonese"
    return names.get("zh", "Chinese") if has_cjk(row["text"]) else names.get("en", "English")


def load_normalizer(spec: str | None):
    """'package.module:function' → callable(str) -> str, or identity."""
    if not spec:
        return lambda s: s
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def audio_duration(path: str) -> float | None:
    try:
        import soundfile as sf

        return sf.info(path).duration
    except Exception:  # noqa: BLE001 - unreadable by soundfile (e.g. mp3 without libsndfile support)
        return None


def _read_lines(path: str | None) -> list[str]:
    if not path:
        return []
    return [w.strip() for w in Path(path).read_text(encoding="utf-8").splitlines() if w.strip()]


def build_sft(rows: list[dict], cfg: dict, train: bool) -> tuple[list[dict], Counter]:
    """SFT rows + counters. Training rows can be context-augmented and repeated by `weight`."""
    dcfg, ccfg, acfg = cfg["data"], cfg["data"]["context"], cfg["data"]["augment"]
    norm = load_normalizer(dcfg.get("normalize_label"))
    stats: Counter = Counter()
    aug = None
    if train and acfg.get("enabled"):
        pool = _read_lines(acfg.get("distractor_pool"))
        for r in rows:
            pool += label_terms(r["text"])
            pool += fields(r, ccfg)[1]
        aug = Augmenter(acfg, ccfg, pool, set(_read_lines(acfg.get("banned_terms"))))
        cursor_pool = [(b, a) for r in rows for _, _, b, a in [fields(r, ccfg)] if b or a]
    max_sec = float(dcfg.get("max_audio_sec") or 0)
    prev_by_spk: dict[str, str] = {}
    out = []
    for r in rows:
        if not Path(r["audio"]).exists():
            stats["missing_audio"] += 1
            continue
        if max_sec:
            dur = audio_duration(r["audio"])
            if dur is not None and dur > max_sec:
                stats["too_long"] += 1
                continue
        label = norm(r["text"]).strip()
        if not label:
            stats["empty_label"] += 1
            continue
        if aug:
            other = aug.rng.choice(cursor_pool) if cursor_pool else None
            ctx = aug.context(r, label, prev_by_spk.get(r["speaker"], ""), other)
        else:
            ctx = build_context(r, ccfg, label)
        prev_by_spk[r["speaker"]] = label
        stats["with_context" if ctx else "no_context"] += 1
        item = {"audio": r["audio"], "prompt": ctx, "text": f"language {language_name(r, dcfg)}<asr_text>{label}",
                "id": r["id"]}
        repeats = max(1, int(r.get("weight", 1))) if train else 1
        out.extend([item] * repeats)
        stats["rows"] += repeats
        stats["clips"] += 1
    return out, stats


def prepare(cfg: dict) -> dict:
    """Write <work_dir>/sft/{train,val}.jsonl and return a summary."""
    work = Path(cfg["work_dir"]) / "sft"
    dcfg = cfg["data"]
    summary = {}
    for split, is_train in (("train", True), ("val", False)):
        if not dcfg.get(split):
            continue
        rows = read_manifests(dcfg[split], dcfg.get("audio_root"))
        sft, stats = build_sft(rows, cfg, train=is_train)
        write_jsonl(sft, work / f"{split}.jsonl")
        summary[split] = dict(stats)
    (work / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
