"""Apply owner transcript corrections (configs/label_corrections.yaml) to manifest records."""

from __future__ import annotations

from pathlib import Path

from whisperflow.data.common import load_yaml


def apply_corrections(records: list[dict], path: str | Path) -> tuple[list[dict], int, list[str]]:
    """Rewrite raw_transcript in place for records whose wav stem has a correction; `drop: true` removes them.

    Returns (kept records, applied count, warnings). A correction whose `replaces` text no longer matches the record is
    skipped with a warning, so a re-cut piece never silently gets text written for different audio.
    """
    table = load_yaml(path).get("corrections") or {}
    kept, applied, warnings, seen = [], 0, [], set()
    for r in records:
        key = Path(r["audio_path"]).stem
        c = table.get(key)
        if c is None:
            kept.append(r)
            continue
        seen.add(key)
        if c.get("drop"):
            applied += 1
            continue
        kept.append(r)
        if r["raw_transcript"] not in (c["replaces"], c["text"]):
            warnings.append(f"{key}: reference slice changed since the correction was written; not applied")
            continue
        r["raw_transcript"] = c["text"]
        r["label_status"] = "human_verified"
        r["reviewer_id"] = c["reviewer_id"]
        r["quality_flags"] = [f for f in r["quality_flags"] if f != "label_corrected"] + ["label_corrected"]
        applied += 1
    warnings += [f"{k}: no matching record" for k in table.keys() - seen]
    return kept, applied, warnings
