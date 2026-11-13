"""Ingest a folder of audio + a metadata CSV/TSV into manifest records.

Input metadata columns (header row required):
    required: audio_path (relative to --audio-dir), transcript, speaker_id
    optional: session_id, formatted_transcript, primary_scenario, language_tags (space-sep),
              hotwords_present ('|'-sep), recording_device, noise_tag, source_utterance_id,
              reviewer_id, parent_utterance_id

Usage:
    wf-ingest --metadata meta.tsv --audio-dir /path/to/corpus --source-id aishell1 \
        --license-id aishell1_apache2 --label-status source_provided \
        --out data/manifests/aishell1_sample.jsonl [--convert-16k] [--limit 100]

Design notes:
  * utterance_id = "<source_id>-<sha256[:16]>" -> content-addressed, stable across re-runs,
    and identical audio from two places collides loudly instead of silently duplicating.
  * Every new record starts with split="unassigned"; splitting is a separate, audited step.
  * Corpus-specific adapters should only produce the metadata table; everything else
    (hashing, probing, ID assignment) goes through this one code path.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from whisperflow.data.audio import convert_to_16k_wav, probe, sha256_file
from whisperflow.data.common import load_yaml
from whisperflow.data.schema import Scenario, Utterance, parse_record, write_manifest
from whisperflow.text.normalize import detect_languages


def _infer_scenario(langs: set[str]) -> str:
    if {"zh", "en"} <= langs:
        return Scenario.zh_en_switch
    if langs == {"en"}:
        return Scenario.en_general
    return Scenario.zh_general


def ingest(metadata: Path, audio_dir: Path, data_root: Path, *, source_id: str, license_id: str,
           label_status: str, convert_16k: bool = False, limit: int | None = None,
           default_device: str = "unknown") -> tuple[list[Utterance], list[str]]:
    delim = "\t" if metadata.suffix.lower() in {".tsv", ".tab"} else ","
    records: list[Utterance] = []
    problems: list[str] = []
    with open(metadata, encoding="utf-8", newline="") as f:
        for row_no, row in enumerate(csv.DictReader(f, delimiter=delim), start=2):
            if limit is not None and len(records) >= limit:
                break
            src = audio_dir / row["audio_path"]
            if not src.is_file():
                problems.append(f"row {row_no}: missing audio {src}")
                continue
            flags: list[str] = []
            try:
                if convert_16k:
                    rel = Path(source_id) / Path(row["audio_path"]).with_suffix(".wav")
                    dst = data_root / rel
                    convert_to_16k_wav(src, dst)
                    flags.append("converted_16k_mono")
                else:
                    dst = src
                    rel = dst.resolve().relative_to(data_root.resolve())
                info = probe(dst)
                digest = sha256_file(dst)
            except ValueError:
                problems.append(f"row {row_no}: {src} is outside data_root {data_root}; "
                                "use --convert-16k or place audio under data_root")
                continue
            except Exception as e:
                problems.append(f"row {row_no}: cannot read {src}: {e}")
                continue

            transcript = row["transcript"]
            langs = detect_languages(transcript)
            lang_tags = (row.get("language_tags") or "").split() or sorted(langs) or ["other"]
            scenario = row.get("primary_scenario") or None
            if not scenario:
                scenario = _infer_scenario(langs)
                flags.append("scenario_inferred")
            obj = {
                "utterance_id": f"{source_id}-{digest[:16]}",
                "audio_path": str(rel),
                "sha256": digest,
                "duration_sec": round(info.duration_sec, 4),
                "speaker_id": f"{source_id}-{row['speaker_id']}",
                "session_id": f"{source_id}-{row.get('session_id') or row['speaker_id']}",
                "source_id": source_id,
                "license_or_consent_id": license_id,
                "split": "unassigned",
                "raw_transcript": transcript,
                "formatted_transcript": row.get("formatted_transcript") or None,
                "primary_scenario": scenario,
                "language_tags": lang_tags,
                "hotwords_present": [h for h in (row.get("hotwords_present") or "").split("|") if h],
                "recording_device": row.get("recording_device") or default_device,
                "noise_tag": row.get("noise_tag") or "unknown",
                "label_status": label_status,
                "reviewer_id": row.get("reviewer_id") or None,
                "quality_flags": flags,
                "source_utterance_id": row.get("source_utterance_id") or None,
                "parent_utterance_id": row.get("parent_utterance_id") or None,
                "sample_rate_hz": info.sample_rate_hz,
                "num_channels": info.num_channels,
            }
            rec, errs = parse_record(obj)
            if errs:
                problems.append(f"row {row_no}: " + "; ".join(errs))
                continue
            records.append(rec)
    return records, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--license-id", required=True)
    ap.add_argument("--label-status", default="source_provided")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--convert-16k", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="unknown", help="default recording_device")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    records, problems = ingest(args.metadata, args.audio_dir, Path(cfg["data_root"]),
                               source_id=args.source_id, license_id=args.license_id,
                               label_status=args.label_status, convert_16k=args.convert_16k,
                               limit=args.limit, default_device=args.device)
    for p in problems:
        print(f"[SKIP] {p}")
    write_manifest(records, args.out)
    print(f"Wrote {len(records)} record(s) to {args.out}; skipped {len(problems)}")
    print("Next: wf-validate", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
