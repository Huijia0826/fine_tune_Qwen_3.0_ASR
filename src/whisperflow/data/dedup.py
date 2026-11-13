"""Duplicate detection.

Tiers (strongest first):
  1. exact_audio      identical sha256                         -> error; keep first, drop rest
  2. likely_reencode  same speaker + same normalized transcript + |Δduration| < 0.1 s
                      (same recording re-encoded / resampled)  -> warning; review
  3. same_transcript  identical normalized transcript across records -> info for leakage
                      analysis. Normal in read-speech corpora (shared prompts), but if it
                      spans train and test it can inflate test scores via text memorization.

Not implemented yet: acoustic fingerprinting (e.g. chromaprint) for re-recorded or
trimmed copies. Add before mixing corpora that may share source material.

Usage:
    wf-dedup data/manifests/x.jsonl [--out deduped.jsonl] [--report dedup.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from whisperflow.data.schema import Utterance, load_manifest, write_manifest
from whisperflow.text.normalize import normalize


def find_duplicates(records: list[Utterance]) -> dict:
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_text: dict[str, list[Utterance]] = defaultdict(list)
    for r in records:
        by_hash[r.sha256].append(r.utterance_id)
        by_text[normalize(r.raw_transcript)].append(r)

    exact = [ids for ids in by_hash.values() if len(ids) > 1]

    reencode = []
    same_text = []
    for text, group in by_text.items():
        if len(group) < 2:
            continue
        same_text.append({
            "normalized_text": text,
            "utterance_ids": [r.utterance_id for r in group],
            "n_speakers": len({r.speaker_id for r in group}),
            "splits": sorted({r.split for r in group}),
        })
        by_spk: dict[str, list[Utterance]] = defaultdict(list)
        for r in group:
            by_spk[r.speaker_id].append(r)
        for spk_group in by_spk.values():
            for a, b in combinations(spk_group, 2):  # groups are tiny; all pairs is fine
                if a.sha256 != b.sha256 and abs(a.duration_sec - b.duration_sec) < 0.1:
                    reencode.append([a.utterance_id, b.utterance_id])

    return {"exact_audio": exact, "likely_reencode": reencode, "same_transcript": same_text}


def drop_exact_duplicates(records: list[Utterance]) -> tuple[list[Utterance], list[str]]:
    seen: set[str] = set()
    kept, dropped = [], []
    for r in records:
        if r.sha256 in seen:
            dropped.append(r.utterance_id)
        else:
            seen.add(r.sha256)
            kept.append(r)
    return kept, dropped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", help="write manifest with exact-audio duplicates removed")
    ap.add_argument("--report", help="write JSON duplicate report")
    args = ap.parse_args(argv)

    records = load_manifest(args.manifest)
    dups = find_duplicates(records)
    cross_split = [g for g in dups["same_transcript"] if len(set(g["splits"]) - {"unassigned"}) > 1]
    print(f"{len(records)} records")
    print(f"  exact audio duplicate groups : {len(dups['exact_audio'])}")
    print(f"  likely re-encoded pairs      : {len(dups['likely_reencode'])}")
    print(f"  repeated-transcript groups   : {len(dups['same_transcript'])} "
          f"({len(cross_split)} span multiple splits)")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(dups, f, ensure_ascii=False, indent=2)
    if args.out:
        kept, dropped = drop_exact_duplicates(records)
        write_manifest(kept, args.out)
        print(f"Wrote {len(kept)} records to {args.out} (dropped {len(dropped)} exact duplicates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
