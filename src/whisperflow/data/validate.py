"""Validate a manifest: schema, provenance/licensing, audio integrity, signal quality.

Usage:
    wf-validate data/manifests/pilot.jsonl [--config configs/dataset.yaml] [--report out.json]

Exit code is 1 if any *error* is found, so it can gate CI and downstream stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from whisperflow.data.audio import probe, sha256_file, signal_stats
from whisperflow.data.common import IssueLog, load_yaml
from whisperflow.data.schema import Utterance, iter_manifest, parse_record


def check_provenance(rec: Utterance, registry: dict, log: IssueLog, line: int) -> None:
    src = registry.get(rec.source_id)
    uid = rec.utterance_id
    if src is None:
        log.error("source_unknown", f"source_id {rec.source_id!r} not in registry", uid, line)
        return
    status = src.get("status")
    if status != "approved":
        log.error("source_not_approved", f"source {rec.source_id!r} status={status}", uid, line)
    if rec.split not in src.get("allowed_splits", []):
        log.error("split_not_allowed",
                  f"source {rec.source_id!r} not allowed in split {rec.split!r}", uid, line)
    if rec.license_or_consent_id not in src.get("license_or_consent_ids", []):
        log.error("license_id_unknown",
                  f"license_or_consent_id {rec.license_or_consent_id!r} not registered "
                  f"for source {rec.source_id!r}", uid, line)


def check_audio(rec: Utterance, data_root: Path, vcfg: dict, log: IssueLog, line: int,
                verify_hash: bool, signal_checks: bool) -> None:
    uid = rec.utterance_id
    path = data_root / rec.audio_path
    if not path.is_file():
        log.error("audio_missing", f"{rec.audio_path} not found under {data_root}", uid, line)
        return
    try:
        info = probe(path)
    except Exception as e:  # soundfile raises various types for corrupt files
        log.error("audio_unreadable", f"{rec.audio_path}: {e}", uid, line)
        return
    if info.frames == 0:
        log.error("audio_empty", f"{rec.audio_path} has zero frames", uid, line)
        return
    if verify_hash:
        actual = sha256_file(path)
        if actual != rec.sha256:
            log.error("sha256_mismatch", f"manifest {rec.sha256[:12]}… file {actual[:12]}…",
                      uid, line)
    tol = vcfg.get("duration_tolerance_sec", 0.05)
    if abs(info.duration_sec - rec.duration_sec) > tol:
        log.error("duration_mismatch",
                  f"manifest {rec.duration_sec:.3f}s vs audio {info.duration_sec:.3f}s", uid, line)
    if info.duration_sec < vcfg.get("min_duration_sec", 0.3):
        log.error("audio_too_short", f"{info.duration_sec:.3f}s", uid, line)
    exp_sr = vcfg.get("expected_sample_rate_hz")
    if exp_sr and info.sample_rate_hz != exp_sr:
        log.warn("sample_rate", f"{info.sample_rate_hz} Hz (expected {exp_sr}; will resample)",
                 uid, line)
    if info.num_channels != 1:
        log.warn("multichannel", f"{info.num_channels} channels (will downmix)", uid, line)
    if info.sample_rate_hz < 16000:
        log.warn("low_sample_rate", f"{info.sample_rate_hz} Hz < 16 kHz (narrowband)", uid, line)

    if signal_checks:
        st = signal_stats(path, vcfg.get("silence_dbfs", -50.0))
        if st.silent:
            log.error("audio_silent", f"RMS {st.rms_dbfs:.1f} dBFS", uid, line)
        if st.clipped_fraction > vcfg.get("clipped_fraction_warn", 0.001):
            log.warn("clipping", f"{st.clipped_fraction:.2%} samples at full scale", uid, line)
        edge = vcfg.get("max_edge_silence_sec", 3.0)
        if st.leading_silence_sec > edge or st.trailing_silence_sec > edge:
            log.warn("edge_silence",
                     f"leading {st.leading_silence_sec:.1f}s / trailing "
                     f"{st.trailing_silence_sec:.1f}s silence", uid, line)


def validate_manifest(manifest: str | Path, cfg: dict, *, check_files: bool = True,
                      verify_hash: bool = True, signal_checks: bool = True,
                      repo_root: Path | None = None) -> tuple[list[Utterance], IssueLog]:
    repo_root = repo_root or Path.cwd()
    log = IssueLog()
    registry = load_yaml(repo_root / cfg["sources_registry"]).get("sources", {})
    data_root = repo_root / cfg["data_root"]
    vcfg = cfg.get("validation", {})
    max_dur = vcfg.get("max_duration_sec", 30.0)

    records: list[Utterance] = []
    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, str] = {}
    for line, obj, jerr in iter_manifest(manifest):
        if jerr:
            log.error("invalid_json", jerr, None, line)
            continue
        rec, errs = parse_record(obj)
        if errs:
            uid = obj.get("utterance_id") if isinstance(obj.get("utterance_id"), str) else None
            for e in errs:
                log.error("schema", e, uid, line)
            continue
        uid = rec.utterance_id
        if uid in seen_ids:
            log.error("duplicate_id", f"also on line {seen_ids[uid]}", uid, line)
            continue
        seen_ids[uid] = line
        if rec.audio_path in seen_paths:
            log.error("duplicate_audio_path", f"same file as {seen_paths[rec.audio_path]}", uid, line)
        seen_paths[rec.audio_path] = uid
        if rec.label_status == "rejected" and rec.split != "unassigned":
            log.error("rejected_in_split", f"rejected record in split {rec.split!r}", uid, line)

        check_provenance(rec, registry, log, line)
        if check_files:
            check_audio(rec, data_root, vcfg, log, line, verify_hash, signal_checks)
        for w in rec.soft_warnings(max_dur):
            log.warn("label_review", w, uid, line)
        records.append(rec)
    return records, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--report", help="write JSON issue report here")
    ap.add_argument("--no-files", action="store_true", help="schema/provenance checks only")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 verification")
    ap.add_argument("--no-signal", action="store_true", help="skip full-decode signal checks")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    records, log = validate_manifest(args.manifest, cfg, check_files=not args.no_files,
                                     verify_hash=not args.no_hash,
                                     signal_checks=not args.no_signal)
    print(f"Validated {args.manifest}: {len(records)} schema-valid record(s)")
    log.print()
    if args.report:
        log.write_json(args.report)
    return 1 if log.errors else 0


if __name__ == "__main__":
    sys.exit(main())
