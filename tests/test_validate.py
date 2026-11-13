from tests.conftest import make_record, write_jsonl, write_wav
from whisperflow.data.validate import main, validate_manifest


def codes(log, level="error"):
    return {i.code for i in log.issues if i.level == level}


def test_clean_manifest_passes(repo, cfg):
    m = write_jsonl(repo / "m.jsonl", [make_record(repo, "u1", seed=1),
                                       make_record(repo, "u2", seed=2)])
    recs, log = validate_manifest(m, cfg)
    assert len(recs) == 2 and log.errors == [], log.issues
    assert main([str(m)]) == 0


def test_detects_broken_audio_and_integrity(repo, cfg):
    good = make_record(repo, "good", seed=1)
    missing = make_record(None, "missing")
    corrupt = make_record(repo, "corrupt", seed=2)
    (repo / "data/private" / corrupt["audio_path"]).write_bytes(b"not a wav file")
    badhash = make_record(repo, "badhash", seed=3, sha256="f" * 64)
    baddur = make_record(repo, "baddur", seed=4, duration_sec=3.5)
    silent = make_record(repo, "silent", kind="silence")
    m = write_jsonl(repo / "m.jsonl", [good, missing, corrupt, badhash, baddur, silent])
    _, log = validate_manifest(m, cfg)
    assert {"audio_missing", "audio_unreadable", "sha256_mismatch", "duration_mismatch",
            "audio_silent"} <= codes(log)
    flagged = {i.utterance_id for i in log.errors}
    assert "good" not in flagged
    assert main([str(m)]) == 1


def test_signal_warnings(repo, cfg):
    clipped = make_record(repo, "clip", kind="clipped")
    rec44 = make_record(None, "sr44", audio="synthetic/sr44.wav", seconds=2.0)
    p = write_wav(repo / "data/private/synthetic/sr44.wav", 2.0, sr=44100, channels=2)
    from whisperflow.data.audio import sha256_file
    rec44["sha256"] = sha256_file(p)
    m = write_jsonl(repo / "m.jsonl", [clipped, rec44])
    _, log = validate_manifest(m, cfg)
    assert {"clipping", "sample_rate", "multichannel"} <= codes(log, "warning")
    assert log.errors == []


def test_provenance_checks(repo, cfg):
    rows = [
        make_record(repo, "unknown_src", seed=1, source_id="mystery"),
        make_record(repo, "pending", seed=2, source_id="pending_corpus",
                    license_or_consent_id="pending_lic"),
        make_record(repo, "wrong_split", seed=3, source_id="train_only_corpus",
                    license_or_consent_id="to_lic", split="test"),
        make_record(repo, "wrong_lic", seed=4, license_or_consent_id="other"),
    ]
    _, log = validate_manifest(write_jsonl(repo / "m.jsonl", rows), cfg)
    assert {"source_unknown", "source_not_approved", "split_not_allowed",
            "license_id_unknown"} <= codes(log)


def test_json_schema_duplicates_and_rejected(repo, cfg):
    a = make_record(repo, "dup", seed=1)
    rejected = make_record(repo, "rej", seed=2, label_status="rejected")
    rows = ["{not json", a, dict(a), make_record(None, "bad", sha256="xyz"), rejected]
    _, log = validate_manifest(write_jsonl(repo / "m.jsonl", rows), cfg)
    assert {"invalid_json", "duplicate_id", "schema", "rejected_in_split"} <= codes(log)
