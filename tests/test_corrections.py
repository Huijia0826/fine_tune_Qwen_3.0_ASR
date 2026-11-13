import yaml

from whisperflow.data.corrections import apply_corrections


def _rec(text):
    return {"audio_path": "segmented/p_00.wav", "raw_transcript": text, "label_status": "source_provided",
            "reviewer_id": None, "quality_flags": ["segmented_from_long"]}


def _table(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"corrections": {"p_00": {"reviewer_id": "owner", "replaces": "old",
                                                          "text": "new"}}}, allow_unicode=True))
    return p


def test_applies_when_slice_matches(tmp_path):
    r = _rec("old")
    assert apply_corrections([r], _table(tmp_path)) == ([r], 1, [])
    assert r["raw_transcript"] == "new" and r["label_status"] == "human_verified"
    assert r["reviewer_id"] == "owner" and r["quality_flags"][-1] == "label_corrected"


def test_idempotent(tmp_path):
    r = _rec("old")
    apply_corrections([r], _table(tmp_path))
    apply_corrections([r], _table(tmp_path))
    assert r["quality_flags"].count("label_corrected") == 1


def test_skips_stale_correction(tmp_path):
    r = _rec("re-cut slice")
    _, applied, warnings = apply_corrections([r], _table(tmp_path))
    assert applied == 0 and r["raw_transcript"] == "re-cut slice" and "changed" in warnings[0]


def test_drop(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"corrections": {"p_00": {"reviewer_id": "owner", "drop": True}}}))
    assert apply_corrections([_rec("old")], p) == ([], 1, [])
