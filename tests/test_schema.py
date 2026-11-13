import pytest

from tests.conftest import make_record
from whisperflow.data.schema import parse_record


def _errs(**over):
    rec, errs = parse_record(make_record(**over))
    return errs


def test_valid_record():
    rec, errs = parse_record(make_record())
    assert errs == [] and rec.utterance_id == "u000001"
    assert rec.soft_warnings(30.0) == []


@pytest.mark.parametrize("over,needle", [
    ({"sha256": "abc"}, "sha256"),
    ({"duration_sec": 0}, "duration_sec"),
    ({"split": "dev"}, "split"),
    ({"raw_transcript": "   "}, "empty"),
    ({"raw_transcript": "。，！"}, "no scoreable tokens"),
    ({"audio_path": "/abs/x.wav"}, "relative"),
    ({"audio_path": "../x.wav"}, ".."),
    ({"utterance_id": "has space"}, "invalid identifier"),
    ({"reviewer_id": None}, "requires reviewer_id"),
    ({"language_tags": ["zh"]}, "requires 'en'"),
    ({"language_tags": []}, "language_tags"),
    ({"primary_scenario": "chitchat"}, "primary_scenario"),
    ({"unexpected_field": 1}, "unexpected_field"),
])
def test_schema_errors(over, needle):
    errs = _errs(**over)
    assert errs and any(needle in e for e in errs), errs


def test_missing_required_field():
    d = make_record()
    del d["speaker_id"]
    _, errs = parse_record(d)
    assert any("speaker_id" in e for e in errs)


def test_soft_warnings():
    rec, _ = parse_record(make_record(hotwords_present=["Kubernetes"], split="test",
                                      label_status="source_provided", reviewer_id=None,
                                      duration_sec=45.0))
    w = " | ".join(rec.soft_warnings(30.0))
    assert "Kubernetes" in w
    assert "not human_verified" in w
    assert "exceeds 30.0s" in w


def test_language_tag_mismatch_warns():
    rec, _ = parse_record(make_record(raw_transcript="今天天气很好", primary_scenario="zh_general",
                                      hotwords_present=[]))
    assert any("declares 'en'" in w for w in rec.soft_warnings(30.0))


def test_implausible_rate_warns():
    rec, _ = parse_record(make_record(raw_transcript="好" * 100, duration_sec=2.0,
                                      primary_scenario="zh_general", language_tags=["zh"],
                                      hotwords_present=[]))
    assert any("implausibly fast" in w for w in rec.soft_warnings(30.0))
