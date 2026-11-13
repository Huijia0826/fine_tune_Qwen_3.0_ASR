import shutil

import pytest

from tests.conftest import make_record, write_wav
from whisperflow.data.dedup import drop_exact_duplicates, find_duplicates
from whisperflow.data.ingest import ingest
from whisperflow.data.schema import parse_record


def _meta(path, rows):
    cols = ["audio_path", "transcript", "speaker_id", "session_id", "hotwords_present"]
    lines = ["\t".join(cols)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_ingest_in_place_is_content_addressed_and_idempotent(repo):
    corpus = repo / "data/private/corpusA"
    write_wav(corpus / "a.wav", 2.0, seed=1)
    write_wav(corpus / "b.wav", 3.0, seed=2)
    meta = _meta(repo / "meta.tsv", [
        ("a.wav", "今天天气很好", "spk1", "", ""),
        ("b.wav", "我们跑一下 benchmark", "spk2", "sessX", "benchmark"),
        ("missing.wav", "丢失", "spk3", "", ""),
    ])
    kw = dict(source_id="synthetic_test", license_id="synthetic", label_status="source_provided")
    recs, problems = ingest(meta, corpus, repo / "data/private", **kw)
    assert len(recs) == 2 and len(problems) == 1 and "missing" in problems[0]
    a, b = recs
    assert a.utterance_id == f"synthetic_test-{a.sha256[:16]}"
    assert a.audio_path == "corpusA/a.wav" and a.split == "unassigned"
    assert a.primary_scenario == "zh_general" and a.language_tags == ["zh"]
    assert b.primary_scenario == "zh_en_switch" and "scenario_inferred" in b.quality_flags
    assert b.speaker_id == "synthetic_test-spk2" and b.session_id == "synthetic_test-sessX"
    assert abs(b.duration_sec - 3.0) < 1e-3
    recs2, _ = ingest(meta, corpus, repo / "data/private", **kw)
    assert [r.model_dump() for r in recs2] == [r.model_dump() for r in recs]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ingest_convert_16k(repo, tmp_path):
    ext = tmp_path / "external"
    write_wav(ext / "x.wav", 1.5, sr=44100, channels=2)
    meta = _meta(repo / "meta.tsv", [("x.wav", "你好", "spk1", "", "")])
    recs, problems = ingest(meta, ext, repo / "data/private", source_id="synthetic_test",
                            license_id="synthetic", label_status="source_provided", convert_16k=True)
    assert problems == []
    r = recs[0]
    assert r.sample_rate_hz == 16000 and r.num_channels == 1
    assert r.audio_path == "synthetic_test/x.wav"
    assert abs(r.duration_sec - 1.5) < 0.01


def test_ingest_rejects_audio_outside_data_root(repo, tmp_path):
    ext = tmp_path / "external"
    write_wav(ext / "x.wav", 1.0)
    meta = _meta(repo / "meta.tsv", [("x.wav", "你好", "spk1", "", "")])
    recs, problems = ingest(meta, ext, repo / "data/private", source_id="synthetic_test",
                            license_id="synthetic", label_status="source_provided")
    assert recs == [] and "outside data_root" in problems[0]


def test_dedup_tiers():
    def rec(uid, sha, spk, text, dur, split="train"):
        return parse_record(make_record(uid=uid, sha256=sha * 64, speaker_id=spk, raw_transcript=text,
                                        duration_sec=dur, split=split, primary_scenario="zh_general",
                                        language_tags=["zh"], hotwords_present=[]))[0]
    recs = [
        rec("a", "a", "s1", "你好世界", 2.0),
        rec("a_copy", "a", "s1", "你好世界", 2.0),         # exact audio dup
        rec("a_reenc", "b", "s1", "你好，世界。", 2.05),    # same spk+text+dur -> re-encode
        rec("other_spk", "c", "s2", "你好世界", 2.6, "test"),  # shared text across splits
    ]
    d = find_duplicates(recs)
    assert d["exact_audio"] == [["a", "a_copy"]]
    assert sorted(map(sorted, d["likely_reencode"])) == [["a", "a_reenc"], ["a_copy", "a_reenc"]]
    assert d["same_transcript"][0]["splits"] == ["test", "train"]
    kept, dropped = drop_exact_duplicates(recs)
    assert dropped == ["a_copy"] and len(kept) == 3
