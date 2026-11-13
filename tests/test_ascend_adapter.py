"""Tests for exercise 1 (ASCEND adapter). Synthetic rows that mimic the real ASCEND parquet schema."""

import csv
import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from whisperflow.data.adapters.ascend import export_ascend


def wav_bytes(seconds: float, freq: float) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 16000,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


def row(id_, name, text, spk, ses, freq):
    return {"id": id_, "path": f"/storage/cache/waves/{name}",  # path on the publisher's machine
            "audio": {"bytes": wav_bytes(1.0, freq), "path": name},
            "transcription": text, "duration": 1.0, "language": "zh",
            "original_speaker_id": spk, "session_id": ses, "topic": "technology"}


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def make_shards(tmp_path):
    # Like the real train split: ids continue across shards, but speaker 1 session 1 and
    # speaker 2 session 1 share session_id=1 (ASCEND session ids are per speaker, not global).
    a = write_parquet(tmp_path / "train-0.parquet", [
        row("00000", "ses1_spk1_L1_0.0_1.0.wav", "我们先跑一下 benchmark", 1, 1, 440),
        row("00001", "ses1_spk1_L2_1.0_1.0.wav", "尤其是在 [UNK] 可能", 1, 1, 450),
    ])
    b = write_parquet(tmp_path / "train-1.parquet", [
        row("00002", "ses1_spk2_L1_0.0_1.0.wav", "and then 看一下 CER", 2, 1, 460),
    ])
    return [a, b]


def test_skips_unk_and_reports_counts(tmp_path):
    stats = export_ascend(make_shards(tmp_path), tmp_path / "audio", tmp_path / "meta.tsv")
    assert stats == {"n_total": 3, "n_skipped_unk": 1, "n_empty_after_fillers": 0, "n_written": 2}
    rows = read_tsv(tmp_path / "meta.tsv")
    assert [r["transcript"] for r in rows] == ["我们先跑一下 benchmark", "and then 看一下 CER"]


def test_tsv_columns_and_audio_files(tmp_path):
    export_ascend(make_shards(tmp_path), tmp_path / "audio", tmp_path / "meta.tsv")
    rows = read_tsv(tmp_path / "meta.tsv")
    assert set(rows[0]) == {"audio_path", "transcript", "speaker_id", "session_id", "source_utterance_id"}
    for r in rows:
        assert (tmp_path / "audio" / r["audio_path"]).is_file()  # relative to audio_out
    assert len({r["audio_path"] for r in rows}) == len(rows)    # no file overwritten another


def test_audio_is_written_unchanged(tmp_path):
    shards = make_shards(tmp_path)
    export_ascend(shards, tmp_path / "audio", tmp_path / "meta.tsv")
    original = pq.read_table(shards[0]).to_pylist()[0]["audio"]["bytes"]
    first = read_tsv(tmp_path / "meta.tsv")[0]
    assert (tmp_path / "audio" / first["audio_path"]).read_bytes() == original


def test_session_ids_do_not_merge_speakers(tmp_path):
    export_ascend(make_shards(tmp_path), tmp_path / "audio", tmp_path / "meta.tsv")
    rows = read_tsv(tmp_path / "meta.tsv")
    assert rows[0]["speaker_id"] != rows[1]["speaker_id"]
    # both rows have ASCEND session_id=1, but they are different conversations' speakers
    assert rows[0]["session_id"] != rows[1]["session_id"]


def test_source_utterance_id_is_unique_across_splits(tmp_path):
    # ASCEND restarts `id` at 00000 in every split, so `id` alone is not a stable key.
    val = write_parquet(tmp_path / "validation.parquet",
                        [row("00000", "ses1_spk7_L1_0.0_1.0.wav", "嗯 好的", 7, 1, 470)])
    export_ascend(make_shards(tmp_path), tmp_path / "audio", tmp_path / "train.tsv")
    export_ascend([val], tmp_path / "audio", tmp_path / "val.tsv")
    ids = [r["source_utterance_id"] for r in read_tsv(tmp_path / "train.tsv") + read_tsv(tmp_path / "val.tsv")]
    assert len(set(ids)) == len(ids)


def filler_shard(tmp_path):
    return write_parquet(tmp_path / "train-f.parquet", [
        row("00000", "ses2_spk3_L1_0.0_1.0.wav", "呃 就是那个 嗯 benchmark", 3, 2, 440),
        row("00001", "ses2_spk3_L2_1.0_1.0.wav", "嗯", 3, 2, 450),            # only a filler
        row("00002", "ses2_spk3_L3_2.0_1.0.wav", "免费额度 um 够用", 3, 2, 460),  # um is a filler (2026-10-30)
    ])


def test_strip_fillers_is_off_by_default(tmp_path):
    export_ascend([filler_shard(tmp_path)], tmp_path / "audio", tmp_path / "meta.tsv")
    assert [r["transcript"] for r in read_tsv(tmp_path / "meta.tsv")] == [
        "呃 就是那个 嗯 benchmark", "嗯", "免费额度 um 够用"]


def test_strip_fillers_cleans_text_and_drops_empty_clips(tmp_path):
    stats = export_ascend([filler_shard(tmp_path)], tmp_path / "audio", tmp_path / "meta.tsv",
                          strip_fillers=True)
    assert stats == {"n_total": 3, "n_skipped_unk": 0, "n_empty_after_fillers": 1, "n_written": 2}
    rows = read_tsv(tmp_path / "meta.tsv")
    assert [r["transcript"] for r in rows] == ["就是那个 benchmark", "免费额度 够用"]
    # the dropped clip's audio must not be written either
    assert sorted(p.name for p in (tmp_path / "audio").iterdir()) == [
        "ses2_spk3_L1_0.0_1.0.wav", "ses2_spk3_L3_2.0_1.0.wav"]


def test_en_digits_and_english_fillers(tmp_path):
    shard = write_parquet(tmp_path / "train-n.parquet", [
        row("00000", "ses3_spk5_L1_0.0_1.0.wav", "um we have like forty five students 一个", 5, 3, 440),
        row("00001", "ses3_spk5_L2_1.0_1.0.wav", "uh", 5, 3, 450),
    ])
    stats = export_ascend([shard], tmp_path / "audio", tmp_path / "meta.tsv", strip_fillers=True, en_digits=True)
    assert stats["n_empty_after_fillers"] == 1 and stats["n_written"] == 1
    assert read_tsv(tmp_path / "meta.tsv")[0]["transcript"] == "we have like 45 students 一个"
