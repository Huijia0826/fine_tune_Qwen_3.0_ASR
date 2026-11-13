"""Training data plumbing: label layout and collator padding (real whisper-small tokenizer, no weights)."""

import numpy as np
import pytest
import torch

from whisperflow.training.data import IGNORE_INDEX, WhisperCollator, make_labels

REV = "973afd24965f72e36ca33b3055d56a652f456b4d"


@pytest.fixture(scope="module")
def tok():
    from transformers import WhisperTokenizer
    try:
        return WhisperTokenizer.from_pretrained("openai/whisper-small", revision=REV, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer files not cached; run scripts/fetch_tokenizer_only.sh")


def test_labels_layout_zh(tok):
    ids = make_labels(tok, "我们先跑一下 benchmark", "zh")
    toks = tok.convert_ids_to_tokens(ids)
    # no <|startoftranscript|>: the model prepends it itself when shifting labels right
    assert toks[:3] == ["<|zh|>", "<|transcribe|>", "<|notimestamps|>"]
    assert toks[-1] == "<|endoftext|>"
    assert tok.decode(ids, skip_special_tokens=True) == "我们先跑一下 benchmark"


def test_labels_english_clip_uses_en_token(tok):
    ids = make_labels(tok, "it's already far away", "en")
    assert tok.convert_ids_to_tokens(ids)[0] == "<|en|>"
    assert tok.decode(ids, skip_special_tokens=True).strip() == "it's already far away"


def test_labels_too_long_raises(tok):
    with pytest.raises(ValueError):
        make_labels(tok, "很长" * 400, "zh", max_tokens=448)


def test_collator_pads_labels_with_ignore_index():
    batch = [{"input_features": np.zeros((80, 3000), np.float32), "labels": [1, 2, 3]},
             {"input_features": np.ones((80, 3000), np.float32), "labels": [4, 5]}]
    out = WhisperCollator()(batch)
    assert out["input_features"].shape == (2, 80, 3000)
    assert out["labels"].tolist() == [[1, 2, 3], [4, 5, IGNORE_INDEX]]
    assert out["labels"].dtype == torch.long


def test_usable_for_loss_drops_long_audio_and_long_labels(tok):
    from types import SimpleNamespace

    from whisperflow.training.data import usable_for_loss

    def rec(uid, dur, text):
        return SimpleNamespace(utterance_id=uid, duration_sec=dur, raw_transcript=text, language_tags=["zh"])

    keep, dropped = usable_for_loss([rec("a", 5.0, "好的"), rec("b", 45.0, "很长的录音"),
                                     rec("c", 20.0, "很长" * 400)], tok)
    assert [r.utterance_id for r in keep] == ["a"] and dropped == 2


def test_labels_with_timestamps(tok):
    ids = make_labels(tok, "我们先跑一下", "zh", end_sec=1.5)
    toks = tok.convert_ids_to_tokens(ids)
    assert toks[:3] == ["<|zh|>", "<|transcribe|>", "<|0.00|>"]
    assert "<|notimestamps|>" not in toks
    assert toks[-2:] == ["<|1.50|>", "<|endoftext|>"]
    assert tok.decode(ids, skip_special_tokens=True) == "我们先跑一下"


def test_timestamp_rounding_and_cap(tok):
    from whisperflow.training.data import timestamp_token
    assert tok.convert_ids_to_tokens(timestamp_token(tok, 2.333)) == "<|2.34|>"   # nearest 0.02
    assert tok.convert_ids_to_tokens(timestamp_token(tok, 31.0)) == "<|30.00|>"   # capped at the window


def test_dataset_timestamp_choice_is_deterministic_and_about_half():
    from types import SimpleNamespace

    from whisperflow.training.data import WhisperDataset
    ds = WhisperDataset([], None, ".", timestamp_ratio=0.5)
    recs = [SimpleNamespace(sha256=f"{i:064x}"[::-1]) for i in range(2000)]
    picks = [ds.uses_timestamps(r) for r in recs]
    assert picks == [ds.uses_timestamps(r) for r in recs]
    assert 0.4 < sum(picks) / len(picks) < 0.6
    assert not any(WhisperDataset([], None, ".", timestamp_ratio=0.0).uses_timestamps(r) for r in recs)
