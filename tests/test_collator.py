"""Label masking in the training collator, with a fake processor (needs torch, not qwen-asr)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qwen_asr_ft import train as T  # noqa: E402


class FakeTok:
    eos_token = "<eos>"
    pad_token_id = 0


class FakeProcessor:
    """Character-level 'tokenizer': one id per character (ord), right padding with 0."""

    tokenizer = FakeTok()

    def __call__(self, text, audio, return_tensors, padding, truncation):
        ids = [[ord(c) for c in t] for t in text]
        n = max(map(len, ids))
        input_ids = torch.tensor([x + [0] * (n - len(x)) for x in ids])
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids])
        return {"input_ids": input_ids, "attention_mask": mask}


def test_only_target_tokens_carry_loss(monkeypatch):
    monkeypatch.setattr(T, "load_audio", lambda path, sr=16000: np.zeros(10, dtype="float32"))
    col = T.DataCollatorForQwen3ASRFinetuning(processor=FakeProcessor())
    batch = col([{"prefix_text": "SYS:", "target": "ab", "audio": "x"},
                 {"prefix_text": "SYSTEM:", "target": "c", "audio": "y"}])
    labels = batch["labels"].tolist()
    keep0 = [chr(t) for t in labels[0] if t != -100]
    keep1 = [chr(t) for t in labels[1] if t != -100]
    assert "".join(keep0).startswith("ab") and "SYS" not in "".join(keep0)
    assert "".join(keep1).startswith("c") and "SYSTEM" not in "".join(keep1)
