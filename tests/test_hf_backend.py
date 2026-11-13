"""Integration test of the real transformers Whisper code path, WITHOUT checkpoint weights.

Uses the real whisper-small tokenizer/processor/generation config (~4 MB, fetched by
scripts/fetch_tokenizer_only.sh) and a tiny randomly-initialised model. Outputs are
gibberish by construction; this verifies device placement, forced-language generation,
beam search, batching and decoding — not accuracy. Skipped if torch or the files are absent.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REV = "973afd24965f72e36ca33b3055d56a652f456b4d"


@pytest.fixture(scope="module")
def tiny_backend():
    from transformers import (
        GenerationConfig,
        WhisperConfig,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", revision=REV,
                                                     local_files_only=True)
        gen = GenerationConfig.from_pretrained("openai/whisper-small", revision=REV,
                                               local_files_only=True)
        real = WhisperConfig.from_pretrained("openai/whisper-small", revision=REV,
                                             local_files_only=True)
    except OSError:
        pytest.skip("tokenizer files not cached; run scripts/fetch_tokenizer_only.sh")
    tiny = WhisperConfig(
        vocab_size=real.vocab_size, num_mel_bins=real.num_mel_bins, d_model=32,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=1500, max_target_positions=448,
        pad_token_id=real.pad_token_id, bos_token_id=real.bos_token_id,
        eos_token_id=real.eos_token_id, decoder_start_token_id=real.decoder_start_token_id)
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(tiny)
    model.generation_config = gen
    return model, processor


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_generate_path(tiny_backend, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    from whisperflow.eval.transcribe import HFWhisperTranscriber

    model, processor = tiny_backend
    dec = {"language": "zh", "task": "transcribe", "num_beams": 2, "max_new_tokens": 8,
           "return_timestamps": False, "initial_prompt": None}
    tr = HFWhisperTranscriber(model, processor, dec, {"device": device, "dtype": "float32", "seed": 1})
    audios = [np.zeros(16000, dtype=np.float32), 0.1 * np.ones(24000, dtype=np.float32)]
    out = tr.transcribe(audios)
    assert len(out) == 2 and all(isinstance(t, str) for t in out)
    assert tr.describe()["device"] == device
    # deterministic decoding: same input -> same output
    assert tr.transcribe(audios) == out


def test_forced_language_prefix(tiny_backend):
    """The decoder must be fed <|startoftranscript|><|zh|><|transcribe|><|notimestamps|>.

    Transformers 5.x strips this prefix from generate()'s return value, so we capture the
    decoder's actual input on its first forward call instead of inspecting the output.
    """
    model, processor = tiny_backend
    model = model.to("cpu")  # fixture is shared; .to() in earlier tests mutates it in place
    seen = []
    hook = model.model.decoder.register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(kwargs.get("input_ids")), with_kwargs=True)
    try:
        feats = processor.feature_extractor([np.zeros(16000, dtype=np.float32)], sampling_rate=16000,
                                            return_tensors="pt")
        model.generate(feats.input_features, language="zh", task="transcribe", num_beams=1,
                       max_new_tokens=3, return_timestamps=False)
    finally:
        hook.remove()
    toks = processor.tokenizer.convert_ids_to_tokens(seen[0][0].tolist())
    assert toks[:4] == ["<|startoftranscript|>", "<|zh|>", "<|transcribe|>", "<|notimestamps|>"]


def test_long_audio_is_not_truncated(tiny_backend):
    """A 45 s clip must take the long-form path and reach the encoder untruncated (>3000 frames)."""
    from whisperflow.eval.transcribe import HFWhisperTranscriber, decode_mode

    model, processor = tiny_backend
    dec = {"language": "auto", "task": "transcribe", "num_beams": 1, "max_new_tokens": 8,
           "return_timestamps": False, "initial_prompt": None,
           "long_form": {"condition_on_prev_tokens": False}}
    tr = HFWhisperTranscriber(model, processor, dec, {"device": "cpu", "dtype": "float32", "seed": 1})
    long_audio = 0.05 * np.random.default_rng(0).standard_normal(45 * 16000).astype(np.float32)
    short_audio = np.zeros(16000, dtype=np.float32)
    assert decode_mode(long_audio) == "long_form_sequential" and decode_mode(short_audio) == "short"

    seen = []
    orig = tr.model.generate
    tr.model.generate = lambda feats, **kw: (seen.append(feats.shape[-1]), orig(feats, **kw))[1]
    try:
        out = tr.transcribe([short_audio, long_audio])
    finally:
        tr.model.generate = orig
    assert len(out) == 2 and all(isinstance(t, str) for t in out)
    assert seen[0] == 3000            # short clip: one padded 30 s window
    assert seen[1] == 4500            # long clip: full 45 s of features (100 frames/s), not cut to 3000


def test_zh_only_prompt_condition(tiny_backend):
    """prompt_condition=zh_only: prompt goes only to clips whose detected language is zh."""
    from whisperflow.eval.transcribe import HFWhisperTranscriber

    model, processor = tiny_backend
    dec = {"language": "auto", "task": "transcribe", "num_beams": 1, "max_new_tokens": 5,
           "return_timestamps": False, "initial_prompt": "好的，我们先过一下这个 feature 的进度。",
           "prompt_condition": "zh_only", "long_form": {"condition_on_prev_tokens": False}}
    tr = HFWhisperTranscriber(model.to("cpu"), processor, dec, {"device": "cpu", "dtype": "float32", "seed": 1})
    tr._detect_languages = lambda audios: ["zh", "en", "cy"][: len(audios)]  # scripted detector
    seen = []
    orig = tr.model.generate
    tr.model.generate = lambda feats, **kw: (seen.append("prompt_ids" in kw), orig(feats, **kw))[1]
    try:
        out = tr.transcribe([np.zeros(16000, dtype=np.float32)] * 3)
    finally:
        tr.model.generate = orig
    assert len(out) == 3
    assert [i["prompt_used"] for i in tr.last_info] == [True, False, False]
    assert sorted(seen) == [False, True]  # one batch without prompt (en, cy), one with (zh)


def test_allowed_languages_restricts_and_forces(tiny_backend):
    """allowed_languages: LID can only return listed languages, and that language is forced
    into generate() per clip (one batch per language)."""
    from whisperflow.eval.transcribe import HFWhisperTranscriber

    model, processor = tiny_backend
    dec = {"language": "auto", "task": "transcribe", "num_beams": 1, "max_new_tokens": 5,
           "return_timestamps": False, "initial_prompt": None, "allowed_languages": ["zh", "en"],
           "long_form": {"condition_on_prev_tokens": False}}
    tr = HFWhisperTranscriber(model.to("cpu"), processor, dec, {"device": "cpu", "dtype": "float32", "seed": 1})
    rng = np.random.default_rng(0)
    audios = [0.1 * rng.standard_normal(16000).astype(np.float32) for _ in range(4)]
    assert set(tr._detect_languages(audios)) <= {"zh", "en"}

    tr._detect_languages = lambda a: ["zh", "en", "zh"][: len(a)]  # scripted detector
    seen = []
    orig = tr.model.generate
    tr.model.generate = lambda feats, **kw: (seen.append((feats.shape[0], kw.get("language"))),
                                             orig(feats, **kw))[1]
    try:
        tr.transcribe(audios[:3])
    finally:
        tr.model.generate = orig
    assert sorted(seen) == [(1, "en"), (2, "zh")]


def test_output_script_simplified(tiny_backend):
    from whisperflow.eval.transcribe import HFWhisperTranscriber

    model, processor = tiny_backend
    dec = {"language": "zh", "task": "transcribe", "num_beams": 1, "max_new_tokens": 5,
           "return_timestamps": False, "initial_prompt": None, "output_script": "simplified"}
    tr = HFWhisperTranscriber(model.to("cpu"), processor, dec, {"device": "cpu", "dtype": "float32", "seed": 1})
    tr._transcribe_short = lambda audios, use_prompt=False, language=None: ["我們的 exam 的話"] * len(audios)
    out = tr.transcribe([np.zeros(16000, dtype=np.float32)])
    assert out == ["我们的 exam 的话"]
    assert tr.last_info[0]["hypothesis_raw"] == "我們的 exam 的話"
