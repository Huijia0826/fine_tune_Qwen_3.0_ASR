"""Qwen3-ASR backend (qwen-asr package, transformers backend) with the same Transcriber interface (2026-11-09).

Runs in .venv-qwen (qwen-asr 0.0.6 pins transformers==4.57.6; .venv has transformers 5.x). Differences from the
Whisper backends:
- decoding is the package's own greedy generate; num_beams / loop guard / long-form settings do not apply.
  Clips up to 1200 s go in as one input unless decoding.max_chunk_sec is set.
- language: decoding.language "auto" lets the model emit "language X<asr_text>..."; any other value is forced
  (Qwen language name, e.g. "Chinese"). Qwen gives no language probabilities, so allowed_languages cannot be
  enforced the Whisper way; the detected language is logged and checked afterwards.
- decoding.context: optional text passed as the system prompt (Qwen's contextual biasing); "" = none.
- decoding.min_tail_sec (default 5): with max_chunk_sec, a last chunk shorter than this is merged into the
  previous one instead of being decoded alone.
- decoding.copy_guard: cut an output tail of >= 8 characters (punctuation / spaces ignored) that appears in the
  clip's Before / After cursor text — J1 sometimes appended the before-cursor text after the spoken sentence.
- decoding.context_file: jsonl {utterance_id, context} — a per-clip context (run J1, 2026-11-09); the runner
  passes the batch's utterance ids through set_batch_ids(). Clips missing from the file get decoding.context.
- runtime.timing (trial server, 2026-11-12): per call, self.last_timing splits the time into feature extraction
  (processor), prefill (generate until the first new token: audio encoder + prompt) and decode (the remaining
  tokens), plus chunk / token counts and post-processing (unglue, t2s). Off by default (evaluation unchanged).
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

from whisperflow.text.normalize import _t2s

_SKIP = re.compile(r"[\s\W_]")
# qwen_asr joins the texts of long-audio chunks with "", which glues English sentences at chunk boundaries
# ("frontend?Well", 2026-11-11). Scoring strips punctuation so MER does not see it; users would.
_GLUED = re.compile(r"(?<=[a-z0-9])([.?!])(?=[A-Z])")


def unglue(text: str) -> str:
    """'frontend?Well' → 'frontend? Well'; leaves 'gmail.com', '3.5', 'Node.js' alone (lower case or digit
    after the mark)."""
    return _GLUED.sub(r"\1 ", text)


def cut_copied_tail(text: str, context: str, min_chars: int = 8) -> str:
    """Remove the longest tail of `text` (>= min_chars non-punctuation characters) that occurs in the cursor text
    of `context`; keep at least min_chars of the original head."""
    cursor = " ".join(line.split(":", 1)[1] for line in context.splitlines()
                      if line.startswith(("Before cursor:", "After cursor:")))
    hay = _SKIP.sub("", cursor.lower())
    if len(hay) < min_chars:
        return text
    for i in range(len(text)):  # earliest start = longest tail
        tail = _SKIP.sub("", text[i:].lower())
        if len(tail) < min_chars:
            break
        if tail in hay and len(_SKIP.sub("", text[:i])) >= min_chars:
            return text[:i].rstrip(" ,，、")
    return text


def merge_short_tail(split, min_tail_sec: float):
    """Wrap qwen_asr's split_audio_into_chunks: a last chunk shorter than min_tail_sec joins the previous chunk."""
    def wrapped(wav, sr, max_chunk_sec, **kw):
        parts = split(wav=wav, sr=sr, max_chunk_sec=max_chunk_sec, **kw)
        if len(parts) > 1 and len(parts[-1][0]) < min_tail_sec * sr:
            (prev, off), (last, _) = parts[-2], parts[-1]
            parts = parts[:-2] + [(np.concatenate([prev, last]), off)]
        return parts
    return wrapped


class _FirstToken:
    """generate() streamer that only records when the first new token arrives (put #1 is the prompt; put #2 the
    first sampled token, already copied to the CPU, so the GPU work before it is finished)."""

    def __init__(self):
        self.n, self.t = 0, None

    def put(self, value) -> None:
        self.n += 1
        if self.n == 2:
            self.t = time.perf_counter()

    def end(self) -> None:
        pass


class QwenASRTranscriber:
    def __init__(self, model_cfg: dict, decoding: dict, runtime: dict):
        import torch
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ASRModel

        self.decoding = decoding
        # model.checkpoint_path: a fine-tuned checkpoint dir (scripts/qwen3_asr_sft.py); otherwise resolve the
        # pinned snapshot locally (the processor is loaded without a revision argument)
        self.model_path = model_cfg.get("checkpoint_path") or snapshot_download(
            model_cfg["name_or_path"], revision=model_cfg["revision"], local_files_only=True)
        self.dtype = runtime.get("dtype", "bfloat16")
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path, dtype=getattr(torch, self.dtype), device_map="cuda:0",
            max_inference_batch_size=int(runtime.get("batch_size", 16)),
            max_new_tokens=int(decoding.get("max_new_tokens", 1024)))
        # decoding.max_chunk_sec: cut long clips at a low-energy point near this length before decoding
        # (package default 1200 s = one pass; zero-shot drifted on > 30 s clips, 2026-11-09)
        if decoding.get("max_chunk_sec"):
            import qwen_asr.inference.qwen3_asr as q

            q.MAX_ASR_INPUT_SECONDS = float(decoding["max_chunk_sec"])
            # a near-silent tail chunk (26.1 s clip → 25 s + 1.1 s) made fine-tuned models invent Chinese text
            # (2026-11-09, dictionary benchmark): a last chunk shorter than min_tail_sec joins the previous one
            q.split_audio_into_chunks = merge_short_tail(q.split_audio_into_chunks,
                                                         float(decoding.get("min_tail_sec", 5.0)))
        self.torch_version = torch.__version__
        self.last_info: list[dict] = []
        self.last_timing: dict = {}
        if runtime.get("timing"):
            self._install_timers()
        self.batch_ids: list[str] | None = None
        self.contexts = {}
        if decoding.get("context_file"):
            with open(decoding["context_file"], encoding="utf-8") as f:
                self.contexts = {r["utterance_id"]: r["context"] for r in map(json.loads, f)}

    def _install_timers(self) -> None:
        """Wrap the package's processor and model.generate on this instance (the qwen_asr package is untouched)."""
        import torch

        q, timing = self.model, self.last_timing
        proc, gen = q.processor, q.model.generate

        class TimedProcessor:
            def __getattr__(self, name):
                return getattr(proc, name)

            def __call__(self, *a, **kw):
                t0 = time.perf_counter()
                out = proc(*a, **kw)
                timing["features_ms"] = timing.get("features_ms", 0) + (time.perf_counter() - t0) * 1000
                return out

        def timed_generate(*a, **kw):
            first = _FirstToken()
            t0 = time.perf_counter()
            out = gen(*a, streamer=first, **kw)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tf = first.t or t1
            n_prompt = kw["input_ids"].shape[1]
            for k, v in (("prefill_ms", (tf - t0) * 1000), ("decode_ms", (t1 - tf) * 1000), ("chunks", 1),
                         ("prompt_tokens", n_prompt), ("new_tokens", out.sequences.shape[1] - n_prompt)):
                timing[k] = timing.get(k, 0) + v
            return out

        q.processor = TimedProcessor()
        q.model.generate = timed_generate

    def set_batch_ids(self, ids: list[str]) -> None:
        self.batch_ids = ids

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        lang = self.decoding.get("language", "auto")
        default = self.decoding.get("context") or ""
        ctxs = ([self.contexts.get(u, default) for u in self.batch_ids] if self.batch_ids and self.contexts
                else [default] * len(audios))
        self.last_timing.clear()
        results = self.model.transcribe(audio=[(a, 16000) for a in audios], context=ctxs,
                                        language=None if lang in (None, "auto") else lang)
        t_post = time.perf_counter()
        out, self.last_info = [], []
        for r, c in zip(results, ctxs, strict=True):
            raw = unglue(r.text.strip())
            if self.decoding.get("copy_guard") and c:
                raw = cut_copied_tail(raw, c)
            self.last_info.append({"detected_language": r.language, "prompt_used": bool(c), "hypothesis_raw": raw})
            out.append(_t2s().convert(raw) if self.decoding.get("output_script") == "simplified" else raw)
        if self.last_timing:
            self.last_timing["post_ms"] = (time.perf_counter() - t_post) * 1000
        return out

    def describe(self) -> dict:
        return {"backend": "qwen3_asr", "device": "cuda:0", "dtype": self.dtype, "model_path": self.model_path,
                "torch": self.torch_version}
