"""faster-whisper (CTranslate2) backend with the same Transcriber interface as HFWhisperTranscriber (E7).

CTranslate2 converts weights, not our Python decoding logic (plan.md Phase 4), so reference decoding v3 is
approximated with faster-whisper's own options. Differences from the transformers backend:
- allowed_languages: faster-whisper's detect_language (first 30 s window), argmax restricted to the allowed
  list, then the language is forced. Same idea as HFWhisperTranscriber._detect_languages.
- short-form loop guard: faster-whisper's temperature fallback with compression_ratio_threshold and
  log_prob_threshold=None, so only repetition triggers it. Sampling RNG differs from PyTorch, so a clip that
  falls back can produce different text.
- long-form: faster-whisper's own sequential windowing (same OpenAI algorithm, separate implementation).
- no max_new_tokens cap for short-form (see _options): faster-whisper's cap behaves as half the value.
- the LoRA adapter must already be merged and converted (scripts/export_ct2.py); model.ct2_path points to it.
"""

from __future__ import annotations

import numpy as np

from whisperflow.eval.transcribe import decode_mode
from whisperflow.text.normalize import _t2s


class FasterWhisperTranscriber:
    def __init__(self, model_cfg: dict, decoding: dict, runtime: dict):
        from faster_whisper import WhisperModel

        self.decoding = decoding
        self.compute_type = runtime.get("compute_type", "float16")
        self.device = "cuda" if runtime.get("device", "auto") in ("auto", "cuda") else runtime["device"]
        self.model_path = model_cfg["ct2_path"]
        self.model = WhisperModel(self.model_path, device=self.device, compute_type=self.compute_type)
        self.last_info: list[dict] = []

    def _language(self, audio: np.ndarray) -> str | None:
        allowed = self.decoding.get("allowed_languages")
        if self.decoding.get("language") not in (None, "auto"):
            return self.decoding["language"]
        if not allowed:
            return None  # faster-whisper detects on its own
        _, _, probs = self.model.detect_language(audio, language_detection_segments=1)
        probs = dict(probs)
        return max(allowed, key=lambda lang: probs.get(lang, 0.0))

    def _options(self, audio: np.ndarray) -> dict:
        d = self.decoding
        opts = dict(task=d["task"], beam_size=int(d["num_beams"]), condition_on_previous_text=False,
                    vad_filter=False, initial_prompt=d.get("initial_prompt"))
        if decode_mode(audio) == "short":
            guard = (d.get("short_form") or {}).get("loop_guard")
            # max_new_tokens is NOT passed: with faster-whisper 1.2.1 / CTranslate2 4.8.4 the output stops at about
            # half of it (200 -> 102 tokens, 300 -> full), which cut off long short-form clips (E7 debug,
            # scripts/debug/ct2_max_tokens.py). The model limit (448 incl. prompt) applies instead; HF uses 200.
            opts.update(without_timestamps=not d.get("return_timestamps", False),
                        temperature=[float(x) for x in guard["temperature_fallback"]] if guard else 0.0,
                        compression_ratio_threshold=guard.get("compression_ratio_threshold", 1.35)
                        if guard else None,
                        log_prob_threshold=None)
        else:
            lf = d.get("long_form", {})
            temps = lf.get("temperature_fallback")
            opts.update(without_timestamps=False,
                        temperature=[float(x) for x in temps] if temps else 0.0,
                        compression_ratio_threshold=lf.get("compression_ratio_threshold", 1.35) if temps else None,
                        log_prob_threshold=lf.get("logprob_threshold", -1.0) if temps else None)
        return opts

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        out, self.last_info = [], []
        for a in audios:
            lang = self._language(a)
            segments, info = self.model.transcribe(a, language=lang, **self._options(a))
            # segments is a lazy generator: decoding happens here, so consume it inside the timed call
            raw = "".join(s.text for s in segments).strip()
            self.last_info.append({"detected_language": lang or info.language, "prompt_used": False,
                                   "hypothesis_raw": raw})
            out.append(_t2s().convert(raw) if self.decoding.get("output_script") == "simplified" else raw)
        return out

    def describe(self) -> dict:
        return {"backend": "ctranslate2", "device": self.device, "compute_type": self.compute_type,
                "model_path": self.model_path}
