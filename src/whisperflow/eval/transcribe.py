"""Run a Whisper model over one split of a manifest and store raw hypotheses + run metadata.

Usage:
    wf-transcribe --config configs/eval.yaml [--split val] [--limit 20] [--run-id NAME]
    wf-transcribe --config configs/eval.yaml --split test --allow-test   # milestones only

Outputs reports/runs/<run_id>/{predictions.jsonl, run_meta.json}. Scoring is a separate
step (wf-score) so runs can be re-scored without re-decoding. Interrupted runs resume:
utterances already present in predictions.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from whisperflow.data.audio import load_16k_mono, sha256_file
from whisperflow.data.common import load_yaml
from whisperflow.data.schema import Utterance, load_manifest
from whisperflow.text.normalize import _t2s

WINDOW_SAMPLES = 30 * 16000  # Whisper encoder window


def decode_mode(audio: np.ndarray) -> str:
    return "short" if len(audio) <= WINDOW_SAMPLES else "long_form_sequential"


class Transcriber(Protocol):
    def transcribe(self, audios: list[np.ndarray]) -> list[str]: ...
    def describe(self) -> dict: ...


def _disable_mps_deferred_stop() -> None:
    """Work around a transformers 5.21 bug: on MPS, sampling uses DeferredStopCheck, whose finish() iterates
    `cache.layers`. Whisper's EncoderDecoderCache has no `.layers`, so temperature fallback crashed. The check is
    only a speed optimisation, so turning it off falls back to the ordinary StopCheck with identical outputs."""
    from transformers.generation import utils as gen_utils

    if hasattr(gen_utils, "DeferredStopCheck"):
        gen_utils.DeferredStopCheck.is_supported = staticmethod(lambda *args, **kwargs: False)


class HFWhisperTranscriber:
    """Transformers Whisper backend (PyTorch). Imported lazily: torch is an optional extra."""

    def __init__(self, model, processor, decoding: dict, runtime: dict):
        import torch

        _disable_mps_deferred_stop()
        self.torch = torch
        self.seed = int(runtime.get("seed", 0))
        self.decoding = decoding
        self.device = self._pick_device(runtime.get("device", "auto"))
        self.dtype = getattr(torch, runtime.get("dtype", "float32"))
        torch.manual_seed(int(runtime.get("seed", 0)))
        self.processor = processor
        self.model = model.to(self.device, dtype=self.dtype).eval()
        self.model_commit = getattr(model.config, "_commit_hash", None)
        self.prompt_ids = None
        self.last_info: list[dict] = []
        if decoding.get("initial_prompt"):
            self.prompt_ids = self.processor.get_prompt_ids(
                decoding["initial_prompt"], return_tensors="pt").to(self.device)

    @classmethod
    def from_config(cls, model_cfg: dict, decoding: dict, runtime: dict) -> HFWhisperTranscriber:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        name, rev = model_cfg["name_or_path"], model_cfg.get("revision")
        processor = WhisperProcessor.from_pretrained(name, revision=rev)
        # merge LoRA in fp32, then cast in __init__: merging directly in bf16/fp16 rounds W + B·A once more
        load_dtype = torch.float32 if model_cfg.get("adapter_path") else getattr(torch, runtime.get("dtype", "float32"))
        model = WhisperForConditionalGeneration.from_pretrained(name, revision=rev, dtype=load_dtype)
        if model_cfg.get("adapter_path"):
            from peft import PeftModel

            # merge W + (alpha/r)·B·A into W: identical outputs, no extra LoRA matmuls at inference
            model = PeftModel.from_pretrained(model, model_cfg["adapter_path"]).merge_and_unload()
        inst = cls(model, processor, decoding, runtime)
        # transformers 5.x dropped config._commit_hash; record the pinned revision instead.
        inst.model_commit = inst.model_commit or rev
        return inst

    def _pick_device(self, want: str) -> str:
        t = self.torch
        if want != "auto":
            return want
        if t.backends.mps.is_available():
            return "mps"
        if t.cuda.is_available():
            return "cuda"
        return "cpu"

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        """Short clips (<=30 s) are batched; longer clips use sequential long-form decoding.

        With decoding.prompt_condition == "zh_only", the language of each clip is detected
        first (first 30 s window) and the initial_prompt is only applied to clips detected as
        Chinese, so a Chinese style prompt cannot bias English clips.

        With decoding.allowed_languages (e.g. [zh, en]) language ID may only pick from that
        list and the detected language is forced for the whole clip, so a 1 s clip can no
        longer come out as Welsh. Long clips use the language of their first 30 s window.

        With decoding.output_script == "simplified" the text is converted Traditional ->
        Simplified (OpenCC t2s) after decoding; the model's raw output is kept in
        hypothesis_raw. Per-clip details are left in self.last_info for the run log.
        """
        n = len(audios)
        allowed = self.decoding.get("allowed_languages")
        need_lid = self.prompt_ids is not None or bool(allowed)
        langs = self._detect_languages(audios) if need_lid else [None] * n
        force = langs if allowed else [None] * n
        use_prompt = [self._prompt_applies(lang) for lang in langs]
        out: list[str | None] = [None] * n
        for key in sorted({(f or "", p) for f, p in zip(force, use_prompt, strict=True)}):
            short = [i for i in range(n) if decode_mode(audios[i]) == "short"
                     and (force[i] or "", use_prompt[i]) == key]
            if short:
                texts = self._transcribe_short([audios[i] for i in short], key[1], key[0] or None)
                for i, t in zip(short, texts, strict=True):
                    out[i] = t
        for i, a in enumerate(audios):
            if out[i] is None:
                out[i] = self._transcribe_long(a, use_prompt[i], force[i])
        self.last_info = [{"detected_language": lang, "prompt_used": up}
                          for lang, up in zip(langs, use_prompt, strict=True)]
        if self.decoding.get("output_script") == "simplified":
            for info, raw in zip(self.last_info, out, strict=True):
                info["hypothesis_raw"] = raw
            out = [_t2s().convert(t) for t in out]
        return out

    def _prompt_applies(self, lang: str | None) -> bool:
        if self.prompt_ids is None:
            return False
        if self.decoding.get("prompt_condition", "always") == "zh_only":
            return lang == "zh"
        return True

    def _detect_languages(self, audios: list[np.ndarray]) -> list[str]:
        """Language of each clip from its first 30 s window (Whisper's own detector).

        Same computation as model.detect_language (logits of the first decoder step, argmax
        over language tokens), but the argmax is taken only over decoding.allowed_languages
        when that is set.
        """
        feats = self.processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
        gc = self.model.generation_config
        allowed = self.decoding.get("allowed_languages") or [
            t.strip("<|>") for t in gc.lang_to_id]
        cand = self.torch.tensor([gc.lang_to_id[f"<|{lang}|>"] for lang in allowed],
                                 device=self.device)
        start = self.torch.full((len(audios), 1), gc.decoder_start_token_id, device=self.device)
        with self.torch.inference_mode():
            logits = self.model(input_features=feats.input_features.to(self.device, dtype=self.dtype),
                                decoder_input_ids=start, use_cache=False).logits[:, -1]
        best = logits[:, cand].argmax(-1).tolist()
        return [allowed[b] for b in best]

    def _common_kwargs(self, use_prompt: bool, language: str | None = None) -> dict:
        kw = dict(language=language or self.decoding["language"], task=self.decoding["task"],
                  num_beams=int(self.decoding["num_beams"]), do_sample=False)
        if kw["language"] in (None, "auto"):
            kw.pop("language")  # let Whisper detect the language per clip / per window
        if use_prompt:
            kw["prompt_ids"] = self.prompt_ids
        return kw

    def _transcribe_short(self, audios: list[np.ndarray], use_prompt: bool = False,
                          language: str | None = None) -> list[str]:
        """Short-form decoding. Optional loop guard (decoding.short_form.loop_guard, E5): the same temperature
        fallback as long-form, triggered only by the compression ratio (repetition loops such as "死死死…"),
        not by low log-probability, so ordinary uncertain clips keep their beam-search output."""
        feats = self.processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        guard = (self.decoding.get("short_form") or {}).get("loop_guard")
        fallback = {}
        if guard:
            fallback = dict(temperature=tuple(float(x) for x in guard["temperature_fallback"]),
                            compression_ratio_threshold=guard.get("compression_ratio_threshold", 1.35),
                            logprob_threshold=None)
            self.torch.manual_seed(self.seed)
        with self.torch.inference_mode():
            out = self.model.generate(
                feats.input_features.to(self.device, dtype=self.dtype),
                attention_mask=feats.attention_mask.to(self.device),
                max_new_tokens=int(self.decoding["max_new_tokens"]),
                return_timestamps=bool(self.decoding.get("return_timestamps", False)),
                **fallback, **self._common_kwargs(use_prompt, language))
        return self._decode(out, use_prompt)

    def _transcribe_long(self, audio: np.ndarray, use_prompt: bool = False,
                         language: str | None = None) -> str:
        """Sequential long-form (OpenAI's algorithm): decode a 30 s window with timestamps,
        move the window to the end of the last complete segment, repeat.

        Optional temperature fallback (decoding.long_form.temperature_fallback): a window whose token
        compression ratio exceeds compression_ratio_threshold (repetition) or whose mean log-prob is below
        logprob_threshold is decoded again at the next temperature. Seeded per clip, so runs are repeatable."""
        lf = self.decoding.get("long_form", {})
        fallback = {}
        if lf.get("temperature_fallback"):
            fallback = dict(temperature=tuple(float(x) for x in lf["temperature_fallback"]),
                            compression_ratio_threshold=lf.get("compression_ratio_threshold", 1.35),
                            logprob_threshold=lf.get("logprob_threshold", -1.0))
            self.torch.manual_seed(self.seed)
        feats = self.processor.feature_extractor(
            [audio], sampling_rate=16000, return_tensors="pt", truncation=False,
            padding="longest", return_attention_mask=True)
        with self.torch.inference_mode():
            out = self.model.generate(
                feats.input_features.to(self.device, dtype=self.dtype),
                attention_mask=feats.attention_mask.to(self.device),
                return_timestamps=True,  # required: timestamps decide where the next window starts
                condition_on_prev_tokens=bool(lf.get("condition_on_prev_tokens", False)),
                **fallback, **self._common_kwargs(use_prompt, language))
        return self._decode(out, use_prompt)[0]

    def _decode(self, out, use_prompt: bool = False) -> list[str]:
        seqs = out["sequences"] if isinstance(out, dict) else out
        texts = self.processor.batch_decode(seqs, skip_special_tokens=True)
        if use_prompt:
            # Older transformers return the prompt as part of the decoded text; strip it.
            p = self.decoding["initial_prompt"].strip()
            texts = [t.strip().removeprefix(p).strip() for t in texts]
        return [t.strip() for t in texts]

    def describe(self) -> dict:
        return {"backend": "transformers", "device": self.device, "dtype": str(self.dtype),
                "model_commit": self.model_commit}


def make_transcriber(cfg: dict) -> Transcriber:
    """runtime.backend: transformers (default) | ctranslate2 (faster-whisper, E7) | qwen3_asr (.venv-qwen)."""
    backend = cfg["runtime"].get("backend", "transformers")
    if backend == "ctranslate2":
        from whisperflow.eval.ct2_backend import FasterWhisperTranscriber

        return FasterWhisperTranscriber(cfg["model"], cfg["decoding"], cfg["runtime"])
    if backend == "qwen3_asr":
        from whisperflow.eval.qwen_asr_backend import QwenASRTranscriber

        return QwenASRTranscriber(cfg["model"], cfg["decoding"], cfg["runtime"])
    return HFWhisperTranscriber.from_config(cfg["model"], cfg["decoding"], cfg["runtime"])


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "peft", "faster-whisper", "ctranslate2", "qwen-asr", "numpy", "soundfile", "opencc",
                "whisperflow"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return r.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def _hardware() -> dict:
    hw = {"platform": platform.platform(), "machine": platform.machine()}
    if sys.platform == "darwin":
        try:
            hw["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       capture_output=True, text=True).stdout.strip()
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            hw["memory_gb"] = round(int(mem.stdout.strip()) / 2**30, 1)
        except Exception:
            pass
    return hw


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run(cfg: dict, records: list[Utterance], transcriber: Transcriber, out_dir: Path,
        *, data_root: Path, batch_size: int, manifest_path: str, split: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_meta.json"
    if meta_path.exists():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        if prev["config_hash"] != config_hash(cfg) or prev["manifest"] != manifest_path:
            raise RuntimeError(f"{out_dir} was produced with a different config/manifest; "
                               "use a new --run-id instead of resuming")
    pred_path = out_dir / "predictions.jsonl"
    done: set[str] = set()
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            done = {json.loads(line)["utterance_id"] for line in f if line.strip()}
    todo = [r for r in records if r.utterance_id not in done]

    audio_sec = 0.0
    t0 = time.perf_counter()
    with open(pred_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            audios = [load_16k_mono(data_root / r.audio_path) for r in batch]
            if hasattr(transcriber, "set_batch_ids"):  # per-clip context (qwen3_asr backend, decoding.context_file)
                transcriber.set_batch_ids([r.utterance_id for r in batch])
            tb = time.perf_counter()
            hyps = transcriber.transcribe(audios)
            dt = time.perf_counter() - tb
            info = getattr(transcriber, "last_info", None) or [{}] * len(batch)
            for r, a, h, extra in zip(batch, audios, hyps, info, strict=True):
                f.write(json.dumps({"utterance_id": r.utterance_id, "hypothesis": h,
                                    "decode_mode": decode_mode(a), **extra,
                                    "audio_sec": round(len(a) / 16000, 3),
                                    "batch_decode_sec": round(dt, 4)},
                                   ensure_ascii=False) + "\n")
                audio_sec += len(a) / 16000
            f.flush()
            print(f"  {min(i + batch_size, len(todo))}/{len(todo)} utterances", flush=True)
    wall = time.perf_counter() - t0

    meta = {
        "run_id": out_dir.name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "manifest": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "n_utterances": len(records),
        "n_decoded_this_invocation": len(todo),
        "transcriber": transcriber.describe(),
        "versions": _versions(),
        "git_commit": _git_commit(),
        "hardware": _hardware(),
        # Offline batch throughput, NOT user-perceived latency (see plan.md Phase 4).
        "batch_rtf_this_invocation": round(wall / audio_sec, 4) if audio_sec else None,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split")
    ap.add_argument("--manifest")
    ap.add_argument("--limit", type=int, help="first N utterances (smoke tests)")
    ap.add_argument("--run-id")
    ap.add_argument("--allow-test", action="store_true",
                    help="required to decode the locked test split; the access is logged")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    split = args.split or cfg["data"]["split"]
    manifest = args.manifest or cfg["data"]["manifest"]
    if split == "test" and not args.allow_test:
        print("Refusing to evaluate the locked test split without --allow-test. "
              "Use val for iteration; test only at predeclared milestones.")
        return 2

    records = [r for r in load_manifest(manifest) if r.split == split]
    records.sort(key=lambda r: r.utterance_id)
    if args.limit:
        records = records[:args.limit]
    if not records:
        print(f"No records with split={split!r} in {manifest}")
        return 1

    run_id = args.run_id or (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{split}_{config_hash(cfg)}")
    out_dir = Path("reports/runs") / run_id
    if split == "test":
        log = Path("reports/test_access_log.jsonl")
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                "run_id": run_id, "config_hash": config_hash(cfg),
                                "model": cfg["model"]}) + "\n")

    print(f"Loading model {cfg['model']['name_or_path']} ...")
    transcriber = make_transcriber(cfg)
    print(f"Decoding {len(records)} utterances from split={split} -> {out_dir}")
    meta = run(cfg, records, transcriber, out_dir, data_root=Path(cfg["data"]["data_root"]),
               batch_size=int(cfg["runtime"]["batch_size"]), manifest_path=manifest, split=split)
    print(json.dumps({k: meta[k] for k in ("run_id", "transcriber", "batch_rtf_this_invocation")},
                     indent=2))
    print(f"Next: wf-score {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
