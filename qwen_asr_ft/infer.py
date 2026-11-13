"""Batch transcription with the base model or a fine-tuned checkpoint (qwen-asr transformers backend).

Long audio is cut at a low-energy point near `decode.max_chunk_sec` (the package default decodes up to 20 min in one
pass, which drifted / looped on multi-minute clips in our tests); a last chunk shorter than `decode.min_tail_sec`
is merged into the previous one (a near-silent tail made fine-tuned models invent text).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .context import build_context
from .data import read_manifests
from .text import to_simplified

_SKIP = re.compile(r"[\s\W_]")
_GLUED = re.compile(r"(?<=[a-z0-9])([.?!])(?=[A-Z])")


def unglue(text: str) -> str:
    """qwen_asr joins chunk texts with "": 'frontend?Well' -> 'frontend? Well' ('gmail.com', '3.5' untouched)."""
    return _GLUED.sub(r"\1 ", text)


def cut_copied_tail(text: str, context: str, min_chars: int = 8) -> str:
    """Remove the longest output tail (>= min_chars) that also occurs in the cursor text of the context."""
    cursor = " ".join(line.split(":", 1)[1] for line in context.splitlines()
                      if line.startswith(("Before cursor:", "After cursor:")))
    hay = _SKIP.sub("", cursor.lower())
    if len(hay) < min_chars:
        return text
    for i in range(len(text)):
        tail = _SKIP.sub("", text[i:].lower())
        if len(tail) < min_chars:
            break
        if tail in hay and len(_SKIP.sub("", text[:i])) >= min_chars:
            return text[:i].rstrip(" ,，、")
    return text


def merge_short_tail(split, min_tail_sec: float):
    def wrapped(wav, sr, max_chunk_sec, **kw):
        parts = split(wav=wav, sr=sr, max_chunk_sec=max_chunk_sec, **kw)
        if len(parts) > 1 and len(parts[-1][0]) < min_tail_sec * sr:
            (prev, off), (last, _) = parts[-2], parts[-1]
            parts = parts[:-2] + [(np.concatenate([prev, last]), off)]
        return parts
    return wrapped


class Transcriber:
    def __init__(self, model_path: str, dcfg: dict):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.d = dcfg
        self.model = Qwen3ASRModel.from_pretrained(
            model_path, dtype=getattr(torch, dcfg.get("dtype", "bfloat16")), device_map=dcfg.get("device", "cuda:0"),
            max_inference_batch_size=int(dcfg.get("batch_size", 16)),
            max_new_tokens=int(dcfg.get("max_new_tokens", 1024)))
        if dcfg.get("max_chunk_sec"):
            import qwen_asr.inference.qwen3_asr as q

            q.MAX_ASR_INPUT_SECONDS = float(dcfg["max_chunk_sec"])
            if not getattr(q.split_audio_into_chunks, "_qasr_wrapped", False):
                q.split_audio_into_chunks = merge_short_tail(q.split_audio_into_chunks,
                                                             float(dcfg.get("min_tail_sec", 5)))
                q.split_audio_into_chunks._qasr_wrapped = True

    def transcribe(self, audios: list[np.ndarray], contexts: list[str] | None = None,
                   sr: int = 16000) -> list[dict]:
        lang = self.d.get("language", "auto")
        contexts = contexts or [""] * len(audios)
        res = self.model.transcribe(audio=[(a, sr) for a in audios], context=contexts,
                                    language=None if lang in (None, "auto") else lang)
        out = []
        for r, c in zip(res, contexts, strict=True):
            text = unglue(r.text.strip())
            if self.d.get("copy_guard") and c:
                text = cut_copied_tail(text, c)
            if self.d.get("to_simplified", True):
                text = to_simplified(text)
            out.append({"hypothesis": text, "detected_language": r.language})
        return out


def transcribe_manifest(model_path: str, manifests, out_path: str | Path, cfg: dict,
                        transcriber: Transcriber | None = None) -> Path:
    """Decode every row of the manifest(s) into out_path (jsonl: id, hypothesis, detected_language, context_used).
    Resumable: ids already in out_path are skipped."""
    import librosa

    dcfg = cfg["decode"]
    rows = read_manifests(manifests, cfg["data"].get("audio_root"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return out_path
    tr = transcriber or Transcriber(model_path, dcfg)
    bs = int(dcfg.get("batch_size", 16))
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            audios = [librosa.load(r["audio"], sr=16000, mono=True)[0] for r in batch]
            ctxs = [build_context(r, cfg["data"]["context"]) if dcfg.get("use_context", True) else "" for r in batch]
            for r, c, h in zip(batch, ctxs, tr.transcribe(audios, ctxs), strict=True):
                f.write(json.dumps({"id": r["id"], **h, "context_used": bool(c)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[transcribe] {min(i + bs, len(todo))}/{len(todo)}", flush=True)
    return out_path
