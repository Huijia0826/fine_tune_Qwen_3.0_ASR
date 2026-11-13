"""YAML config with defaults, ${ENV_VAR} / ${ENV_VAR:-default} expansion and dotted command-line overrides.

    cfg = load_config("configs/example.yaml", ["train.lr=1e-5", "decode.max_chunk_sec=30"])

Relative paths in the config are resolved against the directory you run the command from.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "run_name": "exp",
    "work_dir": "runs/${run_name}",
    "model": {"base": "Qwen/Qwen3-ASR-1.7B", "revision": None},
    "data": {
        "train": [],          # manifest(s) used for training
        "val": [],            # manifest(s) used to pick the checkpoint
        "test": {},           # name -> manifest, evaluated once with the chosen checkpoint
        "audio_root": None,   # prefix for relative audio paths (default: the manifest's folder)
        "max_audio_sec": 180,
        "language_names": {"zh": "Chinese", "en": "English"},
        "default_language": "auto",   # auto = Chinese if the text has Chinese characters, else English
        "normalize_label": None,      # optional "module:function" applied to every training label
        "context": {
            "enabled": True,
            "fields": {"context": "context", "app": "app", "tone": "tone", "vocabulary": "vocabulary",
                       "before_cursor": "before_cursor", "after_cursor": "after_cursor"},
            "max_vocab": 50, "before_chars": 200, "after_chars": 120, "leak_ngram": 8,
        },
        "augment": {
            "enabled": False, "seed": 1234,
            "vocab": {"keep": 0.5, "targets_plus_distractors": 0.4, "distractors_only": 0.1,
                      "n_distractors": [3, 10], "lookalikes": True},
            "cursor": {"keep": 0.45, "previous_utterance": 0.2, "other_clip": 0.15, "none": 0.2},
            "distractor_pool": None,  # optional text file, one term per line (added to terms found in labels)
            "banned_terms": None,     # optional text file: words that must never appear in a training vocabulary
        },
    },
    "train": {
        "lr": 2e-5, "lr_scheduler_type": "linear", "warmup_ratio": 0.05,
        "batch_size": 4, "grad_acc": 4, "epochs": 3, "max_steps": -1,
        "save_steps": 200, "save_total_limit": 50, "eval_steps": 0, "log_steps": 10,
        "num_workers": 8, "seed": 1234, "save_only_model": True, "sr": 16000, "resume": False,
    },
    "decode": {
        "language": "auto", "max_chunk_sec": 25, "min_tail_sec": 5, "max_new_tokens": 1024,
        "batch_size": 16, "dtype": "bfloat16", "device": "cuda:0",
        "use_context": True, "copy_guard": False, "to_simplified": True,
    },
    "eval": {"slice_by": ["language_mix"], "bootstrap": 1000},
}

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)(?::-([^}]*))?\}")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _lookup(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _expand(value, cfg: dict):
    if isinstance(value, dict):
        return {k: _expand(v, cfg) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, cfg) for v in value]
    if isinstance(value, str):
        def rep(m):
            name, default = m.group(1), m.group(2)
            got = os.environ.get(name)
            if got is None:
                got = _lookup(cfg, name)
            return str(got) if got is not None else (default if default is not None else m.group(0))
        return _ENV.sub(rep, value)
    return value


def set_dotted(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"override must look like key.sub=value, got {assignment!r}")
    cur = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    val = yaml.safe_load(raw)
    if isinstance(val, str):  # YAML 1.1 reads "1e-5" as a string
        try:
            val = float(val)
        except ValueError:
            pass
    cur[parts[-1]] = val


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    cfg = _merge(DEFAULTS, user or {})
    for o in overrides or []:
        set_dotted(cfg, o)
    for _ in range(3):  # allow references like ${run_name} inside work_dir
        cfg = _expand(cfg, cfg)
    for key in ("train", "val"):
        if isinstance(cfg["data"][key], str):
            cfg["data"][key] = [cfg["data"][key]]
    return cfg
