"""Text normalization and mixed Chinese / English tokenization used for scoring.

Scoring tokens: one token per Chinese character, one per English word (or alphanumeric run such as
"gpt4o", "node.js", "c++"). Punctuation, whitespace and symbols are dropped, English is lower-cased,
full-width characters are folded (NFKC) and Traditional Chinese is converted to Simplified when
OpenCC is installed (`pip install opencc-python-reimplemented`).

Normalization is applied to evaluation copies only; labels in your manifests are never modified.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from functools import lru_cache

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF))
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*[+#]*")


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def has_cjk(text: str) -> bool:
    return any(is_cjk(c) for c in text)


@lru_cache(maxsize=1)
def _t2s():
    try:
        import opencc
    except ImportError:
        warnings.warn("opencc not installed: Traditional Chinese is not converted to Simplified "
                      "(pip install opencc-python-reimplemented)", stacklevel=2)
        return None
    for cfg in ("t2s", "t2s.json"):
        try:
            return opencc.OpenCC(cfg)
        except Exception:  # noqa: BLE001 - different opencc builds name the config differently
            continue
    return None


def to_simplified(text: str) -> str:
    conv = _t2s()
    return conv.convert(text) if conv else text


def tokenize(text: str) -> list[str]:
    """Scoring tokens: one per Chinese character, one per English word / alphanumeric run."""
    text = to_simplified(unicodedata.normalize("NFKC", text or "")).lower()
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if is_cjk(ch):
            tokens.append(ch)
            i += 1
            continue
        m = _LATIN_TOKEN_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        i += 1
    return tokens


def token_lang(token: str) -> str:
    """'zh' for a Chinese character, 'num' for digits, else 'en'."""
    if len(token) == 1 and is_cjk(token):
        return "zh"
    if token.replace(".", "").isdigit():
        return "num"
    return "en"


def language_mix(text: str) -> str:
    """'zh', 'en', 'zh-en' or 'none' — used to slice results when a manifest has no language field."""
    langs = {token_lang(t) for t in tokenize(text)} - {"num"}
    if langs == {"zh"}:
        return "zh"
    if langs == {"en"}:
        return "en"
    return "zh-en" if langs else "none"
