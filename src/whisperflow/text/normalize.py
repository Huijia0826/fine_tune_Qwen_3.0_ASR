"""Versioned text normalization + mixed-language tokenization for scoring.

Policy `zh_en_v1` (see docs/dataset_spec.md §Normalization). Normalization is applied
to *evaluation copies* only; original labels in the manifest are never modified.

Any change to behaviour here must bump POLICY_VERSION, because scores computed under
different policies are not comparable.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

POLICY_VERSION = "zh_en_v1"

# CJK Unified Ideographs + Extension A + Compatibility Ideographs. Each one is a scoring token.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
)

# A "Latin token" is a maximal run of ASCII letters/digits, allowing a few intra-token
# characters that matter for technical vocabulary: node.js, 3.5, don't, C++, C#.
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*[+#]*")


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


@lru_cache(maxsize=1)
def _t2s():
    import opencc

    return opencc.OpenCC("t2s")


def normalize(text: str) -> str:
    """Return the canonical evaluation form of `text` under policy zh_en_v1.

    Steps (order matters):
      1. Unicode NFKC: full-width ASCII -> half-width, compatibility forms folded.
      2. Traditional -> Simplified Chinese (OpenCC t2s). Whisper frequently emits
         Traditional characters for Mandarin; we score characters, not script choice.
      3. Lowercase (English casing is a formatting concern, not verbatim recognition).
      4. Tokenize into CJK chars + Latin tokens; everything else (punctuation, symbols,
         emoji) is dropped. Punctuation is scored separately, never inside CER/MER.
      5. Join tokens with single spaces.

    Deliberately NOT done in v1 (documented limitations):
      - No numeral normalization (二零二四 vs 2024 count as errors).
      - No filler removal (verbatim scoring keeps 嗯/呃/um).
      - No English spelling variant / compound folding (e-mail vs email).
    """
    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    """Mixed-language scoring tokens: one token per Chinese character, one per English word."""
    text = unicodedata.normalize("NFKC", text)
    text = _t2s().convert(text)
    text = text.lower()

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
        i += 1  # punctuation, whitespace, symbols, other scripts
    return tokens


def token_lang(token: str) -> str:
    """'zh' for a CJK char token, 'num' for pure digits, else 'en'."""
    if len(token) == 1 and is_cjk(token):
        return "zh"
    if token.replace(".", "").isdigit():
        return "num"
    return "en"


def detect_languages(text: str) -> set[str]:
    """Languages observed in a transcript; digits alone do not imply English."""
    langs = {token_lang(t) for t in tokenize(text)}
    langs.discard("num")
    return langs


def contains_term(text: str, term: str) -> bool:
    """Exact, token-boundary-aware term match after normalization (no alias handling in v1).

    "CER" matches "看 CER 吧" but not "CERtain"; "模型" matches "...大模型..." because Chinese
    has no word boundaries at the character level.
    """
    t = tokenize(term)
    if not t:
        return False
    toks = tokenize(text)
    k = len(t)
    return any(toks[i:i + k] == t for i in range(len(toks) - k + 1))
