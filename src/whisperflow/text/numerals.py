"""Numeral-style normalisation for scoring: make "三点五十", "3:50" and "3点50分" score the same.

References mix styles (benchmark_chinese calibrated labels write "三点五十", our review rule writes "3:50"), so a
model is penalised for either choice. Applied to BOTH reference and hypothesis before the unchanged official scorer
(eval/official.py official_mer_num), so it can only remove style differences, never create errors.

Steps, in order:
  1. times: "三点五十(分)" / "3点50分" / "3:50" → "3点50"; "三点半" → "3点30"; "3:00" → "3点"
  2. percent: "百分之五十" → "50%"
  3. digit-by-digit runs ("二零二四" → "2024"; the official parser keeps only the last digit of these), then other
     multi-character Chinese numerals with the official scorer's own parser ("一千万" → "10000000")
  4. remaining single Chinese digits → digits ("两个" → "2个", "第一" → "第1"); symmetric, so "一些" → "1些" on
     both sides is harmless
"""

from __future__ import annotations

import re

from whisperflow.eval.vendor import benchmark_metrics as _bm

_D = "零一二三四五六七八九两十"
_NUM = rf"(?:\d+|[{_D}]+)"
_SINGLE = {c: str(i) for i, c in enumerate("零一二三四五六七八九")} | {"两": "2"}


def _to_int(s: str) -> str:
    """'五十' → '50', '12' → '12', '零五' → '5'; falls back to the input if it is not a numeral."""
    if s.isdigit():
        return str(int(s))
    if len(s) == 1 and s in _SINGLE:
        return _SINGLE[s]
    if all(c in _SINGLE for c in s):  # digit by digit: 零五 → 05 → 5
        return str(int("".join(_SINGLE[c] for c in s)))
    return _bm._parse_zh_numeral(s) or s


def normalize_numerals(text: str) -> str:
    t = re.sub(r"(\d{1,2}):00(?!\d)", r"\1点", text)                       # 3:00 → 3点
    t = re.sub(r"(\d{1,2}):(\d{2})(?!\d)", r"\1点\2", t)                    # 3:50 → 3点50
    t = re.sub(rf"({_NUM})点半", lambda m: f"{_to_int(m[1])}点30", t)        # 三点半 → 3点30
    t = re.sub(rf"({_NUM})点({_NUM})分?", lambda m: f"{_to_int(m[1])}点{_to_int(m[2])}", t)  # 三点五十(分)
    t = re.sub(rf"百分之({_NUM})", lambda m: f"{_to_int(m[1])}%", t)          # 百分之五十 → 50%
    t = re.sub(r"[零一二三四五六七八九]{2,}", lambda m: "".join(_SINGLE[c] for c in m[0]), t)  # 二零二四 → 2024
    t = _bm._ZH_NUMERAL_RE.sub(lambda m: _bm._parse_zh_numeral(m[0]) or m[0], t)
    return "".join(_SINGLE.get(c, c) for c in t)
