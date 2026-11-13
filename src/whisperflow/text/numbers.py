"""English number words -> digits for training labels (decision 2026-10-30).

Matches the benchmark v1 house style: 1-9 stay spelled out ("four of us", "two kids"); 10 and above become
digits ("forty five" -> "45", "one hundred years" -> "100 years").
Chinese numerals are never touched (一个/一些/第一 are words).
Known limitation: years spoken in pairs ("twenty twenty four") become "20 24".
"""

from __future__ import annotations

from text_to_num import alpha2digit

THRESHOLD = 9  # numbers <= 9 keep their spelling unless part of a sequence ("one two three" -> "1 2 3")


def en_numbers_to_digits(text: str) -> str:
    return alpha2digit(text, "en", threshold=THRESHOLD)
