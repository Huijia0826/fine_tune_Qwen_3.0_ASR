"""Remove hesitation fillers from training transcripts (exercise 2, see docs/learning/exercises.md).

Decisions: 2026-10-29 delete 呃 and 嗯 (incl. runs like 嗯嗯); 2026-10-30 also delete English um/uh (lora_v1
inserted 80 more um/uh than the baseline on benchmark v1 English clips). Keep 啊, 哦/噢, 额 (额度).
"""

from __future__ import annotations

import re

FILLERS = re.compile(r"[呃嗯]+")  # one or more 呃/嗯 in a row
# um / uh (umm, uhh) as whole English words. Lookarounds instead of \b: in Python, CJK characters count as
# word characters, so \b would miss "uh" glued to Chinese ("uh正在").
EN_FILLERS = re.compile(r"(?<![A-Za-z'-])(?:um+|uh+)(?![A-Za-z'-])", re.IGNORECASE)
SPACES = re.compile(r"\s+")


def strip_fillers(text: str) -> str:
    """Return text with every 呃/嗯 and English um/uh removed and whitespace tidied.

    - "嗯 我觉得" -> "我觉得"; "我 嗯 觉得" -> "我 觉得"; "嗯嗯" -> ""; "um I found one" -> "I found one"
    - no leading/trailing spaces, no double spaces left behind
    - everything else is unchanged (额度, 好啊, 哦, umbrella, Uber)
    """
    text = FILLERS.sub("", text)             # 1. delete the fillers
    text = EN_FILLERS.sub("", text)
    return SPACES.sub(" ", text).strip()     # 2. collapse leftover spaces, trim the ends
