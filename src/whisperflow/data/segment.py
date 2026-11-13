"""Split long recordings (> 30 s) into <= 30 s training pieces, cutting the HUMAN reference text (experiment E3).

Whisper is used only for *timing*: its long-form segments say when each stretch of speech starts and ends.
Its text is aligned to the reference token by token (edit distance, as in MER) to find where each segment
boundary falls inside the reference. Each piece then gets a slice of the original reference transcript.
Pieces whose Whisper text disagrees too much with their reference slice are dropped (probably cut in the
wrong place).

Pure functions here; `scripts/segment_long_clips.py` runs Whisper and writes the audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from whisperflow.text.normalize import _t2s

# One token = one CJK character or one ASCII word/number run (same units as the MER policy).
TOKEN = re.compile(r"[㐀-鿿豈-﫿]|[A-Za-z0-9]+(?:['.][A-Za-z0-9]+)*")


@dataclass
class Tok:
    key: str    # comparison form: Simplified, lower-case
    start: int  # character span in the original text
    end: int


@dataclass
class Segment:
    start: float  # seconds
    end: float
    text: str


@dataclass
class Piece:
    start: float
    end: float
    reference: str   # slice of the human reference (original punctuation and casing kept)
    hypothesis: str  # Whisper's text for the same span, used only for the quality check
    token_error_rate: float


def tokens(text: str) -> list[Tok]:
    conv = _t2s()
    return [Tok(conv.convert(m.group(0)).lower(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def align(ref: list[str], hyp: list[str]) -> list[int]:
    """Levenshtein alignment. Returns ref_at[j] = number of ref tokens consumed when j hyp tokens are consumed,
    for j = 0..len(hyp), along one optimal edit path."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != hyp[j - 1]))
    # backtrace from (n, m); for each j keep the largest i on the path (cut after deletions, i.e. late)
    ref_at = [0] * (m + 1)
    i, j = n, m
    ref_at[m] = n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
        ref_at[j] = max(ref_at[j], i) if j != m else ref_at[j]
    return ref_at


def edit_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def plan_pieces(reference: str, segments: list[Segment], duration: float, *, max_sec: float = 28.0,
                pad: float = 0.15, max_error: float = 0.3) -> tuple[list[Piece], int]:
    """Group consecutive segments into pieces <= max_sec and cut the reference at the aligned boundaries.

    Returns (kept pieces, number dropped by the quality check)."""
    ref_toks = tokens(reference)
    seg_toks = [tokens(s.text) for s in segments]
    hyp_keys = [t.key for st in seg_toks for t in st]
    ref_at = align([t.key for t in ref_toks], hyp_keys)

    # group segments greedily; a single segment longer than max_sec is skipped
    groups, cur = [], []
    for k, s in enumerate(segments):
        if s.end - s.start > max_sec:
            if cur:
                groups.append(cur)
            cur = []
            continue
        if cur and s.end - segments[cur[0]].start > max_sec:
            groups.append(cur)
            cur = []
        cur.append(k)
    if cur:
        groups.append(cur)

    hyp_offset = [0]
    for st in seg_toks:
        hyp_offset.append(hyp_offset[-1] + len(st))

    kept, dropped = [], 0
    for g in groups:
        first, last = g[0], g[-1]
        r0, r1 = ref_at[hyp_offset[first]], ref_at[hyp_offset[last + 1]]
        if r1 <= r0:
            dropped += 1
            continue
        a = ref_toks[r0].start
        b = ref_toks[r1].start if r1 < len(ref_toks) else len(reference)
        ref_slice = reference[a:b].strip()
        hyp_keys_g = hyp_keys[hyp_offset[first]:hyp_offset[last + 1]]
        err = edit_rate([t.key for t in ref_toks[r0:r1]], hyp_keys_g)
        lo = segments[first - 1].end if first > 0 else 0.0
        hi = segments[last + 1].start if last + 1 < len(segments) else duration
        start = max(lo, segments[first].start - pad, 0.0)
        end = min(hi, segments[last].end + pad, duration)
        if err > max_error or end - start > 30.0:
            dropped += 1
            continue
        kept.append(Piece(round(start, 3), round(end, 3), ref_slice,
                          " ".join(segments[k].text.strip() for k in g), round(err, 4)))
    return kept, dropped
