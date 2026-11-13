"""Official benchmark scorer: vendored copy of ~/Desktop/build_benchmark/metrics.py.

This is the scorer that produced the stored scores in benchmark v1.0 (verified: it reproduces
100% of the arena WER column for all three systems). Use its MER as the headline metric so our
numbers are directly comparable with arena runs. The file is pinned by sha256; if the upstream
scorer changes, re-vendor deliberately and bump OFFICIAL_SCORER_VERSION.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from whisperflow.eval.vendor import benchmark_metrics as _bm

OFFICIAL_SCORER_VERSION = "build_benchmark/metrics.py@413d0a8e"
_EXPECTED_SHA256 = "413d0a8e7933724ce2ab81da2078ee6bddc2bb7a852d6b60b612a8d4123d5f2f"

_actual = hashlib.sha256(Path(_bm.__file__).read_bytes()).hexdigest()
if _actual != _EXPECTED_SHA256:
    raise RuntimeError(f"vendored benchmark_metrics.py changed ({_actual[:12]}), refusing to score")


@dataclass
class OfficialScore:
    errors: int
    n_ref: int
    included: bool  # official scorer excludes empty references from pooled rates

    @property
    def rate(self) -> float:
        return self.errors / self.n_ref if self.n_ref else 0.0


def official_mer(reference: str, hypothesis: str) -> OfficialScore:
    r = _bm.score_one(reference, hypothesis, "mer")
    return OfficialScore(r.edit_distance, r.reference_token_count, r.included_in_batch_summary)


def official_mer_t2s(reference: str, hypothesis: str) -> OfficialScore:
    """Script-insensitive variant: full OpenCC Traditional->Simplified on both sides, then the
    unchanged official scorer. Answers "was the content heard correctly?" whereas plain
    official MER also penalises Traditional output (only a 40-char table is built in)."""
    from whisperflow.text.normalize import _t2s

    t2s = _t2s().convert
    return official_mer(t2s(reference), t2s(hypothesis))


def official_mer_num(reference: str, hypothesis: str) -> OfficialScore:
    """Script- and numeral-style-insensitive: full t2s plus text/numerals.normalize_numerals on both sides, then the
    unchanged official scorer. "三点五十" vs "3:50" no longer counts as errors (2026-11-02)."""
    from whisperflow.text.normalize import _t2s
    from whisperflow.text.numerals import normalize_numerals

    t2s = _t2s().convert
    return official_mer(normalize_numerals(t2s(reference)), normalize_numerals(t2s(hypothesis)))


_LATIN_WORD = re.compile(r"[A-Za-z0-9]+")
MAX_COMPOUND_WORDS = 4


def _latin_runs(text: str) -> list[list[re.Match]]:
    """Latin words grouped into runs; a run continues only across plain spaces ("top step X", not "top, step")."""
    runs: list[list[re.Match]] = []
    for m in _LATIN_WORD.finditer(text):
        if runs and text[runs[-1][-1].end():m.start()].isspace():
            runs[-1].append(m)
        else:
            runs.append([m])
    return runs


def merge_compounds(reference: str, hypothesis: str) -> tuple[str, list[tuple[str, str]]]:
    """Rewrite hypothesis spans that differ from a reference span only by spacing / case of English words.

    A span of 1..4 consecutive hypothesis words is replaced by a reference span of a DIFFERENT word count whose
    words, joined without spaces and lower-cased, are identical: "top step X" -> "TopstepX", "InsideEntry" ->
    "inside entry", "Topstep X" -> "topstepX". Longest hypothesis span first, left to right. Returns the rewritten
    hypothesis and the (hypothesis span, reference span) pairs rewritten. Chinese, misheard words and numerals are
    untouched."""
    ref_spans: dict[str, tuple[str, int]] = {}
    for run in _latin_runs(reference):
        for i in range(len(run)):
            for n in range(1, min(MAX_COMPOUND_WORDS, len(run) - i) + 1):
                ws = run[i:i + n]
                key = "".join(w.group().lower() for w in ws)
                ref_spans.setdefault(key, (reference[ws[0].start():ws[-1].end()], n))
    out, last, merged = [], 0, []
    for run in _latin_runs(hypothesis):
        j = 0
        while j < len(run):
            for n in range(min(MAX_COMPOUND_WORDS, len(run) - j), 0, -1):
                ws = run[j:j + n]
                key = "".join(w.group().lower() for w in ws)
                hit = ref_spans.get(key)
                if hit and hit[1] != n:
                    out.append(hypothesis[last:ws[0].start()])
                    out.append(hit[0])
                    merged.append((hypothesis[ws[0].start():ws[-1].end()], hit[0]))
                    last = ws[-1].end()
                    j += n
                    break
            else:
                j += 1
    out.append(hypothesis[last:])
    return "".join(out), merged


def official_mer_compound(reference: str, hypothesis: str) -> OfficialScore:
    """Diagnostic (2026-11-02): official MER after merge_compounds, i.e. English word spacing / compounding is not
    counted ("TopstepX" vs "top step X" = 0 errors instead of 3). The official MER stays the headline."""
    return official_mer(reference, merge_compounds(reference, hypothesis)[0])
