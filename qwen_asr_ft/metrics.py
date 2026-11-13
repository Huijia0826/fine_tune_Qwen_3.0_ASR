"""Mixed error rate (MER) and speaker-level bootstrap statistics.

MER = (substitutions + deletions + insertions) / reference tokens, with tokens from `text.tokenize`
(Chinese characters + English words). On pure Chinese this is CER, on pure English WER.
Corpus rates are pooled (sum of errors / sum of reference tokens), never the mean of per-utterance rates.
Empty references are left out of pooled rates.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .text import tokenize


@dataclass
class Score:
    n_ref: int
    sub: int
    dele: int
    ins: int

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def mer(self) -> float:
        return self.errors / self.n_ref if self.n_ref else float("nan")


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of a minimum edit-distance alignment."""
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]  # (cost, sub, del, ins)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = prev[j - 1]
            else:
                c_sub, c_del, c_ins = prev[j - 1], prev[j], cur[j - 1]
                best = min((c_sub[0] + 1, c_sub[1] + 1, c_sub[2], c_sub[3]),
                           (c_del[0] + 1, c_del[1], c_del[2] + 1, c_del[3]),
                           (c_ins[0] + 1, c_ins[1], c_ins[2], c_ins[3] + 1))
            cur.append(best)
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


def score(reference: str, hypothesis: str) -> Score:
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    s, d, i = edit_ops(ref, hyp)
    return Score(len(ref), s, d, i)


def pooled_mer(scores: list[Score]) -> float:
    n = sum(s.n_ref for s in scores)
    return 100 * sum(s.errors for s in scores) / n if n else float("nan")


def speaker_bootstrap_ci(scores: list[Score], speakers: list[str], n_boot: int = 1000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile CI of pooled MER (in %), resampling speakers with replacement (clips of one speaker stay
    together). None with fewer than 2 speakers."""
    by_spk: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s, spk in zip(scores, speakers, strict=True):
        by_spk[spk][0] += s.errors
        by_spk[spk][1] += s.n_ref
    groups = list(by_spk.values())
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(n_boot):
        e = n = 0
        for g in rng.choices(groups, k=len(groups)):
            e, n = e + g[0], n + g[1]
        rates.append(100 * e / n if n else 0.0)
    rates.sort()
    return rates[int(alpha / 2 * n_boot)], rates[int((1 - alpha / 2) * n_boot) - 1]


def paired_bootstrap(a: list[Score], b: list[Score], speakers: list[str], n_boot: int = 2000,
                     seed: int = 1234) -> tuple[float, float, float]:
    """Difference B − A of pooled MER (percentage points) on the same clips, with a 95% interval from a paired
    speaker bootstrap. An interval that contains 0 means the difference is not significant."""
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, spk in enumerate(speakers):
        by_spk[spk].append(i)
    spks = sorted(by_spk)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [i for s in rng.choices(spks, k=len(spks)) for i in by_spk[s]]
        diffs.append(pooled_mer([b[i] for i in idx]) - pooled_mer([a[i] for i in idx]))
    diffs.sort()
    return pooled_mer(b) - pooled_mer(a), diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]
