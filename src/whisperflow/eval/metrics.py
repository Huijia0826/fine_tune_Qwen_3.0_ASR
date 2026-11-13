"""Scoring primitives: token alignment, MER/CER, language-attributed errors, bootstrap CIs.

Definitions (policy zh_en_v1, see whisperflow.text.normalize):
  * Tokens: one per Chinese character, one per English word / alphanumeric run.
  * MER (mixed error rate) = (S + D + I) / N_ref over those tokens. On pure-Chinese
    references without digits this equals CER; on pure-English it equals WER.
  * Corpus-level rates are computed as sum(errors) / sum(N_ref) — NOT the mean of
    per-utterance rates, which would over-weight short clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whisperflow.text.normalize import token_lang, tokenize


@dataclass
class Alignment:
    ref: list[str]
    hyp: list[str]
    # ops: ("eq"|"sub"|"del"|"ins", ref_token|None, hyp_token|None)
    ops: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    @property
    def n_ref(self) -> int:
        return len(self.ref)

    def count(self, op: str) -> int:
        return sum(1 for o in self.ops if o[0] == op)

    @property
    def errors(self) -> int:
        return sum(1 for o in self.ops if o[0] != "eq")

    def render(self) -> str:
        """Human-readable diff: matches plain, [ref→hyp] subs, [-ref] dels, [+hyp] ins."""
        out = []
        for op, r, h in self.ops:
            if op == "eq":
                out.append(r)
            elif op == "sub":
                out.append(f"[{r}→{h}]")
            elif op == "del":
                out.append(f"[-{r}]")
            else:
                out.append(f"[+{h}]")
        return " ".join(out)


# Integer costs. A cross-language substitution costs 1 unit more than a same-language one,
# which only breaks ties between equally short edit scripts (as long as an utterance has
# < 1000 tokens), so totals are standard Levenshtein but errors get attributed sensibly:
# "看 CER" -> "看 sir 好" aligns CER->sir + ins 好, not ins sir + CER->好.
_EDIT = 1000
_CROSS_LANG_PENALTY = 1


def _sub_cost(r: str, h: str) -> int:
    if r == h:
        return 0
    return _EDIT + (_CROSS_LANG_PENALTY if token_lang(r) != token_lang(h) else 0)


def align(ref: list[str], hyp: list[str]) -> Alignment:
    """Minimum edit distance alignment, deterministic tie-breaking (see _sub_cost)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1) * _EDIT
    d[0, :] = np.arange(m + 1) * _EDIT
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + _sub_cost(ri, hyp[j - 1]),
                          d[i - 1, j] + _EDIT, d[i, j - 1] + _EDIT)
    ops = []
    i, j = n, m
    # Backtrace preference on ties: match > insertion > deletion > substitution. Walking
    # backwards, this places substitutions as early (leftmost) as possible.
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            ops.append(("eq", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and d[i, j] == d[i, j - 1] + _EDIT:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + _EDIT:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
    ops.reverse()
    return Alignment(ref, hyp, ops)


def align_text(ref_text: str, hyp_text: str) -> Alignment:
    return align(tokenize(ref_text), tokenize(hyp_text))


@dataclass
class UttScore:
    n_ref: int
    sub: int
    dele: int
    ins: int
    # language-attributed: ref-side errors (S+D) by ref token language; insertions by hyp lang
    n_ref_by_lang: dict[str, int]
    err_by_lang: dict[str, int]
    en_correct: int

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def mer(self) -> float:
        return self.errors / self.n_ref if self.n_ref else float("nan")


def score_alignment(a: Alignment) -> UttScore:
    n_by: dict[str, int] = {"zh": 0, "en": 0, "num": 0}
    e_by: dict[str, int] = {"zh": 0, "en": 0, "num": 0}
    en_correct = 0
    for t in a.ref:
        n_by[token_lang(t)] += 1
    for op, r, h in a.ops:
        if op == "eq":
            if token_lang(r) == "en":
                en_correct += 1
        elif op in ("sub", "del"):
            e_by[token_lang(r)] += 1
        else:
            e_by[token_lang(h)] += 1
    return UttScore(a.n_ref, a.count("sub"), a.count("del"), a.count("ins"), n_by, e_by, en_correct)


def corpus_rate(scores: list[UttScore]) -> float:
    n = sum(s.n_ref for s in scores)
    return sum(s.errors for s in scores) / n if n else float("nan")


def cluster_bootstrap_ci(scores: list[UttScore], clusters: list[str], *, n_boot: int = 1000,
                         seed: int = 0, alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile CI for corpus MER, resampling *speakers* (clusters) with replacement.

    Utterances from one speaker are correlated, so resampling utterances independently
    would understate uncertainty. Returns None with fewer than 2 clusters.
    """
    by_c: dict[str, list[int]] = {}
    for s, c in zip(scores, clusters, strict=True):
        e, n = by_c.setdefault(c, [0, 0])
        by_c[c] = [e + s.errors, n + s.n_ref]
    if len(by_c) < 2:
        return None
    arr = np.array(list(by_c.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sampled = arr[idx]  # (n_boot, n_clusters, 2)
    rates = sampled[..., 0].sum(1) / np.maximum(sampled[..., 1].sum(1), 1)
    lo, hi = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
