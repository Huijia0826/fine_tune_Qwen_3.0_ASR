"""Context ("prompt") handling: Qwen3-ASR reads the system prompt as background text for biasing.

A context is built from optional structured fields of a manifest row:

    App: Slack · casual
    Vocabulary: Vercel, PostHog, …
    Before cursor: …<text before the cursor>
    After cursor: <text after the cursor>…

or taken verbatim from a free-text `context` field. All field names, lengths and probabilities come from the
config (`data.context`, `data.augment`), nothing is fixed here.

Context augmentation (optional, training only) teaches the model to USE a vocabulary when the word is spoken and
to IGNORE context that does not match the audio, which prevents two failure modes we saw in practice:
copying the cursor text into the output, and inserting vocabulary words that were not said.
"""

from __future__ import annotations

import difflib
import random
import re

_NORM = re.compile(r"[\s\W_]+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#'-]*")
_STOP = set("""i a an the and or but so to of in on at for with is are was were be been it this that you we they he
    she my your our me us them ok okay yes no not do does did have has had will would can could should just like um
    uh hello hi hey thanks thank please""".split())


def _n(s: str | None) -> str:
    return _NORM.sub("", (s or "").lower())


def leaks(label: str, text: str | None, k: int = 8) -> bool:
    """True if `text` shares a run of k characters (punctuation / spaces ignored) with the label. Such context
    would let the model copy the answer instead of listening, so it is dropped."""
    lab, t = _n(label), _n(text)
    return k > 0 and len(lab) >= k and bool(t) and any(lab[i:i + k] in t for i in range(len(lab) - k + 1))


def as_list(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[;；,，\n]", str(v)) if x.strip()]


def label_terms(label: str) -> list[str]:
    """Proper-noun-like Latin terms in a label (PostHog, GPT4o, Claude Code): inner capital or digit, or a
    capitalised word that is not sentence-initial in English text (any capitalised word in Chinese text)."""
    cjk = bool(re.search(r"[一-鿿]", label))
    terms, cur, last_end = [], [], -1
    for m in _TOKEN.finditer(label):
        t, start = m.group(0).strip("'-"), m.start()
        before = label[:start].rstrip()
        sentence_start = not before or before[-1] in ".!?。！？"
        ok = len(t) >= 2 and t.lower() not in _STOP and (
            bool(re.search(r"[A-Z]", t[1:]) or re.search(r"\d", t))
            or (t[:1].isupper() and (cjk or not sentence_start)))
        if ok and cur and label[last_end:start] == " ":
            cur.append(t)
        else:
            if cur:
                terms.append(" ".join(cur))
            cur = [t] if ok else []
        last_end = m.end()
    if cur:
        terms.append(" ".join(cur))
    return list(dict.fromkeys(terms))


def _clip(text: str, n: int, tail: bool) -> str:
    if n <= 0 or len(text) <= n:
        return text
    return "…" + text[-n:] if tail else text[:n] + "…"


def compose(app: str = "", vocab: list[str] | None = None, before: str = "", after: str = "") -> str:
    lines = [f"App: {app}"] if app else []
    if vocab:
        lines.append("Vocabulary: " + ", ".join(vocab))
    if before:
        lines.append("Before cursor: " + before)
    if after:
        lines.append("After cursor: " + after)
    return "\n".join(lines)


def fields(row: dict, ccfg: dict) -> tuple[str, list[str], str, str]:
    """(app line, vocabulary, before, after) from a manifest row, field names from config."""
    names = ccfg.get("fields", {})
    app = str(row.get(names.get("app", "app")) or "").strip()
    tone = str(row.get(names.get("tone", "tone")) or "").strip()
    if app and tone:
        app = f"{app} · {tone}"
    vocab = as_list(row.get(names.get("vocabulary", "vocabulary")))[: int(ccfg.get("max_vocab", 50))]
    before = str(row.get(names.get("before_cursor", "before_cursor")) or "").strip()
    after = str(row.get(names.get("after_cursor", "after_cursor")) or "").strip()
    return app, vocab, before, after


def build_context(row: dict, ccfg: dict, label: str | None = None) -> str:
    """Context for one row. A free-text `context` field wins; otherwise compose from structured fields.
    With a label (training), cursor text that leaks the label is dropped."""
    if not ccfg.get("enabled", True):
        return ""
    free = row.get(ccfg.get("fields", {}).get("context", "context"))
    if free:
        return str(free)
    app, vocab, before, after = fields(row, ccfg)
    k = int(ccfg.get("leak_ngram", 8))
    if label is not None:
        before = "" if leaks(label, before, k) else before
        after = "" if leaks(label, after, k) else after
    return compose(app, vocab, _clip(before, int(ccfg.get("before_chars", 200)), True),
                   _clip(after, int(ccfg.get("after_chars", 120)), False))


class Augmenter:
    """Seeded context augmentation for training rows (see module docstring). Probabilities from `data.augment`."""

    def __init__(self, acfg: dict, ccfg: dict, pool: list[str], banned: set[str]):
        self.a, self.c = acfg, ccfg
        self.rng = random.Random(int(acfg.get("seed", 1234)))
        self.banned = {b.lower() for b in banned}
        self.pool = sorted({t for t in pool if t.lower() not in self.banned})
        self.k = int(ccfg.get("leak_ngram", 8))

    def _pick(self, probs: dict) -> str:
        r, acc = self.rng.random(), 0.0
        for name, p in probs.items():
            acc += float(p)
            if r < acc:
                return name
        return list(probs)[-1]

    def vocabulary(self, logged: list[str], label: str) -> list[str]:
        v = self.a.get("vocab", {})
        lo, hi = v.get("n_distractors", [3, 10])
        rel = [t for t in label_terms(label) if t.lower() not in self.banned]
        lab_l = label.lower()
        cands = [t for t in self.pool if t.lower() not in lab_l]
        kind = self._pick({"keep": v.get("keep", 0.5), "targets_plus_distractors": v.get("targets_plus_distractors", 0.4),
                           "distractors_only": v.get("distractors_only", 0.1)})
        if kind == "keep" or (kind == "targets_plus_distractors" and not rel) or not cands:
            return logged
        distract = self.rng.sample(cands, min(len(cands), self.rng.randint(int(lo), int(hi))))
        if v.get("lookalikes", True):
            for t in rel:  # look-alike distractors (Vercel / Versal): the model has to listen
                alike = [c for c in difflib.get_close_matches(t, cands, n=3, cutoff=0.6) if c.lower() != t.lower()]
                distract += alike[:1]
        vocab = distract if kind == "distractors_only" else self.rng.sample(rel, min(len(rel), self.rng.randint(1, 3))) + distract
        self.rng.shuffle(vocab)
        return vocab[: int(self.c.get("max_vocab", 50))]

    def context(self, row: dict, label: str, prev_label: str = "", other: tuple[str, str] | None = None) -> str:
        if self.c.get("fields", {}).get("context", "context") in row and row.get(self.c.get("fields", {}).get("context", "context")):
            return build_context(row, self.c, label)  # free-text context: no structured augmentation
        app, logged, before, after = fields(row, self.c)
        vocab = self.vocabulary(logged, label)
        cur = self.a.get("cursor", {})
        kind = self._pick({"keep": cur.get("keep", 0.45), "previous_utterance": cur.get("previous_utterance", 0.2),
                           "other_clip": cur.get("other_clip", 0.15), "none": cur.get("none", 0.2)})
        if kind == "previous_utterance" and prev_label:
            before, after = prev_label, ""
        elif kind == "other_clip" and other:
            before, after = other
        elif kind == "none":
            before, after = "", ""
            app = app if self.rng.random() < 0.5 else ""
        before = "" if leaks(label, before, self.k) else before
        after = "" if leaks(label, after, self.k) else after
        lo_b, hi_b = self.a.get("before_chars_range", [50, int(self.c.get("before_chars", 200))])
        lo_a, hi_a = self.a.get("after_chars_range", [30, int(self.c.get("after_chars", 120))])
        before = _clip(before, self.rng.randint(int(lo_b), int(hi_b)), True) if before else ""
        after = _clip(after, self.rng.randint(int(lo_a), int(hi_a)), False) if after else ""
        return compose(app, vocab, before, after)
