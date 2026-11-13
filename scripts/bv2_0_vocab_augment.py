"""Benchmark v2.0: raise the share of clips whose dictionary is actually used to >= 30% (owner, 2026-11-13).

Before: 155 / 911 Arena clips carried a production vocabulary, but only 7 had a vocabulary term in the ground truth.
After: in every source set >= TARGET of the clips have at least one vocabulary term that occurs in the Arena GT.

Terms come from data/private/benchmark_v2_0/vocab_aug/out_*.jsonl (per clip: 0-3 dictionary-worthy terms copied from the
Arena GT + score 0-3, chosen by LLM agents). Per set, clips already hit count first; then the highest-scoring clips
are added (random tie-break, seed 0). Production vocabularies are kept and the hit terms appended; clips without one get
the hit terms plus 1-4 distractors (score-3 terms picked for other clips that are not in this GT),
shuffled, so the dictionary does not equal the answer.

Clip set and transcript = Arena (reports/arena/bv2_0_J3_step600_ctx_per_clip.csv `ref`), never the old upload sheet.
Outputs (~/Desktop/build_benchmark/benchmark_v2.0/): benchmark_v2.0_vocab_upload.xlsx + vocab_changes.csv
    uv run --offline python scripts/bv2_0_vocab_augment.py
"""

import json
import math
import random
from pathlib import Path

import pandas as pd

OUT = Path.home() / "Desktop/build_benchmark/benchmark_v2.0"
ARENA = Path("reports/arena/bv2_0_J3_step600_ctx_per_clip.csv")
TERMS = Path("data/private/benchmark_v2_0/vocab_aug")
TARGET = 0.32
COLS = ["file", "transcript", "audio_context", "audio_url", "audio_id", "vocabulary", "tone_app_id"]


def split_vocab(v) -> list[str]:
    return [] if pd.isna(v) else [t.strip() for t in str(v).split(";") if t.strip()]


def norm(s: str) -> str:
    return str(s).lower().replace(" ", "")


def hits(terms: list[str], gt: str) -> list[str]:
    return [t for t in terms if len(norm(t)) >= 2 and norm(t) in norm(gt)]


def main() -> None:
    rng = random.Random(0)
    arena = pd.read_csv(ARENA)[["file", "set", "lang", "ref"]]
    up = pd.read_excel(OUT / "benchmark_v2.0_upload.xlsx")
    df = arena.merge(up, on="file", how="left", validate="1:1")
    assert df.audio_id.notna().all(), "Arena clip missing from upload sheet"
    df["transcript"] = df.ref.astype(str)
    picks = {}
    for p in sorted(TERMS.glob("out_*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            picks[r["file"]] = (r["terms"], r["score"])
    assert set(picks) == set(df.file), "term files do not cover the Arena clip set"

    df["orig"] = df.vocabulary.map(split_vocab)
    df["orig_hit"] = [hits(o, g) for o, g in zip(df.orig, df.transcript)]
    df["new_terms"] = [[t for t in picks[f][0] if t in g] for f, g in zip(df.file, df.transcript)]
    df["score"] = [picks[f][1] if nt else 0 for f, nt in zip(df.file, df.new_terms)]
    df["key"] = [rng.random() for _ in range(len(df))]

    df["selected"] = False
    for s, g in df.groupby("set"):
        need = math.ceil(TARGET * len(g)) - len(g[g.orig_hit.str.len() > 0])
        pool = g[(g.orig_hit.str.len() == 0) & (g.score >= 1)].sort_values(["score", "key"], ascending=[False, True])
        df.loc[pool.index[:max(need, 0)], "selected"] = True
        if need > len(pool):
            print(f"WARNING {s}: only {len(pool)} candidates for {need} needed")

    # distractors = strong (score 3) terms picked for other clips; production dictionaries are mostly test junk
    clip_terms = sorted({t for nt, sc in zip(df.new_terms, df.score) if sc == 3 for t in nt})
    vocab, added, distract = [], [], []
    for r in df.itertuples():
        if not r.selected:
            vocab.append(r.vocabulary if r.orig else None); added.append([]); distract.append([])
            continue
        terms = list(r.new_terms)
        if r.orig:
            v, d = r.orig + [t for t in terms if t not in r.orig], []
        else:
            ascii_only = r.lang == "en"
            cands = [t for t in clip_terms
                     if norm(t) not in norm(r.transcript) and (t.isascii() or not ascii_only)]
            d = rng.sample(cands, rng.randint(1, 4))
            v = terms + d
            rng.shuffle(v)
        vocab.append("; ".join(v)); added.append(terms); distract.append(d)
    df["vocabulary"], df["added_terms"], df["distractors"] = vocab, added, distract

    df["has_vocab"] = df.vocabulary.notna()
    df["vocab_hit"] = [len(hits(split_vocab(v), g)) > 0 for v, g in zip(df.vocabulary, df.transcript)]
    summary = df.groupby("set").agg(clips=("file", "size"), has_vocab=("has_vocab", "sum"), hit=("vocab_hit", "sum"))
    summary["hit%"] = (summary.hit / summary.clips * 100).round(1)
    summary.loc["ALL"] = [len(df), df.has_vocab.sum(), df.vocab_hit.sum(), round(df.vocab_hit.mean() * 100, 1)]
    print(summary.to_string())

    df[COLS].to_excel(OUT / "benchmark_v2.0_vocab_upload.xlsx", index=False)
    ch = df[df.selected][["file", "set", "lang", "score", "added_terms", "distractors", "vocabulary", "transcript"]]
    ch.to_csv(OUT / "vocab_changes.csv", index=False, encoding="utf-8-sig")
    print(f"changed {len(ch)} clips -> {OUT / 'benchmark_v2.0_vocab_upload.xlsx'}")


if __name__ == "__main__":
    main()
