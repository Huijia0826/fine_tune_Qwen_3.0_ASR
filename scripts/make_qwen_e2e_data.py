"""Qwen3-ASR SFT data with production context (run J1, owner 2026-11-09): I2's data without ASCEND, labels brought
to annotation policy v1, and the context production logged for each clip passed as Qwen's system prompt.

Context string (Qwen3-ASR reads the system prompt as background text, not as instructions):
    App: 微信 · casual
    Vocabulary: Vercel, PostHog, …          (user dictionary, ≤ 50 terms, logged order)
    Before cursor: …<last 200 chars>
    After cursor: <first 120 chars>…
Screen text is left out in J1 (long, noisy, slows prefill). A cursor field that shares an 8-character run with the
label is dropped (leak: e.g. the owner read recording scripts that were open in the same document). Clips without
production context get an empty context, so the model also learns to work without it.

Also writes contexts.jsonl {utterance_id: context} for every source with pulled context (val / test sets included),
read at decoding time by the qwen3_asr backend (decoding.context_file).

With --augment (run J2, owner 2026-11-09; design in reports/qwen3asr_J1.md "Next" and the 2026-11-09 conversation),
each TRAIN clip gets one seeded variant (same number of rows as J1, so J1 vs J2 differ only by augmentation):
  vocabulary  A 40% logged dictionary as is · B 40% 1–3 terms that occur in the label + 5–30 distractors (incl.
              look-alike terms), shuffled · C 20% distractors only
  cursor text A 45% the clip's own (leak rule) · B 20% "previous utterance" trap: before-cursor = the same
              speaker's previous clip label (previous piece of the same long recording first) · C 15% another
              clip's cursor text · D 20% none
  lengths     before cursor cut to 50–200 chars, after cursor to 30–120
segmented_long pieces get before-cursor text only on the first piece and after-cursor text only on the last.
Distractor / term glossary: Latin terms in our training labels + recording-script targets, minus every target
word of the dictionary benchmark and of recording-script test rows (no test leakage).

    uv run python scripts/make_qwen_e2e_data.py --out runs/qwen_sft/J1
    uv run python scripts/make_qwen_e2e_data.py --out runs/qwen_sft/J2 --augment
"""

import argparse
import csv
import difflib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from whisperflow.text.label_style import to_label_v1

TRAIN = ["benchmark_chinese", "online_phase1", "online_pool_b1full", "recording_v1", "recording_v2",
         "segmented_long", "arena_019f6442", "arena_019f8e51"]
REPEAT = {"recording_v1": 4, "recording_v2": 4}  # oversampling as in I2
VAL = ["benchmark_chinese", "online_phase1"]  # users' val
MAX_AUDIO_SEC = 30.0
QWEN_LANG = {"en": "English", "zh": "Chinese"}


def _n(s: str | None) -> str:
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def leaks(label: str, text: str | None, k: int = 8) -> bool:
    lab, t = _n(label), _n(text)
    return len(lab) >= k and bool(t) and any(lab[i:i + k] in t for i in range(len(lab) - k + 1))


def as_list(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[;；,\n]", str(v)) if x.strip()]


def user_vocab(c: dict) -> list[str]:
    """The user's dictionary only. ctx_vocabulary is words the client extracted from the screen, not a dictionary;
    left out like screen text (in the owner's recordings it holds the script's line ids and test sentences)."""
    return as_list(c.get("effective_vocabulary"))[:50]


def cursor_text(c: dict) -> tuple[str, str]:
    """Before / after cursor text; none for the owner's recordings (the script, incl. test sentences, was open in
    the same document, 2026-11-09)."""
    if (c.get("source") or "").startswith("recording"):
        return "", ""
    return (c.get("text_before_cursor") or "").strip(), (c.get("text_after_cursor") or "").strip()


def build_context(c: dict, label: str, stats: Counter) -> str:
    if not c:
        return ""
    lines = []
    if c.get("app_name"):
        lines.append(f"App: {c['app_name']} · {c.get('tone_mapped') or 'unknown'}")
    vocab = user_vocab(c)
    if vocab:
        lines.append("Vocabulary: " + ", ".join(vocab))
    before, after = cursor_text(c)
    for name, text in (("before", before), ("after", after)):
        if text and leaks(label, text):
            stats[f"leak_dropped_{name}"] += 1
    if before and not leaks(label, before):
        lines.append("Before cursor: " + ("…" if len(before) > 200 else "") + before[-200:])
    if after and not leaks(label, after):
        lines.append("After cursor: " + after[:120] + ("…" if len(after) > 120 else ""))
    return "\n".join(lines)


# ---- J2 augmentation -------------------------------------------------------------------------------------------
STOP = set("""i a an the and or but so to of in on at for with is are was were be been it this that you we they
    he she my your our me us them ok okay yes no not do does did have has had will would can could should just like
    um uh hello hi hey thanks thank please""".split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#'-]*")


def label_terms(label: str) -> list[str]:
    """Proper-noun-like Latin terms in a label. A token counts if it has an inner capital or a digit (PostHog,
    GPT4o), or starts with a capital and is not sentence-initial in English text (in Chinese text any capitalised
    word counts). Adjacent qualifying tokens separated by one space form one term (Claude Code); never across
    punctuation.
    """
    has_cjk = bool(re.search(r"[一-鿿]", label))
    terms, cur, last_end = [], [], -1
    for m in TOKEN.finditer(label):
        t, start = m.group(0).strip("'-"), m.start()
        before = label[:start].rstrip()
        sentence_start = not before or before[-1] in ".!?。！？"
        ok = len(t) >= 2 and t.lower() not in STOP and (
            bool(re.search(r"[A-Z]", t[1:]) or re.search(r"\d", t))
            or (t[0].isupper() and (has_cjk or not sentence_start)))
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


def build_glossary(train_labels: list[str]) -> tuple[list[str], set[str]]:
    """Distractor pool + the set of words that must never appear in a training vocabulary (test targets)."""
    banned = set()
    dict_items = Path("data/private/e2e/items_dict.jsonl")
    if dict_items.exists():
        for it in map(json.loads, open(dict_items, encoding="utf-8")):
            banned |= {w.lower() for w in it["target_vocab"]}
    terms = Counter()
    for f in ("script_v1", "script_v2"):
        path = Path(f"data/private/recording_script/{f}.csv")
        if not path.exists():
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            words = [w for w in r["targets"].split() if len(w) > 1]
            if r.get("split") == "test":
                banned |= {w.lower() for w in words}
            else:
                terms.update(words)
    for lab in train_labels:
        terms.update(label_terms(lab))
    pool = sorted(t for t in terms if t.lower() not in banned)
    return pool, banned


VOCAB_MILD = False  # J3: --vocab-mild


def vocab_variant(rng: random.Random, logged: list[str], label: str, pool: list[str],
                  banned: set[str], rel: list[str] | None = None) -> tuple[str, list[str]]:
    rel = [t for t in (rel if rel is not None else label_terms(label)) if t.lower() not in banned]
    r = rng.random()
    if VOCAB_MILD:  # J3 (J2's heavy English distractors destabilised English output)
        lab_l = label.lower()
        cands = [t for t in pool if t.lower() not in lab_l]
        if r < 0.1:
            return "C", rng.sample(cands, min(len(cands), rng.randint(3, 10)))
        if rel and r < 0.6:
            vocab = rng.sample(rel, min(len(rel), rng.randint(1, 3))) + rng.sample(cands, min(len(cands),
                                                                                         rng.randint(3, 10)))
            rng.shuffle(vocab)
            return "B", vocab
        return "A", logged
    if r < 0.4 or (r < 0.8 and not rel):
        return "A", logged
    lab_l = label.lower()
    cands = [t for t in pool if t.lower() not in lab_l]
    distract = rng.sample(cands, min(len(cands), rng.randint(5, 30)))
    for t in rel:  # look-alike distractors (Vercel / Versal): the model has to listen, not pattern-match
        alike = [c for c in difflib.get_close_matches(t, cands, n=3, cutoff=0.6) if c.lower() != t.lower()]
        distract += alike[:1]
    if r < 0.8:
        vocab = rng.sample(rel, min(len(rel), rng.randint(1, 3))) + distract
        kind = "B"
    else:
        vocab, kind = distract, "C"
    rng.shuffle(vocab)
    return kind, vocab[:50]


def compose(app_line: str, vocab: list[str], before: str, after: str) -> str:
    lines = [app_line] if app_line else []
    if vocab:
        lines.append("Vocabulary: " + ", ".join(vocab))
    if before:
        lines.append("Before cursor: " + before)
    if after:
        lines.append("After cursor: " + after)
    return "\n".join(lines)


def cut(rng: random.Random, text: str, lo: int, hi: int, tail: bool) -> str:
    n = rng.randint(lo, hi)
    if len(text) <= n:
        return text
    return "…" + text[-n:] if tail else text[:n] + "…"


def augmented_context(rng, m, c, label, prev_label, other_ctx, pool, banned, stats) -> str:
    app_line = f"App: {c['app_name']} · {c.get('tone_mapped') or 'unknown'}" if c.get("app_name") else ""
    kind_v, vocab = vocab_variant(rng, user_vocab(c), label, pool, banned)
    stats[f"vocab_{kind_v}"] += 1
    before, after = cursor_text(c)
    pos = m.get("_piece")  # segmented_long: (index, count)
    if pos:
        before = before if pos[0] == 0 else ""
        after = after if pos[0] == pos[1] - 1 else ""
    before = "" if leaks(label, before) else before
    after = "" if leaks(label, after) else after
    r = rng.random()
    if r < 0.45:
        kind = "A"
    elif r < 0.65 and prev_label and not leaks(label, prev_label):
        kind, before, after = "B", prev_label, ""
    elif r < 0.80 and other_ctx:
        o = rng.choice(other_ctx)
        kind, before, after = "C", o[0], o[1]
        before = "" if leaks(label, before) else before
        after = "" if leaks(label, after) else after
    else:
        kind, before, after = "D", "", ""
        app_line = app_line if rng.random() < 0.5 else ""
    stats[f"cursor_{kind}"] += 1
    return compose(app_line, vocab, cut(rng, before, 50, 200, True) if before else "",
                   cut(rng, after, 30, 120, False) if after else "")


def clip_language(tags: list[str]) -> str:
    return "en" if set(tags or []) == {"en"} else "zh"


def extra_train(args, contexts, root, rng, pool, banned, stats, n_ctx) -> list[dict]:
    """J3 additions: whole long real recordings and synthetic TTS clips (with their own written context)."""
    rows = []
    if args.long_real:
        parents = {json.loads(line)["parent_utterance_id"]
                   for line in open("data/manifests/segmented_long.jsonl", encoding="utf-8")}
        for src in ("benchmark_chinese", "arena_019f8e51", "arena_019f6442"):
            for m in map(json.loads, open(f"data/manifests/{src}.jsonl", encoding="utf-8")):
                if m["utterance_id"] not in parents or not 30 < m["duration_sec"] <= 180:
                    continue
                label = to_label_v1(m["raw_transcript"])
                c = contexts.get(m["utterance_id"], {})
                ctx = augmented_context(rng, m, c, label, None, [], pool, banned, stats) if args.augment \
                    else build_context(c, label, stats)
                lang = clip_language(m.get("language_tags"))
                rows.append({"audio": str(root / m["audio_path"]), "prompt": ctx,
                             "utterance_id": m["utterance_id"],
                             "text": f"language {QWEN_LANG[lang]}<asr_text>{label}"})
                n_ctx["long_real"] += 1
    if args.concat:  # J4: real short clips joined into 30–60 s (scripts/synth/concat_long.py); labels already v1
        for m in map(json.loads, open(args.concat, encoding="utf-8")):
            lang = "en" if m["language_tags"] == ["en"] else "zh"
            rows.append({"audio": str(root / m["audio_path"]), "prompt": "", "utterance_id": m["utterance_id"],
                         "text": f"language {QWEN_LANG[lang]}<asr_text>{m['raw_transcript']}"})
            n_ctx["concat_long"] += 1
    drop = set(json.load(open(args.drop))) if Path(args.drop).exists() else set()
    other = [(t["before_cursor"], t["after_cursor"]) for spec in args.synth
             for t in map(json.loads, open(spec.split(":")[1], encoding="utf-8")) if t["before_cursor"]]
    for spec in args.synth:
        man_path, texts_path, aug_dir = spec.split(":")
        texts = {t["id"]: t for t in map(json.loads, open(texts_path, encoding="utf-8"))}
        for m in map(json.loads, open(man_path, encoding="utf-8")):
            if m["utterance_id"] in drop:
                continue
            t = texts[m["source_utterance_id"]]
            label = m["raw_transcript"]
            kind_v, vocab = vocab_variant(rng, [], label, pool, banned, rel=t["terms"])
            stats[f"synth_vocab_{kind_v}"] += 1
            r = rng.random()
            before, after = (t["before_cursor"], t["after_cursor"]) if r < 0.65 else \
                (rng.choice(other) if r < 0.8 and other else ("", ""))
            ctx = compose(f"App: {t['app']} · {t['tone']}", vocab,
                          cut(rng, before, 50, 200, True) if before else "",
                          cut(rng, after, 30, 120, False) if after else "")
            aug = Path(aug_dir) / Path(m["audio_path"]).name
            audio = aug.resolve() if aug.exists() else root / m["audio_path"]
            lang = "en" if m["language_tags"] == ["en"] else "zh"
            rows.append({"audio": str(audio), "prompt": ctx, "utterance_id": m["utterance_id"],
                         "text": f"language {QWEN_LANG[lang]}<asr_text>{label}"})
            n_ctx["synth"] += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-contexts", nargs="*", default=[], help="jsonl {utterance_id, context} appended as is")
    ap.add_argument("--augment", action="store_true", help="J2 context augmentation for the train split")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab-mild", action="store_true", help="J3: lighter vocabulary augmentation")
    ap.add_argument("--long-real", action="store_true",
                    help="J3: add the 100 whole long recordings behind segmented_long (30–180 s)")
    ap.add_argument("--synth", nargs="*", default=[],
                    help="J3: manifest:texts.jsonl:augmented_audio_dir triples of synthetic data")
    ap.add_argument("--drop", default="data/private/synth/qc_drop.json", help="utterance ids failing QC")
    ap.add_argument("--concat", default="", help="J4: concatenated real long clips manifest (30–60 s, no context)")
    args = ap.parse_args()
    global VOCAB_MILD
    VOCAB_MILD = args.vocab_mild
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path("data/private").resolve()
    stats, summary = Counter(), {}
    contexts = {}
    for ctx_file in sorted(Path("data/private/context_pull").glob("[!_]*.jsonl")):
        for c in map(json.loads, open(ctx_file, encoding="utf-8")):
            contexts[c["utterance_id"]] = c
    rng = random.Random(args.seed)
    train_recs = [m for src in TRAIN for m in map(json.loads, open(f"data/manifests/{src}.jsonl",
                                                                       encoding="utf-8"))
                  if m["split"] == "train" and m["duration_sec"] <= MAX_AUDIO_SEC]
    pool, banned = ([], set())
    if args.augment:
        pool, banned = build_glossary([to_label_v1(m["raw_transcript"]) for m in train_recs])
    # previous utterance per clip: previous piece of the same long recording, else the same speaker's previous clip
    prev = {}
    by_parent, by_spk = defaultdict(list), defaultdict(list)
    for m in train_recs:
        if m.get("parent_utterance_id") and "#" in (m.get("source_utterance_id") or ""):
            by_parent[m["parent_utterance_id"]].append(m)
        else:
            ts = contexts.get(m["utterance_id"], {}).get("timestamp") or m["utterance_id"]
            by_spk[m["speaker_id"]].append((ts, m))
    for pieces in by_parent.values():
        pieces.sort(key=lambda x: x["source_utterance_id"])
        for i, m in enumerate(pieces):
            m["_piece"] = (i, len(pieces))
            if i:
                prev[m["utterance_id"]] = to_label_v1(pieces[i - 1]["raw_transcript"])
    for clips in by_spk.values():
        clips.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(clips, clips[1:], strict=False):
            prev[b["utterance_id"]] = to_label_v1(a["raw_transcript"])
    other_ctx = [cursor_text(c) for c in contexts.values() if any(cursor_text(c))]
    piece_of = {m["utterance_id"]: m.get("_piece") for m in train_recs}
    for name, sources, split in (("train", TRAIN, "train"), ("val", VAL, "val")):
        n_ctx, rows = Counter(), []
        for src in sources:
            for m in map(json.loads, open(f"data/manifests/{src}.jsonl", encoding="utf-8")):
                if m["split"] != split or m["duration_sec"] > MAX_AUDIO_SEC:
                    continue
                label = to_label_v1(m["raw_transcript"])
                c = contexts.get(m["utterance_id"], {})
                if args.augment and split == "train":
                    m["_piece"] = piece_of.get(m["utterance_id"])
                    ctx = augmented_context(rng, m, c, label, prev.get(m["utterance_id"]), other_ctx, pool, banned,
                                            stats)
                else:
                    ctx = build_context(c, label, stats)
                n_ctx["with_context" if ctx else "no_context"] += 1
                lang = clip_language(m.get("language_tags"))
                row = {"audio": str(root / m["audio_path"]), "prompt": ctx,
                       "text": f"language {QWEN_LANG[lang]}<asr_text>{label}", "utterance_id": m["utterance_id"]}
                rows += [row] * (REPEAT.get(src, 1) if split == "train" else 1)
        if name == "train":
            rows += extra_train(args, contexts, root, rng, pool, banned, stats, n_ctx)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = {"clips": len(rows), "unique": len({r["utterance_id"] for r in rows}), **n_ctx}
    # contexts for decoding (all sources with pulled context; labels only used for the leak check)
    labels = {}
    for man in Path("data/manifests").glob("*.jsonl"):
        if "." in man.stem:
            continue  # *.pre_strict / *_ingested variants
        for m in map(json.loads, open(man, encoding="utf-8")):
            labels.setdefault(m["utterance_id"], to_label_v1(m["raw_transcript"]))
    with open(out / "contexts.jsonl", "w", encoding="utf-8") as f:
        for uid, c in contexts.items():
            f.write(json.dumps({"utterance_id": uid, "context": build_context(c, labels.get(uid, ""), Counter())},
                               ensure_ascii=False) + "\n")
        for extra in args.extra_contexts:  # e.g. dictionary benchmark (scripts/dictionary_bench_to_manifest.py)
            f.write(Path(extra).read_text(encoding="utf-8"))
    summary["leaks"] = {k: v for k, v in stats.items() if k.startswith("leak")}
    summary["augment"] = {k: v for k, v in stats.items() if not k.startswith("leak")}
    summary["glossary_size"], summary["banned_terms"] = len(pool), len(banned)
    summary["contexts_for_decoding"] = len(contexts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
