"""Context-use metrics for the contextual ASR runs (J1 / J2, 2026-11-09).

  distract  — dictionary-benchmark contexts whose Vocabulary is replaced by 10–20 distractor terms (J2
              glossary; never a word of the reference), App / cursor lines kept. Decoding with it measures false
              insertions: does the model write vocabulary words that were not said?
  report    — per run: MER against the strict transcript (review column D), target-word recall against the
              strict transcript and against the final output (column E: vocabulary fixes belong there, owner
              2026-11-09), copy rate (output holds an 8-character
              run of the clip's cursor text that the reference does not hold), distractor insertion rate.

    uv run python scripts/context_metrics.py distract
    uv run python scripts/context_metrics.py report --manifest data/manifests/dictionary_bench.jsonl \
        --contexts runs/qwen_sft/J2/contexts.jsonl dict_qwen3asr_J2_step_400_ctx …
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from whisperflow.eval.official import official_mer

sys.path.insert(0, str(Path(__file__).parent))
from make_qwen_e2e_data import build_glossary, leaks  # noqa: E402

DISTRACT = Path("data/private/e2e/contexts_dictionary_bench_distract.jsonl")


def _n(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def distract() -> None:
    labels = [json.loads(line)["raw_transcript"] for f in Path("data/manifests").glob("*.jsonl")
              if "." not in f.stem for line in open(f, encoding="utf-8")]
    pool, _ = build_glossary(labels)
    rng = random.Random(1234)
    real = {r["utterance_id"]: r["context"]
            for r in map(json.loads, open("data/private/e2e/contexts_dictionary_bench.jsonl", encoding="utf-8"))}
    with open(DISTRACT, "w", encoding="utf-8") as f:
        for m in map(json.loads, open("data/manifests/dictionary_bench.jsonl", encoding="utf-8")):
            ref = _n(m["raw_transcript"])
            words = [w for w in rng.sample(pool, 40) if _n(w) and _n(w) not in ref][:rng.randint(10, 20)]
            lines = [ln for ln in real.get(m["utterance_id"], "").splitlines() if not ln.startswith("Vocabulary:")]
            lines.insert(1 if lines and lines[0].startswith("App:") else 0, "Vocabulary: " + ", ".join(words))
            f.write(json.dumps({"utterance_id": m["utterance_id"], "context": "\n".join(lines),
                                "distractors": words}, ensure_ascii=False) + "\n")
    print(DISTRACT)


def cursor_of(context: str) -> str:
    return " ".join(ln.split(":", 1)[1] for ln in context.splitlines()
                    if ln.startswith(("Before cursor:", "After cursor:")))


def report(manifest: str, contexts: str, runs: list[str]) -> None:
    man = {m["utterance_id"]: m for m in map(json.loads, open(manifest, encoding="utf-8"))}
    ctx = {r["utterance_id"]: r for r in map(json.loads, open(contexts, encoding="utf-8"))}
    dis = {r["utterance_id"]: r["distractors"] for r in map(json.loads, open(DISTRACT, encoding="utf-8"))} \
        if DISTRACT.exists() else {}
    print("| run | MER vs D (zh/en) | target recall vs D | target recall vs E (final) | copy rate "
          "| distractor ins. |")
    print("|---|---:|---:|---:|---:|---:|")
    for run in runs:
        preds = {p["utterance_id"]: p["hypothesis"]
                 for p in map(json.loads, open(f"reports/runs/{run}/predictions.jsonl", encoding="utf-8"))}
        e = n = hit = tot = copies = ins = ins_n = clips = hit_e = tot_e = 0
        for u, hyp in preds.items():
            m = man.get(u)
            if not m or m["language_tags"] == ["other"]:
                continue
            clips += 1
            ref = m["raw_transcript"]
            o = official_mer(ref, hyp)
            if o.included:
                e, n = e + o.errors, n + o.n_ref
            final = m.get("formatted_transcript") or ref
            for w in m.get("hotwords_present") or []:
                if _n(w) in _n(ref):
                    tot, hit = tot + 1, hit + (_n(w) in _n(hyp))
                if _n(w) in _n(final):
                    tot_e, hit_e = tot_e + 1, hit_e + (_n(w) in _n(hyp))
            cur = cursor_of(ctx.get(u, {}).get("context", ""))
            if cur and leaks(hyp, cur) and not leaks(ref, cur):  # an 8-char run of cursor text that was not said
                copies += 1
            if "distract" in run and u in dis:
                ins_n += len(dis[u])
                ins += sum(_n(w) in _n(hyp) for w in dis[u])
        recall = f"{hit}/{tot} = {100 * hit / tot:.0f}%" if tot else "–"
        recall_e = f"{hit_e}/{tot_e} = {100 * hit_e / tot_e:.0f}%" if tot_e else "–"
        print(f"| {run} | {100 * e / max(n, 1):.2f}% | {recall} | {recall_e} | {copies}/{clips} | "
              + (f"{ins}/{ins_n}" if ins_n else "–") + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("distract")
    r = sub.add_parser("report")
    r.add_argument("--manifest", required=True)
    r.add_argument("--contexts", required=True)
    r.add_argument("runs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "distract":
        distract()
    else:
        report(args.manifest, args.contexts, args.runs)


if __name__ == "__main__":
    main()
