"""QC of synthesised clips by ASR round trip (2026-11-10): decode with Qwen3-ASR zero-shot, compare with the
strict transcript. Numeral style is relaxed (official_mer_num; zero-shot Qwen writes Chinese numerals where
labels write digits), so what remains is mostly TTS misreadings (wrong word, skipped or doubled text, garbled
names) or recognition errors.

  fail   MER > 25% or empty output, all terms recognised → likely a TTS misreading → dropped (--write-drop)
  fail_term  MER > 25% but a term was not recognised    → kept: unknown names are what the data should teach
           (owner-approved rule after the pilot review, 2026-11-10)
  listen 10% < MER ≤ 25%, or a term missing  → kept; the owner / Claude listens to a sample
  pass   otherwise

    uv run python scripts/synth/qc.py data/manifests/synth_pilot.jsonl qc_synth_pilot_qwen_zs [--write-drop]
Writes data/private/synth/qc_<run>.jsonl and prints a summary.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from whisperflow.eval.official import merge_compounds, official_mer_num
from whisperflow.text.numbers import en_numbers_to_digits

SPELL = re.compile(r"\b([A-Za-z0-9])(?:[ -]([A-Za-z0-9]))+\b")


def relax(ref: str, hyp: str) -> tuple[str, str]:
    """Style-only differences a TTS round trip cannot be blamed for: English number words → digits (both sides,
    every number incl. 1–9 and digit-by-digit), spelled letters with or without hyphens, compounds split by ASR
    ("Sweet Green" vs "Sweetgreen")."""
    def spell(t: str) -> str:
        return SPELL.sub(lambda m: m.group(0).replace("-", "").replace(" ", ""), t)
    r, h = (spell(en_numbers_to_digits(x).replace("-", " ")) for x in (ref, hyp))
    h, _ = merge_compounds(r, h)
    return r, h


def _n(s: str) -> str:
    return re.sub(r"[\s\W_-]+", "", (s or "").lower())


def main() -> None:
    manifest, run = sys.argv[1], sys.argv[2]
    man = {m["utterance_id"]: m for m in map(json.loads, open(manifest, encoding="utf-8"))}
    preds = {p["utterance_id"]: p["hypothesis"]
             for p in map(json.loads, open(f"reports/runs/{run}/predictions.jsonl", encoding="utf-8"))}
    rows, verdicts = [], Counter()
    for u, m in man.items():
        hyp = preds.get(u, "")
        o = official_mer_num(*relax(m["raw_transcript"], hyp))
        mer = o.errors / max(o.n_ref, 1)
        missing = [t for t in m["hotwords_present"] if _n(t) and _n(t) not in _n(hyp)]
        if not hyp.strip() or mer > 0.25:
            v = "fail_term" if (missing and hyp.strip()) else "fail"
        else:
            v = "listen" if (mer > 0.10 or missing) else "pass"
        verdicts[v] += 1
        rows.append({"utterance_id": u, "verdict": v, "mer": round(100 * mer, 1), "missing_terms": missing,
                     "voice": m["speaker_id"], "reference": m["raw_transcript"], "hypothesis": hyp})
    out = Path(f"data/private/synth/qc_{run}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scored = [official_mer_num(*relax(man[u]["raw_transcript"], preds.get(u, ""))) for u in man]
    tot_e, tot_n = sum(o.errors for o in scored), sum(o.n_ref for o in scored)
    print(f"{len(rows)} clips: {dict(verdicts)}; pooled MER (numerals relaxed) {100 * tot_e / tot_n:.2f}%")
    by_voice = Counter(r["voice"] for r in rows if r["verdict"] != "pass")
    print("non-pass by voice:", by_voice.most_common())
    miss = Counter(t for r in rows for t in r["missing_terms"])
    print("most often missed terms:", miss.most_common(15))
    print(f"→ {out}")
    if "--write-drop" in sys.argv:  # merge this run's fails into the drop list read by make_qwen_e2e_data.py
        path = Path("data/private/synth/qc_drop.json")
        drop = set(json.load(open(path))) if path.exists() else set()
        drop |= {r["utterance_id"] for r in rows if r["verdict"] == "fail"}
        path.write_text(json.dumps(sorted(drop), indent=1))
        print(f"drop list: {len(drop)} ids → {path}")


if __name__ == "__main__":
    main()
