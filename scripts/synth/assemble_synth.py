"""Assemble the second synthetic text sets (owner, 2026-11-10): long_100 (~45 s dictations, 35-55 s) and more_1500.

Same source format and derivation rules as the pilot (scripts/synth/assemble_pilot.py, reused here):
  data/private/synth/src_long/batch_L*.txt  -> data/private/synth/long_100.jsonl   (ids synth_long_0001 ...)
  data/private/synth/src_more/batch_M*.txt  -> data/private/synth/more_1500.jsonl  (ids synth_more_0001 ...)
  data/private/synth/src_acro/batch_A*.txt  -> data/private/synth/acronym_300.jsonl (ids synth_acro_0001 ...; owner
                                               2026-11-10: spelled-out acronyms inside Chinese / English sentences)
  data/private/synth/src_longen/batch_E*.txt -> data/private/synth/long_en_150.jsonl (ids synth_lonen_0001 ...; 40-60 s
                                               English / zh-en monologues)
  data/private/synth/src/terms.txt (+ src/terms_add_*.txt while batches are being written) -> term_bank.jsonl

One sample per line: app || terms || strict || spoken || before || after || final || tags
("\\n" = line break; "-" in spoken / final = same as strict). Extra tags used by these sets: repeat, false_start,
filler (policy-kept fillers in the strict transcript), spoken_filler (um / uh / 嗯 / 呃 only in spoken_text, so the
audio has hesitations that the label does not), light (final tightened beyond the disfluency clean-up).

    uv run python scripts/synth/assemble_synth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_pilot import OUT, SRC, load_terms, load_tones, parse_batch  # noqa: E402

SETS = {  # name: (source dir, batch glob, output file, id prefix)
    "long": (OUT / "src_long", "batch_L*.txt", OUT / "long_100.jsonl", "synth_long_"),
    "more": (OUT / "src_more", "batch_M*.txt", OUT / "more_1500.jsonl", "synth_more_"),
    "acro": (OUT / "src_acro", "batch_A*.txt", OUT / "acronym_300.jsonl", "synth_acro_"),
    "longen": (OUT / "src_longen", "batch_E*.txt", OUT / "long_en_150.jsonl", "synth_lonen_"),
}


def load_all_terms() -> list[dict]:
    """terms.txt plus any per-batch additions not yet merged into it (first occurrence of a term wins)."""
    rows, seen = [], set()
    extra = sorted(SRC.glob("terms_add_*.txt"))
    for path in [SRC / "terms.txt", *extra]:
        if path.name == "terms.txt":
            part = load_terms()
        else:
            part = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                f = [x.strip() for x in line.split("|")]
                f += [""] * (5 - len(f))
                part.append({"term": f[0], "category": f[1], "language": f[2], "notes": f[3], "lookalike": f[4]})
        for r in part:
            if r["term"] not in seen:
                seen.add(r["term"])
                rows.append(r)
    return rows


def build(name: str, tones: dict) -> list[dict]:
    src, glob, _, prefix = SETS[name]
    rows = []
    for p in sorted(src.glob(glob)):
        rows += parse_batch(p, tones)
    return [{"id": f"{prefix}{i:04d}", **r} for i, r in enumerate(rows, 1)]


def main() -> None:
    tones = load_tones()
    terms = load_all_terms()
    with open(OUT / "term_bank.jsonl", "w", encoding="utf-8") as fh:
        for r in terms:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, (_, _, out, _) in SETS.items():
        rows = build(name, tones)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} samples -> {out.relative_to(OUT.parents[2])}")
    print(f"{len(terms)} bank terms")


if __name__ == "__main__":
    main()
