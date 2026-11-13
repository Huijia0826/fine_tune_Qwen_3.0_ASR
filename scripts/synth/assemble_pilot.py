"""Assemble the synthetic pilot text set (owner, 2026-11-09; plan in docs/synthetic_data_plan.md).

Inputs (hand-written by the text generator, data/private/synth/src/):
  terms.txt        term | category | language | notes | lookalike
  batch_*.txt      one sample per line: app || terms || strict || spoken || before || after || final || tags
                   (trailing fields may be omitted; "\\n" = line break; "-" in spoken / final = same as strict)
Outputs (data/private/synth/):
  term_bank.jsonl  {term, category, language, notes, lookalike}
  pilot_500.jsonl  {id, app, tone, scenario, language, terms, strict_transcript, spoken_text, before_cursor,
                    after_cursor, final_output, tags, batch}

Derived here, so every sample follows the same rules:
  language  zh (no Latin letters) / en (no CJK) / zh-en (both), from the strict transcript
  tags      the hand-written tags, plus "spelled" when the transcript spells letters (V-E-R-C-E-L)
  tone      configs/reference/tone_by_app.tsv
  final     the hand-written final (tightening, applied self-correction / spelling, e-mail line breaks) or the strict
            transcript; then tone casual drops only the sentence-final 。 / . (？ / ? / ！ / ! kept)

    uv run python scripts/synth/assemble_pilot.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/private/synth/src"
OUT = ROOT / "data/private/synth"
TONE_TSV = ROOT / "configs/reference/tone_by_app.tsv"

ALIAS = {
    "wx": "WeChat 微信", "lark": "Lark / 飞书", "imsg": "Messages（Apple 信息）", "x": "X (Twitter)",
    "chrome": "Chrome（浏览器，拿不到网址）", "ios": "无 app 信息（多为 iOS 键盘）", "gsearch": "Google 搜索 app",
    "brave": "Brave（浏览器，拿不到网址）", "firefox": "Firefox（浏览器，拿不到网址）", "doubao": "豆包 Doubao",
    "gbwa": "WhatsApp 第三方改版（GB / MB WhatsApp）",
}
CJK = re.compile(r"[一-鿿]")
SPELLED = re.compile(r"\b[A-Z0-9](?:-[A-Z0-9]){2,}\b")  # policy v1 spelled letters: V-E-R-C-E-L
LATIN = re.compile(r"[A-Za-z]")


def load_tones() -> dict[str, tuple[str, str]]:
    tones = {}
    for line in TONE_TSV.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("tone\t") or not line.strip():
            continue
        tone, scenario, app = line.split("\t")
        tones[app] = (tone, scenario)
    return tones


def language(text: str) -> str:
    zh, en = bool(CJK.search(text)), bool(LATIN.search(text))
    return "zh-en" if zh and en else ("zh" if zh else "en")


def casual_final(text: str) -> str:
    """Casual tone: drop only the sentence-final period (。 / .), keep ？ / ? / ！ / !."""
    t = text.rstrip()
    return re.sub(r"[。.]+$", "", t) if not t.endswith("...") else t


def load_terms() -> list[dict]:
    rows = []
    for line in (SRC / "terms.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split("|")]
        f += [""] * (5 - len(f))
        rows.append({"term": f[0], "category": f[1], "language": f[2], "notes": f[3], "lookalike": f[4]})
    return rows


def parse_batch(path: Path, tones: dict) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split("||")]
        f += [""] * (8 - len(f))
        app, terms, strict, spoken, before, after, final, tags = f[:8]
        app = ALIAS.get(app, app)
        if app not in tones:
            raise SystemExit(f"{path.name}:{n}: unknown app {app!r}")
        unesc = lambda s: s.replace("\\n", "\n")
        tone, scenario = tones[app]
        final = unesc(final) if final and final != "-" else strict
        if tone == "casual":
            final = casual_final(final)
        out.append({
            "app": app, "tone": tone, "scenario": scenario, "language": language(strict),
            "terms": [t.strip() for t in terms.split(";") if t.strip()],
            "strict_transcript": strict,
            "spoken_text": spoken if spoken and spoken != "-" else strict,
            "before_cursor": unesc(before), "after_cursor": unesc(after),
            "final_output": final,
            "tags": [t.strip() for t in tags.split(",") if t.strip()] + (
                ["spelled"] if SPELLED.search(strict) and "spelled" not in tags else []),
            "batch": f"{path.stem}:{n}",
        })
    return out


def main() -> None:
    tones = load_tones()
    terms = load_terms()
    with open(OUT / "term_bank.jsonl", "w", encoding="utf-8") as fh:
        for r in terms:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = []
    for p in sorted(SRC.glob("batch_*.txt")):
        rows += parse_batch(p, tones)
    with open(OUT / "pilot_500.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            fh.write(json.dumps({"id": f"synth_pilot_{i:04d}", **r}, ensure_ascii=False) + "\n")
    print(f"{len(terms)} terms, {len(rows)} samples -> {OUT}")


if __name__ == "__main__":
    main()
