"""Validate the synthetic text sets: long_100 / more_1500 / acronym_300 / long_en_150 (owner, 2026-11-10) and, with
the same code, the pilot.

Hard checks (all sets; the pilot's checks from scripts/synth/validate_pilot.py, whose helpers are reused):
  schema, banned (dictionary-benchmark targets + variants), owner_test_banned, policy_v1, leak, casual_final,
  final_adds, final_identical, terms, duration, spelled_spoken, duplicate (across pilot + long + more)
Added for long / more:
  terms        long: 1-6 terms per sample, more: 0-3 (pilot: 0-3)
  duration     long: estimated speaking time of spoken_text 35-55 s (owner 2026-11-10: ~45 s each; same rate as
               validate_pilot.seconds);
               more: strict <= 25 s as in the pilot and spoken_text <= 28 s
  final        e-mails (Gmail / Outlook) whose final differs only by formatting (line breaks, capitalised first body
               word, no period after the sign-off name) count as rule output, not as tightened
  final_reason a final that differs from the rule output (tag tightened) names a reason tag: self_correction /
               false_start / repeat / spelled / light; those tags (and spelled) imply a changed final
  spoken_drift spoken_text says the same words as the strict transcript: every non-number word / Chinese character
               of the transcript is in spoken_text, and spoken_text adds only number words, read-out letters and
               (tag spoken_filler) um / uh / 嗯 / 呃
  one_line     strict / spoken have no line breaks
Added for acronym_300 (acro) / long_en_150 (longen), owner 2026-11-10:
  acronym      acronyms (all-caps tokens with >= 2 letters such as RL, ARR, FP8, CI/CD, KPIs, GPU's, and MoE / CoT /
               iOS; not the ones read as words, WORD_ACRONYMS) are written letter-spaced in spoken_text ("R L",
               "K P I's") and never as an unspaced all-caps token there; no hyphenated acronym in the transcript
               (policy v1: X-Y-Z only for an explicit spelling-out of a word that is also said as a word, e.g.
               "Parcelwise, P-A-R-C-E-L-W-I-S-E"); acro: 1-3 distinct acronyms per sample, each listed in terms
  terms        acro: 1-4 terms (the acronyms, optionally one more bank term); longen: 2-6; a term also matches its
               plural / possessive (KPI in "KPIs")
  duration     acro: spoken <= 20 s, or 35-55 s for the long ones; longen: spoken 38-55 s (English TTS runs ~12% slower
               than the estimate)
Soft checks (review): negation changes, digits in Chinese spoken_text, shared 8-word / 16-character runs between
samples (uniformity), most common openings.

    uv run python scripts/synth/validate_synth.py                    # long + more (exit code 1 if a hard check fails)
    uv run python scripts/synth/validate_synth.py pilot              # the pilot with the same code
    uv run python scripts/synth/validate_synth.py acro longen        # acronym_300 + long_en_150
    uv run python scripts/synth/validate_synth.py --batch data/private/synth/src_long/batch_L01.txt --kind long
                                                                     # one source batch, before assembling
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_pilot import load_tones, parse_batch  # noqa: E402
from assemble_synth import load_all_terms  # noqa: E402
from validate_pilot import (AI, CHAT, CJK, EMAIL_APPS, WORK, banned_patterns, bucket, chars, has_term,  # noqa: E402
                            is_question, leaks, owner_test_patterns, seconds)

from whisperflow.text.fillers import strip_fillers
from whisperflow.text.label_style import to_label_v1

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/private/synth"
FIELDS = {"id": str, "app": str, "tone": str, "language": str, "terms": list, "strict_transcript": str,
          "spoken_text": str, "before_cursor": str, "after_cursor": str, "final_output": str, "tags": list}
REASONS = {"self_correction", "false_start", "repeat", "spelled", "light"}
SPELLED = re.compile(r"\b[A-Z0-9](?:-[A-Z0-9]){2,}\b")
SP_FILLER = re.compile(r"[嗯呃]|(?<![A-Za-z'-])(?:um+|uh+)(?![A-Za-z'-])", re.I)

BASE_TAGS = {"self_correction": (.10, .03), "spelled": (.05, .02), "question": (.10, .04), "no_terms": (.15, .04),
             "tightened": (.15, .05), "before_nonempty": (.60, .06), "after_nonempty": (.25, .05)}
KINDS = {
    "pilot": dict(path=SYN / "pilot_500.jsonl", prefix="synth_pilot_", terms=(0, 3),
                  targets={"group": ({"chat": .35, "ai": .30, "work": .25, "other": .10}, .04),
                           "language": ({"zh": .40, "zh-en": .40, "en": .20}, .04),
                           "length": ({"short": .40, "medium": .45, "long": .15}, .05)},
                  tags=BASE_TAGS),
    "more": dict(path=SYN / "more_1500.jsonl", prefix="synth_more_", terms=(0, 3),
                 targets={"group": ({"chat": .35, "ai": .30, "work": .25, "other": .10}, .04),
                          "language": ({"en": .35, "zh-en": .35, "zh": .30}, .04),
                          "length": ({"short": .40, "medium": .45, "long": .15}, .05)},
                 tags={**{k: v for k, v in BASE_TAGS.items() if k != "tightened"},
                       "repeat": (.06, .03), "false_start": (.05, .03), "filler": (.12, .05),
                       "spoken_filler": (.05, .03), "light": (.12, .05), "list": (.06, .04),
                       "numbers": (.50, .10)}),
    "long": dict(path=SYN / "long_100.jsonl", prefix="synth_long_", terms=(1, 6),
                 targets={"group": ({"ai": .35, "work": .35, "chat": .20, "other": .10}, .05),
                          "language": ({"en": .45, "zh-en": .30, "zh": .25}, .04),
                          "band": ({"35-45s": .50, "45-55s": .50}, .15)},
                 tags={"before_nonempty": (.50, .06), "after_nonempty": (.20, .05), "self_correction": (.20, .07),
                       "repeat": (.15, .07), "false_start": (.10, .05), "filler": (.50, .15),
                       "spoken_filler": (.10, .04), "spelled": (.05, .02), "light": (.15, .05),
                       "question": (.12, .06), "list": (.15, .08), "numbers": (.60, .15)}),
    "acro": dict(path=SYN / "acronym_300.jsonl", prefix="synth_acro_", terms=(1, 4),
                 targets={"group": ({"work": .40, "chat": .30, "ai": .25, "other": .05}, .06),
                          "language": ({"zh-en": .80, "en": .20}, .04),
                          "band": ({"short <=7s": .40, "medium 7-20s": .50, "long 35-55s": .10}, .06)},
                 tags={"acro_in_context": (.40, .08), "multi_acro": (.35, .12), "before_nonempty": (.60, .10),
                       "after_nonempty": (.20, .08), "question": (.15, .07), "numbers": (.35, .12),
                       "self_correction": (.07, .04), "spoken_filler": (.05, .04)}),
    "longen": dict(path=SYN / "long_en_150.jsonl", prefix="synth_lonen_", terms=(2, 6),
                   targets={"group": ({"ai": .40, "work": .30, "chat": .20, "other": .10}, .07),
                            "language": ({"en": .75, "zh-en": .25}, .04),
                            "band": ({"38-46s": .50, "46-55s": .50}, .15)},
                   tags={"before_nonempty": (.50, .08), "after_nonempty": (.20, .07), "self_correction": (.20, .07),
                         "repeat": (.20, .07), "false_start": (.12, .05), "filler": (.70, .15),
                         "spoken_filler": (.15, .06), "light": (.15, .06), "question": (.12, .06),
                         "list": (.12, .07), "numbers": (.50, .15)}),
}
NEW_KINDS = {"acro", "longen"}

# acronyms (acro / longen): read letter by letter, written letter-spaced in spoken_text
WORD_ACRONYMS = {"CUDA", "SOTA", "RAG", "ARPU", "TAM", "NASA", "JSON", "YAML", "IKEA", "OPPO", "COBRA", "SEVIS",
                 "IELTS", "JAX", "MATLAB", "ROAS", "ASIN", "OK", "GIF", "PIN", "SIM", "LEGO", "FOMO", "YOLO", "ASCII",
                 "LoRA", "SaaS"}  # read as words: kept as they are in spoken_text
MIXED_ACRONYMS = ("MoE", "CoT", "iOS")
ACRO_TOKEN = re.compile(r"(?<![A-Za-z0-9&/'])([A-Z0-9][A-Z0-9&/]*[A-Z0-9]|"
                        + "|".join(MIXED_ACRONYMS) + r")('s|s)?(?![A-Za-z0-9&/])")
HYPHEN_LETTERS = re.compile(r"(?<![A-Za-z0-9-])[A-Za-z](?:-[A-Za-z0-9])+(?![A-Za-z0-9-])")


def acronyms(text: str) -> list[str]:
    """Acronyms of a text (base form, plural / possessive removed), in order; tokens with < 2 letters (Q3, H100) and
    the WORD_ACRONYMS are not acronyms."""
    out = []
    for m in ACRO_TOKEN.finditer(text):
        base = m.group(1)
        if base in WORD_ACRONYMS or len(re.findall(r"[A-Za-z]", base)) < 2:
            continue
        out.append(base)
    return out


def _letter_groups(base: str) -> list[str]:
    if base in MIXED_ACRONYMS:
        return [base]
    return [g for g in re.findall(r"[A-Z]+", base) if len(g) >= 2]  # FP8 -> FP, CI/CD -> CI, CD, B2B -> -


def acronym_issues(strict: str, spoken: str) -> list[str]:
    issues = []
    for base in dict.fromkeys(acronyms(strict)):
        for g in _letter_groups(base):
            pat = r"(?<![A-Za-z])" + r"\s+".join(g) + r"(?![A-Za-z])"
            if not re.search(pat, spoken, 0 if g.isupper() else re.I):
                issues.append(f"{base}: '{' '.join(g)}' not letter-spaced in spoken")
    for base in acronyms(spoken):
        if base in MIXED_ACRONYMS or _letter_groups(base):
            issues.append(f"unspaced acronym {base} in spoken")
    for m in HYPHEN_LETTERS.finditer(strict):
        word = m.group(0).replace("-", "")
        said = [w for w in re.findall(rf"(?<![A-Za-z-]){word}(?![A-Za-z-])", strict, re.I) if not w.isupper()]
        if len(word) < 3 or not said:
            issues.append(f"hyphenated acronym {m.group(0)} (only a spelled-out word that is also said may use X-Y-Z)")
    return issues


def has_term_kind(term: str, text: str, kind: str) -> bool:
    if has_term(term, text):
        return True
    return kind in NEW_KINDS and re.search(rf"(?<![A-Za-z]){re.escape(term)}(?:'s|s)(?![A-Za-z])", text) is not None

# spoken_drift: what spoken_text may add (number words, read-out symbols / letters) or leave out of the transcript
ZH_NUM = set("零〇一二三四五六七八九十百千万亿两幺点半之分号月日年元块毛角")
EN_NUM = re.compile(
    r"(zero|oh|o|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|"
    r"fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|"
    r"hundredth|and|a|point|percent|dollars?|cents?|bucks?|pounds?|euros?|yuan|yen|rmb|k|x|times|by|dot|at|"
    r"slash|dash|colon|hash|plus|minus|degrees?|o'clock|p|m|thirds?|quarters?|half|nil|nought|double|triple|"
    r"st|nd|rd|th|s|v|vs|versus|to|on|hertz|kilohertz|gigs?|gigabytes?|megabytes?|kilometers?|kilos?|seconds?|"
    r"minutes?|hours?|days?|weeks?|months?|years?)")
EN_DROP = {"pm", "am", "vs", "st", "nd", "rd", "th", "k", "x", "s", "p.m", "a.m", "etc", "mins", "min", "hrs",
           "q", "v", "w", "khz", "hz", "gb", "mb", "kb", "tb", "km", "kg", "ms"}


def group_of(scen: str) -> str:
    return "chat" if scen in CHAT else "ai" if scen in AI else "work" if scen in WORK else "other"


def band(sec: float, kind: str = "long") -> str:
    if kind == "acro":
        return "short <=7s" if sec <= 7 else "medium 7-20s" if sec <= 20 else "long 35-55s"
    if kind == "longen":
        return "38-46s" if sec < 46 else "46-55s"
    return "35-45s" if sec < 45 else "45-55s"


def _tokens(text: str) -> tuple[Counter, Counter]:
    """Latin words (no digits, no single letters) and Chinese characters of a text."""
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", strip_fillers(text).lower())  # Kiwi77, 1on1, H100
    t = re.sub(r"[a-z0-9$€£¥.,:%/+'-]*\d[a-z0-9$€£¥.,:%/+'-]*", " ", t)  # numbers are read out differently
    t = SPELLED.sub(" ", t)
    words = Counter(w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", t) if len(w) > 1)
    return words, Counter(re.findall(rf"[{CJK}]", t))


def spoken_drift(strict: str, spoken: str) -> str | None:
    sw, sc = _tokens(SPELLED.sub(" ", strict))
    pw, pc = _tokens(spoken)
    sp_l = " " + re.sub(r"[^a-z]+", " ", spoken.lower()) + " "
    miss_w = [w for w in (sw - pw)
              if w not in EN_DROP and f" {' '.join(w.replace(chr(39), ''))} " not in sp_l]  # SFO -> "S F O", GPU's
    miss_c = list((sc - pc).elements())
    add_w = [w for w in (pw - sw) if not EN_NUM.fullmatch(w) and not re.fullmatch(r"[a-z]('s)?", w)]  # K P I's
    add_c = [c for c in (pc - sc).elements() if c not in ZH_NUM]
    if miss_w or miss_c or add_w or add_c:
        return f"missing {miss_w + miss_c} added {add_w + add_c}"
    return None


def shingles(text: str) -> set[str]:
    t = text.lower()
    words = re.findall(r"[a-z0-9']+", t)
    out = {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}
    cjk = "".join(re.findall(rf"[{CJK}]", t))
    out |= {cjk[i:i + 16] for i in range(len(cjk) - 15)}
    return out


def opening(text: str) -> str:
    t = text.strip()
    if re.match(rf"[{CJK}]", t):
        return re.sub(r"[，。？！、\s].*", "", t)[:4]
    return " ".join(re.findall(r"[A-Za-z']+", t)[:2]).lower()


class Ctx:
    def __init__(self) -> None:
        self.tones = load_tones()
        self.bank = load_all_terms()
        self.bank_terms = {b["term"] for b in self.bank}
        self.pats = banned_patterns()
        self.owner_pats = owner_test_patterns()


def check_bank(ctx: Ctx, hard: dict) -> None:
    for b in ctx.bank:
        for w, p in ctx.pats:
            if p.search(b["term"]):
                hard["banned"].append(f"bank: {b['term']} ~ {w}")
        for w, p in ctx.owner_pats:
            if p.search(b["term"]):
                hard["owner_test_banned"].append(f"bank: {b['term']} ~ {w}")
        if to_label_v1(b["term"]) != b["term"]:
            hard["policy_v1"].append(f"bank: {b['term']!r} -> {to_label_v1(b['term'])!r}")
        if b.get("lookalike") and b["lookalike"] not in ctx.bank_terms:
            hard["schema"].append(f"bank: lookalike {b['lookalike']} of {b['term']} not in bank")


def check_rows(rows: list[dict], kind: str, ctx: Ctx, seen_strict: dict) -> tuple[dict, dict, dict]:
    cfg = KINDS[kind]
    hard: dict[str, list[str]] = defaultdict(list)
    soft: dict[str, list[str]] = defaultdict(list)
    st: dict[str, Counter] = defaultdict(Counter)
    lo_t, hi_t = cfg["terms"]
    for r in rows:
        rid = r.get("id", "?")
        for f, typ in FIELDS.items():
            if not isinstance(r.get(f), typ):
                hard["schema"].append(f"{rid}: field {f} missing / wrong type")
        if not rid.startswith(cfg["prefix"]) and not rid.startswith("batch"):
            hard["schema"].append(f"{rid}: id prefix is not {cfg['prefix']}")
        s, fin, sp = r["strict_transcript"], r["final_output"], r["spoken_text"]
        bef, aft, lang, tags = r["before_cursor"], r["after_cursor"], r["language"], set(r["tags"])
        where = f"{rid} [{r.get('batch', '')}]"
        if r["app"] not in ctx.tones:
            hard["schema"].append(f"{where}: app {r['app']} not in tsv")
            continue
        tone, scen = ctx.tones[r["app"]]
        if r["tone"] != tone:
            hard["schema"].append(f"{where}: tone {r['tone']} != tsv {tone}")
        zh, en = bool(re.search(rf"[{CJK}]", s)), bool(re.search(r"[A-Za-z]", s))
        if lang != ("zh-en" if zh and en else ("zh" if zh else "en")):
            hard["schema"].append(f"{where}: language {lang} does not match the text")
        if s in seen_strict:
            hard["duplicate"].append(f"{where} = {seen_strict[s]}: {s[:80]}")
        seen_strict[s] = rid
        if kind != "pilot" and ("\n" in s or "\n" in sp):
            hard["one_line"].append(f"{where}: line break in strict / spoken")

        # terms
        if not lo_t <= len(r["terms"]) <= hi_t:
            hard["terms"].append(f"{where}: {len(r['terms'])} terms (allowed {lo_t}-{hi_t})")
        for t in r["terms"]:
            if t not in ctx.bank_terms:
                hard["terms"].append(f"{where}: term {t} not in bank")
            if not has_term_kind(t, s, kind):
                hard["terms"].append(f"{where}: term {t} not in transcript")
        if len(set(r["terms"])) != len(r["terms"]):
            hard["terms"].append(f"{where}: duplicate term")

        # banned words: every text field and the terms
        for field in ("strict_transcript", "spoken_text", "final_output", "before_cursor", "after_cursor"):
            for w, p in ctx.pats:
                if p.search(r[field]):
                    hard["banned"].append(f"{where}.{field}: {w}")
            for w, p in ctx.owner_pats:
                if p.search(r[field]):
                    hard["owner_test_banned"].append(f"{where}.{field}: {w}")
        for t in r["terms"]:
            for w, p in ctx.pats:
                if p.search(t):
                    hard["banned"].append(f"{where}.terms: {t} ~ {w}")

        # policy v1
        v1 = to_label_v1(s)
        if v1 != s:
            diff = next((i for i, (a, b) in enumerate(zip(s, v1)) if a != b), min(len(s), len(v1)))
            hard["policy_v1"].append(f"{where}: …{s[max(0, diff - 15):diff + 25]!r} -> …{v1[max(0, diff - 15):diff + 25]!r}")
        if re.search(r"[嗯呃]", s) or re.search(r"(?<![A-Za-z])(um+|uh+)(?![A-Za-z])", s, re.I):
            hard["policy_v1"].append(f"{where}: filler um / uh / 嗯 / 呃 in strict")
        m = re.search(rf"[{CJK}][,.?!]|[,.?!]\s*[{CJK}]", s)
        if m:
            hard["policy_v1"].append(f"{where}: ASCII punctuation next to Chinese: …{s[max(0, m.start() - 10):m.end() + 10]!r}")

        # leaks
        for name, cx in (("before", bef), ("after", aft)):
            if leaks(s, cx) or leaks(fin, cx):
                hard["leak"].append(f"{where}: {name}_cursor shares 8 chars with the transcript")

        # final output
        email = r["app"] in EMAIL_APPS and "\n" in fin
        if tone == "casual":
            if re.search(r"[。.]$", fin.rstrip()) and not fin.rstrip().endswith("..."):
                hard["casual_final"].append(f"{where}: casual final ends with a period")
            rule = re.sub(r"[。.]+$", "", s.rstrip())
        else:
            rule = s
        # e-mail formatting only (line breaks, capitalised first body word, no period after the sign-off name)
        fmt = lambda x: re.sub(r"[\s.。]+", "", x.lower())
        tightened = fin != rule and not (email and fmt(fin) == fmt(rule))
        extra = chars(fin) - chars(s)
        if extra:
            hard["final_adds"].append(f"{where}: final adds {''.join(sorted(extra.elements()))!r}")
        nums_s, nums_f = re.findall(r"\d[\d,.:]*\d|\d", s), re.findall(r"\d[\d,.:]*\d|\d", fin)
        corrected = tags & {"self_correction", "false_start", "repeat"}
        if SPELLED.search(s):
            changed = bool(Counter(re.findall(r"\d", fin)) - Counter(re.findall(r"\d", s)))
        else:
            changed = bool(Counter(nums_f) - Counter(nums_s)) or (bool(Counter(nums_s) - Counter(nums_f)) and not corrected)
        if changed:
            hard["final_adds"].append(f"{where}: numbers changed {nums_s} -> {nums_f}")
        neg = lambda t: len(re.findall(r"[不没别]|(?<![A-Za-z])(not|never|no)(?![A-Za-z])|n't", t, re.I))
        if neg(fin) != neg(s) and not corrected:
            soft["negation"].append(f"{where}: {neg(s)} -> {neg(fin)} negations")
        if tightened and "tightened" not in tags and not email:
            hard["final_identical"].append(f"{where}: final differs from the rule output but not tagged tightened")
        if "tightened" in tags and not tightened:
            hard["final_identical"].append(f"{where}: tagged tightened but final = rule output")
        if kind != "pilot":
            if tightened and not email and not (tags & REASONS):  # e-mail formatting needs no reason
                hard["final_reason"].append(f"{where}: changed final without a reason tag")
            if (tags & REASONS) and not tightened:
                hard["final_reason"].append(f"{where}: {sorted(tags & REASONS)} but final = rule output")

        # spoken text
        spelled = SPELLED.findall(s)
        if bool(spelled) != ("spelled" in tags):
            hard["schema"].append(f"{where}: tag 'spelled' does not match the transcript")
        if spelled and any(x in sp for x in spelled):
            hard["spelled_spoken"].append(f"{where}: spoken text keeps the hyphenated spelling")
        if bool(SP_FILLER.search(sp)) != ("spoken_filler" in tags):
            hard["schema"].append(f"{where}: tag 'spoken_filler' does not match um / uh / 嗯 / 呃 in spoken_text")
        if kind != "pilot":
            d = spoken_drift(s, sp)
            if d:
                hard["spoken_drift"].append(f"{where}: {d}")
        if lang != "en" and re.search(r"(?<![A-Za-z0-9])\d", sp):  # digits inside Latin tokens (STM32) are fine
            soft["zh_digits_spoken"].append(f"{where}: {sp[:60]!r}")

        # acronyms (acro / longen)
        if kind in NEW_KINDS:
            for x in acronym_issues(s, sp):
                hard["acronym"].append(f"{where}: {x}")
            acr = list(dict.fromkeys(acronyms(s)))
            for a in acr:
                st["acronym"][a] += 1
            if kind == "acro":
                if not 1 <= len(acr) <= 3:
                    hard["acronym"].append(f"{where}: {len(acr)} distinct acronyms {acr} (allowed 1-3)")
                for a in acr:
                    if a not in r["terms"]:
                        hard["terms"].append(f"{where}: acronym {a} not listed in terms")
                ctx_acr = [a for a in acr if re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", bef + aft)]
                st["tag"]["acro_in_context"] += bool(ctx_acr)
                st["tag"]["multi_acro"] += len(acr) >= 2
                st["tag"]["acro_at_start"] += bool(acr) and re.match(rf"\W*{re.escape(acr[0])}", s) is not None
                st["tag"]["acro_plural_possessive"] += bool(re.search(
                    r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9&/]*[A-Z0-9]|MoE|CoT)(?:'s|s)(?![A-Za-z])", s))
                st["tag"]["acro_next_to_number"] += bool(re.search(
                    r"(?<![A-Za-z0-9])[A-Z]{2,}(?:'s|s)?[ 的是到了在有约达破从涨降跌掉]{0,4}\$?\d|\d[%万千亿KMB]?[ 个的块张台位条次]{0,3}"
                    r"[A-Z]{2,}(?![a-z])", s))

        # lengths
        sec_s, sec_p = seconds(s, lang), seconds(strip_fillers(sp) if "spoken_filler" in tags else sp, lang)
        if kind == "long":
            if not 35 <= sec_p <= 55:
                hard["duration"].append(f"{where}: spoken ~{sec_p:.0f} s (allowed 35-55)")
            st["band"][band(sec_p)] += 1
        elif kind == "longen":
            if not 38 <= sec_p <= 55:
                hard["duration"].append(f"{where}: spoken ~{sec_p:.0f} s (allowed 38-55)")
            st["band"][band(sec_p, kind)] += 1
        elif kind == "acro":
            if not (sec_p <= 20 or 35 <= sec_p <= 55):
                hard["duration"].append(f"{where}: spoken ~{sec_p:.0f} s (allowed <= 20 or 35-55)")
            st["band"][band(sec_p, kind)] += 1
        else:
            if sec_s > 25 or (kind == "more" and sec_p > 28):
                hard["duration"].append(f"{where}: ~{sec_s:.0f} s / spoken ~{sec_p:.0f} s")
            st["length"][bucket(s, lang)] += 1
        st["seconds"][lang] += sec_p

        st["group"][group_of(scen)] += 1
        st["language"][lang] += 1
        st["nterms"][len(r["terms"])] += 1
        st["app"][r["app"]] += 1
        st["tone"][tone] += 1
        c = st["tag"]
        c["before_nonempty"] += bool(bef.strip())
        c["after_nonempty"] += bool(aft.strip())
        c["question"] += is_question(s)
        c["tightened"] += tightened and not email
        c["email"] += email
        c["numbers"] += bool(re.search(r"\d", s))
        c["no_terms"] += not r["terms"]
        for t in ("self_correction", "spelled", "repeat", "false_start", "filler", "spoken_filler", "light", "list",
                  "continuation", "related"):
            c[t] += t in tags
        c["spoken_differs"] += sp != s
        st["opening"][opening(s)] += 1
    return hard, soft, st


def uniformity(rows: list[dict], others: list[dict]) -> list[str]:
    """Samples that share an 8-word or 16-character run with another sample (this set or the others)."""
    owner = {}
    for r in others:
        for sh in shingles(r["strict_transcript"]):
            owner.setdefault(sh, r["id"])
    hits = []
    for r in rows:
        for sh in shingles(r["strict_transcript"]):
            o = owner.get(sh)
            if o and o != r["id"]:
                hits.append(f"{r['id']} ~ {o}: {sh!r}")
                break
        for sh in shingles(r["strict_transcript"]):
            owner.setdefault(sh, r["id"])
    return hits


def print_report(kind: str, rows: list[dict], hard: dict, soft: dict, st: dict, show: int = 15) -> bool:
    cfg = KINDS[kind]
    n = max(len(rows), 1)

    def line(name, got, target=None, tol=None, cnt=None):
        mark = "" if target is None else ("OK" if abs(got - target) <= tol else "WARN")
        c = f"{cnt:>4}" if cnt is not None else "    "
        t = f"{target:6.0%}" if target is not None else "      "
        print(f"| {name:<24} | {c} | {got:6.1%} | {t} | {mark:<4} |")

    print(f"\n## {kind}: {len(rows)} samples\n")
    print("| item                     |    n |  share | target | ok   |\n|---|---:|---:|---:|---|")
    for key, (targets, tol) in cfg["targets"].items():
        for k, v in targets.items():
            line(f"{key}: {k}", st[key][k] / n, v, tol, st[key][k])
    for k, (tgt, tol) in cfg["tags"].items():
        line(k, st["tag"][k] / n, tgt, tol, st["tag"][k])
    for k in ("tightened", "self_correction", "spelled", "repeat", "false_start", "filler", "spoken_filler", "light",
              "list", "email", "numbers", "continuation", "related", "spoken_differs"):
        if k not in cfg["tags"]:
            line(k, st["tag"][k] / n, cnt=st["tag"][k])
    for k in sorted(st["nterms"]):
        line(f"terms per sample = {k}", st["nterms"][k] / n, cnt=st["nterms"][k])
    for k in sorted(st["tone"]):
        line(f"tone: {k}", st["tone"][k] / n, cnt=st["tone"][k])
    for k in ("acro_at_start", "acro_plural_possessive", "acro_next_to_number"):
        if kind == "acro":
            line(k, st["tag"][k] / n, cnt=st["tag"][k])
    if kind in ("long", "longen"):
        for lg in ("en", "zh-en", "zh"):
            if st["language"][lg]:
                print(f"| mean spoken s ({lg})      |      | {st['seconds'][lg] / st['language'][lg]:5.1f} s |        |      |")
    print(f"\napps ({len(st['app'])}): " + ", ".join(f"{a} {c}" for a, c in st["app"].most_common()))
    top = st["opening"].most_common(12)
    print(f"openings: {len(st['opening'])} distinct; most common: " + ", ".join(f"{o!r} {c}" for o, c in top))
    if st["acronym"]:
        print(f"acronyms ({len(st['acronym'])} distinct, samples per acronym): "
              + ", ".join(f"{a} {c}" for a, c in st["acronym"].most_common()))

    ok = True
    print("\n| hard check | failures |\n|---|---:|")
    for k in ["schema", "banned", "owner_test_banned", "policy_v1", "leak", "casual_final", "final_adds",
              "final_identical", "final_reason", "terms", "duration", "spelled_spoken", "spoken_drift", "one_line",
              "duplicate"] + (["acronym"] if kind in NEW_KINDS else []):
        print(f"| {k} | {len(hard.get(k, []))} |")
        ok &= not hard.get(k)
    print("\n| soft check (review) | count |\n|---|---:|")
    for k in ("zh_digits_spoken", "negation", "shared_run"):
        print(f"| {k} | {len(soft.get(k, []))} |")
    for k, v in list(hard.items()) + list(soft.items()):
        for x in v[:show]:
            print(f"  [{k}] {x}")
        if len(v) > show:
            print(f"  [{k}] … {len(v) - show} more")
    return ok


def load(kind: str) -> list[dict]:
    p = KINDS[kind]["path"]
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", choices=[*KINDS, []], help="default: long more")
    ap.add_argument("--batch", help="check one source batch file (not yet assembled)")
    ap.add_argument("--kind", choices=["long", "more", "acro", "longen"], help="set of the --batch file")
    args = ap.parse_args()
    args.sets = args.sets or ["long", "more"]
    ctx = Ctx()
    bank_hard: dict[str, list[str]] = defaultdict(list)
    check_bank(ctx, bank_hard)
    if len(ctx.bank_terms) != len(ctx.bank):
        bank_hard["schema"].append("bank: duplicate terms")
    ok = True
    if args.batch:
        path = Path(args.batch)
        rows = [{"id": f"batch_{i:03d}", **r} for i, r in enumerate(parse_batch(path, ctx.tones), 1)]
        seen = {}
        hard, soft, st = check_rows(rows, args.kind, ctx, seen)
        for k, v in bank_hard.items():
            hard[k] = v + hard[k]
        soft["shared_run"] = uniformity(rows, load("pilot"))
        ok = print_report(args.kind, rows, hard, soft, st, show=200)
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    all_rows = {k: load(k) for k in KINDS}
    seen: dict[str, str] = {}
    done: list[dict] = []
    for k in KINDS:  # duplicates and shared runs are checked against the sets not being validated as well
        if k not in args.sets:
            for r in all_rows[k]:
                seen.setdefault(r["strict_transcript"], r["id"])
            done += all_rows[k]
    print(f"{len(ctx.bank)} bank terms")
    for kind in args.sets:
        rows = all_rows[kind]
        hard, soft, st = check_rows(rows, kind, ctx, seen)
        for k, v in bank_hard.items():
            hard[k] = v + hard[k]
        soft["shared_run"] = uniformity(rows, done)
        done += rows
        ok &= print_report(kind, rows, hard, soft, st)
        used = Counter(t for r in rows for t in r["terms"])
        print(f"bank terms used: {len(used)} distinct, most used: {used.most_common(8)}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
