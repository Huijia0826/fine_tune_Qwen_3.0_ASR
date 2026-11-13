"""Validate the synthetic pilot text set (data/private/synth/pilot_500.jsonl + term_bank.jsonl).

Checks (owner spec 2026-11-09):
  schema       fields / types, unique ids, app in configs/reference/tone_by_app.tsv, tone = tsv tone, language
               consistent with the strict transcript, 0-3 terms that are in the bank and occur in the transcript
  banned       no dictionary-benchmark target word (data/private/e2e/items_dict.jsonl target_vocab) nor a listed
               variant in the bank or in any text field (Latin: whole word, case-insensitive; Chinese: substring)
  owner test   no term of the owner's own test sets that is unseen in real training data
               (data/private/synth/banned_owner_test.json, owner review 2026-11-09): proper nouns and spelled forms
               strictly (whole word, case-insensitive; spelled forms also in their spoken "F I G M A" form), generic
               entries ("August", "Monday.") exactly as listed (case-sensitive, incl. the final period)
  policy v1    to_label_v1(strict) == strict (diffs listed); no 嗯 / 呃 / um / uh; ASCII , . ? ! not next to CJK
  leaks        before / after cursor never share an 8-character run with the transcript (rule of
               scripts/make_qwen_e2e_data.py `leaks`)
  final        casual: no final 。 / .; final adds no characters that are not in the strict transcript (letters
               case-insensitive), keeps every number; formal / neutral / unknown: identical unless tightened /
               e-mail formatting
  spoken       spelled letters read letter by letter; zh / zh-en transcripts with digits get a spoken text
  lengths      short / medium / long buckets, estimated speaking time <= 25 s
  distributions app group, language, terms, lengths, cursor text, tags vs the targets

    uv run python scripts/synth/validate_pilot.py            # exit code 1 if a hard check fails
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from whisperflow.text.label_style import to_label_v1

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "data/private/synth/pilot_500.jsonl"
BANK = ROOT / "data/private/synth/term_bank.jsonl"
DICT_ITEMS = ROOT / "data/private/e2e/items_dict.jsonl"
OWNER_TEST_BANNED = ROOT / "data/private/synth/banned_owner_test.json"
# entries of banned_owner_test.json that are ordinary words / phrases: matched exactly as listed, not as a word
OWNER_TEST_GENERIC = {"August", "British Britain", "Great Britain", "Hudson", "Hudson River", "Metaphor", "Monday.",
                      "Northern Ireland", "Scouts", "Southeast Asian", "Succession", "Trading", "United Kingdom"}
TONE_TSV = ROOT / "configs/reference/tone_by_app.tsv"

# obvious variants of benchmark targets (spellings, translations, sub-brands) that are banned as well
BANNED_VARIANTS = [
    "GPT", "Chat GPT", "Claude Code", "Anthropic Claude", "Doubao", "CapCut", "Jianying", "希音", "哈弗", "Seedance",
    "Seed Dance", "OpusClip", "Opus Clip", "Wispr", "Whisper Flow", "VoiceCursor", "Type Fast", "TypeFast",
    "Musk", "马斯克", "Thomas", "Kathy", "Catherine", "Damien", "Miles", "Lea", "Leia", "Zach", "Zack", "Danny",
    "Daniel", "Kayla", "Taylor", "Christina", "Krista", "Camila", "Teddy", "Suzy", "Susie", "Susan", "Debbie",
    "Derek", "Jayce", "Ray", "Nathalie", "Amanda", "Stella", "Ariana", "Leo", "蒋", "Chiang",
    "Mobbin", "Sentry", "Langfuse", "Clerk", "Cline", "Jira", "Linear", "Vercel", "Siri", "Waze", "Shein",
    "Lake Tahoe", "macOS Tahoe", "UniFi", "Netgear Orbi", "Astra", "Kairos", "服务器", "dark mode", "深色模式",
    "画中画", "PiP", "Picture in Picture", "次日留存", "内测版", "主分支", "edge case", "edge-case",
]
CHAT = {"个人聊天", "社交发帖 / 评论", "游戏内聊天", "AI 角色聊天"}
AI = {"AI 助手", "AI 编程", "终端", "开发工具"}
WORK = {"邮件", "笔记", "工作 IM", "文档", "办公协作（写作类）"}
EMAIL_APPS = {"Gmail", "Outlook"}
TARGETS = {  # share, tolerance (absolute)
    "group": ({"chat": .35, "ai": .30, "work": .25, "other": .10}, .04),
    "language": ({"zh": .40, "zh-en": .40, "en": .20}, .04),
    "length": ({"short": .40, "medium": .45, "long": .15}, .05),
}
TAG_TARGETS = {"self_correction": (.10, .03), "spelled": (.05, .02), "question": (.10, .04),
               "no_terms": (.15, .04), "tightened": (.15, .05), "before_nonempty": (.60, .06),
               "after_nonempty": (.25, .05)}
CJK = r"一-鿿"
FIELDS = {"id": str, "app": str, "tone": str, "language": str, "terms": list, "strict_transcript": str,
          "spoken_text": str, "before_cursor": str, "after_cursor": str, "final_output": str}


def _n(s: str | None) -> str:  # as scripts/make_qwen_e2e_data.py
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def leaks(label: str, text: str | None, k: int = 8) -> bool:  # as scripts/make_qwen_e2e_data.py
    lab, t = _n(label), _n(text)
    return len(lab) >= k and bool(t) and any(lab[i:i + k] in t for i in range(len(lab) - k + 1))


def banned_patterns() -> list[tuple[str, re.Pattern]]:
    words = set(BANNED_VARIANTS)
    for line in DICT_ITEMS.read_text(encoding="utf-8").splitlines():
        words.update(json.loads(line).get("target_vocab") or [])
    pats = []
    for w in sorted(words, key=str.lower):
        w = w.strip()
        if not w:
            continue
        if re.search(rf"[{CJK}]", w) and not re.search(r"[A-Za-z]", w):
            pats.append((w, re.compile(re.escape(w))))
        else:
            core = re.escape(w.rstrip(".?!"))
            pats.append((w, re.compile(rf"(?<![A-Za-z]){core}(?![A-Za-z])", re.IGNORECASE)))
    return pats


def owner_test_patterns() -> list[tuple[str, re.Pattern]]:
    pats = []
    for w in json.loads(OWNER_TEST_BANNED.read_text(encoding="utf-8"))["banned_owner_test_unseen"]:
        if re.search(rf"[{CJK}]", w):
            pats.append((w, re.compile(re.escape(w))))
        elif w in OWNER_TEST_GENERIC:
            pats.append((w, re.compile(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])")))
        elif re.fullmatch(r"[A-Z0-9](?:-[A-Z0-9])+", w):  # F-I-G-M-A, also read aloud as "F I G M A"
            alt = re.escape(w).replace(r"\-", "[- ]")
            pats.append((w, re.compile(rf"(?<![A-Za-z]){alt}(?![A-Za-z])", re.IGNORECASE)))
        else:
            pats.append((w, re.compile(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", re.IGNORECASE)))
    return pats


def units(text: str, lang: str) -> int:
    """zh / zh-en: CJK characters + Latin words + digit groups; en: words."""
    if lang == "en":
        return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.,:/-]*", text))
    return len(re.findall(rf"[{CJK}]", text)) + len(re.findall(r"[A-Za-z][A-Za-z'&.-]*|\d[\d,.:%]*", text))


def seconds(text: str, lang: str) -> float:
    if lang == "en":
        return units(text, lang) / 2.6
    cjk = len(re.findall(rf"[{CJK}]", text))
    return cjk / 4.2 + 0.45 * (units(text, lang) - cjk)


def bucket(text: str, lang: str) -> str:
    """short: <= 15 words / 25 Chinese units (owner); long: > 35 words / 60 units (~13-25 s) and >= 2 sentences."""
    u = units(text, lang)
    lo, hi = (15, 35) if lang == "en" else (25, 60)
    if u <= lo:
        return "short"
    return "long" if u > hi and sentences(text) >= 2 else "medium"


def sentences(text: str) -> int:
    return max(1, len(re.findall(r"[。？！?!]|\.(?=\s|$)", text.strip())))


def has_term(term: str, text: str) -> bool:
    if re.search(r"[A-Za-z]", term):
        return re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text, re.IGNORECASE) is not None
    return term in text


def chars(s: str) -> Counter:
    return Counter(c for c in s.lower() if c.isalnum())


def is_question(t: str) -> bool:
    return t.rstrip().endswith(("？", "?"))


def main() -> int:
    tones = {}
    for line in TONE_TSV.read_text(encoding="utf-8").splitlines():
        if line.startswith(("#", "tone\t")) or not line.strip():
            continue
        tone, scen, app = line.split("\t")
        tones[app] = (tone, scen)
    bank = [json.loads(l) for l in BANK.read_text(encoding="utf-8").splitlines()]
    bank_terms = {b["term"] for b in bank}
    rows = [json.loads(l) for l in PILOT.read_text(encoding="utf-8").splitlines()]
    pats = banned_patterns()
    owner_pats = owner_test_patterns()
    hard: dict[str, list[str]] = {k: [] for k in
                                  ["schema", "banned", "owner_test_banned", "policy_v1", "leak", "casual_final", "final_adds",
                                   "final_identical", "terms", "duration", "spelled_spoken", "duplicate"]}
    soft: dict[str, list[str]] = {k: [] for k in ["zh_digits_spoken", "long_sentences", "negation"]}

    # ---- term bank
    for b in bank:
        for w, p in pats:
            if p.search(b["term"]):
                hard["banned"].append(f"bank: {b['term']} ~ {w}")
        for w, p in owner_pats:
            if p.search(b["term"]):
                hard["owner_test_banned"].append(f"bank: {b['term']} ~ {w}")
        if to_label_v1(b["term"]) != b["term"]:  # e.g. 茶百道 -> 茶 100 道: the term cannot survive the label pass
            hard["policy_v1"].append(f"bank: {b['term']!r} -> {to_label_v1(b['term'])!r}")
        if b.get("lookalike") and b["lookalike"] not in bank_terms:
            hard["schema"].append(f"bank: lookalike {b['lookalike']} of {b['term']} not in bank")
    if len(bank_terms) != len(bank):
        hard["schema"].append("bank: duplicate terms")

    ids, seen_strict = set(), {}
    stats = Counter()
    group_c, lang_c, len_c, nterm_c = Counter(), Counter(), Counter(), Counter()
    app_c, tone_c = Counter(), Counter()
    for r in rows:
        rid = r.get("id", "?")
        for f, typ in FIELDS.items():
            if not isinstance(r.get(f), typ):
                hard["schema"].append(f"{rid}: field {f} missing / wrong type")
        if rid in ids:
            hard["schema"].append(f"{rid}: duplicate id")
        ids.add(rid)
        s, fin, sp = r["strict_transcript"], r["final_output"], r["spoken_text"]
        bef, aft, lang = r["before_cursor"], r["after_cursor"], r["language"]
        if r["app"] not in tones:
            hard["schema"].append(f"{rid}: app {r['app']} not in tsv")
            continue
        tone, scen = tones[r["app"]]
        if r["tone"] != tone:
            hard["schema"].append(f"{rid}: tone {r['tone']} != tsv {tone}")
        zh, en = bool(re.search(rf"[{CJK}]", s)), bool(re.search(r"[A-Za-z]", s))
        exp_lang = "zh-en" if zh and en else ("zh" if zh else "en")
        if lang != exp_lang:
            hard["schema"].append(f"{rid}: language {lang} but text looks {exp_lang}")
        if s in seen_strict:
            hard["duplicate"].append(f"{rid} = {seen_strict[s]}: {s}")
        seen_strict[s] = rid

        # terms
        if len(r["terms"]) > 3:
            hard["terms"].append(f"{rid}: {len(r['terms'])} terms")
        for t in r["terms"]:
            if t not in bank_terms:
                hard["terms"].append(f"{rid}: term {t} not in bank")
            if not has_term(t, s):
                hard["terms"].append(f"{rid}: term {t} not in transcript")

        # banned
        for field in ("strict_transcript", "spoken_text", "final_output", "before_cursor", "after_cursor"):
            for w, p in pats:
                if p.search(r[field]):
                    hard["banned"].append(f"{rid}.{field}: {w}")
        for t in r["terms"]:
            for w, p in pats:
                if p.search(t):
                    hard["banned"].append(f"{rid}.terms: {t} ~ {w}")

        for field in ("strict_transcript", "spoken_text", "final_output", "before_cursor", "after_cursor"):
            for w, p in owner_pats:
                if p.search(r[field]):
                    hard["owner_test_banned"].append(f"{rid}.{field}: {w}")
        for t in r["terms"]:
            for w, p in owner_pats:
                if p.search(t):
                    hard["owner_test_banned"].append(f"{rid}.terms: {t} ~ {w}")

        # policy v1
        v1 = to_label_v1(s)
        if v1 != s:
            hard["policy_v1"].append(f"{rid}: {s!r} -> {v1!r}")
        if re.search(r"[嗯呃]", s) or re.search(r"(?<![A-Za-z])(um+|uh+)(?![A-Za-z])", s, re.I):
            hard["policy_v1"].append(f"{rid}: filler in {s!r}")
        if re.search(rf"[{CJK}][,.?!]|[,.?!]\s*[{CJK}]", s):
            hard["policy_v1"].append(f"{rid}: ASCII punctuation next to Chinese: {s!r}")

        # leaks (transcript and final output)
        for name, ctx in (("before", bef), ("after", aft)):
            if leaks(s, ctx) or leaks(fin, ctx):
                hard["leak"].append(f"{rid}: {name}_cursor shares 8 chars with the transcript")

        # final output
        email = r["app"] in EMAIL_APPS and "\n" in fin
        if tone == "casual":
            if re.search(r"[。.]$", fin.rstrip()) and not fin.rstrip().endswith("..."):
                hard["casual_final"].append(f"{rid}: casual final ends with a period: {fin!r}")
            rule = re.sub(r"[。.]+$", "", s.rstrip())
        else:
            rule = s
        tightened = fin != rule and not (email and re.sub(r"\s+", " ", fin) == re.sub(r"\s+", " ", rule))
        extra = chars(fin) - chars(s)
        if extra:
            hard["final_adds"].append(f"{rid}: final adds {''.join(sorted(extra.elements()))!r}: {fin!r}")
        nums_s, nums_f = re.findall(r"\d[\d,.:]*\d|\d", s), re.findall(r"\d[\d,.:]*\d|\d", fin)
        if re.search(r"\b[A-Z0-9](?:-[A-Z0-9]){2,}\b", s):  # assembled spelling ("Kiwi77, K-I-W-I-7-7" -> "Kiwi77"):
            changed = bool(Counter(re.findall(r"\d", fin)) - Counter(re.findall(r"\d", s)))  # no new digits
        else:
            changed = bool(Counter(nums_f) - Counter(nums_s)) or (
                bool(Counter(nums_s) - Counter(nums_f)) and "self_correction" not in r["tags"])
        if changed:
            hard["final_adds"].append(f"{rid}: numbers changed {nums_s} -> {nums_f}")
        neg = lambda t: len(re.findall(r"[不没别]|(?<![A-Za-z])(not|never|no)(?![A-Za-z])|n't", t, re.I))
        if neg(fin) != neg(s) and "self_correction" not in r["tags"]:
            soft["negation"].append(f"{rid}: {s!r} -> {fin!r}")
        tagged_tight = "tightened" in r["tags"]
        if tightened and not tagged_tight and not email:
            hard["final_identical"].append(f"{rid}: final differs from the rule output but not tagged tightened")
        if tagged_tight and not tightened:
            hard["final_identical"].append(f"{rid}: tagged tightened but final = rule output")

        # spoken text
        spelled = re.findall(r"\b[A-Z0-9](?:-[A-Z0-9]){2,}\b", s)
        if bool(spelled) != ("spelled" in r["tags"]):
            hard["schema"].append(f"{rid}: tag 'spelled' does not match the transcript")
        if spelled:
            stats["spelled"] += 1
            if any(sp_ in sp for sp_ in spelled):
                hard["spelled_spoken"].append(f"{rid}: spoken text keeps the hyphenated spelling")
        if lang != "en" and re.search(r"(?<![A-Za-z])\d", sp):  # digits outside Latin tokens (S3, B2B ok)
            soft["zh_digits_spoken"].append(f"{rid}: {sp!r}")
        if sp != s:
            stats["spoken_differs"] += 1

        # lengths
        b = bucket(s, lang)
        len_c[b] += 1
        if seconds(s, lang) > 25:
            hard["duration"].append(f"{rid}: ~{seconds(s, lang):.0f} s")
        if b == "long" and not 2 <= sentences(s) <= 4:
            soft["long_sentences"].append(f"{rid}: {sentences(s)} sentences")

        group_c["chat" if scen in CHAT else "ai" if scen in AI else "work" if scen in WORK else "other"] += 1
        lang_c[lang] += 1
        nterm_c[len(r["terms"])] += 1
        app_c[r["app"]] += 1
        tone_c[tone] += 1
        stats["before_nonempty"] += bool(bef.strip())
        stats["after_nonempty"] += bool(aft.strip())
        stats["related_before"] += "related" in r["tags"]
        stats["question"] += is_question(s)
        stats["self_correction"] += "self_correction" in r["tags"]
        stats["repeat"] += "repeat" in r["tags"]
        stats["list"] += "list" in r["tags"]
        stats["tightened"] += tightened and not email
        stats["email"] += email
        stats["numbers"] += bool(re.search(r"\d", s))
        stats["no_terms"] += not r["terms"]
        stats["continuation"] += "continuation" in r["tags"]
        stats["filler"] += "filler" in r["tags"]

    n = len(rows)
    ok = True

    def line(name: str, got: float, target: float | None = None, tol: float | None = None, cnt: int | None = None):
        mark = ""
        if target is not None:
            mark = "OK" if abs(got - target) <= tol else "WARN"
        c = f"{cnt:>4}" if cnt is not None else "    "
        t = f"{target:6.0%}" if target is not None else "      "
        print(f"| {name:<22} | {c} | {got:6.1%} | {t} | {mark:<4} |")

    print(f"\n{n} samples, {len(bank)} bank terms\n")
    print("| item                   |    n |  share | target | ok   |\n|---|---:|---:|---:|---|")
    for key, counter in (("group", group_c), ("language", lang_c), ("length", len_c)):
        targets, tol = TARGETS[key]
        for k, v in targets.items():
            line(f"{key}: {k}", counter[k] / n, v, tol, counter[k])
    for k in ("no_terms", "before_nonempty", "after_nonempty", "self_correction", "spelled", "question",
              "tightened"):
        tgt, tol = TAG_TARGETS[k]
        line(k, stats[k] / n, tgt, tol, stats[k])
    for k in ("repeat", "filler", "list", "email", "numbers", "continuation", "related_before", "spoken_differs"):
        line(k, stats[k] / n, cnt=stats[k])
    for k in sorted(nterm_c):
        line(f"terms per sample = {k}", nterm_c[k] / n, cnt=nterm_c[k])
    for k in sorted(tone_c):
        line(f"tone: {k}", tone_c[k] / n, cnt=tone_c[k])
    print(f"\napps ({len(app_c)}): " + ", ".join(f"{a} {c}" for a, c in app_c.most_common()))
    used = Counter(t for r in rows for t in r["terms"])
    print(f"bank terms used: {len(used)} distinct, most used: {used.most_common(8)}")

    print("\n| hard check | failures |\n|---|---:|")
    for k, v in hard.items():
        print(f"| {k} | {len(v)} |")
        ok &= not v
    print("\n| soft check (review) | count |\n|---|---:|")
    for k, v in soft.items():
        print(f"| {k} | {len(v)} |")
    for k, v in list(hard.items()) + list(soft.items()):
        for x in v[:15]:
            print(f"  [{k}] {x}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
