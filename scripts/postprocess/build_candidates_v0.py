#!/usr/bin/env python3
"""Build candidate (ASR transcript -> teacher output) pairs for the post-processing student, v0.

Read-only: PostHog asr_metrics via the trace-lookup skill helpers (HogQL SELECT only). No LLM calls,
no writes to PostHog / Langfuse / arena. Output contains users' text -> data/private only.

Stages
  pull   per-day HogQL pull of llm_pp dictate rows -> <raw_dir>/<YYYY-MM-DD>.jsonl (cached; past days are
         skipped when the file exists, use --refresh to re-pull). Also stores the per-day dictate total.
  build  filters (audit funnel) -> language tag (zh / zh-en / en) -> dedupe -> classify into the owner's four
         categories + control -> per-user cap -> jsonl per category, _stats.json, spot-check sample.

Usage
  python3 scripts/postprocess/build_candidates_v0.py pull  --since 2026-10-10 --until 2026-11-09
  python3 scripts/postprocess/build_candidates_v0.py build --since 2026-10-10 --until 2026-11-09

Dates are UTC days, `until` exclusive. Credentials come from ~/.config/vc_bench.env (trace-lookup skill).
If a local proxy is set, run with NO_PROXY="*" (PostHog pulls were ~12x faster without the proxy).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

SKILL_DIR = Path(os.environ.get("TRACE_LOOKUP_DIR", "~/.claude/skills/trace-lookup/scripts")).expanduser()
sys.path.insert(0, str(SKILL_DIR))

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/private/postprocess/raw_pull"
OUT_DIR = ROOT / "data/private/postprocess/candidates_v0"
USER_SALT = "vc-pp-candidates-v0"  # pseudonymisation only (not a secret)

# --------------------------------------------------------------------------------------------------
# pull
# --------------------------------------------------------------------------------------------------

# scalar properties pulled per row (no screen text / dictionary payloads)
PROPS = ["audio_id", "user_id", "client_source", "mode", "requested_mode", "model", "lang", "languages",
         "transcription", "processed", "original_len", "processed_len", "asr_seconds", "postprocess_seconds",
         "cleanup_audio_duration_ms", "error_stage", "error", "session_end_reason", "edit_action",
         "edit_no_op_reason", "edit_fallback_used", "postprocess_model", "postprocess_langfuse_prompt_name",
         "postprocess_langfuse_prompt_label", "cleanup_format", "cleanup_format_source", "langfuse_trace_id",
         "app_type", "tone_id", "tone_source", "tone_app_id", "tone_prompt", "invocation_context",
         "multi_modality", "effective_vocabulary_count", "postprocessor_implementation"]
NESTED = {"app_bundle_id": "properties.audio_context.app.bundle_id",
          "app_name": "properties.audio_context.app.name",
          "ctx_language": "properties.audio_context.language",
          "field_kind": "properties.audio_context.app.field_kind"}
# llm_pp = the request went through the text post-processing LLM. For the multimodal-ASR route
# ("no_text_pp") postprocess_model holds the ASR model name and processed == transcription (audit §1).
LLM_PP_SQL = ("properties.postprocess_model IS NOT NULL AND properties.postprocess_model != '' "
              "AND NOT (properties.postprocess_model ILIKE '%transcribe%' OR properties.postprocess_model ILIKE '%asr%' "
              "OR properties.postprocess_model IN ('auto', 'none', 'unknown'))")
BASE_SQL = ("event = 'asr_metrics' AND properties.mode = 'dictate' "
            "AND coalesce(properties.client_source, '') != 'arena'")
PAGE = 4000


def days(since: str, until: str):
    d, end = dt.date.fromisoformat(since), dt.date.fromisoformat(until)
    while d < end:
        yield d.isoformat()
        d += dt.timedelta(days=1)


def pull_day(day: str, q) -> tuple[list[dict], int]:
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    win = f"timestamp >= '{day}' AND timestamp < '{nxt}'"
    total = q(f"SELECT count() FROM events WHERE {BASE_SQL} AND {win}")[0][0]
    sel = ", ".join([f"properties.{p}" for p in PROPS] + list(NESTED.values()))
    cols = PROPS + list(NESTED)
    rows, seen, cursor = [], set(), f"{day}T00:00:00"
    while True:
        res = q(f"SELECT uuid, timestamp, {sel} FROM events WHERE {BASE_SQL} AND {LLM_PP_SQL} "
                f"AND timestamp >= '{cursor}' AND timestamp < '{nxt}' ORDER BY timestamp LIMIT {PAGE}")
        new = 0
        for u, ts, *vals in res:
            if u in seen:
                continue
            seen.add(u)
            new += 1
            r = dict(zip(cols, vals))
            r["timestamp"] = ts
            if isinstance(r.get("languages"), str):
                try:
                    r["languages"] = json.loads(r["languages"])
                except ValueError:
                    r["languages"] = [r["languages"]]
            rows.append(r)
        if len(res) < PAGE or not new:
            break
        cursor = res[-1][1]
    return rows, total


def make_query():
    """HogQL SELECT with a short timeout and retries (the skill's q() can hang on a stalled response)."""
    import time

    import requests
    from trace_lookup import POSTHOG, load_env
    load_env(POSTHOG)
    url = f"{os.environ['POSTHOG_HOST'].rstrip('/')}/api/projects/{os.environ['POSTHOG_PROJECT_ID']}/query"
    hdr = {"Authorization": f"Bearer {os.environ['POSTHOG_PERSONAL_API_KEY']}"}

    def q(sql):
        for i in range(6):
            try:
                r = requests.post(url, headers=hdr, json={"query": {"kind": "HogQLQuery", "query": sql}},
                                  timeout=(10, 90))
                if r.status_code == 200:
                    return r.json()["results"]
                print(f"  PostHog {r.status_code}: {r.text[:200]}", file=sys.stderr)
            except requests.RequestException as e:
                print(f"  PostHog retry {i + 1}: {type(e).__name__}", file=sys.stderr)
            time.sleep(5 * (i + 1))
        sys.exit("PostHog query failed after retries")
    return q


def cmd_pull(a):
    q = make_query()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for day in days(a.since, a.until):
        path = RAW_DIR / f"{day}.jsonl"
        if path.exists() and not a.refresh and day < today:
            continue
        rows, total = pull_day(day, q)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_day": day, "dictate_total": total, "llm_pp_pulled": len(rows)}) + "\n")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"{day}: dictate {total}, llm_pp {len(rows)}", flush=True)


# --------------------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------------------

HAN = re.compile(r"[㐀-䶿一-鿿]")
KANA_HANGUL = re.compile(r"[぀-ヿ가-힯]")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’]*")
OTHER_SCRIPT = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿঀ-৿฀-๿֐-׿]")
STOP = {"en": set("the and is are you to of it that this for with have was what not i my me we they can "
                  "it's i'm don't just so but on in do be at if your a an how about as please got by or our "
                  "hey well will would there their them he she his her from up out all like know get go "
                  "yeah okay ok yes thanks thank want need can't that's i'll you're let's".split()),
        "pt": set("de que não e o é um uma para com eu você vocês isso mas está estão tá tô então muito também "
                  "tudo bem aqui agora pra já da do na no os as a me se meu minha né ele ela aí vai tem ser".split()),
        "es": set("de que no el la es un una para con yo tú eso pero está y estoy muy también aquí ahora ya "
                  "bueno qué los las del a me se mi".split()),
        "fr": set("le la les de et est un une pour avec je vous ce mais pas c'est oui très".split()),
        "de": set("der die das und ist nicht ich du ein eine mit für aber zu ja auch".split()),
        "it": set("di che non il la è un una per con io tu questo ma sono anche molto".split())}


def base_lang(code) -> str:
    return (code or "").split("-")[0].split("_")[0].lower()


def lang_tag(text: str, langs) -> str:
    """zh / zh-en / en / other, from the ASR text (settings are only a tie-breaker for short Latin text)."""
    text = text or ""
    if KANA_HANGUL.search(text) or OTHER_SCRIPT.search(text):
        return "other"
    han = len(HAN.findall(text))
    words = [w.lower() for w in LATIN_WORD.findall(text)]
    if han:
        if han < 2 and len(words) > 3:
            pass  # a stray character inside Latin text -> judge as Latin below
        else:
            return "zh-en" if any(len(w) >= 2 for w in words) else "zh"
    if not words:
        return "other"
    ratio = {l: sum(w in s for w in words) / len(words) for l, s in STOP.items()}
    other = max(v for l, v in ratio.items() if l != "en")
    settings = {base_lang(x) for x in (langs or [])} - {"", "auto"}
    if ratio["en"] > other and (ratio["en"] >= 0.08 or other == 0):
        return "en"
    if ratio["en"] == other == 0 and settings and settings <= {"en"}:
        return "en"
    return "other"


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


TOKEN = re.compile(r"[㐀-䶿一-鿿]|[0-9A-Za-zÀ-ɏ]+(?:['’][A-Za-zÀ-ɏ]+)*")


def content_tokens(s: str) -> list[tuple[str, int, int]]:
    """Content tokens (CJK char / Latin-digit word) with char offsets; punctuation and spacing are ignored."""
    s = nfkc(s)
    return [(m.group(0).lower().replace("’", "'"), m.start(), m.end()) for m in TOKEN.finditer(s)]


def norm_key(s: str) -> str:
    return "".join(t for t, _, _ in content_tokens(s))


EN_FILLERS = {"um", "uh", "uhm", "umm", "uhh", "erm", "er", "hmm", "hm", "mm", "mhm", "ah", "eh", "ahh"}
EN_SOFT = {"like", "so", "well", "yeah", "okay", "ok", "oh", "basically", "just", "you", "know", "kind", "of",
           "sort", "i", "mean", "and", "but", "right", "actually", "literally"}
ZH_FILLER_CHARS = set("呃嗯啊额哦噢唉哎诶欸呐嘛吧呀")
ZH_FILLER_WORDS = ["就是说", "那个", "这个", "就是", "然后", "的话", "对对", "对", "那", "反正", "其实"]


def is_filler_span(toks: list[str]) -> bool:
    """True if a deleted span consists only of hesitation sounds / discourse fillers."""
    if not toks:
        return False
    if all(t in EN_FILLERS or t in EN_SOFT for t in toks if not HAN.match(t)) and not any(HAN.match(t) for t in toks):
        return any(t in EN_FILLERS or t in {"like", "so", "well", "yeah", "okay", "ok", "oh", "you", "know",
                                            "basically", "i", "mean", "actually", "literally"} for t in toks)
    zh = "".join(t for t in toks if HAN.match(t))
    if zh and not any(not HAN.match(t) and t not in EN_FILLERS for t in toks):
        rest = zh
        for w in sorted(ZH_FILLER_WORDS, key=len, reverse=True):
            rest = rest.replace(w, "")
        return all(c in ZH_FILLER_CHARS for c in rest)
    return False


# self-correction cues (input side). Precision is checked by hand in the report.
EN_CUE = re.compile(
    r"\b(no,? wait|wait,? no|scratch that|i mean|i meant|or rather|sorry|actually|correction|let me rephrase|"
    r"make that|my bad|oops|never ?mind|forget that|start over|not \w+(?: \w+)?,? but)\b|[,.;]\s*no\s*[,.;]|[,.;]\s*no no\b|\bno, not\b", re.I)
ZH_CUE = re.compile(r"不对|哦不|噢不|呃不|啊不|不不|我是说|我的意思是|(?:^|[，,。！？\s])应该是|更正|说错了|口误|算了|改成|"
                    r"不好意思|准确地说|确切地说|[，,。]\s*不是[，,]|还是[^，。,.]{1,12}吧")
SPOKEN_PUNCT = re.compile(
    r"\b(comma|period|full stop|question mark|exclamation (?:mark|point)|new line|newline|next line|"
    r"new paragraph|colon|semicolon|open quote|close quote|end quote|unquote|dot com)\b|"
    r"逗号|句号|问号|感叹号|换行|另起一行|冒号|分号|顿号|引号|省略号", re.I)
LIST_CUE = re.compile(
    r"\b(first(ly)?|second(ly)?|third(ly)?|number one|number two|point one|bullet( point)?|step one|"
    r"next item|one,? two,? three)\b|第[一二三四五六七八九十\d]+(点|个|条|项|件|步)?|首先|其次|最后一点|一是|二是",
    re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening)|greetings|你好|您好|亲爱的|各位|"
                      r"[\w一-鿿]{1,12}(老师|总|经理|你好|您好))[\s,，!！:：]", re.I)
SIGNOFF = re.compile(r"^\s*(best( regards)?|kind regards|regards|thanks|thank you|many thanks|sincerely|cheers|"
                     r"warm regards|talk soon|谢谢|感谢|祝好|此致|敬礼|敬上|谨上|顺颂商祺)[\s,，!！.。]*$", re.I)
LIST_LINE = re.compile(r"^\s*(\d+[.)、]|[-•*]\s|[一二三四五六七八九十]+[、.])")


CUE_WORDS = {"no", "not", "sorry", "wait", "mean", "meant", "actually", "correction", "rather", "or", "oops", "my",
             "bad", "make", "that", "scratch", "but", "i", "never", "mind", "nevermind", "forget", "start", "over"}
ZH_CUE_STRINGS = ["不对", "不是", "是", "我是说", "我的意思是", "应该是", "更正", "说错了", "口误", "算了", "改成", "不好意思",
                  "哦不", "噢不", "不不"]
STRONG_CUE = re.compile(r"不对|我是说|我的意思是|说错了|口误|更正|不是|应该是|算了|还是|^(i mean|i meant|no,? wait|"
                        r"wait,? no|scratch that|or rather|correction|make that)$|^[,.;]?\s*no\b", re.I)


def is_cue_false_friend(text: str, m) -> bool:
    """Cue matches that are idioms, not correction operators."""
    pre = text[max(0, m.start() - 12):m.start()].lower()
    cue = m.group(0).lower()
    if cue == "不对" and text[max(0, m.start() - 1):m.start()] == "对":  # 对不对 = tag question
        return True
    if "mean" in cue and re.search(r"(know|see|what)\s*$", pre):  # "you know what I mean"
        return True
    if cue.startswith("sorry") and re.search(r"(i'm|im|i am|so|very|really|feel|be|say)\s*$", pre):  # apology
        return True
    return False


def content_of(toks: list[str]) -> list[str]:
    """Drop hesitation / discourse fillers and correction-cue words; what remains is overridden content."""
    out = []
    zh = "".join(t for t in toks if HAN.match(t))
    for w in sorted(ZH_CUE_STRINGS + ZH_FILLER_WORDS, key=len, reverse=True):
        zh = zh.replace(w, "")
    out.extend(c for c in zh if c not in ZH_FILLER_CHARS)
    out.extend(t for t in toks if not HAN.match(t) and t not in EN_FILLERS and t not in EN_SOFT
               and t not in CUE_WORDS)
    return out


def atok_after(at, pos: int, n: int) -> list[str]:
    return content_of([t for t, s0, _ in at if s0 >= pos][:n])


def contains_seq(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def join_toks(toks: list[str]) -> str:
    return "".join(t if HAN.match(t) else f" {t}" for t in toks).strip()



def final_punct(s: str) -> str:
    s = (s or "").rstrip()
    if not s:
        return "empty"
    if s.endswith("...") or s.endswith("…"):
        return "ellipsis"
    c = s[-1]
    return {"。": "period", ".": "period", "？": "question", "?": "question", "！": "exclaim", "!": "exclaim",
            "，": "comma", ",": "comma"}.get(c, "none" if (c.isalnum() or HAN.match(c)) else "other")


# --------------------------------------------------------------------------------------------------
# classifier (core block; see report §exercises)
# --------------------------------------------------------------------------------------------------

def diff_spans(a: str, b: str):
    """Token-level diff between input a and output b. Returns (opcodes, a_tokens, b_tokens)."""
    at, bt = content_tokens(a), content_tokens(b)
    sm = difflib.SequenceMatcher(None, [t for t, _, _ in at], [t for t, _, _ in bt], autojunk=False)
    return sm.get_opcodes(), at, bt


def classify(inp: str, out: str, row: dict) -> dict:
    """Assign category labels with evidence. Returns {"labels": {cat: {...evidence}}, "op": str, ...}."""
    res: dict = {"labels": {}}
    a, b = nfkc(inp), nfkc(out)
    ops, at, bt = diff_spans(inp, out)
    atok, btok = [t for t, _, _ in at], [t for t, _, _ in bt]
    n_changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    edit_ratio = n_changed / max(1, len(atok))
    res["edit_ratio"] = round(edit_ratio, 3)
    res["final_punct_in"], res["final_punct_out"] = final_punct(inp), final_punct(out)

    # --- 1. self-correction ---------------------------------------------------------------------
    zh_text = bool(HAN.search(a))
    cue_re = ZH_CUE if zh_text else EN_CUE
    cues_in = [m for m in cue_re.finditer(a) if not is_cue_false_friend(a, m)]
    n_cue_out = len([m for m in cue_re.finditer(b) if not is_cue_false_friend(b, m)])
    deleted = {i for tag, i1, i2, j1, j2 in ops if tag in ("delete", "replace") for i in range(i1, i2)}
    sc = None
    for m in cues_in:
        if n_cue_out >= len(cues_in):
            break  # the teacher kept every cue -> no correction applied
        cue_idx = [i for i, (_, s0, e0) in enumerate(at) if s0 < m.end() and e0 > m.start()]
        if not cue_idx or not all(i in deleted for i in cue_idx):
            continue  # this cue occurrence survived -> not the operator of an applied correction
        # the overridden span = the contiguous deleted run that ends at the cue
        i = min(cue_idx) - 1
        run = []
        while i >= 0 and i in deleted and len(run) < 30:
            run.insert(0, atok[i])
            i -= 1
        # the replacement (first content tokens after the cue) must survive in the output ...
        repl = atok_after(at, m.end(), 8)[:2]
        if not repl or not contains_seq(btok, repl):
            continue
        # ... and the deleted stretch must not be a verbatim copy of other text (deleted duplicate sentence)
        j = max(cue_idx) + 1
        while j < len(atok) and j in deleted:
            j += 1
        tail = atok[min(cue_idx):j]  # cue + deleted tokens after it
        if j - max(cue_idx) >= 2 and contains_seq(atok[:i + 1], tail):
            continue
        core = content_of(run)
        after = atok_after(at, m.end(), 6)
        if core and core != after[:len(core)]:  # real content dropped, not a repeat of what follows
            sc = {"subtype": "explicit", "cue": m.group(0).strip(" ,.，。"), "dropped": join_toks(run)}
            break
    if sc is None and cues_in and n_cue_out >= len(cues_in):
        cue = cues_in[0].group(0).strip(" ,.，。")
        if STRONG_CUE.search(cue) and cues_in[0].start() > 0:
            sc = {"subtype": "negative_cue_kept", "cue": cue}
    if sc is None:
        # implicit restatement: a deleted span of >=3 content tokens that is immediately followed by a
        # different version of the same frame (same first token), e.g. "并没有要求…，并没有强制要求…"
        for tag, i1, i2, j1, j2 in ops:
            if tag != "delete" or i2 - i1 < 3 or i2 >= len(atok):
                continue
            span = atok[i1:i2]
            if is_filler_span(span) or len(content_of(span)) < 3 or SPOKEN_PUNCT.search(" ".join(span)):
                continue
            nxt = atok[i2:i2 + len(span) + 6]
            if span == atok[i2:i2 + len(span)] or span == atok[max(0, i1 - len(span)):i1]:
                continue  # exact repetition -> grammar_coherence/repeat_merge
            if nxt and nxt[0] == span[0] and len(set(span) & set(nxt)) / len(set(span)) >= 0.6:
                sc = {"subtype": "implicit_restatement", "dropped": join_toks(span)}
                break
    if sc:
        res["labels"]["self_correction"] = sc

    # --- 2. format: email / list -------------------------------------------------------------------
    lines_out = [l for l in out.split("\n")]
    nl_in, nl_out = inp.count("\n"), out.count("\n")
    n_list = sum(bool(LIST_LINE.match(l)) for l in lines_out)
    greet = bool(lines_out and GREETING.match(lines_out[0])) and nl_out > nl_in
    sign = any(SIGNOFF.match(l) for l in lines_out[1:]) and nl_out > nl_in
    triggers = []
    pname = row.get("postprocess_langfuse_prompt_name") or ""
    if pname.startswith("email_list"):
        triggers.append(f"prompt={pname}")
    if row.get("cleanup_format") in ("list", "email"):
        triggers.append(f"cleanup_format={row.get('cleanup_format')}")
    if LIST_CUE.search(a):
        triggers.append("spoken_cue:" + LIST_CUE.search(a).group(0))
    fmt = None
    if n_list >= 2 and n_list > sum(bool(LIST_LINE.match(l)) for l in inp.split("\n")):
        fmt = "list"
    elif greet or sign:
        fmt = "email"
    elif nl_out > nl_in:
        fmt = "paragraphs"
    if fmt:
        res["labels"]["format_email_list"] = {"subtype": fmt, "triggers": triggers, "newlines_out": nl_out}
    elif any(t.startswith(("prompt=", "cleanup_format=")) for t in triggers):
        res["labels"]["format_email_list"] = {"subtype": "negative_routed_no_layout", "triggers": triggers}

    # --- 3. grammar / coherence + control ----------------------------------------------------------
    if inp == out:
        res["op"] = "unchanged"
    elif atok == btok:
        res["op"] = "punct_case_only"
    else:
        dels = [atok[i1:i2] for tag, i1, i2, j1, j2 in ops if tag == "delete"]
        ins = [btok[j1:j2] for tag, i1, i2, j1, j2 in ops if tag == "insert"]
        reps = [(atok[i1:i2], btok[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == "replace"]
        sp_in, sp_out = len(SPOKEN_PUNCT.findall(a)), len(SPOKEN_PUNCT.findall(b))
        exact_rep = any(atok[i1:i2] == atok[i2:i2 + (i2 - i1)] or atok[i1:i2] == atok[max(0, i1 - (i2 - i1)):i1]
                        for tag, i1, i2, j1, j2 in ops if tag == "delete")
        only_fill = not ins and not reps and dels and all(is_filler_span(d) for d in dels)
        num_edit = any(any(ch.isdigit() for ch in "".join(y)) and not any(ch.isdigit() for ch in "".join(x))
                       for x, y in reps)
        if sp_out < sp_in:
            sub = "spoken_punctuation"
        elif only_fill:
            sub = "filler_only"
        elif exact_rep and not reps and not ins:
            sub = "repeat_merge"
        elif edit_ratio > 0.4:
            sub = "heavy_rewrite"
        elif num_edit and len(reps) <= 2 and not dels and not ins:
            sub = "number_format"
        elif "self_correction" in res["labels"] and res["labels"]["self_correction"]["subtype"] != "negative_cue_kept":
            sub = "correction_only"
        elif fmt in ("list", "email") and edit_ratio <= 0.1:
            sub = "format_only"
        else:
            sub = "word_edit"
        res["op"] = sub
    if res["op"] not in ("unchanged", "correction_only", "format_only"):
        res["labels"]["grammar_coherence"] = {"subtype": res["op"]}
    if res["op"] == "unchanged" or res["op"] == "punct_case_only":
        res["labels"]["control"] = {"subtype": res["op"]}
    res["teacher_suspect"] = teacher_suspect(a, b, ops, atok, btok)
    return res


MARKDOWN = re.compile(r"\*\*|^#{1,4} |```|^\s*[-*] \*\*|^\s*\*[^*\n]+\*\s*$", re.M)


def teacher_suspect(a: str, b: str, ops, atok, btok) -> list[str]:
    """Rule hints that the teacher output may be wrong (not proof; see spot check)."""
    out = []
    if b.lstrip()[:1] in tuple(",，.。;；:：") and a.lstrip()[:1] not in tuple(",，.。;；:："):
        out.append("output_starts_with_punct")
    if MARKDOWN.search(b) and not MARKDOWN.search(a):
        out.append("markdown_or_answer")
    inserted = sum(j2 - j1 for tag, i1, i2, j1, j2 in ops if tag in ("insert", "replace"))
    if len(btok) >= 6 and inserted / len(btok) > 0.3:
        out.append("novel_content_gt30pct")
    if HAN.search(a) and len(HAN.findall(b)) < 0.5 * len(HAN.findall(a)):
        out.append("chinese_dropped_or_translated")
    if len(btok) < 0.5 * len(atok) and len(atok) >= 12:
        out.append("output_lt_half_input")
    return out


# --------------------------------------------------------------------------------------------------
# tone signal
# --------------------------------------------------------------------------------------------------

def tone_kind(row: dict) -> str:
    """Tone as the client reported it. tone_id = '<app category>_<preset>' (e.g. personal_very_casual)
    when the user set a per-category preset, 'default' when not, or a custom tone name with tone_prompt."""
    tid = row.get("tone_id")
    if row.get("tone_prompt"):
        return "custom"
    if tid is None:
        return "missing"
    if tid == "default":
        return "default"
    if tid in ("off", "other_off") or tid.endswith("_off"):
        return "off"
    if "very_casual" in tid:
        return "very_casual"
    if "casual" in tid:
        return "casual"
    if "formal" in tid:
        return "formal"
    return "other_preset"


# --------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------

DROP_FLAGS = ("chatty", "lang_mismatch", "pp_dropped", "pp_expand", "repetition", "text_rate_high", "empty_output")
ONBOARDING_BUNDLES = {"com.voicecursor.android.type"}
CATS = ("self_correction", "format_email_list", "grammar_coherence", "tone_punctuation", "control")


def hash_user(uid) -> str:
    return hashlib.sha256(f"{USER_SALT}:{uid}".encode()).hexdigest()[:12]


def load_raw(since, until):
    totals, rows = {}, []
    for day in days(since, until):
        path = RAW_DIR / f"{day}.jsonl"
        if not path.exists():
            sys.exit(f"missing {path}; run `pull` first")
        with open(path, encoding="utf-8") as f:
            head = json.loads(f.readline())
            totals[day] = head["dictate_total"]
            rows.extend(json.loads(line) for line in f)
    return totals, rows


def cmd_build(a):
    from online import flags_of  # trace-lookup skill rule flags
    rnd = random.Random(a.seed)
    totals, rows = load_raw(a.since, a.until)
    funnel = collections.OrderedDict()
    funnel["dictate_total"] = sum(totals.values())
    funnel["llm_pp"] = len(rows)
    drops = collections.Counter()

    keep = []
    for r in rows:
        if r.get("error_stage") or not (r.get("transcription") or "").strip() or not (r.get("processed") or "").strip():
            drops["error_or_empty"] += 1
            continue
        keep.append(r)
    funnel["no_error_nonempty"] = len(keep)

    rows, keep = keep, []
    for r in rows:
        if (r.get("app_bundle_id") in ONBOARDING_BUNDLES or r.get("tone_app_id") in ONBOARDING_BUNDLES
                or r.get("invocation_context") == "onboarding_practice"
                or "onboarding" in (r.get("app_name") or "").lower()):
            drops["onboarding"] += 1
            continue
        if (r.get("postprocess_langfuse_prompt_label") or "").startswith("test"):
            drops["test_prompt_label"] += 1
            continue
        keep.append(r)
    funnel["no_onboarding_or_test"] = len(keep)

    rows, keep = keep, []
    lang_counts = collections.Counter()
    for r in rows:
        r["lang_tag"] = lang_tag(r["transcription"], r.get("languages"))
        lang_counts[r["lang_tag"]] += 1
        if r["lang_tag"] in ("zh", "zh-en", "en"):
            keep.append(r)
    funnel["lang_zh_zhen_en"] = len(keep)
    by_lang = collections.OrderedDict()
    by_lang["lang_zh_zhen_en"] = dict(collections.Counter(r["lang_tag"] for r in keep))
    funnel["_lang_tag_counts_before_filter"] = dict(lang_counts)

    rows, keep = keep, []
    flag_drop = collections.Counter()
    for r in rows:
        r["flags"] = flags_of(r, 5.0)
        bad = [f for f in r["flags"] if f in DROP_FLAGS or f.startswith("error_")]
        if bad:
            for f in bad:
                flag_drop[f] += 1
            continue
        keep.append(r)
    funnel["quality_flags_ok"] = len(keep)
    by_lang["quality_flags_ok"] = dict(collections.Counter(r["lang_tag"] for r in keep))
    funnel["_flag_drops"] = dict(flag_drop)

    # classify before the pp_shrink check (needs the op)
    for r in keep:
        r["_cls"] = classify(r["transcription"], r["processed"], r)
    rows, keep = keep, []
    for r in rows:
        if "pp_shrink" in r["flags"]:
            c = r["_cls"]
            ok = c["op"] in ("filler_only", "spoken_punctuation", "repeat_merge") or \
                c["labels"].get("self_correction", {}).get("subtype") in ("explicit", "implicit_restatement")
            if not ok:
                drops["pp_shrink_unverified"] += 1
                continue
        keep.append(r)
    funnel["pp_shrink_checked"] = len(keep)
    by_lang["pp_shrink_checked"] = dict(collections.Counter(r["lang_tag"] for r in keep))

    # dedupe on normalised input (keep earliest)
    keep.sort(key=lambda r: r["timestamp"])
    seen, rows, keep = set(), keep, []
    for r in rows:
        k = norm_key(r["transcription"]) or r["transcription"]
        if k in seen:
            drops["duplicate_input"] += 1
            continue
        seen.add(k)
        keep.append(r)
    funnel["deduped"] = len(keep)
    pool = keep
    by_lang["deduped"] = dict(collections.Counter(r["lang_tag"] for r in pool))
    by_lang["deduped_users"] = {l: len({r["user_id"] for r in pool if r["lang_tag"] == l}) for l in ("zh", "zh-en", "en")}
    by_lang["deduped_changed"] = dict(collections.Counter(r["lang_tag"] for r in pool
                                                          if r["transcription"] != r["processed"]))
    funnel["_by_lang"] = by_lang

    # ---- tone / final-period statistics over the whole filtered pool (before per-user caps) ----
    tone_stats = tone_period_stats(pool)

    # ---- assign tone category ----
    for r in pool:
        tk = tone_kind(r)
        r["_tone_kind"] = tk
        if tk not in ("default", "missing"):
            c = r["_cls"]
            ev = {"subtype": tk, "tone_kind": tk, "final_in": c["final_punct_in"], "final_out": c["final_punct_out"]}
            ev["period_action"] = period_action(c["final_punct_in"], c["final_punct_out"])
            if tk == "custom":
                tp = (r.get("tone_prompt") or "").lower()
                ev["custom_mentions_punctuation"] = bool(re.search(r"punctuat|period|full stop|句号|标点", tp))
            c["labels"]["tone_punctuation"] = ev

    # ---- per-category, per-user cap ----
    by_cat = {c: [] for c in CATS}
    for r in pool:
        for c in r["_cls"]["labels"]:
            by_cat[c].append(r)
    capped = {}
    for c, rs in by_cat.items():
        cap = a.control_per_user if c == "control" else a.per_user
        per_user = collections.defaultdict(list)
        for r in rs:
            per_user[r["user_id"]].append(r)
        out = []
        for u, urs in per_user.items():
            rnd.shuffle(urs)
            low = lambda r: r["_cls"]["labels"][c].get("subtype", "").startswith("negative") or \
                (c == "tone_punctuation" and r["_cls"]["labels"][c].get("subtype") == "off")
            neg = [r for r in urs if low(r)]
            pos = [r for r in urs if not low(r)]
            out.extend(pos[:cap] + neg[:a.negative_per_user])
        if c == "control":
            out = stratified_control(out, a.control_total, rnd)
        out.sort(key=lambda r: r["timestamp"])
        capped[c] = out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c, rs in capped.items():
        with open(OUT_DIR / f"{c}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(export_row(r, c), ensure_ascii=False) + "\n")

    stats = {"window": {"since": a.since, "until": a.until}, "funnel": funnel, "drops": dict(drops),
             "params": {"per_user": a.per_user, "control_per_user": a.control_per_user,
                        "negative_per_user": a.negative_per_user,
                        "control_total": a.control_total, "seed": a.seed},
             "pre_cap": {c: summarize(rs) for c, rs in by_cat.items()},
             "post_cap": {c: summarize(rs) for c, rs in capped.items()},
             "tone": tone_stats,
             "apps": top_apps(pool)}
    with open(OUT_DIR / "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    # spot-check sample: per category (and sub-type for the two self_correction kinds), seeded
    with open(OUT_DIR / "_spotcheck_sample.jsonl", "w", encoding="utf-8") as f:
        srnd = random.Random(a.seed + 1)
        for c, rs in capped.items():
            groups = collections.defaultdict(list)
            for r in rs:
                groups[r["_cls"]["labels"][c].get("subtype") if c in ("self_correction", "format_email_list",
                                                                       "grammar_coherence", "tone_punctuation")
                       else "all"].append(r)
            for g, grs in sorted(groups.items()):
                n = a.spot_n if c != "grammar_coherence" else max(3, a.spot_n // 3)
                for r in srnd.sample(grs, min(n, len(grs))):
                    f.write(json.dumps({"category": c, "group": g, **export_row(r, c)}, ensure_ascii=False) + "\n")
    print(json.dumps({"funnel": funnel, "drops": drops,
                      "post_cap": {c: len(rs) for c, rs in capped.items()}}, ensure_ascii=False, indent=1))


def period_action(fin: str, fout: str) -> str:
    if fin == "period" and fout != "period":
        return "removed_period"
    if fin != "period" and fout == "period":
        return "added_period"
    return "kept_period" if fout == "period" else "no_period_kept"


def tone_period_stats(pool):
    """Final-period behaviour by tone signal, app_type, language and prompt (all filtered rows)."""
    def agg(key):
        d = collections.defaultdict(collections.Counter)
        for r in pool:
            c = r["_cls"]
            k = key(r)
            d[k]["n"] += 1
            d[k]["in_period"] += c["final_punct_in"] == "period"
            d[k]["out_period"] += c["final_punct_out"] == "period"
            d[k][period_action(c["final_punct_in"], c["final_punct_out"])] += 1
            d[k]["final_out_" + c["final_punct_out"]] += 1
        return {str(k): dict(v) for k, v in sorted(d.items(), key=lambda x: -x[1]["n"])}
    return {"by_tone_kind": agg(lambda r: tone_kind(r)),
            "by_tone_id": agg(lambda r: r.get("tone_id")),
            "by_tone_kind_lang": agg(lambda r: f"{tone_kind(r)}|{r['lang_tag']}"),
            "by_app_type_lang": agg(lambda r: f"{r.get('app_type')}|{r['lang_tag']}"),
            "by_prompt": agg(lambda r: r.get("postprocess_langfuse_prompt_name")),
            "by_tone_source": agg(lambda r: r.get("tone_source")),
            "by_tone_kind_app_type": agg(lambda r: f"{tone_kind(r)}|{r.get('app_type')}")}


def top_apps(pool, n=40):
    c = collections.Counter((r.get("app_bundle_id") or r.get("tone_app_id") or "(none)", r.get("app_type"))
                            for r in pool)
    by_lang = collections.defaultdict(collections.Counter)
    for r in pool:
        by_lang[r["lang_tag"]][r.get("app_bundle_id") or r.get("tone_app_id") or "(none)"] += 1
    return {"overall": [[b, t, k] for (b, t), k in c.most_common(n)],
            "by_lang": {l: cc.most_common(20) for l, cc in by_lang.items()},
            "bundle_missing": sum(1 for r in pool if not (r.get("app_bundle_id") or r.get("tone_app_id"))),
            "app_type_missing": sum(1 for r in pool if not r.get("app_type")),
            "tone_id_missing": sum(1 for r in pool if r.get("tone_id") is None)}


def stratified_control(rows, total, rnd):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["lang_tag"], r["_cls"]["op"])].append(r)
    keys = sorted(by)
    quota = max(1, total // max(1, len(keys)))
    out = []
    for k in keys:
        rnd.shuffle(by[k])
        out.extend(by[k][:quota])
    return out


def summarize(rs):
    s = {"n": len(rs), "users": len({r["user_id"] for r in rs})}
    s["by_lang"] = dict(collections.Counter(r["lang_tag"] for r in rs))
    s["by_lang_prompt"] = dict(collections.Counter(
        f"{r['lang_tag']}|{r.get('postprocess_langfuse_prompt_name')}" for r in rs))
    sub = collections.Counter()
    for r in rs:
        for c, ev in r["_cls"]["labels"].items():
            if isinstance(ev, dict) and ev.get("subtype"):
                sub[f"{c}.{ev['subtype']}|{r['lang_tag']}"] += 1
    s["subtypes_by_lang"] = dict(sub)
    return s


def export_row(r, cat):
    c = r["_cls"]
    pv = None
    if (r.get("timestamp") or "") >= "2026-10-29" and r.get("postprocess_langfuse_prompt_name") == "transcription-fully-cleanup":
        pv = "v32 (inferred: Langfuse production label after 10-28; not logged per request)"
    return {
        "id": r.get("audio_id"), "langfuse_trace_id": r.get("langfuse_trace_id"),
        "user": hash_user(r.get("user_id")), "timestamp": r.get("timestamp"),
        "lang_tag": r["lang_tag"], "languages_setting": r.get("languages"),
        "platform": r.get("client_source"), "app_bundle_id": r.get("app_bundle_id"), "app_name": r.get("app_name"),
        "app_type": r.get("app_type"), "field_kind": r.get("field_kind"), "mode": r.get("mode"),
        "asr_model": r.get("model"), "multi_modality": r.get("multi_modality"),
        "teacher_model": r.get("postprocess_model"),
        "prompt_name": r.get("postprocess_langfuse_prompt_name"), "prompt_label": r.get("postprocess_langfuse_prompt_label"),
        "prompt_version": pv,
        "tone_id": r.get("tone_id"), "tone_source": r.get("tone_source"), "tone_app_id": r.get("tone_app_id"),
        "tone_kind": r.get("_tone_kind"), "tone_prompt": r.get("tone_prompt"),
        "cleanup_format": r.get("cleanup_format"), "cleanup_format_source": r.get("cleanup_format_source"),
        "transcription": r["transcription"], "processed": r["processed"],
        "categories": sorted(c["labels"]), "category": cat, "evidence": c["labels"].get(cat),
        "all_evidence": c["labels"], "op": c["op"], "edit_ratio": c["edit_ratio"],
        "final_punct_in": c["final_punct_in"], "final_punct_out": c["final_punct_out"],
        "flags": r.get("flags"), "teacher_suspect": c.get("teacher_suspect"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--refresh", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--since", required=True)
    b.add_argument("--until", required=True)
    b.add_argument("--per-user", type=int, default=20)
    b.add_argument("--control-per-user", type=int, default=3)
    b.add_argument("--negative-per-user", type=int, default=5,
                   help="cap for negative_* sub-types and tone 'off' rows (low priority)")
    b.add_argument("--control-total", type=int, default=1200)
    b.add_argument("--spot-n", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"pull": cmd_pull, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
