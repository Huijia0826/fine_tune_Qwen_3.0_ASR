"""Shared helpers for the en-setting vs Chinese-speech probe (2026-11-09).

Population: production dictate requests (PostHog asr_metrics via online.py day scans), 30-day window
2026-10-10 -> 2026-11-08 (UTC, inclusive). Day scans come from data/private/online_scan/bv2_days (10-10..11-03)
and data/private/lid_probe/scan_days (11-04..11-08, same online.py scan --mode dictate).

Exclusions: onboarding (app com.voicecursor.android.type or invocation_context=onboarding_practice), test recordings
and tester accounts (scripts/test_utterances.py), arena replays (already dropped by online.py).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from test_utterances import TestDetector  # noqa: E402

OUT = ROOT / "data/private/lid_probe"
SINCE, UNTIL = dt.date(2026, 10, 10), dt.date(2026, 11, 9)  # [since, until)
DAY_DIRS = [ROOT / "data/private/online_scan/bv2_days", OUT / "scan_days"]
ONBOARDING_APP = "com.voicecursor.android.type"

HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힯]")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏ]+(?:'[A-Za-z]+)?")
ZH_BASES = {"zh", "yue", "cmn"}


def day_files():
    d = SINCE
    while d < UNTIL:
        for base in DAY_DIRS:
            p = base / f"{d}.jsonl"
            if p.exists():
                yield p
                break
        else:
            raise SystemExit(f"missing day scan for {d}")
        d += dt.timedelta(days=1)


def load_rows() -> list[dict]:
    rows = []
    for p in day_files():
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "_scan" in r or r.get("mode") != "dictate":
                    continue
                rows.append(r)
    return rows


def base(code: str) -> str:
    return (code or "").replace("_", "-").split("-")[0].lower()


def setting_class(langs) -> str:
    """User language setting (PostHog `languages`, the list the client sends) -> coarse class."""
    if isinstance(langs, str):
        try:
            langs = json.loads(langs)
        except ValueError:
            langs = [langs]
    if not langs:
        return "none"
    b = {base(x) for x in langs}
    if "auto" in b:
        return "auto"
    zh = bool(b & ZH_BASES)
    if b == {"en"}:
        return "en_only"
    if zh and b - ZH_BASES == set():
        return "zh_only"
    if zh and b - ZH_BASES == {"en"}:
        return "zh_en"
    if zh:
        return "zh_plus_other"
    if "en" in b:
        return "en_plus_other"
    return "other"


SETTING_ORDER = ["en_only", "zh_only", "zh_en", "zh_plus_other", "en_plus_other", "auto", "other", "none"]
SETTING_LABEL = {"en_only": "仅英语", "zh_only": "仅中文", "zh_en": "中文+英语", "zh_plus_other": "含中文+其他",
                 "en_plus_other": "英语+其他（无中文）", "auto": "auto", "other": "其他（无中英）", "none": "空 / 无"}


def text_stats(t: str) -> dict:
    t = t or ""
    han, kana, hangul = len(HAN.findall(t)), len(KANA.findall(t)), len(HANGUL.findall(t))
    words = len(LATIN_WORD.findall(t))
    return {"han": han, "kana": kana, "hangul": hangul, "latin_words": words}


def text_class(t: str) -> str:
    """Script mix of a transcript: zh / zh-en / en(latin) / ja / ko / empty / other.
    zh-en: Han present and >= 2 Latin words with Han share of (Han chars + Latin words) between 15% and 90%."""
    s = text_stats(t)
    if s["kana"] >= 2:
        return "ja"
    if s["hangul"] >= 2:
        return "ko"
    tot = s["han"] + s["latin_words"]
    if tot == 0:
        return "empty" if not (t or "").strip() else "other"
    share = s["han"] / tot
    if s["han"] >= 2 and s["latin_words"] >= 2 and 0.15 <= share <= 0.9:
        return "zh-en"
    if s["han"] >= 2 and share > 0.5:
        return "zh"
    if s["han"] >= 2:
        return "zh-en"
    return "latin"


def exclusion(rows: list[dict]) -> dict:
    """audio_id -> reason for every excluded row (onboarding / test kinds / tester account)."""
    onboarding = set()
    p = OUT / "onboarding_ids.txt"
    if p.exists():
        onboarding = set(p.read_text().split())
    det = TestDetector([r for r in rows if r.get("user_id")])
    why = {}
    for r in rows:
        if r.get("app") == ONBOARDING_APP or r["audio_id"] in onboarding:
            why[r["audio_id"]] = "onboarding"
        elif r.get("user_id"):
            w = det.why(r)
            if w:
                why[r["audio_id"]] = w
    return why


def duration_s(r: dict) -> float:
    try:
        return float(r.get("cleanup_audio_duration_ms") or 0) / 1000
    except (TypeError, ValueError):
        return 0.0
