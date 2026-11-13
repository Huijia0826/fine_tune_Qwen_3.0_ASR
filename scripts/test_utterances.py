"""Detect test / mic-check recordings in production traffic (owner, 2026-11-01: "一二三四…, many are test data").

Three kinds:
  counting     an ascending run of ≥ 4 numbers (一二三四, 1234, 1 2 3 4, one two three four) or a numbers-only utterance
  test_script  mic checks and QA scripts: "testing, testing", "hello hello", "喂喂", "测试测试", "能听到吗",
               "这是一段…测试", "validation number …", and short (≤ 25 chars) first tries: "开始测试", "试一下这个"
  multi_user   the same short text (8–30 chars) spoken by ≥ 2 different users: read-aloud scripts. Longer shared
               texts are duplicates (one person, two accounts), handled by text dedup, not treated as tests
Tester accounts: users with ≥ 5 test-like requests making up ≥ 50% of their traffic; all their requests are excluded.
Usage: python3 scripts/test_utterances.py <scan.jsonl>   (prints counts and examples)
"""

from __future__ import annotations

import collections
import json
import re
import sys

ZH_NUM = {c: i for i, c in enumerate("零一二三四五六七八九十", 0)}
EN_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
SCRIPT = re.compile(r"(?:testing\W*){2,}|^\W*test(?:ing)?\W*$|^\W*(?:hello\W*){2,}$|^\W*(?:喂\W*){2,}|测试测试|"
                    r"^\W*测试\W*(?:一下)?\W*$|能听到(?:我)?(?:说话)?吗|听得到吗|麦克风测试|这是一[段条句].{0,12}测试|"
                    r"语音光标的.{0,4}测试|validation number", re.I)
TRIAL = re.compile(r"测试|试一下|试试|测一下|试用|test", re.I)  # only for short utterances (first tries)
ONLY_NUMBERS = re.compile(r"^[\W\d零一二三四五六七八九十]+$")


def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", (t or "").lower())


def ascending_run(t: str) -> int:
    """Longest run of numbers increasing by exactly 1 (Chinese numerals, digits, or English number words)."""
    seqs = [[ZH_NUM[c] for c in run] for run in re.findall(r"[零一二三四五六七八九十]+", t)]  # 一二三四 written together
    seqs += [[int(c) for c in tok] for tok in re.findall(r"\d{4,}", t)]                    # 1234 inside one number
    seqs.append([int(x) for x in re.findall(r"(?<![\d:.])\d(?![\d:.])", t)])              # 1 2 3 4 said one by one
    seqs.append([EN_NUM[w] for w in re.findall(r"[a-z]+", t.lower()) if w in EN_NUM])
    best = 0
    for s in seqs:
        run = 1
        for a, b in zip(s, s[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
    return best


class TestDetector:
    def __init__(self, rows: list[dict]):
        users = collections.defaultdict(set)
        for r in rows:
            n = norm(r.get("transcription"))
            if 8 <= len(n) <= 30:
                users[n].add(r["user_id"])
        self.multi = {n for n, u in users.items() if len(u) >= 2}
        per = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            per[r["user_id"]][0] += 1
            per[r["user_id"]][1] += bool(self.kind(r))
        self.testers = {u for u, (n, k) in per.items() if k >= 5 and k / n >= 0.5}

    def kind(self, r: dict) -> str | None:
        t = (r.get("transcription") or "").strip()
        if ascending_run(t) >= 4 or (t and ONLY_NUMBERS.match(t)):
            return "counting"
        if SCRIPT.search(t) or (TRIAL.search(t) and len(norm(t)) <= 25):
            return "test_script"
        if norm(t) in self.multi:
            return "multi_user"
        return None

    def why(self, r: dict) -> str | None:
        return self.kind(r) or ("tester_account" if r["user_id"] in self.testers else None)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][1:]
    det = TestDetector(rows)
    c = collections.Counter(det.why(r) or "kept" for r in rows)
    print(dict(c), "| tester accounts:", len(det.testers))
    for k in ("counting", "test_script", "multi_user", "tester_account"):
        ex = [r["transcription"] or "" for r in rows if det.why(r) == k]
        print(f"\n== {k}: longest examples (false-positive check)")
        for t in sorted(set(ex), key=len, reverse=True)[:5]:
            print("  ", t[:90].replace("\n", " "))
