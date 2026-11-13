"""Assign data-collection scenarios (docs/data_collection_plan.md) to an online.py scan and count capacity.

Rules use the production ASR text (`transcription`); they are coarse and only size the pool, the reviewer confirms.
Usage: python3 scripts/scan_scenarios.py data/private/online_scan/<scan>.jsonl [--exclude-users users.txt]
"""

import argparse
import collections
import json
import re

LAT = re.compile(r"[A-Za-z][A-Za-z'.+-]*")
CJK = re.compile(r"[一-鿿]")
KANA = re.compile(r"[぀-ヿ]")  # Japanese kana: kanji-only CJK matching would count Japanese as Chinese
SELF_CORR = re.compile(r"不对|不是不是|我是说|改成|换行|第一[个点]|sorry|I mean|actually", re.I)
TERM = re.compile(r"^(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z]\w*|\w*\d\w*)$")  # acronym, CamelCase, or contains a digit


def scenario(r: dict) -> str:
    t = r.get("transcription") or ""
    d = (r.get("cleanup_audio_duration_ms") or 0) / 1000
    if not t.strip():
        return "empty"
    if not CJK.search(t) or KANA.search(t):
        return "no_cjk"
    if 0 < d <= 3:
        return "S4_short"
    if SELF_CORR.search(t) or "pp_shrink" in r.get("flags", []):
        return "S5_selfcorr"
    words = LAT.findall(t)
    if words:
        return "S3_terms" if any(TERM.match(w) for w in words) else "S1_zh_en"
    return "S2_zh"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--exclude-users", help="file with one user_id per line (e.g. users already in our datasets)")
    args = ap.parse_args()
    skip = set(open(args.exclude_users).read().split()) if args.exclude_users else set()
    rows = [json.loads(l) for l in open(args.scan, encoding="utf-8")][1:]
    keep = [r for r in rows if not r.get("error_stage") and r.get("s3_key") and r["user_id"] not in skip
            and r.get("app") != "com.voicecursor.android.type"]
    by = collections.defaultdict(list)
    for r in keep:
        by[scenario(r)].append(r)
    print(f"{len(rows)} requests, {len({r['user_id'] for r in rows})} users; kept {len(keep)}")
    print(f"{'scenario':<12}{'clips':>7}{'users':>7}{'≤3/user':>9}{'hours':>7}")
    for s in sorted(by):
        per_user = collections.Counter(r["user_id"] for r in by[s])
        hours = sum((r.get("cleanup_audio_duration_ms") or 0) for r in by[s]) / 3.6e6
        print(f"{s:<12}{len(by[s]):>7}{len(per_user):>7}{sum(min(3, n) for n in per_user.values()):>9}{hours:>7.2f}")


if __name__ == "__main__":
    main()
