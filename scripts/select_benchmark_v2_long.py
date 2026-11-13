"""benchmark v2, long part (owner, 2026-11-09): 30–180 s production dictation clips, same review page and steps as the
short part (scripts/select_benchmark_v2.py → fetch_benchmark_v2.py → prepare_review_benchmark_v2.py).

Spec (owner-approved defaults): decoded duration 30–180 s; 80 clips after review drops, zh 30 / zh-en 30 / en 20, so
TARGET × OVERSAMPLE (1.3) clips go to review; ONE clip per user; unseen users first, users seen in training / the pool
only if needed (tagged seen=True); never users of a val / test set; no audio of the short part (537 clips) and,
preferably, none of its users (unseen users outside the short part first; overlap tagged short_user=True). en has plenty
of users, so en drops seen users and short-part users outright (as the short part dropped seen en users).

Production metadata durations are unreliable (2026-11-01: a clip with metadata 1.5 s decoded to 64 s), so the duration
rule is applied twice: `select` keeps clips whose metadata says 25–240 s OR whose production text is long enough for
≥ 30 s of speech (≥ 120 Chinese characters, or ≥ 75 English words); `fetch` downloads them and keeps decoded 30–180 s.
Each user gets up to BACKUP clips in a seeded order (metadata-in-range first) so that a user whose first clip is out of
range or unusable still counts; fetch keeps the first usable in-range clip per user, users in rank order.
Two deviations: the "counting" test rule does not drop texts of ≥ 60 characters (lists, not mic checks; see test_kind),
and a text whose Chinese characters are < 20% of (Chinese characters + Latin words) is classed WITHOUT them: production
often prefixes Portuguese / English audio with one hallucinated Chinese word ("对了 O cara é…"), which the short-part rule
calls zh-en (12 of the short part's 134 zh-en clips are such Portuguese / French clips; see lang_class_long).
Every other filter (test recordings, val / test users, our datasets, language class, duplicate text) and the
audio-quality gate are the short part's, imported from its scripts.

Output: data/private/benchmark_v2_long/{candidates.jsonl, audio/*.ogg, items.jsonl, summary.md}.
Usage: uv run --offline python scripts/select_benchmark_v2_long.py select
       NO_PROXY="*" uv run --offline --with boto3 python scripts/select_benchmark_v2_long.py fetch
"""

import collections
import glob
import importlib.util
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dump_online_pool as dop  # noqa: E402  (quality / traits / high_quality / bench helpers)
from fetch_benchmark_v2 import usable  # noqa: E402
from select_benchmark_v2 import CJK, known, lang_class, norm_text, pseudo, test_utterances  # noqa: E402

TARGET = {"zh": 30, "zh-en": 30, "en": 20}  # after review drops
OVERSAMPLE = 1.3  # review gets round(TARGET × 1.3) clips per class
DUR = (30.0, 180.0)  # decoded duration, s
META = (25.0, 240.0)  # metadata window for `select` (margin around DUR)
LONG_TEXT = {"zh_chars": 120, "en_words": 75}  # or: production text long enough for ≥ 30 s of speech
BACKUP = 3  # clips downloaded per user, tried in order
COUNTING_MIN_CHARS = 60  # texts at least this long are not dropped by the "counting" rule alone
MIN_ZH_SHARE = 0.2  # zh / zh-en need Chinese characters ≥ 20% of (Chinese characters + Latin words)
EN_USERS = 2.0  # en: draw 2 × the review count of users (en has thousands; the rest are not downloaded)
SEED = 20261109
OUT = Path("data/private/benchmark_v2_long")
SHORT = Path("data/private/benchmark_v2")
KEYS = ("audio_id", "user_id", "user_hash", "lang_class", "seen", "short_user", "user_rank", "clip_rank", "timestamp",
        "client_source", "model", "app", "lang", "s3_key", "langfuse_trace_id", "langfuse", "flags", "transcription",
        "processed", "cleanup_audio_duration_ms", "mode")


def maybe_long(r: dict) -> bool:
    d, t = (r.get("cleanup_audio_duration_ms") or 0) / 1000, r.get("transcription") or ""
    return (META[0] <= d <= META[1] or len(CJK.findall(t)) >= LONG_TEXT["zh_chars"]
            or len(re.findall(r"[A-Za-z']+", t)) >= LONG_TEXT["en_words"])


def test_kind(tests, r: dict) -> str | None:
    """The short part's test detector, except "counting" (an ascending run of 4 numbers) on long texts: in a 30 s+
    dictation that is a list ("arm one, two, three, four", "1. … 2. … 3. … 4.") far more often than a mic check."""
    k = tests.why(r)
    if k == "counting" and len(norm_text(r.get("transcription"))) >= COUNTING_MIN_CHARS:
        return "tester_account" if r["user_id"] in tests.testers else None
    return k


def lang_class_long(t: str) -> str | None:
    c = lang_class(t)
    n, w = len(CJK.findall(t)), len(re.findall(r"[A-Za-z][A-Za-z']*", t))
    if c in ("zh", "zh-en") and n / (n + w) < MIN_ZH_SHARE:
        return lang_class(CJK.sub("", t))  # en, or None (pt / es / …)
    return c


def in_range(r: dict) -> bool:
    return DUR[0] <= (r.get("cleanup_audio_duration_ms") or 0) / 1000 <= DUR[1]


def short_part() -> tuple[set[str], set[str], set[str]]:
    """audio_ids (every downloaded candidate), user_ids (final items) and texts of the short part."""
    cands = [json.loads(l) for l in open(SHORT / "candidates.jsonl", encoding="utf-8")]
    items = [json.loads(l) for l in open(SHORT / "items.jsonl", encoding="utf-8")]
    return ({r["audio_id"] for r in cands}, {r["user_id"] for r in items},
            {norm_text(r.get("transcription")) for r in cands})


def select() -> None:
    rows = []
    for p in sorted(glob.glob("data/private/online_scan/bv2_days/*.jsonl")):
        rows += [r for r in map(json.loads, open(p, encoding="utf-8")) if "_scan" not in r]
    ids, seen_spk, eval_spk, known_text = known()
    short_ids, short_users, short_text = short_part()
    ids |= short_ids
    known_text |= short_text
    tests = test_utterances.TestDetector(rows)  # built on ALL rows, as in the short part (multi-user / tester rules)
    funnel, pool, seen_text = collections.Counter({"scanned": len(rows)}), [], set()
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        t, u = r.get("transcription") or "", pseudo(r["user_id"])
        if r.get("error_stage") or not r.get("s3_key") or r.get("app") == "com.voicecursor.android.type":
            funnel["error / no audio / onboarding"] += 1
        elif test_kind(tests, r):
            funnel["test recording / tester account"] += 1
        elif not maybe_long(r):
            funnel["not long (metadata 25–240 s or long text)"] += 1
        elif r["audio_id"] in ids:
            funnel["audio already in our datasets / v2 short"] += 1
        elif u in eval_spk:
            funnel["user of a val / test set"] += 1
        elif (c := lang_class_long(t)) is None:
            funnel["not zh / zh-en / en"] += 1
        elif c == "en" and u in seen_spk:
            funnel["en: user seen in training / pool"] += 1
        elif c == "en" and r["user_id"] in short_users:
            funnel["en: user in v2 short"] += 1
        elif norm_text(t) in seen_text or norm_text(t) in known_text:
            funnel["duplicate text (incl. our datasets)"] += 1
        else:
            seen_text.add(norm_text(t))
            funnel["(kept, reclassified: Chinese share < 20%)"] += c != lang_class(t)
            r.update(lang_class=c, seen=u in seen_spk, short_user=r["user_id"] in short_users, user_hash=u)
            pool.append(r)
    funnel["eligible (metadata)"] = len(pool)

    rng, picked, stats = random.Random(SEED), [], {}
    for c, target in TARGET.items():
        by_user = collections.defaultdict(list)
        for r in pool:
            if r["lang_class"] == c:
                by_user[r["user_id"]].append(r)
        users = sorted(by_user)
        rng.shuffle(users)
        users.sort(key=lambda u: (by_user[u][0]["seen"], by_user[u][0]["short_user"]))  # unseen first, then not in short
        if c == "en":
            users = users[:round(target * OVERSAMPLE * EN_USERS)]
        for ur, u in enumerate(users):
            rs = by_user[u]
            rng.shuffle(rs)
            rs.sort(key=lambda r: not in_range(r))  # metadata already in 30–180 s first
            for cr, r in enumerate(rs[:BACKUP]):
                picked.append({**r, "user_rank": ur, "clip_rank": cr})
        stats[c] = (sum(map(len, by_user.values())), len(by_user), sum(by_user[u][0]["seen"] for u in by_user),
                    sum(by_user[u][0]["short_user"] for u in by_user),
                    sum(any(in_range(r) for r in by_user[u]) for u in by_user))
        funnel[f"to download {c} (≤ {BACKUP} per user)"] = sum(r["lang_class"] == c for r in picked)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({k: r.get(k) for k in KEYS}, ensure_ascii=False) + "\n")
    lines = ["## select (metadata)", "", "| step | clips |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in funnel.items()]
    lines += ["", "| class | eligible clips | eligible users | seen users | short-part users | users with metadata 30–180 s |",
              "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {c} | {' | '.join(map(str, s))} |" for c, s in stats.items()]
    (OUT / "select.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def fetch() -> None:
    spec = importlib.util.spec_from_file_location("bench", dop.BENCH_PY)
    bench = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(dop.BENCH_PY.parent))
    spec.loader.exec_module(bench)
    cands = [json.loads(l) for l in open(OUT / "candidates.jsonl", encoding="utf-8")]
    adir = OUT / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in cands if not (adir / f"{r['audio_id']}.ogg").exists()]
    if todo:
        bench.load_env(())
        s3 = bench.s3_client()
        print(f"downloading {len(todo)} of {len(cands)} clips from S3 …", flush=True)

        def get(r):
            try:
                bench.s3_download(r["s3_key"], str(adir / f"{r['audio_id']}.ogg"), s3)
            except Exception as e:  # keep going; the clip counts as missing
                print(f"  {r['audio_id']}: {type(e).__name__}", flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, todo))
    have = [r for r in cands if (adir / f"{r['audio_id']}.ogg").exists()]
    with ThreadPoolExecutor(8) as ex:
        qs = list(ex.map(lambda r: dop.quality(adir / f"{r['audio_id']}.ogg"), have))
    funnel = collections.Counter({"candidates (clips)": len(cands), "downloaded": len(have)})
    ok = collections.defaultdict(list)  # (class, user_rank) → usable in-range clips, in clip_rank order
    for r, q in sorted(zip(have, qs), key=lambda x: (x[0]["lang_class"], x[0]["user_rank"], x[0]["clip_rank"])):
        dur = q.get("duration_s", 0)
        if not DUR[0] <= dur <= DUR[1]:
            funnel[f"decoded duration outside 30–180 s ({r['lang_class']})"] += 1
            continue
        if not usable(q):
            funnel[f"unusable audio ({r['lang_class']})"] += 1
            continue
        # training high_quality rule without its ≤ 30 s cap, recorded for slicing only
        ok[r["lang_class"], r["user_rank"]].append({**r, "quality": q, "high_quality": dop.high_quality({**q, "duration_s": 30}),
                                                    "traits": dop.traits(r, dur)})
    items, avail, used = [], {}, set()
    for c, target in TARGET.items():  # zh, zh-en, en: the scarcest class keeps a user who has clips in two classes
        firsts = [ok[k][0] for k in sorted(k for k in ok if k[0] == c)]  # one clip per user, users in rank order
        funnel[f"user already picked in another class ({c})"] = sum(r["user_id"] in used for r in firsts)
        firsts = [r for r in firsts if r["user_id"] not in used]
        funnel[f"usable 30–180 s clips incl. 2nd / 3rd per user ({c})"] = sum(
            len(v) for k, v in ok.items() if k[0] == c and v[0]["user_id"] not in used)
        avail[c] = firsts
        items += firsts[:round(target * OVERSAMPLE)]
        used |= {r["user_id"] for r in firsts}
        funnel[f"users with a usable 30–180 s clip ({c})"] = len(firsts)
        funnel[f"picked {c}"] = min(len(firsts), round(target * OVERSAMPLE))
    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    h = lambda R: sum(r["quality"]["duration_s"] for r in R) / 3600  # noqa: E731
    lines = ["# benchmark v2 long (30–180 s) candidates", "", (OUT / "select.md").read_text(encoding="utf-8"),
             "## fetch (decoded audio)", "", "| step | clips |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in funnel.items()]
    lines += ["", "| class | target after review | picked | available users | hours | mean s | users | seen | short-part users |"
              " high_quality | proper_noun | platform |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in [*TARGET, None]:
        sel = [r for r in items if c is None or r["lang_class"] == c]
        plat = collections.Counter(r.get("client_source") for r in sel).most_common()
        lines.append(f"| {c or 'total'} | {TARGET[c] if c else sum(TARGET.values())} | {len(sel)} | "
                     f"{len(avail[c]) if c else sum(map(len, avail.values()))} | {h(sel):.2f} | "
                     f"{h(sel) * 3600 / max(1, len(sel)):.0f} | {len({r['user_id'] for r in sel})} | {sum(r['seen'] for r in sel)} | "
                     f"{sum(r['short_user'] for r in sel)} | {sum(r['high_quality'] for r in sel)} | "
                     f"{sum('proper_noun' in r['traits'] for r in sel)} | {plat} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"select": select, "fetch": fetch}[sys.argv[1]]()
