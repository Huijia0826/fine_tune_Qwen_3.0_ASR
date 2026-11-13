"""benchmark v2 candidates (owner, 2026-11-04): a larger locked test set of short production dictation clips.

Why: benchmark v1 (412 clips, 58% of its token weight in 69 clips > 30 s) cannot resolve the 0.1–0.3 point steps between
fine-tunes. Planned: ~1,000 reviewed clips of 1–30 s, zh 40% / zh-en 30% / en 30%, speakers unseen in training. Production
has too little Chinese for that (30 days: ~130 zh / zh-en users, most already in our data), so the owner chose
(2026-11-04): take ALL eligible zh / zh-en clips (≤ 5 per user; unseen users first, then users seen in training or the
pool, tagged seen=True; never users of a val / test set), and trim en to 240 unseen users (1 clip each).
→ ≈ 206 zh + 154 zh-en + 240 en ≈ 600 candidates, ≈ 520 after review drops.

Input: online.py scans (data/private/online_scan/bv2_days/*.jsonl, dictate, 2026-10-05 → 11-04).
Filters, in order (counted in the funnel): error / no audio / onboarding app; test recordings and tester accounts
(scripts/test_utterances.py); outside 1–30 s; audio already in our datasets; user of a val / test set; en from users seen in
training or the pool; language class (from the production ASR text) not zh / zh-en / en; duplicate text (incl. our data).
Language class: Chinese characters and no kana → "zh-en" if it has a Latin word of ≥ 2 letters, else "zh"; no Chinese →
"en" if ≥ 15% of its words are English function words and it has ≥ 3 words, else dropped (pt / es / … traffic).
Sampling per class: seeded shuffle, ≤ PER_USER clips per user, one clip per user first (as many speakers as possible), then
fill; OVERSAMPLE × target is drawn so the audio-quality filter (applied after download) can trim back to target.
Output: data/private/benchmark_v2/candidates.jsonl + summary.md.
Usage: python3 scripts/select_benchmark_v2.py
"""

import collections
import glob
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location("test_utterances", Path(__file__).with_name("test_utterances.py"))
test_utterances = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_utterances)

TARGET = {"zh": None, "zh-en": None, "en": 240}  # None = every eligible clip
OVERSAMPLE = 1.3  # en only: drawn × 1.3, trimmed back to 240 after the audio-quality check
PER_USER = {"zh": 5, "zh-en": 5, "en": 1}
SEED = 20261104
OUT = Path("data/private/benchmark_v2")
CJK, KANA = re.compile(r"[一-鿿]"), re.compile(r"[぀-ヿ]")
LAT2 = re.compile(r"[A-Za-z]{2,}")
EN_STOP = set("the and is are you to of it that this for with have was what not i my me we they can be do so on in "
              "just like if but at all your can't don't i'm it's".split())


def pseudo(uid: str) -> str:
    return "u" + hashlib.sha256(uid.encode()).hexdigest()[:12]


def lang_class(t: str) -> str | None:
    if CJK.search(t) and not KANA.search(t):
        return "zh-en" if LAT2.search(t) else "zh"
    if CJK.search(t) or KANA.search(t):
        return None
    words = re.findall(r"[a-z']+", t.lower())
    if len(words) >= 3 and sum(w in EN_STOP for w in words) / len(words) >= 0.15:
        return "en"
    return None


def norm_text(t: str) -> str:
    return re.sub(r"[\W_]+", "", (t or "").lower())


def known() -> tuple[set[str], set[str], set[str], set[str]]:
    """audio_ids, speakers seen in training / the pool, speakers of a val / test set, and every known text."""
    ids, seen, evals, texts = set(), set(), set(), set()
    for p in glob.glob("data/manifests/*.jsonl"):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            ids.add(re.sub(r"\.\w+$", "", (r.get("source_utterance_id") or "").split("#")[0]))
            (evals if r["split"] in ("val", "test") else seen).add(r["speaker_id"].split("-", 1)[-1])
            texts.add(norm_text(r.get("raw_transcript")))
    for p in ("data/private/online_pool/pool.jsonl", "data/private/online_phase1/candidates.jsonl"):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            ids.add(r["audio_id"])
            (evals if r.get("split") == "val" and "phase1" in p else seen).add(pseudo(r["user_id"]))  # phase-1 "test" became train
            texts.add(norm_text(r.get("transcription")))
    return ids, seen, evals, texts


def main() -> None:
    rows = []
    for p in sorted(glob.glob("data/private/online_scan/bv2_days/*.jsonl")):
        rows += [r for r in map(json.loads, open(p, encoding="utf-8")) if "_scan" not in r]
    ids, seen_spk, eval_spk, known_text = known()
    tests = test_utterances.TestDetector(rows)
    funnel, pool, seen_text = collections.Counter({"scanned": len(rows)}), [], set()
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        t, d = r.get("transcription") or "", (r.get("cleanup_audio_duration_ms") or 0) / 1000
        if r.get("error_stage") or not r.get("s3_key") or r.get("app") == "com.voicecursor.android.type":
            funnel["error / no audio / onboarding"] += 1
        elif tests.why(r):
            funnel["test recording / tester account"] += 1
        elif not 1 <= d <= 30:
            funnel["outside 1–30 s"] += 1
        elif r["audio_id"] in ids:
            funnel["audio already in our datasets"] += 1
        elif pseudo(r["user_id"]) in eval_spk:
            funnel["user of a val / test set"] += 1
        elif (c := lang_class(t)) is None:
            funnel["not zh / zh-en / en"] += 1
        elif c == "en" and pseudo(r["user_id"]) in seen_spk:
            funnel["en: user seen in training / pool"] += 1
        elif norm_text(t) in seen_text or norm_text(t) in known_text:
            funnel["duplicate text (incl. our datasets)"] += 1
        else:
            seen_text.add(norm_text(t))
            r["lang_class"], r["seen"] = c, pseudo(r["user_id"]) in seen_spk
            pool.append(r)
    funnel["eligible"] = len(pool)

    rng, picked = random.Random(SEED), []
    for c, target in TARGET.items():
        by_user = collections.defaultdict(list)
        for r in pool:
            if r["lang_class"] == c:
                by_user[r["user_id"]].append(r)
        users = sorted(by_user)
        rng.shuffle(users)
        users.sort(key=lambda u: by_user[u][0]["seen"])  # unseen users first (stable: keeps the shuffle within groups)
        for u in users:
            rng.shuffle(by_user[u])
        rounds = [[by_user[u][k] for u in users if len(by_user[u]) > k] for k in range(PER_USER[c])]
        take = [r for rnd in rounds for r in rnd]
        if target:
            take = take[:round(target * OVERSAMPLE)]
        for r in take:
            r["round"] = next(k for k, rnd in enumerate(rounds) if r in rnd)
        picked += take
        funnel[f"picked {c}"] = len(take)

    OUT.mkdir(parents=True, exist_ok=True)
    keys = ("audio_id", "user_id", "lang_class", "seen", "round", "timestamp", "client_source", "model", "app", "lang", "s3_key",
            "langfuse_trace_id", "langfuse", "flags", "transcription", "processed", "cleanup_audio_duration_ms", "mode")
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False) + "\n")
    lines = ["# benchmark v2 candidates", "", "| step | clips |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in funnel.items()]
    lines += ["", "| class | eligible clips | eligible users | picked | picked users | seen clips | hours | platform |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in TARGET:
        el = [r for r in pool if r["lang_class"] == c]
        pk = [r for r in picked if r["lang_class"] == c]
        plat = collections.Counter(r.get("client_source") for r in pk).most_common(4)
        lines.append(f"| {c} | {len(el)} | {len({r['user_id'] for r in el})} | {len(pk)} | {len({r['user_id'] for r in pk})} | {sum(r['seen'] for r in pk)} | "
                     f"{sum(r['cleanup_audio_duration_ms'] for r in pk) / 3.6e6:.2f} | {plat} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
