#!/usr/bin/env python3
"""Pull every context / vocabulary field VoiceCursor logged for the clips already in our datasets (2026-11-09).

Why: the next model (Qwen3-ASR-1.7B, end-to-end contextual ASR) takes audio + context (dictionary / hotwords, text before
and after the cursor, screen text, app -> tone, mode, language setting). Before designing the input format we need to
know, per clip we already have, which of these fields production actually recorded.

Read-only: PostHog asr_metrics (HogQL SELECT) and Langfuse v2 observations (GET), through the trace-lookup skill
helpers and ~/.config/vc_bench.env. No LLM calls, no writes to PostHog / Langfuse / arena. Output contains users' text
-> data/private only; the report (reports/context_pull/) is aggregated + anonymised.

Sources and how the production audio_id is found
  benchmark_chinese, online_phase1, online_pool_b1full (superset of online_pool), recording_freeform_test
                         source_utterance_id = audio_id (also the audio file name)
  recording_v1 / v2 / v2_test   source_utterance_id = "<audio_id>#<script id>"; recorded on the TEST environment
                         (scripts/fetch_recordings.py) -> not in production PostHog, looked up in Langfuse by sessionId
  benchmark_v1           arena file_name (= source_utterance_id, a UUID) plus data/private/benchmark_v1/audio/*.json
                         (audio-fetch sidecar: production s3 key and, when found, the production audio_id); joined by
                         audio_id first, then by s3_key
  benchmark_v2           data/private/benchmark_v2/items.jsonl (audio_id; no human label yet -> production ASR text is
                         used as a stand-in and flagged)
  benchmark_v2_long      data/private/benchmark_v2_long/items.jsonl (same as benchmark_v2; added 2026-11-11)
  segmented_long         benchmark_chinese long clips cut into segments: context of the parent clip (arena parts skipped)
  arena_019f8e51         arena file name = production audio_id
  arena_019f6442         production audio_id from data/private/arena_019f6442/dataset_audios.csv (arena benchmark-pool
                         provenance, exported by scripts/arena_dataset_to_ingest.py)
  ASCEND, val_longform   not production -> skipped

Stages
  pull     PostHog by audio_id / s3_key (+ Langfuse sessionId fallback for unmatched ids)
           -> data/private/context_pull/_posthog.json (cache)
  traces   Langfuse trace per matched clip (root input = audio_context as received, gen_ai.transcribe input = what was
           actually sent to the ASR model) -> data/private/context_pull/_langfuse/<audio_id>.json (resumable, ~2 s each)
  textmatch  arena datasets: production audio id by transcript + duration match in PostHog 2026-05-18 -> 07-23
           (-> _posthog_window.jsonl cache, _textmatch.json)
  build    per-source jsonl + bundle_to_app.tsv
  report   reports/context_pull/context_coverage.md (+ before-cursor timing check on the cached 30-day scan)

Usage
  NO_PROXY="*" python3 scripts/context_pull/pull_context.py pull
  NO_PROXY="*" python3 scripts/context_pull/pull_context.py traces
  python3 scripts/context_pull/pull_context.py build
  python3 scripts/context_pull/pull_context.py report
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SKILL = Path.home() / ".claude/skills/trace-lookup/scripts"
sys.path.insert(0, str(SKILL))
import trace_lookup as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "data/manifests"
OUT = ROOT / "data/private/context_pull"
LF_DIR = OUT / "_langfuse"
PH_CACHE = OUT / "_posthog.json"
TONE_TSV = ROOT / "configs/reference/tone_by_app.tsv"
REPORT = ROOT / "reports/context_pull/context_coverage.md"
SCAN_DAYS = ROOT / "data/private/online_scan/bv2_days"
SINCE = "2026-05-01"  # asr_metrics starts 2026-05-18

WORDISH = re.compile(r"[\w一-鿿]")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MANIFEST_SOURCES = ["benchmark_chinese", "online_phase1", "online_pool_b1full", "recording_v1", "recording_v2",
                    "recording_v2_test", "recording_freeform_test", "benchmark_v1"]
SKIPPED = {"ascend*": "public corpus (CC BY-SA), not production", "val_longform": "built from ASCEND"}

# ---------- sources ----------


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_sources() -> dict[str, list[dict]]:
    """source -> [{utterance_id, split, label, label_is_human, ids: [audio_id candidates], s3_key, extra}]"""
    out: dict[str, list[dict]] = {}
    pool_ids = {r["utterance_id"] for r in read_jsonl(MAN / "online_pool.jsonl")}
    bv1_meta = {}
    for p in (ROOT / "data/private/benchmark_v1/audio").glob("*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        bv1_meta[m["input"]] = m
    bv1_keys = {r["file_name"]: r["object_key"]
                for r in csv.DictReader(open(ROOT / "data/private/benchmark_v1/arena_object_keys.csv", encoding="utf-8"))}
    for src in MANIFEST_SOURCES:
        rows = []
        for r in read_jsonl(MAN / f"{src}.jsonl"):
            sid = r["source_utterance_id"]
            d = {"utterance_id": r["utterance_id"], "split": r["split"], "label": r["raw_transcript"],
                 "label_is_human": True, "duration_sec": r.get("duration_sec"), "ids": [], "s3_key": None, "extra": {}}
            if src == "benchmark_v1":
                fname = sid
                d["ids"].append(fname.rsplit(".", 1)[0])
                key = bv1_keys.get(fname)
                m = bv1_meta.get(key) or {}
                if m.get("audio_id") and UUID.fullmatch(m["audio_id"]):
                    d["ids"].insert(0, m["audio_id"])
                if key and key.startswith("s3://"):
                    d["s3_key"] = key.split("/", 3)[3]
                d["extra"]["bv1_origin"] = "production_s3" if key and key.startswith("s3://") else "arena_upload"
            else:
                d["ids"].append(sid.split("#")[0])
            if src == "online_pool_b1full":
                d["extra"]["in_online_pool"] = r["utterance_id"] in pool_ids
            rows.append(d)
        out[src] = rows
    # arena datasets (owner, 2026-11-09: also production audio). 019f6442: production audio_id from the arena benchmark
    # pool provenance (exported by scripts/arena_dataset_to_ingest.py) + the original S3 key's base name; 019f8e51: the
    # arena file name is the production audio_id. Label = arena ground truth; counted as human when reviewed / edited.
    for short in ("019f8e51", "019f6442"):
        d = ROOT / f"data/private/arena_{short}"
        meta = {r["file_name"]: r for r in csv.DictReader(open(d / "dataset_audios.csv", encoding="utf-8"))}
        prov = {r["file_name"]: r["provenance"] for r in csv.DictReader(open(d / "gt_provenance.csv", encoding="utf-8"))}
        rows = []
        for r in read_jsonl(MAN / f"arena_{short}.jsonl"):
            fn = r["source_utterance_id"]
            m = meta.get(fn) or {}
            ids = [i for i in (m.get("audio_id"), fn.rsplit(".", 1)[0]) if i and UUID.fullmatch(i)]
            rows.append({"utterance_id": r["utterance_id"], "split": r["split"], "label": r["raw_transcript"],
                         "label_is_human": prov.get(fn) in ("human_edited", "reviewed_unedited"),
                         "duration_sec": r.get("duration_sec"), "ids": list(dict.fromkeys(ids)), "s3_key": None,
                         "extra": {"gt_provenance": prov.get(fn), "key_basename": m.get("key_basename"),
                                   "pool_audio_id": m.get("audio_id") if short == "019f6442" else None}})
        out[f"arena_{short}"] = rows
    out["benchmark_v2"] = [{"utterance_id": f"benchmark_v2-{r['audio_id']}", "split": "test_candidate",
                            "label": r.get("transcription") or "", "label_is_human": False,
                            "duration_sec": (r.get("cleanup_audio_duration_ms") or 0) / 1000, "ids": [r["audio_id"]],
                            "s3_key": r.get("s3_key"), "extra": {"lang_class": r.get("lang_class")}}
                           for r in read_jsonl(ROOT / "data/private/benchmark_v2/items.jsonl")]
    out["benchmark_v2_long"] = [{"utterance_id": f"benchmark_v2_long-{r['audio_id']}", "split": "test_candidate",
                                 "label": r.get("transcription") or "", "label_is_human": False,
                                 "duration_sec": float(r.get("cleanup_audio_duration_ms") or 0) / 1000,
                                 "ids": [r["audio_id"]], "s3_key": r.get("s3_key"),
                                 "extra": {"lang_class": r.get("lang_class")}}
                                for r in read_jsonl(ROOT / "data/private/benchmark_v2_long/items.jsonl")]
    # segmented_long: map benchmark_chinese segments to the parent's audio id; arena parts are skipped
    seg = []
    for r in read_jsonl(MAN / "segmented_long.jsonl"):
        if r["source_id"] != "benchmark_chinese":
            continue
        aid, _, part = r["source_utterance_id"].partition("#")
        seg.append({"utterance_id": r["utterance_id"], "split": r["split"], "label": r["raw_transcript"],
                    "label_is_human": True, "duration_sec": r.get("duration_sec"), "ids": [aid], "s3_key": None,
                    "extra": {"parent_utterance_id": r.get("parent_utterance_id"), "segment": part}})
    out["segmented_long"] = seg
    return out


# ---------- pull (PostHog + Langfuse sessionId fallback) ----------


def posthog_by(field: str, values: list[str], since: str = None, until: str = "2100-01-01",
               batch: int = 150) -> dict[str, list[dict]]:
    """asr_metrics rows (all client_source, incl. arena) keyed by properties.<field>. HogQL applies LIMIT 100 when the
    query has none, so the explicit LIMIT matters (a batch of 150 ids can return > 100 events)."""
    res: dict[str, list[dict]] = collections.defaultdict(list)
    for i in range(0, len(values), batch):
        inl = ",".join("'" + v.replace("'", "") + "'" for v in values[i:i + batch])
        for ts, p in q_retry(f"SELECT timestamp, properties FROM events WHERE event='asr_metrics' "
                             f"AND properties.{field} IN ({inl}) AND timestamp >= '{since or SINCE}' "
                             f"AND timestamp < '{until}' ORDER BY timestamp LIMIT 10000"):
            p = tl.props(p)
            p = {k: v for k, v in p.items() if not k.startswith("$")}
            p["_timestamp"] = ts
            res[p.get(field)].append(p)
        print(f"  PostHog {field}: {min(i + batch, len(values))}/{len(values)}", file=sys.stderr)
    return res


def langfuse_sessions(ids: list[str]) -> dict[str, list[str]]:
    out = {}
    for n, i in enumerate(ids):
        obs = tl.lf_observations({"sessionId": i, "fields": "core,basic"})
        out[i] = list(dict.fromkeys(o["traceId"] for o in obs))
        if n % 20 == 0:
            print(f"  Langfuse sessionId: {n + 1}/{len(ids)}", file=sys.stderr)
    return out


def cmd_pull(a) -> None:
    tl.load_env(tl.POSTHOG + tl.LANGFUSE)
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = load_sources()
    ids = sorted({i for rows in srcs.values() for r in rows for i in r["ids"]})
    keys = sorted({r["s3_key"] for rows in srcs.values() for r in rows if r["s3_key"]})
    print(f"{len(ids)} audio ids, {len(keys)} s3 keys")
    by_id = posthog_by("audio_id", ids)
    by_key = posthog_by("s3_key", keys)
    found = set(by_id) | {p["audio_id"] for ps in by_key.values() for p in ps}
    missing = [i for i in ids if i not in found]
    print(f"PostHog matched {len(ids) - len(missing)}/{len(ids)} ids; Langfuse sessionId lookup for {len(missing)}")
    old = json.load(open(PH_CACHE)).get("lf_sessions", {}) if PH_CACHE.exists() else {}
    sessions = {i: old[i] for i in missing if i in old}
    if a.langfuse_fallback:
        sessions.update(langfuse_sessions([i for i in missing if i not in old]))
    json.dump({"by_id": by_id, "by_key": by_key, "lf_sessions": sessions}, open(PH_CACHE, "w"), ensure_ascii=False)
    print(f"-> {PH_CACHE}")


# ---------- textmatch (arena datasets whose audio id is not in PostHog) ----------

TM_WINDOW = ("2026-05-18", "2026-07-24")  # asr_metrics start -> day after the newest arena dataset (019f8e51, 07-23)
TM_CACHE = OUT / "_posthog_window.jsonl"
TM_OUT = OUT / "_textmatch.json"
TM_SOURCES = ("arena_019f8e51", "arena_019f6442")
REPLAY_MIN = 20


def q_retry(sql: str, tries: int = 5):
    """tl.q with retries: large PostHog responses were sometimes cut off (JSON decode error / read timeout)."""
    import time
    for i in range(tries):
        try:
            return tl.q(sql)
        except Exception as e:  # noqa: BLE001
            print(f"  PostHog retry {i}: {type(e).__name__}", file=sys.stderr)
            time.sleep(10)
    raise SystemExit("PostHog: giving up")


def tm_norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def cmd_textmatch(a) -> None:
    """Recover production audio ids by transcript: arena labels are (human-edited) production transcripts, so the
    original request has a near-identical `transcription` / `processed` text and the same audio duration.
    Match = SequenceMatcher ratio >= 0.85 on normalised text, |duration diff| <= 0.6 s, and the best candidate beats the
    runner-up by >= 0.05 (else ambiguous, left unmatched)."""
    import difflib
    tl.load_env(tl.POSTHOG)
    if not TM_CACHE.exists() or a.refresh:
        cols = ["audio_id", "user_id", "client_source", "transcription", "processed", "audio_duration_ms",
                "cleanup_audio_duration_ms"]
        cursor, seen, n = TM_WINDOW[0], set(), 0
        with open(TM_CACHE, "w", encoding="utf-8") as f:
            while True:
                res = q_retry(f"SELECT uuid, timestamp, {', '.join('properties.' + c for c in cols)} FROM events "
                           f"WHERE event='asr_metrics' AND coalesce(properties.client_source, '') != 'arena' "
                           f"AND timestamp >= '{cursor}' AND timestamp < '{TM_WINDOW[1]}' ORDER BY timestamp LIMIT 3000")
                new = 0
                for u, ts, *vals in res:
                    if u in seen:
                        continue
                    seen.add(u); new += 1
                    f.write(json.dumps({"timestamp": ts, **dict(zip(cols, vals))}, ensure_ascii=False) + "\n")
                n += new
                print(f"  PostHog window: {n} (to {res[-1][1] if res else cursor})", file=sys.stderr)
                if len(res) < 3000 or not new:
                    break
                cursor = res[-1][1]
    evs = [json.loads(l) for l in open(TM_CACHE, encoding="utf-8")]
    index = collections.defaultdict(set)  # 4-char shingles of normalised text -> event index
    texts = []
    for i, e in enumerate(evs):
        t = {tm_norm(e.get("transcription")), tm_norm(e.get("processed"))} - {""}
        texts.append(t)
        for x in t:
            for k in range(0, max(1, len(x) - 3), 3):
                index[x[k:k + 4]].add(i)
    srcs = load_sources()
    cands = {}
    for src in TM_SOURCES:
        for r in srcs[src]:
            g = tm_norm(r["label"])
            if len(g) < 4:
                continue
            votes = collections.Counter()
            for k in range(0, max(1, len(g) - 3)):
                for i in index.get(g[k:k + 4], ()):
                    votes[i] += 1
            scored = []
            for i, _ in votes.most_common(40):
                ratio = max(difflib.SequenceMatcher(None, g, t).ratio() for t in texts[i])
                if ratio >= 0.85:
                    scored.append((round(ratio, 3), evs[i]))
            cands[r["utterance_id"]] = (r, scored)
    # Replay accounts: an account that reproduces >= REPLAY_MIN dataset labels and, for most of them (>= 50%), is NOT
    # the first account to say that text is re-playing dataset audio (internal tester), not the original speaker ->
    # its requests are not candidates. (2026-11-09: exactly one account, 451 labels, 84% not first.)
    per_user, not_first = collections.Counter(), collections.Counter()
    for _, sc in cands.values():
        first = min(sc, key=lambda t: t[1]["timestamp"])[1]["user_id"] if sc else None
        for u in {e["user_id"] for _, e in sc}:
            per_user[u] += 1
            not_first[u] += u != first
    replay = {u for u, n in per_user.items() if n >= REPLAY_MIN and not_first[u] / n >= 0.5}
    out = {}
    for uid, (r, scored) in cands.items():
        sc = [(x, e) for x, e in scored if e["user_id"] not in replay]
        best_by_id = {}
        for x, e in sorted(sc, key=lambda t: (-t[0], t[1]["timestamp"])):
            best_by_id.setdefault(e["audio_id"], (x, e))
        ranked = sorted(best_by_id.values(), key=lambda t: (-t[0], t[1]["timestamp"]))
        if not ranked:
            out[uid] = {"accepted": False, "reason": "no_candidate" if not scored else "replay_account_only",
                        "n_candidates": len(scored)}
            continue
        (x1, e1) = ranked[0]
        x2 = ranked[1][0] if len(ranked) > 1 else 0.0
        users = {e["user_id"] for _, e in ranked if e is not None and _ >= x1 - 0.05}
        ok = x1 - x2 >= 0.05
        reason = "unique" if ok else ("same_user_repeats" if len(users) == 1 else "several_users")
        out[uid] = {"audio_id": e1["audio_id"], "ratio": x1, "runner_up": x2, "accepted": ok, "reason": reason,
                    "n_candidates": len(ranked), "timestamp": e1["timestamp"]}
    for src in TM_SOURCES:
        v = [out.get(r["utterance_id"], {"accepted": False, "reason": "label_too_short"}) for r in srcs[src]]
        print(f"{src}: text-matched {sum(x['accepted'] for x in v)}/{len(v)} "
              f"{dict(collections.Counter(x['reason'] for x in v))}; replay accounts excluded: {len(replay)}")
    json.dump(out, open(TM_OUT, "w"), ensure_ascii=False, indent=0)
    # full properties of the accepted ids -> PostHog cache, so build treats them like any other match
    ids = sorted({v["audio_id"] for v in out.values() if v["accepted"]})
    cache = json.load(open(PH_CACHE))
    cache["by_id"].update(posthog_by("audio_id", [i for i in ids if i not in cache["by_id"]],
                                     since=TM_WINDOW[0], until=TM_WINDOW[1], batch=40))
    json.dump(cache, open(PH_CACHE, "w"), ensure_ascii=False)
    print(f"-> {TM_OUT}; {len(ids)} ids added to {PH_CACHE}")


# ---------- traces (Langfuse) ----------


def slim_trace(t: dict) -> dict:
    """Keep only what matters here: root input / metadata and the ASR generation(s) input + prompt version."""
    asr = []
    for o in t["observations"]:
        if o.get("name") == "gen_ai.transcribe" or o.get("name") == "multimodal":
            md = o.get("metadata") or {}
            asr.append({"name": o["name"], "promptName": o.get("promptName"), "promptVersion": o.get("promptVersion"),
                        "model": md.get("attributes.gen_ai.request.model") or o.get("model"),
                        "env": md.get("resourceAttributes.deployment.environment.name"),
                        "input": o.get("input"), "level": o.get("level")})
    md = t.get("metadata") or {}
    return {"trace_id": t["id"], "sessionId": t.get("sessionId"), "timestamp": t.get("timestamp"),
            "root_input": t.get("input"),
            "root_metadata": {k: v for k, v in md.items() if not k.startswith(("scope.", "resourceAttributes.telemetry"))},
            "asr": asr, "obs_names": [o.get("name") for o in t["observations"]]}


def cmd_traces(a) -> None:
    tl.load_env(tl.POSTHOG + tl.LANGFUSE)
    LF_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(PH_CACHE))
    todo = []
    for aid, ps in list(cache["by_id"].items()) + [(p["audio_id"], [p]) for ps in cache["by_key"].values() for p in ps]:
        p = prod_event(ps)
        if p and p.get("langfuse_trace_id"):
            todo.append((aid, p["langfuse_trace_id"], p.get("started_at") or p["_timestamp"]))
    # Langfuse-only clips (TEST environment recordings) first: they have no other source of context
    todo = [(aid, tids[0], None) for aid, tids in cache.get("lf_sessions", {}).items() if tids] + todo
    todo = [x for x in dict((t[0], t) for t in reversed(todo)).values() if not (LF_DIR / f"{x[0]}.json").exists()]
    todo.sort(key=lambda x: x[2] is not None)
    print(f"Langfuse traces to fetch: {len(todo)} ({a.workers} workers)")

    def one(job: tuple) -> None:
        aid, tid, ts = job
        try:
            t = tl.fetch_trace(tid, ts)
            rec = slim_trace(t) if t.get("observations") else {"trace_id": tid, "missing": True}
        except SystemExit as e:  # tl.lf exits on 4xx / gives up
            rec = {"trace_id": tid, "error": str(e)[:300]}
        json.dump(rec, open(LF_DIR / f"{aid}.json", "w"), ensure_ascii=False, default=str)

    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            if n % 50 == 0:
                print(f"  {n + 1}/{len(todo)}", file=sys.stderr, flush=True)
    print(f"-> {LF_DIR}")


# ---------- build ----------


def prod_event(ps: list[dict]) -> dict | None:
    """The original production request: first non-arena event (arena replays of the same id are ignored)."""
    ps = [p for p in ps if p.get("client_source") != "arena"]
    return ps[0] if ps else None


def as_obj(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def nonempty(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def term_in(term: str, text: str) -> bool:
    """Dictionary term occurs in text: case-insensitive; Latin terms on word boundaries, spaces optional between
    CJK and Latin (labels write 'Claude 改了' while terms are bare)."""
    t, x = norm(term), norm(text)
    if not t:
        return False
    if re.search(r"[A-Za-z0-9]", t) and not re.search(r"[一-鿿]", t):
        pat = r"(?<![a-z0-9])" + r"\s*".join(re.escape(w) for w in t.split()) + r"(?![a-z0-9])"
        return re.search(pat, x) is not None
    return t.replace(" ", "") in x.replace(" ", "")


def load_tone_map() -> dict[str, tuple[str, str]]:
    """app display name (as in the owner's TSV) -> (tone, scenario)."""
    m = {}
    for line in open(TONE_TSV, encoding="utf-8"):
        if line.startswith("#") or line.startswith("tone\t") or not line.strip():
            continue
        tone, scen, app = line.rstrip("\n").split("\t")
        m[app] = (tone, scen)
    return m


# bundle id (lower-case) / production app name -> TSV app name. Only ids whose identity is certain; everything else is
# left unmatched for the owner. Browsers are mapped to their TSV row ("no URL available") even when a URL was logged.
BUNDLE_ALIASES = {
    "com.whatsapp": "WhatsApp", "net.whatsapp.whatsapp": "WhatsApp", "desktop.whatsapp": "WhatsApp",
    "com.whatsapp.w4b": "WhatsApp Business", "net.whatsapp.whatsappsmb": "WhatsApp Business",
    "com.gbwhatsapp": "WhatsApp 第三方改版（GB / MB WhatsApp）", "com.mbwhatsapp": "WhatsApp 第三方改版（GB / MB WhatsApp）",
    "com.google.android.apps.messaging": "Google Messages", "com.apple.mobilesms": "Messages（Apple 信息）",
    "com.apple.messages": "Messages（Apple 信息）", "com.apple.mobilesms.compose": "Messages（Apple 信息）",
    "com.anthropic.claude": "Claude", "com.anthropic.claudefordesktop": "Claude",
    "com.openai.chatgpt": "ChatGPT", "com.openai.chat": "ChatGPT",
    "com.facebook.aura": "Muse", "com.twitter.android": "X (Twitter)", "com.atebits.tweetie2": "X (Twitter)",
    "com.roblox.client": "Roblox", "com.google.android.googlequicksearchbox": "Google 搜索 app",
    "com.google.gemini": "Gemini", "com.google.android.apps.bard": "Gemini",
    "com.google.chrome": "Chrome（浏览器，拿不到网址）", "com.android.chrome": "Chrome（浏览器，拿不到网址）",
    "com.google.chrome.ios": "Chrome（浏览器，拿不到网址）",
    "com.openai.codex": "Codex", "com.todesktop.230313mzl4w4u92": "Cursor",
    "com.tencent.mm": "WeChat 微信", "com.tencent.xinwechat": "WeChat 微信", "com.tencent.xin": "WeChat 微信",
    "com.whatsamb": "WhatsApp 第三方改版（GB / MB WhatsApp）", "zeta.app": "zeta",
    "com.instagram.android": "Instagram", "com.burbn.instagram": "Instagram",
    "com.facebook.orca": "Messenger", "com.facebook.messenger": "Messenger",
    "ai.x.grok": "Grok", "com.zhiliaoapp.musically": "TikTok", "com.ss.android.ugc.trill": "TikTok",
    "com.facebook.katana": "Facebook", "com.facebook.facebook": "Facebook",
    "com.google.android.gm": "Gmail", "com.google.gmail": "Gmail", "com.apple.notes": "Apple Notes",
    "com.apple.mobilenotes": "Apple Notes", "com.electron.lark": "Lark / 飞书", "com.ss.android.lark": "Lark / 飞书",
    "com.bytedance.lark": "Lark / 飞书", "com.larksuite.suite": "Lark / 飞书",
    "org.telegram.messenger": "Telegram", "ru.keepcoder.telegram": "Telegram", "org.telegram.desktop": "Telegram",
    "ph.telegra.telegraph": "Telegram", "com.tinyspeck.slackmacgap": "Slack", "com.slack": "Slack",
    "com.google.antigravity": "Antigravity", "com.google.android.youtube": "YouTube",
    "org.thoughtcrime.securesms": "Signal", "org.whispersystems.signal": "Signal",
    "com.apple.terminal": "Terminal", "com.microsoft.vscode": "VS Code", "com.googlecode.iterm2": "iTerm2",
    "com.discord": "Discord", "com.hnc.discord": "Discord", "com.linkedin.linkedin": "LinkedIn", "com.hammerandchisel.discord": "Discord", "dev.warp.warp-stable": "Warp",
    "com.samsung.android.app.notes": "Samsung Notes", "com.linkedin.android": "LinkedIn",
    "com.android.vending": "Play Store", "com.google.android.apps.dynamite": "Google Chat",
    "com.deepseek.chat": "DeepSeek", "com.google.android.keep": "Keep", "com.google.android.apps.docs.editors.docs":
    "Google Docs", "com.android.systemui": "System UI", "com.google.android.apps.googlevoice": "Google Voice",
    "com.snapchat.android": "Snapchat", "com.toyopagroup.picaboo": "Snapchat", "com.enflick.android.textnow": "TextNow",
    "com.microsoft.office.outlook": "Outlook", "com.microsoft.outlook": "Outlook", "com.grindrapp.android": "Grindr",
    "com.janitorai.app": "Janitor", "com.larus.nova": "豆包 Doubao", "com.brave.browser": "Brave（浏览器，拿不到网址）",
    "com.brave.browser.ios": "Brave（浏览器，拿不到网址）", "notion.id": "Notion", "notion.so": "Notion",
    "com.microsoft.teams": "Microsoft Teams", "com.microsoft.teams2": "Microsoft Teams",
    "com.amazon.mshop.android.shopping": "Amazon", "com.vivaldi.browser": "Vivaldi", "com.lemon.lvoverseas": "CapCut",
    "com.mercadolibre": "Mercado Livre", "com.google.android.calendar": "Calendar",
    "com.duckduckgo.mobile.android": "DuckDuckGo", "org.mozilla.firefox": "Firefox（浏览器，拿不到网址）",
    "org.mozilla.firefox.ios": "Firefox（浏览器，拿不到网址）", "com.google.android.apps.maps": "Maps",
    "md.obsidian": "Obsidian", "com.opera.browser": "Opera", "com.perplexity.app": "Perplexity",
    "ai.perplexity.app.android": "Perplexity", "ai.perplexity.mac": "Perplexity",
}
# production app.name -> TSV name, used when the bundle id is not in BUNDLE_ALIASES but the logged display name is an
# exact (case-insensitive) TSV name.
NAME_ALIASES = {"messages": None, "google chrome": "Chrome（浏览器，拿不到网址）", "chrome": "Chrome（浏览器，拿不到网址）",
                "x": "X (Twitter)", "feishu": "Lark / 飞书", "lark": "Lark / 飞书", "飞书": "Lark / 飞书",
                "wechat": "WeChat 微信", "微信": "WeChat 微信", "豆包": "豆包 Doubao", "doubao": "豆包 Doubao",
                "visual studio code": "VS Code", "code": "VS Code", "google docs": "Google Docs",
                "firefox": "Firefox（浏览器，拿不到网址）", "brave browser": "Brave（浏览器，拿不到网址）",
                "brave": "Brave（浏览器，拿不到网址）", "google": "Google 搜索 app", "play store": "Play Store",
                "google play store": "Play Store", "keep notes": "Keep", "google keep": "Keep", "notes": None}


def map_app(bundle: str | None, name: str | None, tone_map: dict) -> tuple[str | None, str]:
    """-> (TSV app name or None, how): how in alias_bundle | name_exact | name_alias | no_app | unmatched."""
    if not bundle and not name:
        return "无 app 信息（多为 iOS 键盘）", "no_app"
    b = (bundle or "").lower()
    if b in BUNDLE_ALIASES and BUNDLE_ALIASES[b] in tone_map:
        return BUNDLE_ALIASES[b], "alias_bundle"
    n = (name or "").strip()
    lower = {k.lower(): k for k in tone_map}
    if n.lower() in lower:
        return lower[n.lower()], "name_exact"
    if NAME_ALIASES.get(n.lower()) in tone_map:
        return NAME_ALIASES[n.lower()], "name_alias"
    return None, "unmatched"


def ctx_vocab_meaning_hint(ctx: dict) -> float | None:
    """Fraction of audio_context.vocabulary terms that appear in screen_text (macOS) -- hint that the client extracts
    them from the screen rather than from the user's dictionary."""
    v, st = ctx.get("vocabulary") or [], ctx.get("screen_text") or ""
    if not v or not st:
        return None
    return sum(norm(t) in norm(st) for t in v) / len(v)


def parse_sent(asr: list[dict]) -> dict:
    """What the ASR request actually carried (Langfuse gen_ai.transcribe input). Formats seen (2026-09 -> 11):
    transcription-for-multimodality v31 / -chinese v25: <CURSOR_CONTEXT>{"before_cursor_suffix", "after_cursor_prefix"}
      and <VOCABULARY_CONTEXT>{"keywords": [...]}; v14 / v18 (system prompt): <CURSOR_CONTEXT_CONTINUITY> with
      TEXT_BEFORE_CURSOR / TEXT_AFTER_CURSOR, <VOCABULARY_CONTEXT>; qwen streaming: context text
      "before_cursor_suffix:..." / "after_cursor_prefix:..." + parameters.vocabulary {term: weight};
      gpt-live: session.update transcription.keywords. Screen text is never sent."""
    out = {"asr_prompt": None, "asr_request_model": None, "sent_blocks": [], "sent_vocabulary": None,
           "sent_before_cursor": None, "sent_after_cursor": None}
    for o in asr:
        inp = o.get("input")
        if inp in (None, "null"):
            continue  # hedged request that was not used / not logged
        s = json.dumps(inp, ensure_ascii=False)
        out["asr_request_model"] = o.get("model")
        out["asr_prompt"] = f"{o['promptName']}@v{o.get('promptVersion')}" if o.get("promptName") else None
        out["sent_blocks"] = sorted(set(re.findall(r"<([A-Z_]+_CONTEXT(?:_CONTINUITY)?)>", s)))
        vocab = None
        m = re.search(r"<VOCABULARY_CONTEXT>(.*?)</VOCABULARY_CONTEXT>", s)
        if m:
            vocab = re.findall(r'\\+"(.*?)\\+"', m.group(1).split("keywords", 1)[-1])
        for pat in (r'\\"vocabulary\\":\{(.*?)\}', r'\\"keywords\\":\[(.*?)\]'):
            m = re.search(pat, s)
            if m and vocab is None:
                vocab = re.findall(r'\\+"(.*?)\\+"', m.group(1))
                vocab = [v for v in vocab if v not in (":",)]
        vocab = [v for v in (vocab or []) if v != "keywords" and re.search(r"\w", v)]
        out["sent_vocabulary"] = vocab or None
        out["sent_before_cursor"] = "before_cursor_suffix" in s or "TEXT_BEFORE_CURSOR:" in s
        out["sent_after_cursor"] = "after_cursor_prefix" in s or "TEXT_AFTER_CURSOR:" in s
    return out


def platform_of(p: dict | None, md: dict, root_in: dict, ctx: dict) -> str:
    """client_source; before ~2026-09 PostHog often lacks it (source = 'voicecursor-services'), then it is inferred:
    audio_input / screen_text only come from the macOS client; otherwise 'unknown'."""
    for v in ((p or {}).get("client_source"), md.get("source"), root_in.get("source")):
        if v and v not in ("voicecursor-services", "arena"):
            return v
    if ctx.get("audio_input") or ctx.get("screen_text") is not None:
        return "macos(inferred)"
    return "unknown"


def proper_terms(text: str) -> list[str]:
    """Proper-noun-like Latin tokens in a label: contain an upper-case letter after the first character or are
    all-caps / CamelCase, or start upper-case in the middle of CJK text (e.g. 'Langfuse', 'VoiceCursor', 'PR', 'iOS')."""
    out = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9.+\-]*[A-Za-z0-9]|[A-Z]", text or ""):
        t = m.group(0)
        prev = text[max(0, m.start() - 2):m.start()]
        cjk_ctx = re.search(r"[一-鿿]", prev) is not None or m.start() == 0 and re.search(r"[一-鿿]", text or "")
        if t in ("I", "OK", "Ok") or len(t) < 2:
            continue
        if re.search(r"[A-Z]", t[1:]) or (t[0].isupper() and cjk_ctx):
            out.append(t)
    return list(dict.fromkeys(out))


def build_row(src: str, r: dict, p: dict | None, via: str, lf: dict | None, tone_map: dict) -> dict:
    ctx: dict = {}
    if p:
        ctx = tl.ctx_of(p)
    elif lf and isinstance(lf.get("root_input"), dict):
        ctx = as_obj(lf["root_input"].get("audio_context")) or {}
        via = "langfuse_root"
    md = (lf or {}).get("root_metadata") or {}
    app = dict(ctx.get("app") or {})
    app_source = "audio_context" if app.get("bundle_id") or app.get("name") else None
    if not app_source and ((p or {}).get("tone_app_id") or md.get("tone_app_id")):
        # before ~2026-08 audio_context is missing, but tone_app_id (lower-cased bundle id) is logged
        app = {"bundle_id": (p or {}).get("tone_app_id") or md.get("tone_app_id")}
        app_source = "tone_app_id"
    root_in = (lf or {}).get("root_input") if isinstance((lf or {}).get("root_input"), dict) else {}
    get = lambda k: (p or {}).get(k) if p else (md.get(k) if md.get(k) is not None else root_in.get(k))
    eff_vocab = as_obj(get("effective_vocabulary")) or []
    ctx_vocab = ctx.get("vocabulary") or []
    label = r["label"] or ""
    tsv_app, how = map_app(app.get("bundle_id"), app.get("name"), tone_map) if (p or ctx) else (None, "unmatched")
    tone = tone_map.get(tsv_app, (None, None)) if tsv_app else (None, None)
    row = {
        "utterance_id": r["utterance_id"], "source": src, "split": r["split"], "audio_id": None,
        "matched": via != "none", "match_via": via, "label": label, "label_is_human": r["label_is_human"],
        "duration_sec": r.get("duration_sec"), **{f"x_{k}": v for k, v in r["extra"].items()},
    }
    if via == "none":
        row["audio_id_candidates"] = r["ids"]
        if r["extra"].get("pool_audio_id"):  # production id known from arena provenance, but no logged request
            row["audio_id"], row["audio_id_source"] = r["extra"]["pool_audio_id"], "arena_pool_provenance"
        return row
    row.update({
        "audio_id": (p or {}).get("audio_id") or (lf or {}).get("sessionId") or r["ids"][0],
        "timestamp": (p or {}).get("started_at") or (p or {}).get("_timestamp") or (lf or {}).get("timestamp"),
        "client_source": platform_of(p, md, root_in, ctx), "client_source_logged": bool(get("client_source")),
        "mode": get("mode") or get("effective_mode"),
        "mode_inferred": get("mode") or get("effective_mode") or ("edit" if get("edit_action") else "dictate"),
        "requested_mode": get("requested_mode"), "asr_model": get("model"),
        "lang": get("lang"), "lang_source": get("lang_source"), "languages": as_obj(get("languages")),
        "ctx_language": ctx.get("language"),
        "app_bundle_id": app.get("bundle_id"), "app_name": app.get("name"), "app_url": app.get("url"),
        "app_field_kind": app.get("field_kind"), "app_return_action": app.get("return_action"),
        "app_type": get("app_type"), "app_source": app_source,
        "text_before_cursor": ctx.get("text_before_cursor"), "text_after_cursor": ctx.get("text_after_cursor"),
        "selected_text": ctx.get("selected_text"), "screen_text": ctx.get("screen_text"),
        "placeholder_text": ctx.get("placeholder_text"), "participants": ctx.get("participants"),
        "subject": ctx.get("subject"), "ctx_vocabulary": ctx_vocab,
        "use_dictionary": get("use_dictionary"), "effective_vocabulary": eff_vocab,
        "effective_vocabulary_count": get("effective_vocabulary_count"),
        "tone_id": get("tone_id"), "tone_source": get("tone_source"), "tone_app_id": get("tone_app_id"),
        "tone_prompt": get("tone_prompt"), "invocation_context": get("invocation_context"),
        "prod_transcription": (p or {}).get("transcription"), "prod_processed": (p or {}).get("processed"),
        "langfuse_trace_id": get("langfuse_trace_id") or (lf or {}).get("trace_id"),
        "audio_context_keys": sorted(ctx),
        # derived
        "tone_mapped": tone[0], "tone_scenario": tone[1], "tone_map_app": tsv_app, "tone_map_via": how,
        "len_before_cursor": len(ctx.get("text_before_cursor") or ""),
        "len_after_cursor": len(ctx.get("text_after_cursor") or ""),
        "len_screen_text": len(ctx.get("screen_text") or ""), "len_selected_text": len(ctx.get("selected_text") or ""),
        "dict_size": len(eff_vocab), "ctx_vocab_size": len(ctx_vocab),
        "dict_terms_in_label": [t for t in eff_vocab if term_in(t, label)],
        "ctx_vocab_in_label": [t for t in ctx_vocab if term_in(t, label)],
        "ctx_vocab_in_screen_frac": ctx_vocab_meaning_hint(ctx),
        "label_in_before_cursor": bool(label) and norm(label)[:40] in norm(ctx.get("text_before_cursor") or ""),
        "label_in_after_cursor": bool(label) and norm(label)[:40] in norm(ctx.get("text_after_cursor") or ""),
        "label_in_screen_text": bool(label) and norm(label)[:40] in norm(ctx.get("screen_text") or ""),
        "label_terms": proper_terms(label),
        "label_terms_in_context": [t for t in proper_terms(label) if any(
            term_in(t, ctx.get(k) or "") for k in ("text_before_cursor", "text_after_cursor", "screen_text"))],
    })
    if lf and lf.get("asr"):
        row.update(parse_sent(lf["asr"]))
        row["langfuse_env"] = next((o.get("env") for o in lf["asr"] if o.get("env")), None)
    row["langfuse_found"] = bool(lf and not lf.get("missing") and not lf.get("error"))
    return row


def cmd_build(a) -> None:
    cache = json.load(open(PH_CACHE))
    by_id, by_key = cache["by_id"], cache["by_key"]
    lf_sessions = cache.get("lf_sessions", {})
    textmatch = json.load(open(TM_OUT)) if TM_OUT.exists() else {}
    tone_map = load_tone_map()
    srcs = load_sources()
    bundles = collections.Counter()
    bundle_names: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for src, rows in srcs.items():
        out = []
        for r in rows:
            p, via, aid = None, "none", None
            tm = textmatch.get(r["utterance_id"]) or {}
            if tm.get("accepted") and tm["audio_id"] not in r["ids"]:
                p = prod_event(by_id.get(tm["audio_id"], []))
                if p:
                    via, aid = "posthog_text_match", tm["audio_id"]
            for i in ([] if p else r["ids"]):
                p = prod_event(by_id.get(i, []))
                if p:
                    via, aid = ("posthog_audio_id", i)
                    break
                if by_id.get(i):  # only arena replays of this id
                    via = "arena_only"
            if not p and r["s3_key"]:
                p = prod_event(by_key.get(r["s3_key"], []))
                if p:
                    via, aid = "posthog_s3_key", p["audio_id"]
            lf = None
            if p:
                f = LF_DIR / f"{p['audio_id']}.json"
                lf = json.load(open(f)) if f.exists() else None
            else:
                for i in r["ids"]:
                    f = LF_DIR / f"{i}.json"
                    if lf_sessions.get(i) and f.exists():
                        lf, via, aid = json.load(open(f)), "langfuse_session", i
                        break
            if via == "arena_only":
                via = "none"
            row = build_row(src, r, p, via, lf, tone_map)
            if r["utterance_id"] in textmatch:
                row["text_match"] = textmatch[r["utterance_id"]]
            if row["matched"] and src != "segmented_long":
                key = row.get("app_bundle_id") or "(none)"
                bundles[key] += 1
                bundle_names[key][row.get("app_name") or ""] += 1
            out.append(row)
        with open(OUT / f"{src}.jsonl", "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"{src}: {len(out)} rows, matched {sum(r['matched'] for r in out)} -> {OUT / (src + '.jsonl')}")
    # proposed join, over every bundle id seen in our matched clips plus the 30-day production scan
    scan_bundles, scan_names = scan_bundle_counts()
    allb = set(bundles) | set(scan_bundles)
    with open(OUT / "bundle_to_app.tsv", "w", encoding="utf-8") as f:
        f.write("bundle_id\tapp_name_logged\tclips_in_our_data\tprod_requests_30d\ttsv_app\ttone\tmatched\tmatch_via\n")
        for b in sorted(allb, key=lambda x: (-bundles[x], -scan_bundles[x], x)):
            name = (bundle_names[b] or scan_names[b]).most_common(1)[0][0] if (bundle_names[b] or scan_names[b]) else ""
            tsv_app, how = map_app(None if b == "(none)" else b, name, tone_map)
            tone = tone_map.get(tsv_app, ("", ""))[0] if tsv_app else ""
            f.write(f"{b}\t{name}\t{bundles[b]}\t{scan_bundles[b]}\t{tsv_app or ''}\t{tone}\t"
                    f"{'yes' if tsv_app else 'no'}\t{how}\n")
    print(f"-> {OUT / 'bundle_to_app.tsv'} ({len(allb)} bundle ids)")


def scan_bundle_counts() -> tuple[collections.Counter, dict]:
    """bundle id frequency in the cached 30-day dictate scan (online.py, 2026-10-05 -> 11-04)."""
    cnt, names = collections.Counter(), collections.defaultdict(collections.Counter)
    for fp in sorted(glob.glob(str(SCAN_DAYS / "*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            if line.startswith('{"_scan'):
                continue
            r = json.loads(line)
            app = (tl.ctx_of(r).get("app") or {})
            b = app.get("bundle_id") or "(none)"
            cnt[b] += 1
            names[b][app.get("name") or ""] += 1
    return cnt, names


# ---------- report ----------

FIELDS = [("before-cursor", "text_before_cursor"), ("after-cursor", "text_after_cursor"),
          ("用户词典", "effective_vocabulary"), ("屏幕词(macOS)", "ctx_vocabulary"), ("屏幕文字", "screen_text"),
          ("选中文字", "selected_text"), ("App", "app_bundle_id"), ("tone(映射)", "tone_mapped"), ("mode(有记录)", "mode"),
          ("语言设置", "languages"), ("客户端语言", "ctx_language"), ("自定义 tone", "tone_prompt")]
PII = [(re.compile(r"(发送给|发给|To:?)\s*[^\s⏎|]{1,20}"), r"\1 <name>"), (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"), (re.compile(r"https?://\S+"), "<url>"),
       (re.compile(r"\+?\d[\d\s\-()]{6,}\d"), "<phone>"), (re.compile(r"@[\w\-]{2,}"), "@<handle>")]


def pct(v: list[float], q: float):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


def has(r: dict, k: str) -> bool:
    v = r.get(k)
    if k == "tone_mapped":
        return v in ("casual", "neutral", "formal")
    if k == "mode":
        return nonempty(v)
    if k == "app_bundle_id":
        return nonempty(v) or nonempty(r.get("app_name"))
    return nonempty(v)


def anon(s: str, n: int = 60, names: list[str] | None = None) -> str:
    s = (s or "").replace("\n", " ⏎ ").replace("|", "\\|")
    for pat, rep in PII:
        s = pat.sub(rep, s)
    for nm in names or []:
        if nm and len(nm) >= 2:
            s = re.sub(re.escape(nm), "<name>", s, flags=re.I)
    return s if len(s) <= n else s[:n] + "…"


def coverage_table(groups: list[tuple[str, list[dict]]]) -> list[str]:
    head = "| 来源 | split | 条数 | 匹配到线上请求 | " + " | ".join(f for f, _ in FIELDS) + " |"
    lines = [head, "|" + "---|" * (len(FIELDS) + 4)]
    for name, rows in groups:
        by_split = collections.defaultdict(list)
        for r in rows:
            by_split[r["split"]].append(r)
        for sp, rs in sorted(by_split.items()) + ([("全部", rows)] if len(by_split) > 1 else []):
            m = [r for r in rs if r["matched"]]
            cells = [f"{sum(has(r, k) for r in m) / len(m):.0%}" if m else "—" for _, k in FIELDS]
            lines.append(f"| {name} | {sp} | {len(rs)} | {len(m)} ({len(m) / len(rs):.0%}) | " + " | ".join(cells) + " |")
    return lines


def timing_check() -> dict:
    """Is text_before_cursor captured after the previous dictation was inserted? Consecutive dictations by the same user
    in the same app within 90 s (cached 30-day scan): does the next request's before-cursor text contain the previous
    output? And does a request's own output already sit in its before / after-cursor text (would be label leakage)?"""
    by_user = collections.defaultdict(list)
    for fp in sorted(glob.glob(str(SCAN_DAYS / "*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            if line.startswith('{"_scan'):
                continue
            r = json.loads(line)
            if r.get("error_stage") or not (r.get("processed") or "").strip():
                continue
            c = tl.ctx_of(r)
            by_user[r["user_id"]].append((r["timestamp"], r.get("client_source"), (c.get("app") or {}).get("bundle_id"),
                                          r["processed"], c.get("text_before_cursor"), c.get("text_after_cursor")))
    pairs = collections.Counter()
    own = collections.Counter()
    import datetime as dt
    ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    for rs in by_user.values():
        rs.sort()
        for i, (t, cs, app, out, tbc, tac) in enumerate(rs):
            key = norm(out)[:30]
            if len(key) >= 12:
                own[(cs, "n")] += 1
                own[(cs, "in_tbc")] += bool(tbc) and key in norm(tbc)
                own[(cs, "in_tac")] += bool(tac) and key in norm(tac)
            if i == 0:
                continue
            pt, pcs, papp, pout, _, _ = rs[i - 1]
            if papp != app or (ts(t) - ts(pt)).total_seconds() > 90 or len(norm(pout)) < 12 or not tbc:
                continue
            tail = norm(pout)[-30:]
            pairs[(cs, "n")] += 1
            pairs[(cs, "prev_in_tbc")] += tail in norm(tbc)
    return {"pairs": pairs, "own": own}


def load_scrub() -> list[str]:
    """Private list of strings to mask in report examples (names etc.), one per line; kept out of git."""
    f = OUT / "_scrub.txt"
    return [l.strip() for l in open(f, encoding="utf-8") if l.strip() and not l.startswith("#")] if f.exists() else []


def by_source_leak(data: dict, main: list[str]) -> list[str]:
    lines = ["| 来源 | 匹配条数 | 标注已在 before-cursor | 标注已在 after-cursor | 标注已在屏幕文字 | 标注里的专名（拉丁词） | 其中在 before / after / 屏幕文字里出现 |",
             "|---|---|---|---|---|---|---|"]
    for k in main:
        m = [r for r in data[k] if r["matched"]]
        terms = sum(len(r.get("label_terms") or []) for r in m)
        hit = sum(len(r.get("label_terms_in_context") or []) for r in m)
        lines.append(f"| {k} | {len(m)} | {sum(bool(r.get('label_in_before_cursor')) for r in m)} | "
                     f"{sum(bool(r.get('label_in_after_cursor')) for r in m)} | {sum(bool(r.get('label_in_screen_text')) for r in m)} | "
                     f"{terms}（{sum(bool(r.get('label_terms')) for r in m)} 条）| {hit}（{hit / max(1, terms):.0%}；"
                     f"{sum(bool(r.get('label_terms_in_context')) for r in m)} 条）|")
    return lines


def cmd_report(a) -> None:
    data = {}
    for src in ["benchmark_chinese", "online_phase1", "online_pool_b1full", "benchmark_v1", "benchmark_v2",
                "recording_v1", "recording_v2", "recording_v2_test", "recording_freeform_test",
                "arena_019f8e51", "arena_019f6442", "segmented_long"]:
        data[src] = read_jsonl(OUT / f"{src}.jsonl")
    main = [k for k in data if k != "segmented_long"]
    allm = [r for k in main for r in data[k] if r["matched"]]
    human = [r for r in allm if r["label_is_human"]]
    scrub = load_scrub()
    n_all = sum(len(data[k]) for k in main)
    pc = lambda xs, f: f"{sum(bool(f(r)) for r in xs) / max(1, len(xs)):.0%}"
    L: list[str] = []
    w = L.append

    # ---- numbers used in the summary
    dict_rows = [r for r in human if r.get("dict_size")]
    dict_hit = [r for r in dict_rows if r["dict_terms_in_label"]]
    tac = [r for r in allm if nonempty(r.get("text_after_cursor"))]
    tc = timing_check()
    p_, o_ = tc["pairs"], tc["own"]
    bt = list(csv.DictReader(open(OUT / "bundle_to_app.tsv", encoding="utf-8"), delimiter="\t"))
    tot_our = sum(int(b["clips_in_our_data"]) for b in bt)
    tot_prod = sum(int(b["prod_requests_30d"]) for b in bt)
    m_our = sum(int(b["clips_in_our_data"]) for b in bt if b["matched"] == "yes")
    m_prod = sum(int(b["prod_requests_30d"]) for b in bt if b["matched"] == "yes")
    sent = [r for r in allm if r.get("asr_request_model")]

    w("# 我们数据集里每条音频，线上记录了哪些上下文（context pull）\n")
    w("- 生成：2026-11-09，`scripts/context_pull/pull_context.py`（pull → traces → build → report）。只读：PostHog `asr_metrics` "
      "HogQL SELECT + Langfuse v2 observations GET；没有调用任何 LLM，没有写线上系统。")
    w("- 明细（含用户原文，只在本地）：`data/private/context_pull/<source>.jsonl`（每条 utterance_id 一行）、`bundle_to_app.tsv`。"
      "本报告的例子都截断过，并去掉了邮箱、电话、网址、@账号和人名。")
    w("- 跳过（不是线上请求）：ASCEND 全部、val_longform（由 ASCEND 拼成）。segmented_long 里切自 arena 的 100 段没有单独拉"
      "（父音频属于 arena 数据集，见 §1.1）。\n")

    w("## 0. 结论\n")
    n_lf = sum(1 for r in allm if r["match_via"] == "langfuse_root")
    n_tm = sum(1 for r in allm if r["match_via"] == "posthog_text_match")
    arena = [r for k in ("arena_019f8e51", "arena_019f6442") for r in data[k]]
    arena_ctx = sum(1 for r in arena if r["matched"] and r.get("audio_context_keys"))
    w(f"1. **{len(allm)}/{n_all} 条对上了线上请求**。非 arena 来源 {sum(len(data[k]) for k in main if not k.startswith('arena'))} 条全部对上"
      f"（benchmark_v2 候选 537 条也在内）；其中 recording_v1（138）和 recording_freeform_test（21）只在 Langfuse **test 环境**，"
      f"上下文取自 Langfuse 根 span 的 input（{n_lf} 条）。arena 两个数据集（{len(arena)} 条）文件名 / 池子 id 在 PostHog 基本查不到，"
      f"改用**转写文本匹配**找回 {n_tm} 条线上 audio_id（§1.1）；但这些请求都在 2026-05 → 07，那时还不记 `audio_context`，"
      f"所以 arena 只有 {arena_ctx} 条有光标 / 屏幕上下文，大多只能拿到 App（从 `tone_app_id`）和语言。")
    w(f"2. **光标后文字（text_after_cursor）有记录**：字段在 `audio_context.text_after_cursor`，我们的数据里 {len(tac)}/{len(allm)} 条非空"
      f"（{len(tac) / len(allm):.0%}），主要来自 macOS（{pc([r for r in allm if str(r.get('client_source')).startswith('macos')], lambda r: nonempty(r.get('text_after_cursor')))}），"
      f"Android / iOS 只有 3–4%。线上 30 天 macOS 约 45–55% 的 dictate 请求带它。部分模型的请求里它也真的发给了 ASR（§5）。")
    w(f"3. **用户词典基本没用上**：有人工标注且带词典的 {len(dict_rows)} 条里，只有 {len(dict_hit)} 条（{len(dict_hit) / max(1, len(dict_rows)):.0%}）"
      f"的标注里出现了至少一个词典词；按词条算命中率 {sum(len(r['dict_terms_in_label']) for r in dict_rows) / max(1, sum(r['dict_size'] for r in dict_rows)):.1%}。"
      "词典是用户自己加的，长期不变，跟当前这句话大多无关；拿它当 hotword 训练，绝大多数是干扰项（这其实正好可以用来训练模型不乱插词）。")
    w(f"4. **App → tone**：{m_our}/{tot_our} 条（{m_our / max(1, tot_our):.0%}）能把 bundle id 对上 `tone_by_app.tsv`；"
      f"线上 30 天 dictate 请求能对上 {m_prod / max(1, tot_prod):.0%}。没对上的主要是 Safari、国产笔记 App、VoiceCursor 自带教程等（§4）。")
    w(f"5. **before-cursor 在开始录音时抓**：同一 App 里 90 秒内连续两次听写，后一次的 before-cursor 里有前一次输出的占 "
      f"{p_[('android', 'prev_in_tbc')] / max(1, p_[('android', 'n')]):.0%}（Android）/ {p_[('macos', 'prev_in_tbc')] / max(1, p_[('macos', 'n')]):.0%}（macOS）/ "
      f"{p_[('ios', 'prev_in_tbc')] / max(1, p_[('ios', 'n')]):.0%}（iOS）；本条输出出现在自己的 before-cursor 里的只有 0.1–0.3%。"
      "也就是说，before-cursor 里是上一句话（已插入），不会泄露本句答案。**例外：recording_v2 / v2_test 是对着打开的朗读稿录的，"
      "朗读稿本身就在 before / after-cursor 和屏幕文字里（§7），这两个集合的上下文不能直接用。**")
    w(f"6. **线上 ASR 并没有用上大部分上下文**：屏幕文字从来不发；gpt-transcribe 请求里一次也没看到词典（"
      f"{sum(1 for r in sent if r['asr_request_model'] == 'gpt-transcribe' and r.get('dict_size'))} 条有词典的请求，0 条发出去了）；"
      "before / after-cursor 只有一部分请求发了（§5）。所以线上输出不能当作“用了上下文的结果”，训练要用 PostHog 里的原始字段。\n")

    w("## 1. 覆盖率（按来源 × split）\n")
    w("百分比的分母是**匹配到线上请求的条数**。字段非空才算有（空字符串、空列表不算）。"
      "`tone(映射)` = App 能通过 bundle id 对上 `configs/reference/tone_by_app.tsv`，且 tone 不是 unknown。"
      "`mode(有记录)` / `语言设置`：2026-09 之前的 PostHog 事件没有 `mode` 和 `languages` 字段（只有路由用的 `lang`），"
      "Langfuse test 环境的根 span 也不带 `languages`；所以这两列是“有没有记录”，不是“有没有设置”。"
      "benchmark_v2 的 split 是 test_candidate（还没审）。\n")
    L.extend(coverage_table([(k, data[k]) for k in main]))
    w("")
    w("匹配方式：" + "；".join(f"{k} {dict(collections.Counter(r['match_via'] for r in data[k]))}" for k in main) + "。")
    w("online_pool_b1full 包含 online_pool 的全部 267 条，所以没有单列 online_pool（明细里 `x_in_online_pool` 标出了这 267 条）。"
      "benchmark_v1 有 108 条是手工传到 arena 的，arena 文件名就是线上 audio_id，按 audio_id 都能对上。\n")

    w("### 1.1 arena 数据集怎么找回线上 audio_id\n")
    for k in ("arena_019f8e51", "arena_019f6442"):
        rs = data[k]
        reasons = collections.Counter((r.get("text_match") or {}).get("reason", "label_too_short") for r in rs)
        w(f"- **{k}**（{len(rs)} 条）：匹配方式 {dict(collections.Counter(r['match_via'] for r in rs))}；"
          f"文本匹配结果 {dict(reasons)}。")
    w("- 方法：arena_019f8e51 的文件名是 arena 自己生成的 UUID，PostHog / Langfuse 都查不到，arena 库里也没有来源记录"
      "（`arena_dataset_audios.audio_context` / `vocabulary` / `tone_app_id` 全空，也不在 `arena_audio_benchmark_datasets` 里）。"
      "arena_019f6442 在 arena 池子里有原始 audio_id（`gt_provenance.audio_id`，105/105），其中 34 条能在 PostHog 直接查到；"
      "arena 自己 2026-07-17 的检查也记的是 `no_posthog_event`。")
    w("- 文本匹配：把 PostHog 2026-05-18 → 07-23 的全部 asr_metrics（48,817 条，排除 arena 重放）拿来，"
      "arena 标注和线上 `transcription` / `processed` 归一化后 SequenceMatcher ≥ 0.85，且第一名比第二名（不同 audio_id）高 ≥ 0.05 才算。"
      "有 1 个账号在 2026-06 → 07 把数据集音频重放了一遍（命中 451 条标注，其中 84% 不是第一个说这句话的账号），"
      "它的请求不算候选。音频时长在这段时间基本没记（1,702/48,817 条有），没法用来加验证。")
    w("- 验证：arena_019f6442 里池子 audio_id 能在 PostHog 查到的 34 条，文本匹配独立地给出了**同一个 audio_id 的有 31 条**，"
      "另 3 条文本匹配没下结论（2 条只剩重放账号、1 条多个账号说过同一句），**没有一条给出不同的 audio_id**。")
    w("- 没找回的：`replay_account_only` = 只有重放账号说过这句话（原始请求不在 PostHog 里，可能早于 05-18）；"
      "`no_candidate` = 没有足够相似的线上文本；`same_user_repeats` / `several_users` = 同一句话有多次请求，分不清是哪一条。"
      "arena_019f6442 没找回的 70 条仍然在明细里保留了池子里的 audio_id（`audio_id_source = arena_pool_provenance`），上下文为空。\n")
    w("### 1.2 按平台\n")
    plat = collections.defaultdict(list)
    for r in allm:
        plat[r.get("client_source") or "?"].append(r)
    w("| 平台 | 条数 | " + " | ".join(f for f, _ in FIELDS) + " |")
    w("|" + "---|" * (len(FIELDS) + 2))
    for k, rs in sorted(plat.items(), key=lambda x: -len(x[1])):
        w(f"| {k} | {len(rs)} | " + " | ".join(f"{sum(has(r, f) for r in rs) / len(rs):.0%}" for _, f in FIELDS) + " |")
    w("\n`macos(inferred)`：2026-08 的事件很多没有 `client_source`，有 `audio_input` / `screen_text` 的只可能来自 macOS 客户端；"
      "`unknown` 是推断不出来的：主要是 arena_019f8e51 文本匹配到的 2026-07 请求（那时不记平台，`tone_app_id` 看多是 Cursor / Chrome / Claude 桌面版，"
      "大概率是 macOS），其余是 2026-08 的早期 iOS / Android 请求。\n")
    seg = data["segmented_long"]
    w("### 1.3 segmented_long\n")
    w(f"{len(seg)} 段（只算切自 benchmark_chinese 的），全部能对到父音频的线上请求。父音频的 before-cursor 只对第 00 段成立，"
      "after-cursor 只对最后一段成立；用作训练输入时要按段处理，或者只给第一段 / 最后一段。\n")

    w("## 2. 文本字段长度（字符，只算非空）\n")
    w("| 字段 | 非空条数 | p50 | p95 | max |")
    w("|---|---|---|---|---|")
    for name, k in [("before-cursor", "text_before_cursor"), ("after-cursor", "text_after_cursor"),
                    ("屏幕文字", "screen_text"), ("选中文字", "selected_text"), ("自定义 tone", "tone_prompt")]:
        v = [len(r[k]) for r in allm if nonempty(r.get(k))]
        w(f"| {name} | {len(v)} | {pct(v, .5)} | {pct(v, .95)} | {max(v) if v else None} |")
    ds = [r["dict_size"] for r in allm if r.get("dict_size")]
    cs = [r["ctx_vocab_size"] for r in allm if r.get("ctx_vocab_size")]
    w("")
    w(f"before-cursor 和屏幕文字的上限看起来是客户端截断的（before 最长 2000、屏幕 1800 字符）。"
      f"用户词典大小（有词典的 {len(ds)} 条）：p50 {pct(ds, .5)}，p95 {pct(ds, .95)}，max {max(ds) if ds else None}。"
      f"macOS 屏幕词（`audio_context.vocabulary`，{len(cs)} 条）：p50 {pct(cs, .5)}，p95 {pct(cs, .95)}，max {max(cs) if cs else None}（上限 30）。")
    acc = [r for r in allm if re.search(r"Screen Reader|not accessible|accessibility help|Terminal \d+,", r.get("text_before_cursor") or "")]
    w(f"噪声：{len(acc)} 条 before-cursor 是 VS Code / Cursor 终端的无障碍提示文字（“Terminal 3, … Toggle Screen Reader …”），不是用户写的内容。\n")

    w("## 3. 词典有没有用\n")
    w("看词典里的词有没有出现在我们的人工标注里（不区分大小写；拉丁词按词边界匹配，中英之间空格可有可无）。"
      "benchmark_v2 还没有人工标注，用线上 ASR 原文代替，所以不计入“全部”。\n")
    w("| 来源 | 有用户词典的条数 | 至少 1 个词出现在标注里 | 词条命中率（出现的词 / 全部词） | 有屏幕词的条数 | 屏幕词至少 1 个出现 | 屏幕词命中率 |")
    w("|---|---|---|---|---|---|---|")
    for k in main + ["全部（人工标注）"]:
        rs = human if k.startswith("全部") else [r for r in data[k] if r["matched"]]
        d = [r for r in rs if r.get("dict_size")]
        c = [r for r in rs if r.get("ctx_vocab_size")]
        if not d and not c:
            continue
        def hit(xs, a_, b_):
            if not xs:
                return "—", "—"
            n = sum(bool(r[a_]) for r in xs)
            return f"{n} ({n / len(xs):.0%})", f"{sum(len(r[a_]) for r in xs) / max(1, sum(r[b_] for r in xs)):.1%}"
        h1, h2 = hit(d, "dict_terms_in_label", "dict_size")
        h3, h4 = hit(c, "ctx_vocab_in_label", "ctx_vocab_size")
        w(f"| {k} | {len(d)} | {h1} | {h2} | {len(c)} | {h3} | {h4} |")
    w("")
    users = collections.Counter(tuple(sorted(r["effective_vocabulary"])) for r in dict_rows)
    junk = collections.Counter()
    for r in allm:
        for t in r.get("effective_vocabulary") or []:
            if len(t) > 40:
                junk["长句（>40 字符）"] += 1
            elif re.fullmatch(r"[\W\d_]+", t):
                junk["纯数字 / 符号"] += 1
            elif re.search(r"@|\.com|https?:", t):
                junk["邮箱 / 网址"] += 1
    w(f"- 有词典的 {len(dict_rows)} 条人工标注样本里，只有 {len(users)} 种不同的词典（同一个用户的每条请求带的是同一份词典），"
      f"最常见的一份占 {users.most_common(1)[0][1] if users else 0} 条。词典里还混着测试串、纯数字、邮箱、整句：{dict(junk)}。")
    fr = [r["ctx_vocab_in_screen_frac"] for r in allm if r.get("ctx_vocab_in_screen_frac") is not None]
    if fr:
        w(f"- `audio_context.vocabulary`（只有 macOS）平均 {sum(fr) / len(fr):.0%} 的词能在同一请求的 screen_text 里找到，"
          "其余像是文件路径、窗口标题之类——它是客户端从屏幕 / 窗口里抽的词，**不是**用户词典。")
    terms = sum(len(r.get("label_terms") or []) for r in human)
    hit_ = sum(len(r.get("label_terms_in_context") or []) for r in human)
    w(f"- 对比：人工标注里的专名（含大写字母的拉丁词，如 Langfuse、PR、iOS）共 {terms} 个，其中 {hit_}（{hit_ / max(1, terms):.0%}）"
      "能在同一条的 before / after-cursor 或屏幕文字里找到——**光标附近的文字比词典更像有用的 hotword 来源**（按来源分见 §7 表）。\n")

    w("## 4. App → tone 映射（bundle id join）\n")
    via = collections.Counter(r.get("tone_map_via") for r in allm)
    tones = collections.Counter(r.get("tone_mapped") or "（未映射）" for r in allm)
    w(f"join 规则（`bundle_to_app.tsv` 的 `match_via`）：先查脚本里的 bundle id 别名表（`alias_bundle`，只收确定的），"
      f"再按线上显示名精确匹配 TSV（`name_exact`）或常见别名（`name_alias`，如 Feishu → Lark / 飞书、Code → VS Code）；"
      f"没有 App 信息的记为“无 app 信息”（tone = unknown）。我们的匹配条数 {len(allm)}：{dict(via)}；tone 分布 {dict(tones)}。\n")
    w(f"- 我们的数据：{m_our}/{tot_our} 条（{m_our / max(1, tot_our):.0%}）能对上。")
    w(f"- 线上 30 天 dictate（缓存 scan，10-05 → 11-04，{tot_prod} 条）：{m_prod / max(1, tot_prod):.0%} 能对上；bundle id 共 {len(bt)} 个，对上 "
      f"{sum(b['matched'] == 'yes' for b in bt)} 个（长尾很长）。")
    w("- TSV 里有 4 行没有任何 bundle id 能对上：muse.ai（网页）、www.reeval.io（网页）、Meta 内部工具、SocialAI。"
      "前两个是网页——macOS 浏览器请求有一部分带 `app.url`，可以按域名对；Android Chrome 不带网址。")
    w("- 注意 `com.openai.codex` 在线上的显示名是 “ChatGPT”（macOS），按 bundle id 映射成了 Codex（两者都是 neutral，不影响 tone）。\n")
    w("**没对上的 App，按我们数据里的条数排（前 20）**\n")
    w("| bundle id | 线上显示名 | 我们的条数 | 线上 30 天请求 |")
    w("|---|---|---|---|")
    for b in sorted([b for b in bt if b["matched"] == "no" and int(b["clips_in_our_data"])],
                    key=lambda b: -int(b["clips_in_our_data"]))[:20]:
        w(f"| `{b['bundle_id']}` | {b['app_name_logged']} | {b['clips_in_our_data']} | {b['prod_requests_30d']} |")
    w("\n**没对上的 App，按线上 30 天请求量排（前 20）**\n")
    w("| bundle id | 线上显示名 | 线上 30 天请求 | 我们的条数 |")
    w("|---|---|---|---|")
    for b in sorted([b for b in bt if b["matched"] == "no"], key=lambda b: -int(b["prod_requests_30d"]))[:20]:
        w(f"| `{b['bundle_id']}` | {b['app_name_logged']} | {b['prod_requests_30d']} | {b['clips_in_our_data']} |")
    w("")
    tid = collections.Counter(r.get("tone_id") for r in allm)
    w(f"另外，线上 `tone_id` 是用户在 VoiceCursor 里给这个 App 选的 tone（不是我们的映射）：{dict(tid.most_common(6))}；"
      f"有 `tone_prompt`（自定义 tone 文本）的只有 {sum(nonempty(r.get('tone_prompt')) for r in allm)} 条。\n")

    w("## 5. 实际发给 ASR 模型的上下文（Langfuse `gen_ai.transcribe` 的 input）\n")
    lfr = [r for r in allm if r.get("langfuse_found")]
    w(f"有 Langfuse trace 的：{len(lfr)}/{len(allm)}（2026-09 之前的 PostHog 事件没有 `langfuse_trace_id`，benchmark_v1 和大部分 benchmark_chinese 因此没有）；"
      f"能看到 ASR 请求输入的：{len(sent)}。按实际发出请求的模型分：\n")
    grp = collections.defaultdict(list)
    for r in sent:
        grp[r["asr_request_model"]].append(r)
    w("| ASR 模型 | 条数 | prompt（前 2） | 有用户词典 | 其中词典发出去了 | 有 before-cursor | 其中发出去了 | 有 after-cursor | 其中发出去了 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for mdl, rs in sorted(grp.items(), key=lambda x: -len(x[1])):
        d = [r for r in rs if r.get("dict_size")]
        prom = collections.Counter(r.get("asr_prompt") or "（无 prompt）" for r in rs).most_common(2)
        hb = [r for r in rs if r.get("text_before_cursor")]
        ha = [r for r in rs if r.get("text_after_cursor")]
        w(f"| {mdl} | {len(rs)} | {', '.join(f'{k} ×{v}' for k, v in prom)} | {len(d)} | {sum(bool(r.get('sent_vocabulary')) for r in d)} | "
          f"{len(hb)} | {sum(bool(r.get('sent_before_cursor')) for r in hb)} | {len(ha)} | {sum(bool(r.get('sent_after_cursor')) for r in ha)} |")
    w("")
    w("- 格式：`transcription-for-multimodality` v31（及 -chinese v25）用 `<APPLICATION_CONTEXT>`（一句 App 描述）+ `<CURSOR_CONTEXT>`"
      "（JSON：`before_cursor_suffix` / `after_cursor_prefix`）+ `<VOCABULARY_CONTEXT>`（`keywords`）；旧版 v14 / v18 写在 system prompt 的 "
      "`<CURSOR_CONTEXT_CONTINUITY>` 里（TEXT_BEFORE_CURSOR / TEXT_AFTER_CURSOR，截断）；qwen 流式发 `before_cursor_suffix:` / "
      "`after_cursor_prefix:` 文本 + `vocabulary {词: 权重}`；gpt-live 只发 `keywords` 和 `languages`。")
    w("- **屏幕文字（screen_text）在任何请求里都没有发出去**；`<APPLICATION_CONTEXT>` 里只有 App 类别描述，不含 App 名以外的上下文。")
    w(f"- 少数请求发出去的词表 = 用户词典 + macOS 屏幕词（{sum(1 for r in sent if r.get('sent_vocabulary') and not set(r['sent_vocabulary']) <= set(r.get('effective_vocabulary') or []))} 条）。")
    w("- 同一请求可能有两条 `gen_ai.transcribe`（对冲请求，没用上的那条 input 为空），表里按有 input 的那条算。\n")

    w("## 6. 光标后文字（text after cursor）\n")
    w("**有记录。** 字段：PostHog `asr_metrics.properties.audio_context.text_after_cursor`（Langfuse 根 span input 里同一份）。")
    w(f"- 我们的数据：{len(tac)}/{len(allm)} 条非空；按平台见 §1.2。线上 30 天按平台：macOS 约 45–55%，Android 4–5%，iOS 3–6%，Windows / web 0。")
    w("- 内容：光标后面的输入框文字（macOS 常常只有换行或一个句号，因为光标在末尾；在文档中间插入时才有实质内容）。"
      f"长度 p50 {pct([len(r['text_after_cursor']) for r in tac], .5)}、p95 {pct([len(r['text_after_cursor']) for r in tac], .95)} 字符。"
      f"只含空白 / 标点的占 {sum(1 for r in tac if not WORDISH.search(r['text_after_cursor'])) / max(1, len(tac)):.0%}。")
    w("- 发给 ASR：v31 `<CURSOR_CONTEXT>` 的 `after_cursor_prefix`、qwen 的 `after_cursor_prefix:`、旧版 v18 的 TEXT_AFTER_CURSOR（见 §5）。\n")

    w("## 7. before-cursor 什么时候抓、会不会泄露答案\n")
    w("用缓存的 30 天 dictate scan（`data/private/online_scan/bv2_days`），看同一用户、同一 App、90 秒内的连续两次听写（只算后一次 before-cursor 非空的）：\n")
    w("| 平台 | 连续对数 | 后一次的 before-cursor 里有前一次的输出 | 请求数 | 本条输出已在自己的 before-cursor 里 | 本条输出已在自己的 after-cursor 里 |")
    w("|---|---|---|---|---|---|")
    for cs_ in ("android", "macos", "ios"):
        w(f"| {cs_} | {p_[(cs_, 'n')]} | {p_[(cs_, 'prev_in_tbc')] / max(1, p_[(cs_, 'n')]):.0%} | {o_[(cs_, 'n')]} | "
          f"{o_[(cs_, 'in_tbc')] / max(1, o_[(cs_, 'n')]):.1%} | {o_[(cs_, 'in_tac')] / max(1, o_[(cs_, 'n')]):.1%} |")
    w("")
    w("结论：before-cursor 是**按下录音那一刻**的输入框内容，上一句听写已经插进去了，本句还没插（不到 40% 的连续对里没看到上一句，"
      "多半是用户发出 / 清空了输入框）。我们自己数据里的情况：\n")
    L.extend(by_source_leak(data, main))
    w("")
    w("- recording_v2 / v2_test：owner 是在 VS Code 里打开朗读稿（每行 `V0xx 句子`）录的，**要读的句子本身就在 before / after-cursor 和屏幕文字里**。"
      "这两个集合的上下文等于给了答案，训练 / 评测都不能直接用，至少要把朗读稿行删掉。")
    w("- recording_v1：在 Apple Notes 里一句接一句录，before-cursor 是前面几句的 ASR 输出（同一句重录时会包含上一遍的结果）。")
    w("- “专名在上下文里出现”的比例高，一部分是真实的有用信号（用户在同一个对话里反复提到同一个词），recording_v2 那部分是泄露。\n")

    w("## 8. 例子（10 条，已匿名、截断）\n")
    w("| 来源 | 平台 | App → tone | before-cursor（末尾） | after-cursor | 用户词典 | 人工标注 |")
    w("|---|---|---|---|---|---|---|")
    for r in pick_examples(allm, 10):
        w(f"| {r['source']} | {r.get('client_source')} | {anon(r.get('app_name') or '-', 20, scrub)} → {r.get('tone_mapped') or '-'} | "
          f"{anon(tail(r.get('text_before_cursor'), 40), 45, scrub)} | {anon(r.get('text_after_cursor'), 30, scrub)} | "
          f"{anon(', '.join((r.get('effective_vocabulary') or [])[:3]), 36, scrub)} | {anon(r.get('label'), 50, scrub)} |")
    w("")

    w("## 9. 端到端训练样本可以长什么样（3 条真实例子，已匿名）\n")
    w("输入 = 音频 + 下面的 context；输出 = 我们的严格转写（strict transcript）。之后要做最终成品文本时，再加一个 target。"
      "字段都来自 PostHog 原始记录；`tone` 是按 TSV 映射的，不是线上 `tone_id`。\n")
    for r in pick_examples(allm, 3, rich=True):
        sample = {"audio": f"{r['source']}/{r['utterance_id']}.wav ({r.get('duration_sec') or 0:.1f}s)",
                  "context": {"hotwords": [anon(t, 30, scrub) for t in (r.get("effective_vocabulary") or [])][:8],
                              "screen_terms": [anon(t, 30, scrub) for t in (r.get("ctx_vocabulary") or [])][:8],
                              "text_before_cursor": anon(tail(r.get("text_before_cursor"), 160), 170, scrub),
                              "text_after_cursor": anon(r.get("text_after_cursor"), 80, scrub),
                              "app": r.get("app_name"), "tone": r.get("tone_mapped"),
                              "mode": r.get("mode") or r.get("mode_inferred"), "languages": r.get("languages")},
                  "target_strict_transcript": anon(r.get("label"), 200, scrub)}
        w("```json\n" + json.dumps(sample, ensure_ascii=False, indent=1) + "\n```\n")
    L.append(FIELD_DEFS)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {REPORT}")


def tail(s: str | None, n: int = 50) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def pick_examples(rows: list[dict], n: int, rich: bool = False) -> list[dict]:
    """Deterministic spread over sources / platforms; short clips; skip recording_v2* (script leakage). rich=True wants
    before-cursor + a dictionary or screen terms (e2e format samples)."""
    def score(r):
        return sum(has(r, k) for _, k in FIELDS) + (2 if r.get("dict_terms_in_label") else 0)
    cand = [r for r in rows if r["label_is_human"] and r.get("label") and (r.get("duration_sec") or 99) <= 15
            and not r["source"].startswith("recording_v2")]
    cand.sort(key=lambda r: (-score(r), r["utterance_id"]))
    if rich:
        cand = [r for r in cand if r.get("text_before_cursor") and (r.get("effective_vocabulary") or r.get("ctx_vocabulary"))]
    out, seen = [], collections.Counter()
    for r in cand:
        k = r["source"] if rich else (r["source"], r.get("client_source"))
        if seen[k] >= 1:
            continue
        seen[k] += 1
        out.append(r)
        if len(out) == n:
            break
    return out


FIELD_DEFS = """## 10. 字段定义和注意事项

| 明细 jsonl 里的列 | 来源 | 含义 |
|---|---|---|
| `text_before_cursor` | PostHog `audio_context.text_before_cursor`（= Langfuse 根 span input） | 按下录音时输入框里光标前的文字（最长约 2000 字符，取末尾）。上一句听写已插入（§7） |
| `text_after_cursor` | `audio_context.text_after_cursor` | 光标后的文字（§6）。主要是 macOS |
| `selected_text` | `audio_context.selected_text` | 选中的文字；dictate 时几乎总是空 |
| `screen_text` | `audio_context.screen_text` | **只有 macOS**：当前窗口 / 输入框的可见文字（最长约 1800 字符），常常 = 输入框全文（before + after），拼进模型输入会重复 |
| `placeholder_text` / `participants` / `subject` | `audio_context.*` | macOS 少量：输入框占位文字、邮件参与人、邮件主题 |
| `ctx_vocabulary` | `audio_context.vocabulary` | **只有 macOS**：客户端从屏幕 / 窗口抽的词（最多 30 个），不是用户词典（§3） |
| `effective_vocabulary` / `use_dictionary` / `effective_vocabulary_count` | PostHog | **用户词典**：用户在 VoiceCursor 里加的词，服务端整理后的那份；列表只在非空时记录 |
| `app_bundle_id` / `app_name` / `app_url` / `app_field_kind` / `app_return_action` | `audio_context.app.*` | 目标 App；macOS 浏览器部分带 url；Android / iOS 带输入框类型（text / search …）和回车键类型；iOS 很多拿不到 bundle id |
| `app_type` | PostHog | 服务端的 App 粗分类（other / browser / chat / code …），决定 `<APPLICATION_CONTEXT>` 那句描述 |
| `tone_id` / `tone_source` / `tone_app_id` / `tone_prompt` | PostHog | 用户在 VoiceCursor 里选的 tone（default / other_off / personal_very_casual / 自定义 id）；`tone_prompt` 只在自定义 tone 时有 |
| `tone_mapped` / `tone_scenario` / `tone_map_app` / `tone_map_via` | 派生 | 按 `configs/reference/tone_by_app.tsv` 映射；join 见 `bundle_to_app.tsv` |
| `mode` / `mode_inferred` / `requested_mode` | PostHog | dictate / edit；2026-09 前没有 `mode`，`mode_inferred` 按有没有 `edit_action` 推断 |
| `languages` / `lang` / `lang_source` / `ctx_language` | PostHog + `audio_context.language` | `languages` = 用户设置的语言列表（2026-09 起）；`lang` = 路由用的单一语言（`lang_source=fallback` 表示没设置）；`ctx_language` = 客户端带的语言设置字符串（如 `en,zh-CN`） |
| `client_source` | PostHog | 平台；早期缺失时按 `audio_input` / `screen_text` 推断为 `macos(inferred)` |
| `asr_model` / `prod_transcription` / `prod_processed` | PostHog | 线上 ASR 路由、ASR 原文、后处理成品 |
| `asr_request_model` / `asr_prompt` / `sent_blocks` / `sent_vocabulary` / `sent_before_cursor` / `sent_after_cursor` | Langfuse `gen_ai.transcribe` | 真正发给 ASR 的内容（§5） |
| `dict_terms_in_label` / `ctx_vocab_in_label` / `label_terms` / `label_terms_in_context` / `label_in_*` | 派生 | 词典词 / 屏幕词 / 专名在标注里的命中；标注是否已出现在上下文里（泄露检查） |

注意事项：
- 时间缺口：PostHog `asr_metrics` 从 2026-05-18 开始；我们的线上数据最早是 2026-08-13（benchmark_v1）。2026-09 之前的事件没有 `mode`、`languages`、`langfuse_trace_id`，很多没有 `client_source`。
- 平台缺口：Windows / web 基本没有上下文；iOS 多数没有 bundle id 和 `ctx_language`；屏幕文字和屏幕词只有 macOS。
- arena 重放（client_source = arena）用的是评测账号，没有原用户的词典和上下文，已排除，只用原始请求。
- recording_v1 和 recording_freeform_test 在 test 环境，只有 Langfuse 记录，没有 `languages` / `tone_id` 等 PostHog 字段。
- before-cursor 在 VS Code / Cursor 终端里常常是无障碍提示文字，不是用户内容（§2）。
- 本报告的“词典 / 专名命中”只看标注文本，没有看音频；没有命中不代表那个词对识别完全没用（例如可以用来防止误插）。
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--no-langfuse-fallback", dest="langfuse_fallback", action="store_false")
    p = sub.add_parser("traces")
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("textmatch")
    p.add_argument("--refresh", action="store_true")
    sub.add_parser("build")
    sub.add_parser("report")
    a = ap.parse_args()
    {"pull": cmd_pull, "traces": cmd_traces, "textmatch": cmd_textmatch, "build": cmd_build, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
