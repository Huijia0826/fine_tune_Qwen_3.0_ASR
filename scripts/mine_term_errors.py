"""Term errors for the recording script (2026-11-04): which English words / names inside Chinese dictation our models get
wrong, and how often production users say them.

Evidence (human-verified labels only, never a test or val set — recording data picked from test errors would inflate
the test):
  batch1  online_pool batch 1 (451 ok/fix clips): owner's final_transcript vs the C step 600 draft (C had not been
          trained on batch 1, so these are held-out errors)
  phase1  online_phase1 TRAIN clips (230): owner's label vs the zero-shot whisper-large-v3 draft
Per clip the label and the draft are aligned (zh characters / English words, case-insensitive); every error run whose
label side contains a Latin word is a term error: label term → what the model wrote.
Frequency: Latin words in the production ASR text of zh / zh-en dictation (online.py scans, 2026-08-31 → 11-04), counted
as distinct users (one heavy user should not make a word "frequent").
Output: data/private/recording_script/term_errors.csv (term, errors, variants, sources) and term_freq.csv.
Usage: uv run python scripts/mine_term_errors.py
"""

import collections
import csv
import glob
import json
import re
from pathlib import Path

from whisperflow.eval.metrics import align_text

OUT = Path("data/private/recording_script")
LAT = re.compile(r"[A-Za-z][A-Za-z0-9.+'\-]*")
CJK = re.compile(r"[一-鿿]")


def term_errors(ref: str, hyp: str) -> list[tuple[str, str]]:
    """(label term, model output) for each error run whose label side has a Latin word."""
    out, run = [], []
    ops = align_text(ref, hyp).ops + [("eq", None, None)]
    for op, r, h in ops:
        if op != "eq":
            run.append((r, h))
            continue
        if run:
            rs = [x for x, _ in run if x is not None]
            hs = [y for _, y in run if y is not None]
            if any(LAT.fullmatch(x or "") for x in rs) and " ".join(rs).lower() != " ".join(hs).lower():
                out.append((" ".join(t for t in rs if LAT.fullmatch(t)), "".join(hs) if all(CJK.match(y) for y in hs) else " ".join(hs)))
            run = []
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errs = collections.defaultdict(lambda: {"n": 0, "variants": collections.Counter(), "src": collections.Counter()})
    b1 = "data/private/online_pool/online-pool-batch1/reviewed/review_2026-11-03_full.csv"
    for r in csv.DictReader(open(b1, encoding="utf-8-sig")):
        if r["verdict"] in ("ok", "fix"):
            for t, h in term_errors(r["final_transcript"], r["draft_1"]):
                e = errs[t.lower()]; e["n"] += 1; e["variants"][h or "(漏掉)"] += 1; e["src"]["batch1"] += 1
    drafts = {d["audio_id"]: d["drafts"]["whisper:large-v3"]
              for d in map(json.loads, open("data/private/online_phase1/bench/online-zh-phase1/drafts.jsonl", encoding="utf-8"))}
    for r in map(json.loads, open("data/manifests/online_phase1.jsonl", encoding="utf-8")):
        if r["split"] == "train" and r["source_utterance_id"] in drafts:
            for t, h in term_errors(r["raw_transcript"], drafts[r["source_utterance_id"]]):
                e = errs[t.lower()]; e["n"] += 1; e["variants"][h or "(漏掉)"] += 1; e["src"]["phase1"] += 1

    users = collections.defaultdict(set)
    seen = set()
    for p in glob.glob("data/private/online_scan/*.jsonl") + glob.glob("data/private/online_scan/bv2_days/*.jsonl"):
        for r in map(json.loads, open(p, encoding="utf-8")):
            if "_scan" in r or r.get("audio_id") in seen or r.get("mode") != "dictate":
                continue
            seen.add(r.get("audio_id"))
            t = r.get("transcription") or ""
            if CJK.search(t):
                for w in set(m.group(0).lower() for m in LAT.finditer(t)):
                    users[w].add(r["user_id"])
    with open(OUT / "term_freq.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "users"])
        for t, u in sorted(users.items(), key=lambda x: -len(x[1])):
            w.writerow([t, len(u)])
    with open(OUT / "term_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "errors", "prod_users", "variants", "sources"])
        for t, e in sorted(errs.items(), key=lambda x: (-x[1]["n"], -len(users.get(x[0].split()[0], ())))):
            w.writerow([t, e["n"], len(users.get(t, ())), json.dumps(e["variants"].most_common(4), ensure_ascii=False),
                        json.dumps(dict(e["src"]))])
    print(f"term errors: {sum(e['n'] for e in errs.values())} runs, {len(errs)} distinct terms; "
          f"production zh/zh-en clips scanned: {len(seen)}, Latin words: {len(users)}")


if __name__ == "__main__":
    main()
