"""Steps 3d-4: spoken-language classes from Qwen3-ASR (auto LID), prevalence with CIs, per-user view, damage proxy.

Inputs: data/private/lid_probe/sample.jsonl (sample.py), qwen_lid.jsonl (run_qwen_lid.py on the cluster),
population.jsonl (landscape.py), optional failure_overrides.json ({audio_id: {"category": ..., "note": ...}} from a
manual read of the flagged clips).
Outputs: data/private/lid_probe/clips.jsonl (one row per clip, all fields) and analysis.json (all report numbers).

Spoken class (Qwen is NOT ground truth):
  none   Qwen transcript empty (silence / noise)
  zh     Qwen tag Chinese/Cantonese, or transcript script class zh
  zh-en  transcript has >= 2 Chinese characters and >= 2 Latin words (common.text_class), or tag "Chinese,English"
  en     Latin transcript, tag English (or Chinese tag with no Chinese characters: Qwen auto does not translate)
Clips whose Qwen transcript is a mic check / counting (scripts/test_utterances.py rules) are dropped from the analysis.
  other  anything else (Portuguese, Japanese, ...)
Damage proxy: official MER of the production raw ASR text / final processed text against Qwen's transcript as a
pseudo-reference. Failure category of the production RAW text (rules, then manual overrides):
  ok              MER <= 15 %
  translated      reference has Chinese chars, production has < 20 % of them, >= 4 Latin words that read as English
                  (>= 25 % common English words)
  romanized       same script switch, but the Latin text is short or not fluent English: Chinese rendered as
                  English-ish / pinyin-like words ("买一个吧" -> "Maigoba")
  empty           production text empty while the reference is not
  dropped         production has < 60 % of the reference tokens (content missing)
  misrecognized   everything else above 15 % MER (wrong characters / words, script still right)
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, text_class, text_stats  # noqa: E402
from test_utterances import TestDetector  # noqa: E402
from whisperflow.eval.official import official_mer, official_mer_num, official_mer_t2s  # noqa: E402

BOOT, SEED = 2000, 7
ZH = ("zh", "zh-en")
EN_COMMON = set("""the a an and or but if so to of in on at for with from by as is are was were be been am i you he she
it we they me him her us them my your his its our their this that these those there here what which who when where why
how not no yes do does did have has had will would can could should just like okay ok yeah really very also then than
too all some any one about up out get got go going know think want need make let see say said time now well right""".split())


def spoken_class(q: dict) -> str:
    hyp = (q.get("hypothesis") or "").strip()
    tag = (q.get("detected_language") or "").lower()
    if not hyp:
        return "none"
    tc = text_class(hyp)
    zh_tag = "chinese" in tag or "cantonese" in tag
    if tc == "zh-en" or (zh_tag and "english" in tag and tc in ZH):
        return "zh-en"
    if tc == "zh" or (zh_tag and text_stats(hyp)["han"] >= 1):
        return "zh"
    if tc == "latin" and (tag in ("", "english") or zh_tag):  # Qwen auto does not translate: Latin text = English words
        return "en"
    return "other"


def n_tokens(t: str) -> int:
    s = text_stats(t)
    return s["han"] + s["latin_words"]


def category(ref: str, hyp: str, mer: float) -> str:
    if mer <= 0.15:
        return "ok"
    if not (hyp or "").strip():
        return "empty"
    sr, sh = text_stats(ref), text_stats(hyp)
    if sr["han"] >= 1 and sh["han"] < 0.2 * sr["han"] and sh["latin_words"] >= 1:
        words = re.findall(r"[a-z']+", hyp.lower())
        fluent = sum(w in EN_COMMON for w in words) / max(len(words), 1)
        return "translated" if fluent >= 0.25 and len(words) >= 4 else "romanized"
    if n_tokens(hyp) < 0.6 * n_tokens(ref):
        return "dropped"
    return "misrecognized"


def boot_ratio(groups: dict, rng, iters=BOOT):
    """groups: user -> list of (num, den, weight). Cluster bootstrap over users of sum(w*num)/sum(w*den)."""
    users = list(groups)
    arr = [(sum(w * a for a, b, w in groups[u]), sum(w * b for a, b, w in groups[u])) for u in users]
    arr = np.array(arr, dtype=float)
    if len(arr) == 0 or arr[:, 1].sum() == 0:
        return None, None, None
    est = arr[:, 0].sum() / arr[:, 1].sum()
    idx = rng.integers(0, len(arr), size=(iters, len(arr)))
    num, den = arr[idx, 0].sum(1), arr[idx, 1].sum(1)
    reps = num[den > 0] / den[den > 0]
    return est, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def main() -> None:
    sample = [json.loads(line) for line in open(OUT / "sample.jsonl", encoding="utf-8")]
    qwen = {d["audio_id"]: d for d in map(json.loads, open(OUT / "qwen_lid.jsonl", encoding="utf-8"))}
    ov_path = OUT / "failure_overrides.json"
    overrides = json.load(open(ov_path, encoding="utf-8")) if ov_path.exists() else {}
    design = json.load(open(OUT / "sample_design.json"))
    rng = np.random.default_rng(SEED)
    det = TestDetector([])  # counting / mic-check rules only, applied to Qwen's transcript
    clips = []
    for r in sample:
        q = qwen.get(r["audio_id"])
        if not q:
            continue
        c = dict(r)
        c.update({"qwen_lang": q["detected_language"], "qwen_text": q["hypothesis"], "spoken": spoken_class(q),
                  "test_like": det.kind({"transcription": q["hypothesis"]}) or ""})
        ref = q["hypothesis"]
        for side, txt in (("raw", r["transcription"]), ("out", r["processed"])):
            m = official_mer(ref, txt or "")
            m2 = official_mer_t2s(ref, txt or "")
            c[f"err_{side}"], c["n_ref"], c[f"err_{side}_t2s"] = m.errors, m.n_ref, m2.errors
            c[f"err_{side}_num"], c["n_ref_num"] = (lambda x: (x.errors, x.n_ref))(official_mer_num(ref, txt or ""))
            c[f"mer_{side}"] = m.errors / m.n_ref if m.n_ref else None
        c["category_rule"] = category(ref, r["transcription"], c["mer_raw"] or 0.0) if c["n_ref"] else "no_ref"
        o = overrides.get(r["audio_id"], {})
        c["category"] = o.get("category", c["category_rule"])
        c["category_note"] = o.get("note", "")
        clips.append(c)
    print(f"{len(clips)}/{len(sample)} clips have Qwen output")
    res = {"n_clips": len(clips), "test_like": {s: sum(1 for c in clips if c["stratum"] == s and c["test_like"])
                                                for s in ("H", "Z", "R", "C")},
           "test_like_clips": [(c["stratum"], c["spoken"], c["qwen_text"][:30], c["transcription"][:30])
                               for c in clips if c["test_like"]]}
    all_clips = clips
    clips = [c for c in clips if not c["test_like"]]  # mic checks the metadata filter could not see
    print("test-like (excluded):", res["test_like"])

    # ---- spoken class by stratum (clip counts) and weighted prevalence
    tab = {}
    for s in ("H", "Z", "R", "C"):
        cs = [c for c in clips if c["stratum"] == s]
        tab[s] = dict(collections.Counter(c["spoken"] for c in cs))
        print(s, len(cs), tab[s])
    res["spoken_by_stratum"] = tab

    def prevalence(target):
        """Request-level prevalence among en-only requests (1-60 s): Hajek within stratum, N-weighted across strata."""
        out = {}
        Ns = {s: design[s]["N"] for s in ("H", "Z", "R")}
        per = {}
        for s in ("H", "Z", "R"):
            g = collections.defaultdict(list)
            for c in clips:
                if c["stratum"] == s:
                    g[c["user_id"]].append((float(c["spoken"] in target), 1.0, c["weight"]))
            per[s] = g
            est, lo, hi = boot_ratio(g, rng)
            out[s] = {"est": est, "lo": lo, "hi": hi, "n": sum(len(v) for v in g.values()),
                      "k": int(sum(a for v in g.values() for a, _, _ in v))}
        # combined: bootstrap each stratum independently, then combine with N_s
        reps, hid = np.zeros(BOOT), np.zeros(BOOT)
        for s in ("H", "Z", "R"):
            users = list(per[s])
            arr = np.array([(sum(w * a for a, b, w in per[s][u]), sum(w * b for a, b, w in per[s][u])) for u in users])
            idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
            rep_s = Ns[s] * arr[idx, 0].sum(1) / arr[idx, 1].sum(1)
            reps += rep_s
            if s != "H":
                hid += rep_s
        tot = sum(Ns.values())
        point = sum(Ns[s] * out[s]["est"] for s in Ns) / tot
        out["all"] = {"est": point, "lo": float(np.percentile(reps / tot, 2.5)),
                      "hi": float(np.percentile(reps / tot, 97.5)), "N": tot,
                      "requests_est": point * tot}
        # without the text-visible stratum H: Chinese speech whose production text shows no Chinese at all
        hidden = (Ns["Z"] * out["Z"]["est"] + Ns["R"] * out["R"]["est"])
        out["hidden_requests_est"] = hidden
        out["hidden_requests_ci"] = [float(np.percentile(hid, 2.5)), float(np.percentile(hid, 97.5))]
        return out

    res["prevalence_zh_or_zhen"] = prevalence(ZH)
    res["prevalence_zh"] = prevalence(("zh",))
    res["prevalence_zhen"] = prevalence(("zh-en",))
    res["prevalence_other"] = prevalence(("other",))
    for k in ("prevalence_zh_or_zhen", "prevalence_zh", "prevalence_zhen", "prevalence_other"):
        p = res[k]
        print(k, {s: (round(v["est"], 4), round(v["lo"], 4), round(v["hi"], 4), v.get("k")) for s, v in p.items()
                  if isinstance(v, dict)}, "hidden≈", round(p["hidden_requests_est"]))

    # ---- per-user view (en-only users with sampled clips)
    pop = [json.loads(line) for line in open(OUT / "population.jsonl", encoding="utf-8")]
    en_pop = [p for p in pop if p["setting"] == "en_only" and 1 <= p["dur"] <= 60]
    zh_text_users = {p["user_id"] for p in pop if p["raw_class"] in ZH or p["out_class"] in ZH}

    def strat(p):
        if p["raw_class"] in ZH or p["out_class"] in ZH:
            return "H"
        return "Z" if p["user_id"] in zh_text_users else "R"

    n_us = collections.Counter((p["user_id"], strat(p)) for p in en_pop)
    users = collections.defaultdict(dict)
    for c in clips:
        if c["stratum"] in ("H", "Z", "R"):
            users[c["user_id"]].setdefault(c["stratum"], []).append(c["spoken"] in ZH)
    strat_mean = {s: res["prevalence_zh_or_zhen"][s]["est"] for s in ("H", "Z", "R")}
    uview = []
    for u in {p["user_id"] for p in en_pop if p["user_id"] in zh_text_users}:
        tot, zh = 0, 0.0
        for s in ("H", "Z"):
            n = n_us.get((u, s), 0)
            obs = users.get(u, {}).get(s)
            share = sum(obs) / len(obs) if obs else strat_mean[s]
            tot += n
            zh += n * share
        uview.append({"user": u, "requests": tot, "zh_share": zh / tot if tot else 0,
                      "han_text_requests": n_us.get((u, "H"), 0), "sampled": sum(len(v) for v in users.get(u, {}).values())})
    uview.sort(key=lambda x: -x["requests"])
    r_users = {c["user_id"] for c in clips if c["stratum"] == "R"}
    r_users_zh = {c["user_id"] for c in clips if c["stratum"] == "R" and c["spoken"] in ZH}
    res["users"] = {"en_only_users": len({p["user_id"] for p in en_pop}),
                    "zh_text_users": len(uview),
                    "zh_text_users_mostly_zh": sum(1 for x in uview if x["zh_share"] >= 0.5),
                    "zh_text_users_share_ge20": sum(1 for x in uview if x["zh_share"] >= 0.2),
                    "zh_text_users_requests": sum(x["requests"] for x in uview),
                    "zh_text_users_est_zh_requests": sum(x["requests"] * x["zh_share"] for x in uview),
                    "R_users_sampled": len(r_users), "R_users_with_zh_clip": len(r_users_zh),
                    "top": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k != "user"}
                            for x in uview[:25]]}
    print("users", {k: v for k, v in res["users"].items() if k != "top"})

    # ---- damage proxy
    def pooled(cs, side="raw", weighted=False, t2s=False, num=False):
        g = collections.defaultdict(list)
        for c in cs:
            if c["n_ref"]:
                err = c[f"err_{side}" + ("_t2s" if t2s else "_num" if num else "")]
                g[c["user_id"]].append((err, c["n_ref_num"] if num else c["n_ref"], c["weight"] if weighted else 1.0))
        est, lo, hi = boot_ratio(g, rng)
        return {"mer": est, "lo": lo, "hi": hi, "clips": sum(len(v) for v in g.values()), "users": len(g)}

    groups = {
        "en_only_zh": [c for c in clips if c["stratum"] != "C" and c["spoken"] == "zh"],
        "en_only_zhen": [c for c in clips if c["stratum"] != "C" and c["spoken"] == "zh-en"],
        "en_only_zh_all": [c for c in clips if c["stratum"] != "C" and c["spoken"] in ZH],
        "en_only_zh_all_H": [c for c in clips if c["stratum"] == "H" and c["spoken"] in ZH],
        "en_only_zh_all_ZR": [c for c in clips if c["stratum"] in ("Z", "R") and c["spoken"] in ZH],
        "control_zh": [c for c in clips if c["stratum"] == "C" and c["spoken"] == "zh"],
        "control_zhen": [c for c in clips if c["stratum"] == "C" and c["spoken"] == "zh-en"],
        "control_zh_all": [c for c in clips if c["stratum"] == "C" and c["spoken"] in ZH],
        "en_only_en (noise floor)": [c for c in clips if c["stratum"] != "C" and c["spoken"] == "en"],
        "control_en": [c for c in clips if c["stratum"] == "C" and c["spoken"] == "en"],
    }
    dmg = {}
    print("\n| 组 | 条 | 用户 | raw MER [95% CI] | final MER | raw MER (t2s) | raw MER (num) |")
    for k, cs in groups.items():
        a, b, t, nm = pooled(cs, "raw"), pooled(cs, "out"), pooled(cs, "raw", t2s=True), pooled(cs, "raw", num=True)
        dmg[k] = {"raw": a, "out": b, "raw_t2s": t, "raw_num": nm,
                  "categories": dict(collections.Counter(c["category"] for c in cs)),
                  "models": dict(collections.Counter(c["model"] for c in cs))}
        if a["mer"] is not None:
            print(f"| {k} | {a['clips']} | {a['users']} | {a['mer']:.1%} [{a['lo']:.1%}, {a['hi']:.1%}] | {b['mer']:.1%} | {t['mer']:.1%} | {nm['mer']:.1%} |",
                  dmg[k]["categories"])
    # weighted (request-level) for en-only Chinese speech
    dmg["en_only_zh_all_weighted"] = {"raw": pooled(groups["en_only_zh_all"], "raw", weighted=True),
                                      "out": pooled(groups["en_only_zh_all"], "out", weighted=True)}
    # by production model, Chinese speech
    bym = {}
    for grp in ("en_only_zh_all", "control_zh_all"):
        for m in sorted({c["model"] for c in groups[grp]}):
            cs = [c for c in groups[grp] if c["model"] == m]
            bym[f"{grp}|{m}"] = {"raw": pooled(cs, "raw"), "categories": dict(collections.Counter(c["category"] for c in cs))}
    dmg["by_model"] = bym
    for k, v in bym.items():
        if v["raw"]["mer"] is not None:
            print(k, v["raw"]["clips"], f"{v['raw']['mer']:.1%}", v["categories"])
    # per-clip severity buckets
    for grp in ("en_only_zh_all", "control_zh_all"):
        cs = [c for c in groups[grp] if c["n_ref"]]
        dmg[grp]["severity"] = {"<=15%": sum(c["mer_raw"] <= 0.15 for c in cs),
                                "15-40%": sum(0.15 < c["mer_raw"] <= 0.4 for c in cs),
                                ">40%": sum(c["mer_raw"] > 0.4 for c in cs)}
    res["damage"] = dmg
    json.dump(res, open(OUT / "analysis.json", "w"), ensure_ascii=False, indent=1, default=float)
    with open(OUT / "clips.jsonl", "w", encoding="utf-8") as f:
        for c in all_clips:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"-> {OUT / 'analysis.json'}, {OUT / 'clips.jsonl'}")


if __name__ == "__main__":
    main()
