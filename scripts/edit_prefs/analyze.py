"""Analyse edit-mode behaviour: volume, instruction taxonomy, per-user preference stability, dictation -> edit
linkage, custom-tone overlap. Rules only (no LLM calls). Reads data/private/edit_prefs/*.jsonl from pull.py.

Outputs (data/private/edit_prefs/):
  edits_labeled.jsonl  one row per real edit (after exclusions) with category, language, linkage
  pairs.jsonl          dictation output -> edited result pairs (linked edits)
  user_prefs.jsonl     per-user category counts, detected preference, period-split stability
  stats.json           all numbers used in reports/edit_prefs/edit_behaviour.md
Usage: python3 scripts/edit_prefs/analyze.py
"""
from __future__ import annotations

import collections
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from classify import ONE_OFF, STYLE, labels, primary, text_lang  # noqa: E402
from test_utterances import TestDetector  # noqa: E402

D = os.path.join(HERE, "..", "..", "data", "private", "edit_prefs")
SALT = "vc-edit-prefs-v0"  # pseudonymisation only
TUTORIAL_BUNDLE = "com.voicecursor.android.type"
INTERNAL_USERS = {"user_3Fq6DfuCmTOmUEWjKZLHTkfqlXu"}  # internal test account (reports/postprocess_data/candidates_v0.md)
# fixed onboarding sample texts that also show up outside the tutorial bundle
TUTORIAL_TEXT = re.compile(r"^\W*(jason[,，]|joão[,，]|milk[,\n]|leite[,\n]|牛奶|hi tina|tina[,，]|"
                           r"here'?s my to-do list|o chuveiro|the shower is leaking|淋浴)", re.I)
PERIOD1 = ("2026-10-12", "2026-10-26")  # weeks 1-2 of the last 28 days
PERIOD2 = ("2026-10-26", "2026-11-09")  # weeks 3-4
LINK_MIN = (2, 10, 30)
MEANINGFUL_EXCLUDE = {"empty", "non_edit"}


def uhash(u: str) -> str:
    return hashlib.sha256(f"{SALT}:{u}".encode()).hexdigest()[:12]


def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm(t: str | None) -> str:
    return re.sub(r"[\W_]+", "", (t or "").lower())


def load(name: str) -> list[dict]:
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f][1:]


def similar(sel: str, out: str) -> str | None:
    """How the selected text relates to a dictation output: exact / contained / fuzzy / None."""
    a, b = norm(sel), norm(out)
    if len(a) < 2 or len(b) < 2:
        return None
    if a == b:
        return "exact"
    if (a in b and len(a) >= max(4, 0.3 * len(b))) or (b in a and len(b) >= max(4, 0.5 * len(a))):
        return "contained"
    if min(len(a), len(b)) >= 6 and difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.8:
        return "fuzzy"
    return None


def main() -> None:
    edits_raw = load("edits.jsonl")
    dic = load("dictate.jsonl")
    ustats = load("user_stats.jsonl")
    S: dict = {}

    # ---------- test detection (dictations of edit users + instructions) ----------
    det = TestDetector(dic)
    testers = det.testers | INTERNAL_USERS

    def test_like(text: str | None) -> bool:
        k = det.kind({"transcription": text or "", "user_id": ""})
        return k in ("counting", "test_script")

    funnel = collections.Counter()
    real = []
    for r in edits_raw:
        funnel["0_all_non_arena"] += 1
        if r["bundle_id"] == TUTORIAL_BUNDLE:
            funnel["1_tutorial_bundle"] += 1; continue
        if r["invocation_context"] == "onboarding_practice":
            funnel["2_onboarding_practice"] += 1; continue
        if TUTORIAL_TEXT.search(r["selected_text"] or ""):
            funnel["3_tutorial_sample_text"] += 1; continue
        if not r["edit_action"]:
            funnel["4_cancel_or_disconnect"] += 1; continue
        if r["user_id"] in testers:
            funnel["5_tester_or_internal_account"] += 1; continue
        instr = (r["intent"] or {}).get("instruction")
        if test_like(instr) or test_like(r["selected_text"]):
            funnel["6_test_utterance"] += 1; continue
        funnel["7_kept"] += 1
        real.append(r)
    S["funnel"] = dict(sorted(funnel.items()))
    S["testers"] = len(testers)

    # ---------- user language from dictations ----------
    dic_ok = [d for d in dic if d["processed"] and d["bundle_id"] != TUTORIAL_BUNDLE
              and d["invocation_context"] != "onboarding_practice" and d["user_id"] not in testers]
    ul = collections.defaultdict(collections.Counter)
    for d in dic_ok:
        l = text_lang(d["processed"])
        if l:
            ul[d["user_id"]][l] += 1
    for r in real:
        i = (r["intent"] or {}).get("instruction")
        r["instr_lang"] = text_lang(i)
        r["sel_lang"] = text_lang(r["selected_text"])
    user_lang = {}
    for u in {r["user_id"] for r in real}:
        c = ul.get(u)
        if not c or sum(c.values()) < 3:
            c = collections.Counter(x for r in real if r["user_id"] == u for x in (r["instr_lang"], r["sel_lang"]) if x)
        if not c:
            user_lang[u] = "unknown"; continue
        zh = c["zh"] + c["zh-en"]
        tot = sum(c.values())
        # a user counts as Chinese if >= 30% of their text is Chinese (many zh users also type English)
        if zh / tot >= 0.3:
            user_lang[u] = "zh-en" if c["zh-en"] >= c["zh"] else "zh"
        else:
            user_lang[u] = c.most_common(1)[0][0]
    for r in real:
        r["user_lang"] = user_lang[r["user_id"]]
        r["lang_group"] = "zh" if r["user_lang"] in ("zh", "zh-en") else r["user_lang"]

    # ---------- labels ----------
    for r in real:
        it = r["intent"] or {}
        r["has_instruction_event"] = bool(r["intent"])
        r["instruction"] = it.get("instruction")
        r["category"] = primary(it.get("instruction"), it.get("intent_category")) if r["intent"] else "no_log"
        r["all_labels"] = labels(it.get("instruction"))
        r["style"] = r["category"] in STYLE
        r["t"] = ts(r["timestamp"])

    # ---------- volume ----------
    def vol(rows, key):
        c, u = collections.Counter(), collections.defaultdict(set)
        for r in rows:
            k = key(r); c[k] += 1; u[k].add(r["user_id"])
        return {k: {"requests": v, "users": len(u[k])} for k, v in c.most_common()}
    S["volume_total"] = {"requests": len(real), "users": len({r["user_id"] for r in real}),
                         "apply_edit": sum(r["edit_action"] == "apply_edit" for r in real),
                         "no_op": sum(r["edit_action"] == "no_op" for r in real),
                         "with_instruction_event": sum(r["has_instruction_event"] for r in real),
                         "with_nonempty_instruction": sum(bool((r["instruction"] or "").strip()) for r in real),
                         "with_selected_text": sum(bool(r["selected_text"]) for r in real),
                         "with_output": sum(bool(r["processed"]) for r in real),
                         "with_langfuse_trace_id": sum(bool(r["langfuse_trace_id"]) for r in real),
                         "with_s3_audio": sum(bool(r["s3_key"]) for r in real)}
    S["volume_platform"] = vol(real, lambda r: r["platform"])
    S["volume_lang"] = vol(real, lambda r: r["user_lang"])
    S["volume_app"] = dict(list(vol(real, lambda r: r["app_name"] or r["bundle_id"] or "?").items())[:20])
    S["volume_week"] = vol(real, lambda r: (r["t"] - dt.timedelta(days=r["t"].weekday())).strftime("%Y-%m-%d"))
    first_instr = min(r["timestamp"] for r in real if r["has_instruction_event"])
    S["first_instruction_event"] = first_instr
    S["no_op_reason"] = dict(collections.Counter(r["edit_no_op_reason"] for r in real if r["edit_action"] == "no_op"))
    S["intent_category"] = dict(collections.Counter((r["intent"] or {}).get("intent_category") for r in real
                                                    if r["has_instruction_event"]))
    # instruction text vs transcription_result
    same = sum(norm((r["intent"] or {}).get("instruction")) == norm((r["intent"] or {}).get("transcription_result"))
               for r in real if r["has_instruction_event"])
    S["instruction_equals_transcription_result"] = same
    tte = sum(norm((r["intent"] or {}).get("text_to_edit")) == norm(r["selected_text"]) for r in real if r["has_instruction_event"])
    S["text_to_edit_equals_selected"] = tte
    eo = sum(norm((r["intent"] or {}).get("edited_text")) == norm(r["processed"]) for r in real if r["has_instruction_event"])
    S["edited_text_equals_processed"] = eo

    # ---------- taxonomy (rows with instruction event) ----------
    tx = [r for r in real if r["has_instruction_event"]]
    def shares(rows):
        c = collections.Counter(r["category"] for r in rows)
        n = len(rows)
        return {"n": n, "users": len({r["user_id"] for r in rows}),
                "cat": {k: [v, round(100 * v / n, 1)] for k, v in c.most_common()},
                "style_share": round(100 * sum(r["style"] for r in rows) / max(n, 1), 1)}
    S["taxonomy_all"] = shares(tx)
    S["taxonomy_by_lang"] = {g: shares([r for r in tx if r["lang_group"] == g]) for g in ("zh", "en", "other", "unknown")}
    def user_shares(rows):
        uc = collections.defaultdict(set)
        for r in rows:
            uc[r["category"]].add(r["user_id"])
        n = len({r["user_id"] for r in rows})
        return {"users": n, "pct_users_asking": {k: round(100 * len(v) / n, 1) for k, v in
                                                 sorted(uc.items(), key=lambda x: -len(x[1]))},
                "top1_user_share_of_requests": {k: round(100 * collections.Counter(r["user_id"] for r in rows
                                                                                   if r["category"] == k).most_common(1)[0][1]
                                                         / sum(r["category"] == k for r in rows), 1) for k in uc}}
    S["taxonomy_users_by_lang"] = {g: user_shares([r for r in tx if r["lang_group"] == g]) for g in ("zh", "en", "other")}
    S["taxonomy_users_all"] = user_shares(tx)
    S["taxonomy_zh_split"] = {g: shares([r for r in tx if r["user_lang"] == g]) for g in ("zh", "zh-en")}
    S["taxonomy_applied"] = {k: round(100 * sum(r["edit_action"] == "apply_edit" for r in tx if r["category"] == k)
                                      / max(1, sum(r["category"] == k for r in tx)), 1)
                             for k in STYLE + ONE_OFF}
    # translate targets
    tgt = collections.Counter()
    for r in tx:
        if r["category"] == "translate":
            i = (r["instruction"] or "").lower()
            for name, rx in [("en", r"english|ingl|英文|英语|英語"), ("zh", r"chinese|中文|chin[eê]s|mandarin|简体|繁体"),
                             ("es", r"spanish|espanhol|español|西班牙|西语"), ("pt", r"portugu|葡萄牙|葡语"),
                             ("ja", r"japan|日语|日文|japon"), ("ko", r"korean|韩|corean")]:
                if re.search(rx, i):
                    tgt[(r["lang_group"], name)] += 1; break
            else:
                tgt[(r["lang_group"], "?")] += 1
    S["translate_targets"] = {f"{a}->{b}": v for (a, b), v in tgt.most_common()}
    # explicit "from now on" preference statements
    fut = re.compile(r"以后|今后|from now on|always|every time|sempre que|daqui pra frente|de agora em diante|下次", re.I)
    S["explicit_future_statements"] = [{"user": uhash(r["user_id"]), "lang": r["lang_group"], "cat": r["category"],
                                        "instruction": (r["instruction"] or "")[:120]}
                                       for r in tx if fut.search(r["instruction"] or "")]

    # ---------- linkage dictate -> edit ----------
    by_user = collections.defaultdict(list)
    for d in dic_ok:
        d["t"] = ts(d["timestamp"])
        by_user[d["user_id"]].append(d)
    for u in by_user:
        by_user[u].sort(key=lambda d: d["t"])
    edits_by_user = collections.defaultdict(list)
    for r in sorted(real, key=lambda r: r["t"]):
        edits_by_user[r["user_id"]].append(r)
    pairs = []
    for r in real:
        r["link"] = None
        if not r["selected_text"]:
            continue
        best = None
        for d in reversed(by_user.get(r["user_id"], [])):
            if d["t"] >= r["t"]:
                continue
            gap = (r["t"] - d["t"]).total_seconds() / 60
            if gap > 24 * 60:
                break
            how = similar(r["selected_text"], d["processed"])
            if how:
                best = (d, gap, how); break
        # re-edit: selected text equals output of the user's previous edit
        prev_edit = None
        for e in reversed(edits_by_user[r["user_id"]]):
            if e["t"] < r["t"] and (r["t"] - e["t"]).total_seconds() <= 30 * 60 and e["processed"] \
                    and similar(r["selected_text"], e["processed"]) in ("exact", "contained"):
                prev_edit = e; break
        r["re_edit"] = bool(prev_edit)
        if best:
            d, gap, how = best
            r["link"] = {"gap_min": round(gap, 2), "how": how, "same_app": d["bundle_id"] == r["bundle_id"],
                         "dictate_audio_id": d["audio_id"], "dictate_model": d["model"],
                         "dictate_pp_model": d["postprocess_model"], "dictate_tone_id": d["tone_id"],
                         "dictate_has_custom_tone": bool(d["tone_prompt"])}
            d.setdefault("edited_by", []).append(r)
            if r["edit_action"] == "apply_edit":
                pairs.append({"user": uhash(r["user_id"]), "lang_group": r["lang_group"], "category": r["category"],
                              "all_labels": r["all_labels"], "instruction": r["instruction"],
                              "dictate_transcription": d["transcription"], "dictate_output": d["processed"],
                              "selected_text": r["selected_text"], "edited_result": r["processed"],
                              "gap_min": round(gap, 2), "how": how, "same_app": d["bundle_id"] == r["bundle_id"],
                              "app": r["app_name"] or r["bundle_id"], "platform": r["platform"],
                              "edit_audio_id": r["audio_id"], "dictate_audio_id": d["audio_id"],
                              "timestamp": r["timestamp"]})
    with_sel = [r for r in real if r["selected_text"]]
    def link_rate(rows, minutes, same_app=False):
        k = sum(1 for r in rows if r["link"] and r["link"]["gap_min"] <= minutes and (r["link"]["same_app"] or not same_app))
        return [k, round(100 * k / max(1, len(rows)), 1)]
    S["linkage"] = {f"<= {m} min": link_rate(with_sel, m) for m in LINK_MIN}
    S["linkage"]["<= 30 min same app"] = link_rate(with_sel, 30, True)
    S["linkage"]["<= 24 h"] = link_rate(with_sel, 24 * 60)
    S["linkage"]["re_edit_of_previous_edit_30min"] = [sum(r.get("re_edit", False) for r in with_sel), None]
    S["linkage_how"] = dict(collections.Counter(r["link"]["how"] for r in with_sel if r["link"] and r["link"]["gap_min"] <= 30))
    S["linkage_by_cat"] = {}
    for k in STYLE + ONE_OFF + ["no_log"]:
        rows = [r for r in with_sel if r["category"] == k]
        if rows:
            S["linkage_by_cat"][k] = {"n": len(rows), "linked_30min": link_rate(rows, 30)}
    S["linkage_by_lang"] = {g: {"n": len(rs), "linked_30min": link_rate(rs, 30)}
                            for g in ("zh", "en", "other") for rs in [[r for r in with_sel if r["lang_group"] == g]]}
    S["linked_dictate_route"] = dict(collections.Counter(
        ("custom_tone" if r["link"]["dictate_has_custom_tone"] else
         "llm_pp" if (r["link"]["dictate_pp_model"] or "") .startswith(("gemini-3.5-flash-lite", "gpt-5")) else "other")
        for r in with_sel if r["link"] and r["link"]["gap_min"] <= 30))

    # ---------- per-user preference ----------
    uprefs = []
    for u, rows in edits_by_user.items():
        m = [r for r in rows if r["has_instruction_event"] and r["category"] not in MEANINGFUL_EXCLUDE]
        c = collections.Counter(r["category"] for r in m)
        sc = collections.Counter(r["category"] for r in m if r["style"])
        top, topn = (sc.most_common(1)[0] if sc else (None, 0))
        def per(p):
            return collections.Counter(r["category"] for r in m if r["style"] and p[0] <= r["timestamp"][:10] < p[1])
        p1, p2 = per(PERIOD1), per(PERIOD2)
        # exact same instruction repeated (macro-like)
        ninstr = collections.Counter(norm(r["instruction"]) for r in m if norm(r["instruction"]))
        rep = ninstr.most_common(1)[0][1] if ninstr else 0
        nd = len(by_user.get(u, []))
        top_days = len({r["timestamp"][:10] for r in m if r["category"] == top}) if top else 0
        uprefs.append({"user": uhash(u), "_uid": u, "lang": user_lang[u], "platform": rows[0]["platform"],
                       "n_edits": len(rows), "n_meaningful": len(m), "n_style": sum(sc.values()),
                       "cats": dict(c), "top_style": top, "top_style_n": topn,
                       "top_share_meaningful": round(topn / len(m), 2) if m else 0, "top_style_days": top_days,
                       "p1": dict(p1), "p2": dict(p2), "max_same_instruction": rep, "n_dictations": nd,
                       "days_active_edit": len({r["timestamp"][:10] for r in rows})})
    # A user can hold several learned preferences: (user, style category) pairs.
    PREF_MIN_N, PREF_MIN_DAYS, PREF_MIN_SHARE = 3, 2, 0.3
    for p, (u, rows) in zip(uprefs, edits_by_user.items()):
        m = [r for r in rows if r["has_instruction_event"] and r["style"]]
        sc = collections.Counter(r["category"] for r in m)
        days = {k: len({r["timestamp"][:10] for r in m if r["category"] == k}) for k in sc}
        p["prefs"] = sorted([k for k, n in sc.items() if n >= PREF_MIN_N and days[k] >= PREF_MIN_DAYS
                             and n / len(m) >= PREF_MIN_SHARE], key=lambda k: -sc[k])
        p["pref_n"] = {k: sc[k] for k in p["prefs"]}
        p["pref_days"] = {k: days[k] for k in p["prefs"]}
        # spec-literal rule: >= 3 meaningful edits and >= 50% of them in one style category
        p["stable_spec"] = p["n_meaningful"] >= 3 and p["top_style_n"] >= 2 and p["top_share_meaningful"] >= 0.5
        p["stable"] = bool(p["prefs"])
        p["stable_strict"] = any(sc[k] >= 5 and days[k] >= 3 and sc[k] / len(m) >= 0.5 for k in p["prefs"])
    lg = lambda p: "zh" if p["lang"] in ("zh", "zh-en") else p["lang"]
    pref_pairs = [(p, k) for p in uprefs for k in p["prefs"]]
    S["users"] = {
        "edit_users": len(uprefs),
        "users_with_instruction": sum(1 for p in uprefs if p["n_meaningful"] >= 1),
        ">=3 meaningful edits": sum(p["n_meaningful"] >= 3 for p in uprefs),
        ">=3 style edits": sum(p["n_style"] >= 3 for p in uprefs),
        "same style cat >=3 times": sum(p["top_style_n"] >= 3 for p in uprefs),
        "stable_spec (>=3 meaningful edits, top style cat >=50% of them)": sum(p["stable_spec"] for p in uprefs),
        "users with >=1 learned pref [main: cat >=3 edits, >=2 days, >=30% of style edits]": sum(p["stable"] for p in uprefs),
        "users with strict pref (>=5 edits, >=3 days, >=50%)": sum(p["stable_strict"] for p in uprefs),
        "users with >=2 prefs": sum(len(p["prefs"]) >= 2 for p in uprefs),
        "edit_users_by_lang": dict(collections.Counter(lg(p) for p in uprefs)),
        "pref_users_by_lang": dict(collections.Counter(lg(p) for p in uprefs if p["stable"])),
        "stable_spec_by_top": dict(collections.Counter(p["top_style"] for p in uprefs if p["stable_spec"])),
    }
    S["edits_per_user_hist"] = dict(collections.Counter(
        "1" if p["n_edits"] == 1 else "2" if p["n_edits"] == 2 else "3-4" if p["n_edits"] <= 4 else
        "5-9" if p["n_edits"] <= 9 else "10-29" if p["n_edits"] <= 29 else "30+" for p in uprefs))
    total_style = sum(r["style"] for r in tx)
    st_users = [p for p in uprefs if p["stable"]]
    covered = sum(p["pref_n"][k] for p, k in pref_pairs)
    S["stable_coverage"] = {
        "users": len(st_users), "pref_pairs": len(pref_pairs),
        "their_edits": sum(p["n_edits"] for p in st_users),
        "edits_in_pref_categories": covered,
        "share_of_all_real_edits": round(100 * covered / len(real), 1),
        "share_of_style_edits": round(100 * covered / max(1, total_style), 1),
        "share_of_edit_users": round(100 * len(st_users) / len(uprefs), 1),
        "pairs_by_cat": dict(collections.Counter(k for _, k in pref_pairs).most_common()),
        "pairs_by_lang_cat": {f"{a}|{b}": v for (a, b), v in
                              collections.Counter((lg(p), k) for p, k in pref_pairs).most_common()},
        "edits_by_lang_cat": {f"{a}|{b}": v for (a, b), v in
                              sum((collections.Counter({(lg(p), k): p["pref_n"][k]}) for p, k in pref_pairs),
                                  collections.Counter()).most_common()},
        "macro_like_users (same instruction >=3x)": sum(p["max_same_instruction"] >= 3 for p in st_users),
        "top5_users_share_of_covered_edits": round(100 * sum(sorted((sum(p["pref_n"].values()) for p in st_users),
                                                                    reverse=True)[:5]) / max(1, covered), 1),
    }
    # stability across periods (weeks 1-2 vs 3-4)
    def p_prefs(c):
        n = sum(c.values())
        return {k for k, v in c.items() if v >= 2 and v / n >= PREF_MIN_SHARE}
    rows_p1 = [(p, k) for p in uprefs for k in p_prefs(collections.Counter(p["p1"]))]
    with_p2 = [(p, k) for p, k in rows_p1 if sum(p["p2"].values()) >= 1]
    p2_users = {r["user_id"] for r in real if PERIOD2[0] <= r["timestamp"][:10] < PERIOD2[1]}
    both = [p for p in uprefs if sum(p["p1"].values()) >= 2 and sum(p["p2"].values()) >= 2]
    same_top = sum(collections.Counter(p["p1"]).most_common(1)[0][0] == collections.Counter(p["p2"]).most_common(1)[0][0]
                   for p in both)
    S["stability"] = {
        "p1_prefs (cat >=2 edits, >=30% of user's p1 style edits)": len(rows_p1),
        "p1_prefs_users": len({p["user"] for p, _ in rows_p1}),
        "of_which_user_made_any_edit_in_p2": sum(1 for p, _ in rows_p1 if p["_uid"] in p2_users),
        "of_which_user_made_style_edit_in_p2": len(with_p2),
        "of_which_same_cat_again_in_p2": sum(1 for p, k in with_p2 if p["p2"].get(k)),
        "p2_style_edits_of_those_users": sum(sum(p["p2"].values()) for p in {p["user"]: p for p, _ in with_p2}.values()),
        "p2_style_edits_in_p1_pref_cat": sum(p["p2"].get(k, 0) for p, k in with_p2),
        "users_>=2_style_edits_in_both_periods": len(both),
        "of_which_same_top_category": same_top,
    }
    # ---------- Q5: would the preference have helped ----------
    def units(t):
        return len(re.findall(r"[一-鿿]", t or "")) + len(re.findall(r"[A-Za-zÀ-ɏ0-9']+", t or ""))
    def lb(n):
        return "<20" if n < 20 else "20-59" if n < 60 else "60-149" if n < 150 else ">=150"
    q5, cond = [], collections.defaultdict(lambda: [0, 0])
    for p, k in pref_pairs:
        u = p["_uid"]
        ds = by_user.get(u, [])
        hits = [any(e["category"] == k for e in d.get("edited_by", [])) for d in ds]
        pref_edits = [r for r in edits_by_user[u] if r["category"] == k]
        apps = collections.Counter(r["bundle_id"] for r in pref_edits)
        top_app = apps.most_common(1)[0][0] if apps else None
        for d, hit in zip(ds, hits):
            for key in [(k, "len", lb(units(d["processed"]))),
                        (k, "app", "top_pref_app" if d["bundle_id"] == top_app else "other_apps")]:
                cond[key][0] += 1; cond[key][1] += hit
        q5.append({"user": p["user"], "lang": lg(p), "pref": k, "n_dict": len(ds), "n_dict_edited_pref": sum(hits),
                   "n_dict_edited_any": sum(1 for d in ds if d.get("edited_by")),
                   "n_pref_edits": len(pref_edits), "pref_edits_linked": sum(1 for r in pref_edits if r["link"]),
                   "top_app_share": round(apps.most_common(1)[0][1] / len(pref_edits), 2) if pref_edits else None})
    S["q5_conditional"] = {"|".join(k): [v[0], v[1], round(100 * v[1] / max(1, v[0]), 1)] for k, v in sorted(cond.items())}
    shares_ = sorted(100 * x["n_dict_edited_pref"] / x["n_dict"] for x in q5 if x["n_dict"])
    S["q5"] = {"pref_pairs": len(q5), "users": len({x["user"] for x in q5}),
               "their_dictations (sum over pairs)": sum(x["n_dict"] for x in q5),
               "dictations_later_edited_with_pref": sum(x["n_dict_edited_pref"] for x in q5),
               "pref_edits": sum(x["n_pref_edits"] for x in q5),
               "pref_edits_linked_to_dictation": sum(x["pref_edits_linked"] for x in q5),
               "median_pair_share_pct": round(shares_[len(shares_) // 2], 1) if shares_ else None,
               "p75_pair_share_pct": round(shares_[int(len(shares_) * 0.75)], 1) if shares_ else None,
               "pairs_with_share_>=10pct": sum(1 for s_ in shares_ if s_ >= 10),
               "median_top_app_share": sorted(x["top_app_share"] for x in q5 if x["top_app_share"] is not None)[len(q5) // 2],
               "by_pref": {}}
    for k in sorted({x["pref"] for x in q5}):
        xs = [x for x in q5 if x["pref"] == k]
        S["q5"]["by_pref"][k] = {"pairs": len(xs), "dict": sum(x["n_dict"] for x in xs),
                                 "dict_edited_pref": sum(x["n_dict_edited_pref"] for x in xs),
                                 "pref_edits": sum(x["n_pref_edits"] for x in xs),
                                 "pref_edits_linked": sum(x["pref_edits_linked"] for x in xs)}
    with open(os.path.join(D, "pref_pairs.jsonl"), "w", encoding="utf-8") as f:
        for x in q5:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    # ---------- Q6: custom tone ----------
    ct_users = {s["user_id"] for s in ustats if s["n_custom_tone"] > 0} - testers
    all_users = {s["user_id"] for s in ustats} - testers
    edit_users = {r["user_id"] for r in real}
    st_ids = {p["_uid"] for p in st_users}
    ct_prompt = {}
    for d in dic:
        if d["tone_prompt"]:
            ct_prompt.setdefault(d["user_id"], d["tone_prompt"])
    for r in edits_raw:
        if r["tone_prompt"]:
            ct_prompt.setdefault(r["user_id"], r["tone_prompt"])
    dict_n = {s["user_id"]: s["n_dictate_or_edit"] - s["n_edit"] for s in ustats}
    real_n = collections.Counter(r["user_id"] for r in real)
    def edit_rate(users):
        d = sum(max(0, dict_n.get(u, 0)) for u in users)
        e = sum(real_n.get(u, 0) for u in users)
        return {"users": len(users), "edit_users": len(users & edit_users),
                "pct_users_editing": round(100 * len(users & edit_users) / max(1, len(users)), 1),
                "edits_per_100_dictations": round(100 * e / max(1, d), 2)}
    S["custom_tone"] = {
        "custom_tone_users": edit_rate(ct_users), "other_users": edit_rate(all_users - ct_users),
        "stable_pref_users_with_custom_tone": len(st_ids & ct_users),
        "stable_pref_users": len(st_ids),
        "tone_prompt_mentions_their_pref": sum(1 for p in st_users if p["_uid"] in ct_prompt
                                               and set(p["prefs"]) & set(labels(ct_prompt[p["_uid"]]))),
        "custom_tone_editors_style_cats": dict(collections.Counter(
            r["category"] for r in tx if r["user_id"] in ct_users).most_common()),
        "custom_tone_edit_labels_in_tone_prompt": None,
    }
    # do custom-tone users' edits ask for something their tone prompt already says?
    m = n = 0
    for r in tx:
        if r["user_id"] in ct_prompt and r["style"]:
            n += 1
            m += r["category"] in labels(ct_prompt[r["user_id"]])
    S["custom_tone"]["custom_tone_edit_labels_in_tone_prompt"] = [m, n]
    # preference already declared in a custom tone but the edited dictation ran without it
    decl = collections.Counter()
    decl_users = set()
    for r in tx:
        u = r["user_id"]
        if u in ct_prompt and r["style"] and r["category"] in labels(ct_prompt[u]):
            decl["style_edits_matching_declared_tone"] += 1
            decl_users.add(u)
            if r["link"]:
                decl["linked"] += 1
                decl["linked_source_had_custom_tone" if r["link"]["dictate_has_custom_tone"] else
                     "linked_source_without_custom_tone"] += 1
    decl["users"] = len(decl_users)
    tone_cov = {}
    for u in ct_users & edit_users:
        ds = by_user.get(u, [])
        if ds:
            tone_cov[u] = sum(bool(d["tone_prompt"]) for d in ds) / len(ds)
    v = sorted(tone_cov.values())
    decl["custom_tone_editors_median_share_of_dictations_with_tone"] = round(100 * v[len(v) // 2], 1) if v else None
    S["custom_tone"]["declared_vs_applied"] = dict(decl)
    S["linked_edits_with_custom_tone_dictation"] = sum(1 for r in real if r["link"] and r["link"]["dictate_has_custom_tone"])

    # ---------- write ----------
    with open(os.path.join(D, "edits_labeled.jsonl"), "w", encoding="utf-8") as f:
        for r in real:
            o = {k: v for k, v in r.items() if k not in ("t", "intent", "user_id")}
            o["user"] = uhash(r["user_id"])
            o["intent_category"] = (r["intent"] or {}).get("intent_category")
            f.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
    with open(os.path.join(D, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(D, "user_prefs.jsonl"), "w", encoding="utf-8") as f:
        for p in sorted(uprefs, key=lambda p: -p["n_edits"]):
            f.write(json.dumps({k: v for k, v in p.items() if k != "_uid"}, ensure_ascii=False) + "\n")
    S["pairs_written"] = len(pairs)
    S["pairs_by_cat"] = dict(collections.Counter(p["category"] for p in pairs).most_common())
    S["pairs_by_lang_cat"] = {f"{a}|{b}": v for (a, b), v in
                              collections.Counter((p["lang_group"], p["category"]) for p in pairs).most_common(30)}
    with open(os.path.join(D, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(S, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
