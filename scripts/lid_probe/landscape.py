"""Steps 1-2 of the en-setting probe: language-setting landscape + text-based lower bound (metadata only).

Reads the 30-day dictate day scans (see common.py), drops onboarding / tests / tester accounts, then writes
  data/private/lid_probe/population.jsonl   one slim row per kept request (input to sampling)
  data/private/lid_probe/landscape.json     all numbers used in the report
and prints markdown tables.
Usage: python3 scripts/lid_probe/landscape.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, SETTING_LABEL, SETTING_ORDER, duration_s, exclusion, load_rows, setting_class, text_class, text_stats  # noqa: E402

PLATFORMS = ["android", "ios", "macos", "windows", "web"]


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def main() -> None:
    rows = load_rows()
    why = exclusion(rows)
    n_all = len(rows)
    kept = [r for r in rows if r["audio_id"] not in why]
    has_asr = [r for r in kept if r.get("model") not in (None, "unknown", "None") and r.get("languages")]
    res = {"n_dictate": n_all, "excluded": dict(collections.Counter(why.values())), "n_kept": len(kept),
           "n_with_asr": len(has_asr)}
    print(f"dictate {n_all}; excluded {len(why)} {res['excluded']}; kept {len(kept)}; with ASR + setting {len(has_asr)}")

    slim = []
    for r in has_asr:
        sc = setting_class(r["languages"])
        raw, out = r.get("transcription") or "", r.get("processed") or ""
        s_raw, s_out = text_stats(raw), text_stats(out)
        slim.append({"audio_id": r["audio_id"], "user_id": r["user_id"], "platform": r.get("client_source"),
                     "model": r.get("model"), "languages": r["languages"], "lang": r.get("lang"), "setting": sc,
                     "dur": round(duration_s(r), 2), "raw_class": text_class(raw), "out_class": text_class(out),
                     "han_raw": s_raw["han"], "han_out": s_out["han"], "words_raw": s_raw["latin_words"],
                     "transcription": raw, "processed": out, "s3_key": r.get("s3_key"),
                     "pp_model": r.get("postprocess_model"), "pp_prompt": r.get("postprocess_langfuse_prompt_name"),
                     "app": r.get("app"), "ts": r.get("timestamp"), "langfuse": r.get("langfuse"),
                     "flags": r.get("flags")})
    with open(OUT / "population.jsonl", "w", encoding="utf-8") as f:
        for s in slim:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- step 1: settings landscape
    N = len(slim)
    by_set = collections.defaultdict(list)
    for s in slim:
        by_set[s["setting"]].append(s)
    user_sets = collections.defaultdict(collections.Counter)
    for s in slim:
        user_sets[s["user_id"]][s["setting"]] += 1
    primary = {u: c.most_common(1)[0][0] for u, c in user_sets.items()}
    U = len(user_sets)
    land = []
    print("\n| 设置 | 请求数 | 占比 | 用户数（用过） | 用户数（主设置） | 占用户 | " + " | ".join(PLATFORMS) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(PLATFORMS))
    plat_tot = collections.Counter(s["platform"] for s in slim)
    for k in SETTING_ORDER:
        rs = by_set.get(k, [])
        users_any = len({s["user_id"] for s in rs})
        users_pri = sum(1 for v in primary.values() if v == k)
        pc = collections.Counter(s["platform"] for s in rs)
        land.append({"setting": k, "requests": len(rs), "users_any": users_any, "users_primary": users_pri,
                     "by_platform": {p: pc.get(p, 0) for p in PLATFORMS}})
        print(f"| {SETTING_LABEL[k]} | {len(rs)} | {pct(len(rs), N)} | {users_any} | {users_pri} | {pct(users_pri, U)} | "
              + " | ".join(pct(pc.get(p, 0), plat_tot[p]) for p in PLATFORMS) + " |")
    print(f"| 合计 | {N} | | {U} | | | " + " | ".join(str(plat_tot[p]) for p in PLATFORMS) + " |")
    res.update({"n_requests": N, "n_users": U, "landscape": land, "platform_totals": dict(plat_tot),
                "users_multi_setting": sum(1 for c in user_sets.values() if len(c) > 1)})

    # routed ASR model per setting class
    route = {}
    print("\n| 设置 | 路由到的 ASR 模型（占该设置请求） |\n|---|---|")
    for k in SETTING_ORDER:
        rs = by_set.get(k, [])
        if not rs:
            continue
        mc = collections.Counter(s["model"] for s in rs)
        route[k] = dict(mc.most_common(8))
        print(f"| {SETTING_LABEL[k]} | " + "，".join(f"{m} {v / len(rs):.0%}" for m, v in mc.most_common(5)) + " |")
    res["routing"] = route

    # ---- step 2: lower bound within en_only
    en = by_set["en_only"]
    n_en = len(en)
    han1 = [s for s in en if max(s["han_raw"], s["han_out"]) >= 1]
    han2 = [s for s in en if s["raw_class"] in ("zh", "zh-en") or s["out_class"] in ("zh", "zh-en")]
    rc = collections.Counter(s["raw_class"] for s in en)
    oc = collections.Counter(s["out_class"] for s in en)
    en_users = {s["user_id"] for s in en}
    han_users = collections.Counter(s["user_id"] for s in han2)
    en_req_per_user = collections.Counter(s["user_id"] for s in en)
    lb = {"n_en_only": n_en, "users_en_only": len(en_users), "any_han_char": len(han1), "zh_or_zhen_text": len(han2),
          "raw_class": dict(rc), "out_class": dict(oc), "users_with_han": len(han_users),
          "han_by_platform": dict(collections.Counter(s["platform"] for s in han2)),
          "en_by_platform": dict(collections.Counter(s["platform"] for s in en)),
          "han_by_model": dict(collections.Counter(s["model"] for s in han2)),
          "en_by_model": dict(collections.Counter(s["model"] for s in en).most_common(8))}
    # concentration of the Han-text requests among users
    shares = sorted(((han_users[u] / en_req_per_user[u], han_users[u], en_req_per_user[u]) for u in han_users), reverse=True)
    lb["users_han_share_ge50"] = sum(1 for x in shares if x[0] >= 0.5)
    lb["users_han_share_ge10"] = sum(1 for x in shares if x[0] >= 0.1)
    lb["users_han_ge3"] = sum(1 for x in shares if x[1] >= 3)
    top = sorted(han_users.values(), reverse=True)
    lb["han_top5_users_share"] = sum(top[:5]) / max(len(han2), 1)
    lb["han_top20_users_share"] = sum(top[:20]) / max(len(han2), 1)
    # requests from "Chinese-speaking en-only users" (>=1 zh/zh-en text request) whose text is Latin: where hidden
    # translations could sit
    zh_users = set(han_users)
    lat_zhusers = [s for s in en if s["user_id"] in zh_users and s["raw_class"] == "latin" and s["out_class"] == "latin"]
    lb["latin_requests_of_zh_users"] = len(lat_zhusers)
    lb["requests_of_zh_users"] = sum(en_req_per_user[u] for u in zh_users)
    # raw has Han but output does not (post-processing translated) and the reverse
    lb["raw_han_out_latin"] = sum(1 for s in en if s["raw_class"] in ("zh", "zh-en") and s["out_class"] == "latin")
    lb["raw_latin_out_han"] = sum(1 for s in en if s["raw_class"] == "latin" and s["out_class"] in ("zh", "zh-en"))
    # comparison: other settings, share with zh/zh-en text
    comp = {}
    for k in SETTING_ORDER:
        rs = by_set.get(k, [])
        if rs:
            comp[k] = sum(1 for s in rs if s["raw_class"] in ("zh", "zh-en") or s["out_class"] in ("zh", "zh-en")) / len(rs)
    lb["zh_text_share_by_setting"] = comp
    res["lower_bound"] = lb
    print(f"\nen_only: {n_en} requests / {len(en_users)} users; any Han char {len(han1)} ({pct(len(han1), n_en)}); "
          f"zh or zh-en text {len(han2)} ({pct(len(han2), n_en)}); users {len(han_users)} ({pct(len(han_users), len(en_users))})")
    print("raw class", dict(rc.most_common()), "\nout class", dict(oc.most_common()))
    print("users share>=50%", lb["users_han_share_ge50"], ">=10%", lb["users_han_share_ge10"], ">=3 req", lb["users_han_ge3"],
          "top5", f"{lb['han_top5_users_share']:.0%}", "top20", f"{lb['han_top20_users_share']:.0%}")
    print("Latin-text requests of zh users", len(lat_zhusers), "of", lb["requests_of_zh_users"])
    print("raw Han -> out Latin", lb["raw_han_out_latin"], "; raw Latin -> out Han", lb["raw_latin_out_han"])
    print("Han by platform", lb["han_by_platform"], "\nHan by model", lb["han_by_model"])
    print("zh text share by setting", {k: f"{v:.2%}" for k, v in comp.items()})
    json.dump(res, open(OUT / "landscape.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
