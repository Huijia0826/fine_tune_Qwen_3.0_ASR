"""Build review items for the end-to-end dataset (owner, 2026-11-09) for scripts/build_e2e_review.py.

  dict  — the owner's dictionary (hotword) benchmark: 207 clips with ground truth, target vocabulary and the
          production audio_context (app, language, text before / after cursor, screen text, user
          vocabulary). Audio is copied to data/private/dictionary_bench/audio/ (independent of ~/Downloads).
  i2    — I2's data without ASCEND (owner 2026-11-09): manifests + the production context pulled by
          scripts/context_pull/pull_context.py (data/private/context_pull/<source>.jsonl, joined on utterance_id).
          --group val (users' val), prod (production-origin training clips), rest (segmented_long + arena).
  bv2   — benchmark v2 test candidates (owner, 2026-11-11): short (537) or long (55) part, not labelled yet. Prefill =
          production ASR text brought to policy v1 (old_label = the production text); our model's draft (D step 600 /
          I2 step 600, the drafts made when the candidates were picked) is shown for reference.
Prefills (owner): transcript = label brought to policy v1 (text/label_style.py; the old label is kept for the
diff), final_output = transcript (the owner writes the final output during review).

    uv run --with openpyxl python scripts/make_e2e_items.py dict <xlsx> <audio dir> \
        data/private/e2e/items_dict.jsonl
"""

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path

import soundfile as sf

from whisperflow.text.label_style import to_label_v1

TONE_TSV = Path("configs/reference/tone_by_app.tsv")


# production app names that the owner's table lists under another name
ALIASES = {"feishu": "lark", "teams": "microsoft teams", "notes": "apple notes", "messages": "messages",
           "brave browser": "brave"}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", " ", name.lower()).strip()


def load_tone_map() -> dict[str, str]:
    """app alias -> tone: 'WeChat 微信' → 'wechat 微信', 'wechat', '微信'; 'Lark / 飞书' → 'lark', '飞书'."""
    lines = csv.reader(open(TONE_TSV, encoding="utf-8"), delimiter="\t")
    rows = [r for r in lines if r and not r[0].startswith("#")]
    out = {}
    for tone, _scene, app in rows[1:]:
        for name in re.split(r"\s*/\s*", re.sub(r"（.*?）|\(.*?\)", "", app)):
            k = _key(name)
            for alias in {k, *re.findall(r"[a-z0-9 ]*[a-z0-9]", k), *re.findall(r"[一-鿿]+", k)}:
                if alias.strip():
                    out.setdefault(alias.strip(), tone)
    return out


def tone_for(app_name: str | None, tones: dict[str, str]) -> str | None:
    """Tone from the owner's table; None = the app is not in the table (different from the table's 'unknown')."""
    if not app_name:
        return "unknown"  # no app information: the table's "无 app 信息" row
    key = _key(app_name)
    key = ALIASES.get(key, key)
    if key in tones:
        return tones[key]
    hits = [k for k in tones if re.search(rf"(^|\s){re.escape(k)}($|\s)", key)]  # 'google chrome' → 'chrome'
    return tones[max(hits, key=len)] if hits else None


def as_list(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[;；\n]", str(v)) if x.strip()]


def build_dict(xlsx: str, audio_dir: str, out: str) -> None:
    import openpyxl

    ws = openpyxl.load_workbook(xlsx).active
    rows = list(ws.iter_rows(values_only=True))
    rows = [dict(zip(rows[0], r, strict=True)) for r in rows[1:]]
    tones = load_tone_map()
    dst_dir = Path("data/private/dictionary_bench/audio")
    dst_dir.mkdir(parents=True, exist_ok=True)
    items, unknown_apps = [], {}
    for r in rows:
        src = Path(audio_dir) / r["file_name"]
        dst = dst_dir / r["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
        ctx = json.loads(r["audio_context"] or "{}")
        app = ctx.get("app") or {}
        tone = tone_for(app.get("name"), tones)
        if tone is None:
            unknown_apps[app["name"]] = unknown_apps.get(app["name"], 0) + 1
            tone = "unknown"
        gt = (r["ground_truth"] or "").strip()
        label = to_label_v1(gt)
        items.append({
            "item_id": Path(r["file_name"]).stem, "source": "dictionary_bench", "split": "test",
            "audio_src": str(dst), "duration_s": round(sf.info(str(dst)).duration, 1),
            "language": ctx.get("language") or "", "app": app.get("name") or "",
            "bundle_id": app.get("bundle_id") or "",
            "tone": tone, "target_vocab": as_list(r["vocabulary"]), "user_vocab": as_list(ctx.get("vocabulary")),
            "before": ctx.get("text_before_cursor") or "", "after": ctx.get("text_after_cursor") or "",
            "screen": ctx.get("screen_text") or "", "old_label": gt, "transcript": label, "final_output": label,
            "note": r.get("notes") or "", "arena_item_id": r["id"], "object_key": r["object_key"]})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{out}: {len(items)} items; v1 changed {sum(i['old_label'] != i['transcript'] for i in items)}; "
          f"with user vocabulary {sum(bool(i['user_vocab']) for i in items)}; before-cursor "
          f"{sum(bool(i['before']) for i in items)}; after-cursor {sum(bool(i['after']) for i in items)}")
    print("tone:", {t: sum(i["tone"] == t for i in items) for t in ("casual", "neutral", "formal", "unknown")})
    print("apps not in the tone table:", sorted(unknown_apps.items(), key=lambda x: -x[1]))

I2_GROUPS = {"val": [("benchmark_chinese", "val"), ("online_phase1", "val")],
             "prod": [("benchmark_chinese", "train"), ("online_phase1", "train"), ("online_pool_b1full", "train"),
                      ("recording_v1", "train"), ("recording_v2", "train")],
             "rest": [("segmented_long", "train"), ("arena_019f6442", "train"), ("arena_019f8e51", "train")]}


def build_i2(group: str, out: str) -> None:
    items, no_ctx = [], 0
    for src, split in I2_GROUPS[group]:
        ctx_path = Path(f"data/private/context_pull/{src}.jsonl")
        ctx = {r["utterance_id"]: r for r in map(json.loads, open(ctx_path, encoding="utf-8"))} \
            if ctx_path.exists() else {}
        for m in map(json.loads, open(f"data/manifests/{src}.jsonl", encoding="utf-8")):
            if m["split"] != split:
                continue
            c = ctx.get(m["utterance_id"], {})
            no_ctx += not c
            label = to_label_v1(m["raw_transcript"])
            items.append({
                "item_id": m["utterance_id"], "source": src, "split": split,
                "audio_src": str(Path("data/private") / m["audio_path"]),
                "duration_s": round(m["duration_sec"], 1),
                "language": c.get("ctx_language") or "+".join(m.get("language_tags") or []),
                "app": c.get("app_name") or "", "bundle_id": c.get("app_bundle_id") or "",
                "tone": c.get("tone_mapped") or "unknown", "target_vocab": [],
                "user_vocab": as_list(c.get("effective_vocabulary") or c.get("ctx_vocabulary")),
                "before": c.get("text_before_cursor") or "", "after": c.get("text_after_cursor") or "",
                "screen": c.get("screen_text") or "", "old_label": m["raw_transcript"], "transcript": label,
                "final_output": label, "ref_final": c.get("prod_processed") or "", "note": "",
                "audio_id": c.get("audio_id") or ""})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{out}: {len(items)} items, {no_ctx} without context, v1 changed "
          f"{sum(i['old_label'] != i['transcript'] for i in items)}, with production output "
          f"{sum(bool(i['ref_final']) for i in items)}")


BV2_PARTS = {"short": ("benchmark_v2", "drafts_D600.jsonl", "D600"),
             "long": ("benchmark_v2_long", "drafts_I2_600.jsonl", "I2-600")}


def build_bv2(part: str, out: str) -> None:
    name, drafts_file, draft_model = BV2_PARTS[part]
    root = Path("data/private") / name
    ctx = {r["audio_id"]: r for r in map(json.loads, open(f"data/private/context_pull/{name}.jsonl", encoding="utf-8"))}
    drafts = {r["audio_id"]: r["hypothesis"] for r in map(json.loads, open(root / drafts_file, encoding="utf-8"))}
    items = []
    for r in map(json.loads, open(root / "items.jsonl", encoding="utf-8")):
        aid = r["audio_id"]
        c = ctx[aid]
        audio = root / "audio" / f"{aid}.ogg"
        prod = (r.get("transcription") or "").strip()
        label = to_label_v1(prod)
        # decoded duration from the selection step (some production ".ogg" files are WebM / WAV, which soundfile can't open)
        q = r["quality"] if isinstance(r["quality"], dict) else ast.literal_eval(r["quality"])
        items.append({
            "item_id": aid, "source": name, "split": "test", "audio_src": str(audio),
            "duration_s": round(q["duration_s"], 1), "language": r["lang_class"],
            "app": c.get("app_name") or "", "bundle_id": c.get("app_bundle_id") or "",
            "tone": c.get("tone_mapped") or "unknown", "target_vocab": [],
            "user_vocab": as_list(c.get("effective_vocabulary") or c.get("ctx_vocabulary")),
            "before": c.get("text_before_cursor") or "", "after": c.get("text_after_cursor") or "",
            "screen": c.get("screen_text") or "", "old_label": prod, "transcript": label, "final_output": label,
            "ref_final": c.get("prod_processed") or "", "note": "", "audio_id": aid,
            "draft": drafts.get(aid) or "", "draft_model": draft_model,
            "seen": r.get("seen") == "True", "s3_key": r.get("s3_key")})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{out}: {len(items)} items, {sum(i['duration_s'] for i in items) / 60:.0f} min, v1 changed "
          f"{sum(i['old_label'] != i['transcript'] for i in items)}, with draft {sum(bool(i['draft']) for i in items)}, "
          f"languages {dict(sorted({i['language']: sum(j['language'] == i['language'] for j in items) for i in items}.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dict")
    d.add_argument("xlsx")
    d.add_argument("audio_dir")
    d.add_argument("out")
    i = sub.add_parser("i2")
    i.add_argument("--group", choices=list(I2_GROUPS), required=True)
    i.add_argument("out")
    b = sub.add_parser("bv2")
    b.add_argument("--part", choices=list(BV2_PARTS), required=True)
    b.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "dict":
        build_dict(args.xlsx, args.audio_dir, args.out)
    elif args.cmd == "bv2":
        build_bv2(args.part, args.out)
    else:
        build_i2(args.group, args.out)


if __name__ == "__main__":
    main()
