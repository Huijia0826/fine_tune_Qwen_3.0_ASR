"""Pull edit-mode requests, edit instructions and the same users' dictations from PostHog (read-only).

Outputs (data/private/edit_prefs/):
  edits.jsonl      asr_metrics mode=edit (all apps, tutorial flagged later), joined with edit_intention_detect
                   by audio_id (instruction text, text_to_edit, edited_text, intent_category)
  dictate.jsonl    asr_metrics mode=dictate of every user who has >= 1 non-tutorial edit in the window
  user_stats.jsonl per-user dictate totals and custom-tone usage for ALL dictate users (for the custom-tone comparison)

Usage: NO_PROXY="*" python3 scripts/edit_prefs/pull.py --since 2026-09-10 --until 2026-11-09
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/trace-lookup/scripts"))
from trace_lookup import POSTHOG, load_env, q  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "private", "edit_prefs")
PAGE = 10000
TUTORIAL = "com.voicecursor.android.type"

EDIT_COLS = {
    "audio_id": "properties.audio_id", "user_id": "properties.user_id", "platform": "properties.client_source",
    "lang": "properties.lang", "languages": "toString(properties.languages)", "edit_action": "properties.edit_action",
    "edit_no_op_reason": "properties.edit_no_op_reason", "error_stage": "properties.error_stage",
    "edit_pipeline": "properties.edit_pipeline", "session_end_reason": "properties.session_end_reason",
    "processed": "properties.processed", "tone_id": "properties.tone_id", "tone_prompt": "properties.tone_prompt",
    "invocation_context": "properties.invocation_context", "app_type": "properties.app_type",
    "langfuse_trace_id": "properties.langfuse_trace_id", "s3_key": "properties.s3_key",
    "bundle_id": "JSONExtractString(properties.audio_context, 'app', 'bundle_id')",
    "app_name": "JSONExtractString(properties.audio_context, 'app', 'name')",
    "selected_text": "JSONExtractString(properties.audio_context, 'selected_text')",
    "text_before_cursor_len": "length(JSONExtractString(properties.audio_context, 'text_before_cursor'))",
}
INTENT_COLS = {
    "audio_id": "properties.audio_id", "instruction": "properties.real_edit_instruction_text",
    "transcription_result": "properties.transcription_result", "text_to_edit": "properties.text_to_edit",
    "edited_text": "properties.real_edited_text", "intent_category": "properties.intent_category",
    "detect_result": "properties.detect_result", "p_not_edit": "properties.p_not_edit",
    "real_edit_action": "properties.real_edit_action", "edit_rule_guard": "properties.edit_rule_guard",
    "state": "properties.state",
}
DICT_COLS = {
    "audio_id": "properties.audio_id", "user_id": "properties.user_id", "platform": "properties.client_source",
    "lang": "properties.lang", "transcription": "properties.transcription", "processed": "properties.processed",
    "model": "properties.model", "postprocess_model": "properties.postprocess_model",
    "tone_id": "properties.tone_id", "tone_prompt": "properties.tone_prompt",
    "invocation_context": "properties.invocation_context", "error_stage": "properties.error_stage",
    "bundle_id": "JSONExtractString(properties.audio_context, 'app', 'bundle_id')",
    "app_name": "JSONExtractString(properties.audio_context, 'app', 'name')",
}


def esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def paged(event_where: str, cols: dict, since: str, until: str, label: str) -> list[dict]:
    sel = ", ".join(cols.values())
    rows, seen, cursor = [], set(), since
    while True:
        res = q(f"SELECT uuid, timestamp, {sel} FROM events WHERE {event_where} AND timestamp < '{esc(until)}' "
                f"AND timestamp >= '{esc(cursor)}' ORDER BY timestamp LIMIT {PAGE}")
        new = 0
        for u, ts, *vals in res:
            if u in seen:
                continue
            seen.add(u)
            new += 1
            r = dict(zip(cols, vals))
            r["timestamp"] = ts
            rows.append(r)
        print(f"  {label}: +{new} (total {len(rows)}, up to {res[-1][1] if res else cursor})", file=sys.stderr)
        if len(res) < PAGE or not new:
            break
        cursor = res[-1][1]
    return rows


def dump(path: str, rows: list[dict], meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {len(rows)} -> {path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-10")
    ap.add_argument("--until", default="2026-11-09")
    ap.add_argument("--skip", nargs="*", default=[], choices=["edits", "dictate", "stats"])
    a = ap.parse_args()
    load_env(POSTHOG)
    os.makedirs(OUT, exist_ok=True)
    meta = {"since": a.since, "until": a.until, "source": "posthog asr_metrics / edit_intention_detect"}
    not_arena = "coalesce(properties.client_source, '') != 'arena'"

    if "edits" not in a.skip:
        edits = paged(f"event = 'asr_metrics' AND properties.mode = 'edit' AND {not_arena}",
                      EDIT_COLS, a.since, a.until, "edit")
        intents = paged("event = 'edit_intention_detect'", INTENT_COLS, a.since, a.until, "intent")
        by_id: dict[str, dict] = {}
        for i in intents:
            by_id.setdefault(i["audio_id"], i)  # first event per audio_id
        for e in edits:
            i = by_id.get(e["audio_id"])
            e["intent"] = {k: v for k, v in i.items() if k not in ("audio_id",)} if i else None
        meta["n_intent_events"] = len(intents)
        meta["n_intent_audio_ids"] = len(by_id)
        dump(os.path.join(OUT, "edits.jsonl"), edits, meta)

    if "dictate" not in a.skip:
        users = (f"SELECT properties.user_id FROM events WHERE event = 'asr_metrics' AND properties.mode = 'edit' "
                 f"AND {not_arena} AND timestamp >= '{esc(a.since)}' AND timestamp < '{esc(a.until)}' "
                 f"AND JSONExtractString(properties.audio_context, 'app', 'bundle_id') != '{TUTORIAL}' "
                 f"AND coalesce(properties.invocation_context, '') != 'onboarding_practice'")
        dic = paged(f"event = 'asr_metrics' AND properties.mode = 'dictate' AND {not_arena} "
                    f"AND properties.user_id IN ({users})", DICT_COLS, a.since, a.until, "dictate")
        dump(os.path.join(OUT, "dictate.jsonl"), dic, meta)

    if "stats" not in a.skip:
        res = q(f"""SELECT properties.user_id, count(), countIf(length(coalesce(properties.tone_prompt, '')) > 0),
                    any(properties.client_source), countIf(properties.mode = 'edit')
                    FROM events WHERE event = 'asr_metrics' AND {not_arena}
                    AND properties.mode IN ('dictate', 'edit')
                    AND timestamp >= '{esc(a.since)}' AND timestamp < '{esc(a.until)}'
                    AND JSONExtractString(properties.audio_context, 'app', 'bundle_id') != '{TUTORIAL}'
                    GROUP BY 1 LIMIT 100000""")
        rows = [{"user_id": u, "n_dictate_or_edit": n, "n_custom_tone": c, "platform": p, "n_edit": e}
                for u, n, c, p, e in res]
        dump(os.path.join(OUT, "user_stats.jsonl"), rows, meta)


if __name__ == "__main__":
    main()
