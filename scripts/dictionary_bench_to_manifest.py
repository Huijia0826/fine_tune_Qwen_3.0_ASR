"""Owner's dictionary (hotword) benchmark → a test manifest + per-clip contexts (2026-11-09).

Reads data/private/e2e/items_dict.jsonl (scripts/make_e2e_items.py dict). Reference = ground truth brought to
annotation policy v1 (until the owner's review replaces it). Each clip is its own speaker (no user ids). The target
vocabulary is the dictionary under test, so it goes into the context (plus the user's logged vocabulary, app, tone
and cursor text, with the same 8-character leak rule as training: scripts/make_qwen_e2e_data.py).
Language: pt / other clips are tagged `other` (out of the zh / en scope; scored separately).

    uv run python scripts/dictionary_bench_to_manifest.py
Outputs: data/manifests/dictionary_bench.jsonl, data/private/e2e/contexts_dictionary_bench.jsonl
"""

import json
import re
import sys
from pathlib import Path

from whisperflow.data.audio import sha256_file
from whisperflow.data.schema import Utterance

sys.path.insert(0, str(Path(__file__).parent))
from make_qwen_e2e_data import leaks  # noqa: E402


def lang_tags(text: str, ctx_lang: str) -> list[str]:
    zh, en = bool(re.search(r"[一-鿿]", text)), bool(re.search(r"[A-Za-z]{2,}", text))
    if ctx_lang.startswith("pt") and not zh:
        return ["other"]
    return (["zh"] if zh else []) + (["en"] if en or not zh else [])


def main() -> None:
    items = [json.loads(line) for line in open("data/private/e2e/items_dict.jsonl", encoding="utf-8")]
    man, ctx = [], []
    for it in items:
        tags = lang_tags(it["transcript"], it["language"] or "")
        u = Utterance(
            utterance_id=f"dictionary_bench-{it['item_id'][:16]}",
            audio_path=str(Path(it["audio_src"]).relative_to("data/private")), sha256=sha256_file(it["audio_src"]),
            duration_sec=it["duration_s"],
            speaker_id=f"dictionary_bench-{it['item_id'][:8]}", session_id=f"dictionary_bench-{it['item_id'][:8]}",
            source_id="dictionary_bench", license_or_consent_id="voicecursor_production", split="test",
            raw_transcript=it["transcript"],
            primary_scenario="technical_terms" if tags != ["other"] else "en_general", language_tags=tags,
            hotwords_present=it["target_vocab"], label_status="human_verified",
            reviewer_id="owner_dictionary_bench",
            source_utterance_id=it["item_id"])
        man.append(u.model_dump())
        lines = [f"App: {it['app']} · {it['tone']}"] if it["app"] else []
        vocab = list(dict.fromkeys(it["target_vocab"] + it["user_vocab"]))[:50]
        if vocab:
            lines.append("Vocabulary: " + ", ".join(vocab))
        b, a = it["before"].strip(), it["after"].strip()
        if b and not leaks(it["transcript"], b):
            lines.append("Before cursor: " + ("…" if len(b) > 200 else "") + b[-200:])
        if a and not leaks(it["transcript"], a):
            lines.append("After cursor: " + a[:120] + ("…" if len(a) > 120 else ""))
        ctx.append({"utterance_id": u.utterance_id, "context": "\n".join(lines)})
    with open("data/manifests/dictionary_bench.jsonl", "w", encoding="utf-8") as f:
        for r in man:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/private/e2e/contexts_dictionary_bench.jsonl", "w", encoding="utf-8") as f:
        for r in ctx:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(man), "clips;", Counter("+".join(r["language_tags"]) for r in man))


if __name__ == "__main__":
    main()
