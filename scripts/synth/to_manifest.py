"""Synthetic texts + synthesised audio → a training manifest (2026-11-10).

raw_transcript = strict_transcript (ASR target), formatted_transcript = final_output; speaker_id = the TTS voice
(so per-speaker statistics and bootstraps treat a voice as a speaker); split = train; label_status
machine_draft until QC passes (scripts/synth/qc.py sets human-level trust only for clips that round-trip). The
context fields stay in the text jsonl (joined by utterance_id = synth id).

    uv run python scripts/synth/to_manifest.py data/private/synth/pilot_500.jsonl data/private/synth/audio \
        --audio-subdir synth/audio --out data/manifests/synth_pilot.jsonl
"""

import argparse
import json
from pathlib import Path

from whisperflow.data.audio import sha256_file
from whisperflow.data.schema import Utterance

LANG = {"zh": ["zh"], "zh-en": ["zh", "en"], "en": ["en"]}
SCEN = {"zh": "zh_general", "zh-en": "zh_en_switch", "en": "en_general"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("audio_dir")
    ap.add_argument("--audio-subdir", required=True, help="audio path relative to data/private")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    synth = {m["id"]: m for m in map(json.loads, open(Path(args.audio_dir) / "manifest.jsonl", encoding="utf-8"))
             if not m.get("error")}
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in map(json.loads, open(args.texts, encoding="utf-8")):
            s = synth.get(r["id"])
            if not s:
                continue
            wav = Path(args.audio_dir) / f"{r['id']}.wav"
            tags = r.get("tags", [])
            u = Utterance(
                utterance_id=r["id"].replace("_", "-"), audio_path=f"{args.audio_subdir}/{r['id']}.wav",
                sha256=sha256_file(str(wav)), duration_sec=s["duration_s"],
                speaker_id=f"tts-minimax-{s['voice'].replace(' ', '_').replace('(', '').replace(')', '')}"[:128],
                session_id=f"tts-minimax-{r['id']}", source_id="synth_minimax", split="train",
                license_or_consent_id="synthetic_minimax_stock_voice", raw_transcript=r["strict_transcript"],
                formatted_transcript=r["final_output"],
                primary_scenario="self_correction" if "self_correction" in tags else SCEN[r["language"]],
                language_tags=LANG[r["language"]], hotwords_present=r["terms"], label_status="machine_draft",
                reviewer_id=None, quality_flags=["synthetic_tts"], source_utterance_id=r["id"])
            f.write(json.dumps(u.model_dump(), ensure_ascii=False) + "\n")
            n += 1
    print(f"{args.out}: {n} clips")


if __name__ == "__main__":
    main()
