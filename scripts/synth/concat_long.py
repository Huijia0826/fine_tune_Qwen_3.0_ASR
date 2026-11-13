"""Long-audio augmentation from real short clips (owner, 2026-11-10: "专门做一下长音频增广").

Concatenates TRAINING clips into 30–60 s samples with 0.3–1.0 s pauses (room tone: the quietest stretch of the
first clip, else silence); the label is the clip labels joined in order (policy v1). Two kinds:
  same_speaker  clips of one production user (online_pool / online_phase1 / benchmark_chinese), in time order —
                real voice, device and topic continuity, close to a real long dictation
  arena_en      English arena clips (no speaker ids) in random order — less natural, adds long English
Only train-split clips ≤ 30 s; never a val / test speaker. Output: data/private/concat/<id>.wav and
data/manifests/concat_long.jsonl (source_id concat_long, quality flag concatenated).

    uv run python scripts/synth/concat_long.py --same 150 --arena 150
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from whisperflow.data.audio import load_16k_mono, sha256_file
from whisperflow.data.schema import Utterance
from whisperflow.text.label_style import to_label_v1

SR = 16000
ROOT = Path("data/private")


def joiner(a: str, b: str) -> str:
    """Join two labels: keep their own punctuation, add a sentence end if the first has none."""
    a = a.rstrip()
    if a and a[-1] not in "。！？.!?，,":
        a += "。" if any("一" <= ch <= "鿿" for ch in a[-3:]) else "."
    return a + ("" if any("一" <= ch <= "鿿" for ch in b[:1]) else " ") + b.lstrip()


def build(group: list[dict], rng: random.Random, out_id: str, kind: str) -> dict | None:
    parts, labels, dur = [], [], 0.0
    for m in group:
        x = load_16k_mono(ROOT / m["audio_path"])
        if parts:
            gap = int(rng.uniform(0.3, 1.0) * SR)
            q = parts[0][: SR // 2]
            parts.append(np.resize(q * 0.5, gap) if q.std() < 0.01 else np.zeros(gap, dtype=np.float32))
        parts.append(x.astype(np.float32))
        labels.append(to_label_v1(m["raw_transcript"]))
        dur += len(x) / SR
    audio = np.concatenate(parts)
    path = ROOT / "concat" / f"{out_id}.wav"
    sf.write(path, audio, SR, subtype="PCM_16")
    text = labels[0]
    for lab in labels[1:]:
        text = joiner(text, lab)
    tags = sorted({t for m in group for t in m["language_tags"]})
    u = Utterance(utterance_id=out_id, audio_path=f"concat/{out_id}.wav", sha256=sha256_file(str(path)),
                  duration_sec=round(len(audio) / SR, 3), speaker_id=group[0]["speaker_id"],
                  session_id=f"concat-{out_id}", source_id="concat_long", split="train",
                  license_or_consent_id=group[0]["license_or_consent_id"], raw_transcript=text,
                  primary_scenario="en_general" if tags == ["en"] else ("zh_en_switch" if "en" in tags else
                                                                         "zh_general"), language_tags=tags,
                  label_status="human_verified", reviewer_id=group[0].get("reviewer_id") or "inherited",
                  quality_flags=["concatenated", kind],
                  parent_utterance_id=None, source_utterance_id="+".join(m["utterance_id"] for m in group))
    return u.model_dump()


def chunks(clips: list[dict], rng: random.Random, lo: float, hi: float) -> list[list[dict]]:
    """Consecutive runs of clips whose total duration (with ~0.6 s pauses) lands in [lo, hi]."""
    out, cur, d = [], [], 0.0
    for m in clips:
        cur.append(m)
        d += m["duration_sec"] + 0.6
        if d >= lo:
            if d <= hi and len(cur) >= 2:
                out.append(cur)
            cur, d = [], 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--same", type=int, default=150)
    ap.add_argument("--arena", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    (ROOT / "concat").mkdir(parents=True, exist_ok=True)
    held_out = {json.loads(line)["speaker_id"] for f in Path("data/manifests").glob("*.jsonl") if "." not in f.stem
                for line in open(f, encoding="utf-8") if json.loads(line)["split"] in ("val", "test")}
    by_spk = defaultdict(list)
    for src in ("online_pool_b1full", "online_phase1", "benchmark_chinese"):
        for m in map(json.loads, open(f"data/manifests/{src}.jsonl", encoding="utf-8")):
            if m["split"] == "train" and m["duration_sec"] <= 30 and m["speaker_id"] not in held_out:
                by_spk[m["speaker_id"]].append(m)
    same = []
    for clips in by_spk.values():
        clips.sort(key=lambda m: m["utterance_id"])  # ids are not time-ordered; context_pull timestamps below
        same += chunks(clips, rng, 30, 60)
    ctx = {}
    for f in Path("data/private/context_pull").glob("[!_]*.jsonl"):
        for c in map(json.loads, open(f, encoding="utf-8")):
            ctx[c["utterance_id"]] = c.get("timestamp") or ""
    same = [sorted(g, key=lambda m: ctx.get(m["utterance_id"], m["utterance_id"])) for g in same]
    rng.shuffle(same)
    arena = [m for m in map(json.loads, open("data/manifests/arena_019f8e51.jsonl", encoding="utf-8"))
             if m["split"] == "train" and m["duration_sec"] <= 30]
    rng.shuffle(arena)
    groups = [("same_speaker", g) for g in same[:args.same]] + \
             [("arena_en", g) for g in chunks(arena, rng, 30, 60)[:args.arena]]
    rows = []
    for k, (kind, g) in enumerate(groups, 1):
        rows.append(build(g, rng, f"concat-{kind.replace('_', '')}-{k:04d}", kind))
    with open("data/manifests/concat_long.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips ({sum(1 for k, _ in groups if k == 'same_speaker')} same-speaker, "
          f"{sum(1 for k, _ in groups if k == 'arena_en')} arena English), "
          f"{sum(r['duration_sec'] for r in rows) / 3600:.2f} h → data/manifests/concat_long.jsonl")


if __name__ == "__main__":
    main()
