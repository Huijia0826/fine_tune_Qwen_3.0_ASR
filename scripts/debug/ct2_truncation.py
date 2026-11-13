"""E7 debug (2026-11-02): why does faster-whisper drop the end of some short clips? Vary one option at a time."""
import json

from faster_whisper import WhisperModel

from whisperflow.data.audio import load_16k_mono

ids = ["fa0433a9", "422b012e", "f2b59870"]
recs = [json.loads(l) for m in ("data/manifests/benchmark_chinese.jsonl", "data/manifests/online_phase1.jsonl")
        for l in open(m) if json.loads(l)["split"] == "val"]
recs = [r for r in recs if r["utterance_id"][-8:] in ids]
m = WhisperModel("runs/ct2/C600_fp16", device="cuda", compute_type="float16")
base = dict(language="zh", beam_size=5, condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
            max_new_tokens=200, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], compression_ratio_threshold=1.35,
            log_prob_threshold=None)
variants = {"as_run": {}, "temp0_only": dict(temperature=0.0, compression_ratio_threshold=None),
            "no_max_new": dict(max_new_tokens=None), "timestamps": dict(without_timestamps=False),
            "greedy": dict(beam_size=1), "no_suppress": dict(suppress_tokens=[], suppress_blank=False)}
for r in recs:
    a = load_16k_mono("data/private/" + r["audio_path"])
    print(f"\n### {r['utterance_id'][-8:]} {len(a) / 16000:.1f}s REF {r['raw_transcript'][-40:]}")
    for name, over in variants.items():
        segs, info = m.transcribe(a, **{**base, **over})
        segs = list(segs)
        txt = "".join(s.text for s in segs)
        print(f"  {name:12s} n_seg={len(segs)} temp={[s.temperature for s in segs]} "
              f"cr={[round(s.compression_ratio, 2) for s in segs]} end={[round(s.end, 1) for s in segs]} | ...{txt[-40:]}")
