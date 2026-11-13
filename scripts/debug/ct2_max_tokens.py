"""E7 debug (2026-11-02): where does faster-whisper actually stop for a given max_new_tokens?"""
import json

from faster_whisper import WhisperModel

from whisperflow.data.audio import load_16k_mono

r = [json.loads(l) for m in ("data/manifests/benchmark_chinese.jsonl", "data/manifests/online_phase1.jsonl")
     for l in open(m) if l.strip()]
r = [x for x in r if x["utterance_id"].endswith("fa0433a9")][0]
a = load_16k_mono("data/private/" + r["audio_path"])
m = WhisperModel("runs/ct2/C600_fp16", device="cuda", compute_type="float16")
for mnt in (100, 150, 200, 250, 300, 400, None):
    for beam in (1, 5):
        try:
            segs = list(m.transcribe(a, language="zh", beam_size=beam, without_timestamps=True, max_new_tokens=mnt,
                                     temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0])
            print(f"max_new_tokens={mnt} beam={beam}: out tokens={sum(len(s.tokens) for s in segs)} "
                  f"...{''.join(s.text for s in segs)[-20:]}")
        except Exception as e:
            print(f"max_new_tokens={mnt} beam={beam}: {type(e).__name__} {str(e)[:80]}")
