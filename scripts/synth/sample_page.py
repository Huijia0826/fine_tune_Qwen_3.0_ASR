"""Listen-and-read page for a random sample of the synthetic pilot (owner, 2026-11-10).

Synthesises N random samples of data/private/synth/pilot_500.jsonl with MiniMax (a random stock voice per sample:
English voices for English text, Chinese voices otherwise — they also read the English words), saves the wavs under
data/private/synth/audio_sample/, and writes one self-contained HTML page (audio embedded) showing the context, the
text the TTS read, the strict transcript (= ASR ground truth) and the final output (= final ground truth).

    MINIMAX_API_KEY=… MINIMAX_API_BASE=https://api.minimax.io uv run python scripts/synth/sample_page.py --n 20
"""

import argparse
import base64
import html
import io
import json
import random
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tts_compare"))
from synth import TARGET_SR, VENDORS, call_minimax  # noqa: E402

OUT = Path("data/private/synth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(OUT / "pilot_500.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.n)
    voices = VENDORS["minimax"]["voices"]
    (OUT / "audio_sample").mkdir(parents=True, exist_ok=True)
    cards = []
    for k, r in enumerate(sample, 1):
        pool = [v for v in voices if v.lang == ("en" if r["language"] == "en" else "zh")]
        v = rng.choice(pool)
        wav = OUT / "audio_sample" / f"{r['id']}.wav"
        if not wav.exists():
            audio = call_minimax(r["spoken_text"], v, "en" if r["language"] == "en" else "zh")
            sf.write(str(wav), audio, TARGET_SR, subtype="PCM_16")
        buf = io.BytesIO()
        data, sr = sf.read(str(wav))
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        b64 = base64.b64encode(buf.getvalue()).decode()
        e = lambda s: html.escape(s or "").replace("\n", "<br>")  # noqa: E731
        ctx = "".join(f"<div><b>{n}</b> {e(t)}</div>" for n, t in (("光标前", r["before_cursor"]),
                                                                    ("光标后", r["after_cursor"])) if t)
        cards.append(f"""<div class=card><div class=meta>#{k} · {e(r['id'])} · {e(r['app'])} → {e(r['tone'])} ·
 {e(r['language'])} · 术语：{e(', '.join(r['terms'])) or '—'} · 音色：{e(v.label)} ·
 标签：{e(', '.join(r.get('tags', [])))}</div>
<audio controls preload=none src="data:audio/wav;base64,{b64}"></audio>
{f'<div class=ctx>{ctx}</div>' if ctx else ''}
<p><b>TTS 读的文字</b>（不是答案）<br><span class=sp>{e(r['spoken_text'])}</span></p>
<p><b>① 严格转写 = ASR 答案（D）</b><br>{e(r['strict_transcript'])}</p>
<p><b>② 最终输出 = 最终答案（E）</b><br>{e(r['final_output'])}</p></div>""")
    page = f"""<!doctype html><meta charset=utf-8><title>合成数据抽样 {args.n} 条</title>
<style>body{{font-family:-apple-system,'PingFang SC';max-width:960px;margin:20px auto;background:#f5f5f7}}
.card{{background:#fff;border-radius:10px;padding:14px 18px;margin:12px 0}}.meta{{color:#666;font-size:13px}}
.ctx{{background:#f2f5f9;font-size:13px;padding:6px 10px;border-radius:6px;margin:6px 0;color:#444}}
.ctx b{{color:#778;font-weight:500;margin-right:6px}}.sp{{color:#777}}audio{{width:100%;margin:6px 0}}
p{{line-height:1.6;margin:8px 0}}</style>
<h2>合成数据抽样：{args.n} 条（随机，MiniMax 随机音色，未加噪声）</h2>
<p>看三件事：音频是否读对了（特别是英文名、数字、拼读字母）；① 严格转写是否就是音频里说的话；
② 最终输出是否是你想要的样子。
要改的地方告诉我 id 和改法，或者直接在 pilot_500_edit.xlsx 里改。</p>
{''.join(cards)}"""
    (OUT / "pilot_sample_20.html").write_text(page, encoding="utf-8")
    print(OUT / "pilot_sample_20.html")


if __name__ == "__main__":
    main()
