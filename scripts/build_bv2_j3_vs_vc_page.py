"""Side-by-side page: Qwen3-ASR J3 step 600 (+context) vs VoiceCursor streaming on benchmark v2.0 (owner, 2026-11-12).

Only the clips J3 never trained on (split test + val, 270 clips). Per clip: the audio (embedded Ogg; open in Chrome),
the arena ground truth, and both outputs with their errors marked against the ground truth on the scorer's own tokens
(policy v1 MER tokens = scripts/score_arena_export.py, so casing / punctuation / spacing are not shown as errors): red strike = missed reference token,
yellow = wrong token (hover shows the reference token), green = extra token. Filters: duration (< 30 s / ≥ 30 s and
finer buckets), source set, outcome. Default order: clips where J3 loses most come first.

    python scripts/build_bv2_j3_vs_vc_page.py
"""
import base64
import html
import re
from pathlib import Path

import pandas as pd

from whisperflow.eval.metrics import align_text

VC = Path("reports/arena/benchmark2.0_voicecursorstreaming_per_clip.csv")
J3 = Path("reports/arena/bv2_0_J3_step600_ctx_per_clip.csv")
AUDIO = Path.home() / "Desktop/Build_Benchmark/benchmark_v2.0/audio"
OUT = Path("data/private/bv2_compare/bv2_J3_vs_voicecursor.html")
BUCKETS = [(0, 5, "<5s"), (5, 15, "5-15s"), (15, 30, "15-30s"), (30, 1e9, "≥30s")]
_WORD = re.compile(r"[a-z0-9']")


def marked(ref: str, hyp: str) -> tuple[str, int]:
    """hyp tokens with the errors against ref marked up, and the error count."""
    parts, errs = [], 0
    for op, r, h in align_text(ref, hyp).ops:
        if op == "eq":
            parts.append(("", html.escape(h)))
            continue
        errs += 1
        if op == "sub":
            parts.append(("s", f'<span class="sub" title="参考：{html.escape(r)}">{html.escape(h)}</span>'))
        elif op == "ins":
            parts.append(("s", f'<span class="ins">{html.escape(h)}</span>'))
        else:
            parts.append(("s", f'<span class="del">{html.escape(r)}</span>'))
    # space between two Latin / number tokens, none around Chinese characters
    out, prev = "", ""
    for _, frag in parts:
        tok = re.sub(r"<[^>]+>", "", frag)
        if out and prev and _WORD.match(prev[-1]) and tok and _WORD.match(tok[0]):
            out += " "
        out += frag
        prev = tok or prev
    return out, errs


def pct(e: int, n: int) -> str:
    return f"{100 * e / n:.2f}%" if n else "—"


def main() -> None:
    vc, j3 = pd.read_csv(VC, keep_default_na=False), pd.read_csv(J3, keep_default_na=False)
    m = vc.merge(j3[["file", "hyp", "err"]], on="file", suffixes=("_vc", "_j3"))
    m = m[m.split.isin(["test", "val"])].copy()
    m["delta"] = m.err_j3 - m.err_vc
    m = m.sort_values(["delta", "dur"], ascending=[False, False])

    cards, mismatch = [], 0
    for k, r in enumerate(m.itertuples(), 1):
        vc_html, vc_err = marked(r.ref, r.hyp_vc)
        j3_html, j3_err = marked(r.ref, r.hyp_j3)
        mismatch += (vc_err, j3_err) != (r.err_vc, r.err_j3)
        bucket = next(b for lo, hi, b in BUCKETS if lo <= r.dur < hi)
        outcome = "j3worse" if r.delta > 0 else "j3better" if r.delta < 0 else "tie"
        label = {"j3worse": "J3 更差", "j3better": "J3 更好", "tie": "一样"}[outcome]
        b64 = base64.b64encode((AUDIO / r.file).read_bytes()).decode()
        cards.append(f"""<div class="card" data-long="{int(r.dur >= 30)}" data-bucket="{bucket}" data-set="{r.set}"
 data-split="{r.split}" data-outcome="{outcome}">
<div class="meta">#{k} <span class="tag {outcome}">{label}（J3 {r.err_j3} vs VC {r.err_vc} 个错误）</span>
 {r.dur:.1f}s · {r.set} · {r.lang} · {'没训练过' if r.split == 'test' else 'J3 的验证集'}
 <code>{r.file}</code></div>
<audio controls preload="none" src="data:audio/ogg;base64,{b64}"></audio>
<div class="lbl">参考答案（arena ground truth）</div><div class="txt">{html.escape(r.ref)}</div>
<div class="lbl">VoiceCursor · {vc_err} 错 / {r.n_ref} 词（{pct(vc_err, r.n_ref)}）</div><div class="txt">{vc_html}</div>
<div class="lbl">J3 · {j3_err} 错 / {r.n_ref} 词（{pct(j3_err, r.n_ref)}）</div><div class="txt">{j3_html}</div>
</div>""")

    def summary(rows: pd.DataFrame, name: str) -> str:
        n = rows.n_ref.sum()
        return (f"<tr><td>{name}</td><td>{len(rows)}</td><td>{rows.dur.sum() / 60:.0f} 分钟</td>"
                f"<td>{pct(rows.err_vc.sum(), n)}</td><td>{pct(rows.err_j3.sum(), n)}</td></tr>")
    t = m[m.split == "test"]
    table = "".join([summary(t, "全部（没训练过）"), summary(t[t.dur < 30], "&lt; 30 秒"),
                     summary(t[t.dur >= 30], "≥ 30 秒")]
                    + [summary(t[(t.dur >= lo) & (t.dur < hi)], b) for lo, hi, b in BUCKETS])
    sets = "".join(f'<option value="{s}">{s}</option>' for s in sorted(m.set.unique()))
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>bv2.0：J3 vs VoiceCursor</title><style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#f5f5f7;margin:0;color:#1d1d1f}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;z-index:2}}
main{{max-width:980px;margin:16px auto;padding:0 20px}}
.card{{background:#fff;border-radius:12px;padding:14px 20px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.meta{{color:#666;font-size:13px;margin-bottom:6px}} code{{background:#f0f0f0;padding:1px 5px;border-radius:4px;font-size:11px}}
.tag{{padding:1px 8px;border-radius:10px;font-size:12px}} .j3worse{{background:#fde2e2}} .j3better{{background:#d9f5df}}
.tie{{background:#eee}} .lbl{{font-size:13px;color:#555;margin-top:10px}} .txt{{font-size:16px;line-height:1.7}}
.sub{{background:#fff1a8;border-bottom:2px solid #e0b000}} .ins{{background:#d9f5df}}
.del{{background:#fde2e2;color:#a33;text-decoration:line-through}} audio{{width:100%}}
table{{border-collapse:collapse;font-size:13px;margin:6px 0}} td,th{{border:1px solid #ddd;padding:2px 10px}}
select{{margin-right:10px}} .hide{{display:none}} details{{font-size:13px}}
</style></head><body><header><b>benchmark 2.0：J3 step 600（带上下文）vs VoiceCursor streaming</b>
　只含 J3 没训练过的 {len(m)} 条（test {len(t)} + J3 验证集 {len(m) - len(t)}）· 显示 <span id="n"></span> 条
<div style="margin-top:6px">时长 <select id="f-len"><option value="">全部</option><option value="0">&lt; 30 秒</option>
<option value="1">≥ 30 秒</option>{''.join(f'<option value="b:{b}">{b}</option>' for *_, b in BUCKETS)}</select>
结果 <select id="f-out"><option value="">全部</option><option value="j3worse">J3 更差</option>
<option value="j3better">J3 更好</option><option value="tie">一样</option></select>
来源 <select id="f-set"><option value="">全部</option>{sets}</select>
数据 <select id="f-split"><option value="">全部</option><option value="test">没训练过（test）</option>
<option value="val">J3 验证集（val）</option></select></div>
<details><summary>汇总（test 213 条，MER）和颜色说明</summary><table><tr><th>时长</th><th>条数</th><th>音频</th>
<th>VoiceCursor</th><th>J3</th></tr>{table}</table>
<span class="del">删掉的</span> 参考里有、输出里漏了 · <span class="sub">错的</span> 认错了（鼠标悬停看参考）·
<span class="ins">多的</span> 参考里没有、输出多出来的。按打分用的词切分对齐，大小写、标点、空格不算错。默认排序：J3 比 VC 多错最多的在前。
音频是 Ogg，请用 Chrome 打开。</details></header>
<main>{''.join(cards)}</main>
<script>
const f=["len","out","set","split"].map(k=>document.getElementById("f-"+k));
function apply(){{let n=0;document.querySelectorAll(".card").forEach(c=>{{const[l,o,s,sp]=f.map(x=>x.value);
const okL=!l||(l.startsWith("b:")?c.dataset.bucket===l.slice(2):c.dataset.long===l);
const ok=okL&&(!o||c.dataset.outcome===o)&&(!s||c.dataset.set===s)&&(!sp||c.dataset.split===sp);
c.classList.toggle("hide",!ok);n+=ok}});document.getElementById("n").textContent=n}}
f.forEach(x=>x.onchange=apply);apply();
</script></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT}: {len(m)} clips, {OUT.stat().st_size / 1e6:.1f} MB; error counts differing from the scorer csv: "
          f"{mismatch}")


if __name__ == "__main__":
    main()
