"""Read-only page of the benchmark v2 English clips whose final output (box ②) differs from the strict transcript (box ①)
(owner, 2026-11-12): the upload Excel uses box ①, but some hearing corrections were typed only into box ②, so the owner
checks these clips and fixes the ground truth by hand in arena.

Shows per clip: the arena file name (copy button), the audio (embedded Ogg; open in Chrome), box ① = what the upload
Excel has, box ② with a word-level diff against ①, and the production output. Clips with a word difference come first
(punctuation / case-only differences are listed after them and can be hidden).

    python3 scripts/build_bv2_en_diff_page.py
"""

import base64
import csv
import difflib
import html
import json
import re
from pathlib import Path

REVIEW = Path("data/private/e2e/reviewed/bv2-en-short-review.csv")
ITEMS = Path("data/private/e2e/items_bv2_en_short.jsonl")
AUDIO = Path.home() / "Desktop/build_benchmark/benchmark_v2.0/audio"
OUT = Path("data/private/e2e/review/bv2-en-diff/bv2-en-diff.html")


def norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def word_diff(a: str, b: str) -> str:
    """b with words not in a marked <ins>, words of a missing from b marked <del>."""
    ta, tb = re.findall(r"\S+", a), re.findall(r"\S+", b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out.append(f"<del>{html.escape(' '.join(ta[i1:i2]))}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins>{html.escape(' '.join(tb[j1:j2]))}</ins>")
        if op == "equal":
            out.append(html.escape(" ".join(ta[i1:i2])))
    return " ".join(out)


def main() -> None:
    items = {it["item_id"]: it for it in map(json.loads, open(ITEMS, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(REVIEW, encoding="utf-8-sig"))
            if r["verdict"] != "drop" and r["final_output"].strip() != r["transcript"].strip()]
    rows.sort(key=lambda r: norm(r["final_output"]) == norm(r["transcript"]))
    cards = []
    for k, r in enumerate(rows, 1):
        f = f"{r['item_id']}.ogg"
        words = norm(r["final_output"]) != norm(r["transcript"])
        b64 = base64.b64encode((AUDIO / f).read_bytes()).decode()
        prod = items[r["item_id"]].get("ref_final") or ""
        cards.append(f"""<div class="card {'words' if words else 'punct'}">
<div class="meta">{k}/{len(rows)} <span class="tag {'w' if words else ''}">{'用词不同' if words else '仅标点 / 大小写'}</span>
 {items[r['item_id']]['duration_s']}s <code>{f}</code> <button onclick="cp(this,'{f}')">复制文件名</button></div>
<audio controls preload="none" src="data:audio/ogg;base64,{b64}"></audio>
<div class="lbl">① 严格转写（现在 Excel / arena 里的）<button onclick="cp(this,{html.escape(json.dumps(r['transcript']))})">复制</button></div>
<div class="txt">{html.escape(r['transcript'])}</div>
<div class="lbl">② 最终输出（与 ① 的差异：<ins>新增</ins> <del>删除</del>）<button onclick="cp(this,{html.escape(json.dumps(r['final_output']))})">复制</button></div>
<div class="txt">{word_diff(r['transcript'], r['final_output'])}</div>
{f'<div class="prod">线上输出：{html.escape(prod)}</div>' if prod and prod != r['transcript'] else ''}
</div>""")
    n_words = sum(norm(r["final_output"]) != norm(r["transcript"]) for r in rows)
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>bv2 英文：① vs ② 差异</title><style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#f5f5f7;margin:0;color:#1d1d1f}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;z-index:2}}
main{{max-width:960px;margin:16px auto;padding:0 20px}}
.card{{background:#fff;border-radius:12px;padding:14px 20px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.meta{{color:#666;font-size:13px;margin-bottom:6px}} code{{background:#f0f0f0;padding:1px 5px;border-radius:4px}}
.tag{{padding:1px 8px;border-radius:10px;background:#eee;font-size:12px}} .tag.w{{background:#fff1c2}}
.lbl{{font-size:13px;color:#555;margin-top:10px}} .txt{{font-size:16px;line-height:1.6;margin-top:3px}}
.prod{{font-size:13px;color:#777;margin-top:8px}} audio{{width:100%}}
ins{{background:#d9f5df;text-decoration:none}} del{{background:#fde2e2;color:#a33}}
button{{font-size:12px;padding:1px 8px;margin-left:6px}} body.only .punct{{display:none}}
</style></head><body><header><b>benchmark v2 英文：最终输出 ≠ 严格转写</b>　共 {len(rows)} 条（用词不同 {n_words}，仅标点 / 大小写 {len(rows) - n_words}）
<label style="margin-left:12px"><input type="checkbox" onchange="document.body.classList.toggle('only',this.checked)"> 只看用词不同</label>
<div style="font-size:13px;color:#666;margin-top:4px">只读页面。听音频 → 如果 ① 听错了，在 arena 里按文件名找到这条，把 transcript 改成正确的严格转写。音频是 Ogg，请用 Chrome 打开。</div></header>
<main>{''.join(cards)}</main>
<script>function cp(b,t){{navigator.clipboard.writeText(t);const o=b.textContent;b.textContent="已复制";setTimeout(()=>b.textContent=o,900)}}</script>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT}: {len(rows)} clips ({n_words} with a word difference), {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
