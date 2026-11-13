# ruff: noqa: E501  (HTML / JS template on long lines, as in build_review_html.py)
"""Blind A/B preference review (owner, 2026-11-10): which final text is better, production or ours?

References edited from production output cannot score a rewrite fairly, so the owner judges pairs blind: each item
shows the audio and two texts in random order (A / B); the page never contains which system is which — the key is
written to a separate local file (<out>/key.json). The owner picks A better / B better / about the same / both bad,
and can flag "added content or answered the question" for either side. Export review.csv → scripts/ab_score.py.

    uv run python scripts/build_ab_review.py --name ab-bv1 --n 100 \
        --sys production=voicecursor_e2e_benchmark_v1 --sys ours=bv1_qwen3asr_J3_step_600_noctx_pp_q3_4b
"""

import argparse
import base64
import csv
import json
import random
import subprocess
import tempfile
from pathlib import Path

HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>盲测 __NAME__</title>
<style>body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
main{max-width:1000px;margin:18px auto;padding:0 20px}.card{background:#fff;border-radius:12px;padding:18px 24px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.meta{color:#666;font-size:13px}.opt{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
.box{border:2px solid #ddd;border-radius:10px;padding:12px 14px;font-size:16px;line-height:1.6;white-space:pre-wrap}
.box h3{margin:0 0 6px;font-size:14px;color:#555}.box.sel{border-color:#0071e3;background:#f0f7ff}
button{font-size:14px;padding:7px 14px;border-radius:8px;border:1px solid #ccc;background:#fff;cursor:pointer}
button.on{background:#0071e3;color:#fff;border-color:#0071e3}.row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;align-items:center}
audio{width:100%;margin:8px 0}input.note{width:100%;box-sizing:border-box;padding:6px 10px;margin-top:8px;border:1px solid #ccc;border-radius:6px}
kbd{background:#eee;border-radius:4px;padding:0 5px}</style></head><body>
<header><b>__NAME__</b><span id="prog"></span><button onclick="go(-1)">← 上一条</button><button onclick="go(1)">下一条 →</button>
<button onclick="nextTodo()">下一条未评</button><button onclick="exportCsv()">导出 review.csv</button></header>
<main><div class="card" id="card"></div>
<p style="font-size:13px;color:#444">听音频，比较两段文字作为<b>用户最终看到的文字</b>哪个更好：意思是否忠实、口头禅/重复是否清理干净、标点格式是否合适、有没有听错的词。
A / B 的来源是随机的，每条都不同。快捷键：<kbd>1</kbd> A 更好 · <kbd>2</kbd> B 更好 · <kbd>3</kbd> 差不多 · <kbd>4</kbd> 都不好 · <kbd>⌥P</kbd> 播放/暂停 · <kbd>⌥J</kbd>/<kbd>⌥K</kbd> 上/下一条。进度自动保存；评完点"导出 review.csv"发给 Claude。</p></main>
<script>const NAME="__NAME__",DATA=__DATA__,AUDIO=__AUDIO__,KEY="wf_ab_"+NAME;
let st=JSON.parse(localStorage.getItem(KEY)||"{}"),pos=0,t0=Date.now();const save=()=>localStorage.setItem(KEY,JSON.stringify(st));
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));const S=r=>st[r.item]||(st[r.item]={});
const LAB={A:"A 更好",B:"B 更好",tie:"差不多",bad:"都不好"};
function render(){const r=DATA[pos],s=S(r),done=DATA.filter(x=>st[x.item]&&st[x.item].choice).length;
 document.getElementById("prog").textContent=`已评 ${done}/${DATA.length}`;
 document.getElementById("card").innerHTML=`<div class="meta">第 ${pos+1}/${DATA.length} 条 · ${r.duration}s ${s.choice?"· 已选："+LAB[s.choice]:""}</div>
 <audio id="au" controls src="data:audio/mp4;base64,${AUDIO[r.item]}"></audio>
 <div class="opt"><div class="box ${s.choice=="A"?"sel":""}"><h3>A</h3>${esc(r.A)}</div><div class="box ${s.choice=="B"?"sel":""}"><h3>B</h3>${esc(r.B)}</div></div>
 <div class="row">${Object.entries(LAB).map(([k,v],i)=>`<button class="${s.choice==k?"on":""}" onclick="pick('${k}')">${i+1} ${v}</button>`).join("")}</div>
 <div class="row"><label><input type="checkbox" id="fa" ${s.addA?"checked":""} onchange="flag('addA',this.checked)"> A 凭空加了内容 / 回答了问题</label>
 <label><input type="checkbox" id="fb" ${s.addB?"checked":""} onchange="flag('addB',this.checked)"> B 凭空加了内容 / 回答了问题</label></div>
 <input class="note" id="nt" placeholder="备注（可选）" value="${esc(s.note||"")}" onchange="S(DATA[pos]).note=this.value;save()">`;
 const au=document.getElementById("au");au.play().catch(()=>{});t0=Date.now()}
function pick(c){const s=S(DATA[pos]);s.choice=c;s.seconds=(s.seconds||0)+Math.min(300,(Date.now()-t0)/1000);s.note=document.getElementById("nt").value;save();if(pos<DATA.length-1)pos++;render()}
function flag(k,v){S(DATA[pos])[k]=v;save()}
function go(d){pos=Math.min(Math.max(0,pos+d),DATA.length-1);render()}
function nextTodo(){const k=DATA.findIndex(x=>!(st[x.item]&&st[x.item].choice));if(k>=0){pos=k;render()}}
document.addEventListener("keydown",e=>{if(e.target.tagName=="INPUT"&&e.target.type!="checkbox")return;const au=document.getElementById("au");
 if(!e.altKey&&["1","2","3","4"].includes(e.key)){pick(["A","B","tie","bad"][+e.key-1]);return}
 if(e.altKey){const k={KeyP:()=>au.paused?au.play():au.pause(),KeyJ:()=>go(-1),KeyK:()=>go(1)}[e.code];if(k){e.preventDefault();k()}}});
function exportCsv(){const cols=["item","choice","addA","addB","note","seconds"];const cell=v=>{v=v==null?"":String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
 const lines=[cols.join(",")].concat(DATA.map(r=>{const s=st[r.item]||{};return [r.item,s.choice||"",s.addA?1:0,s.addB?1:0,s.note||"",Math.round(s.seconds||0)].map(cell).join(",")}));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+lines.join("\r\n")],{type:"text/csv"}));a.download=NAME+"-review.csv";a.click()}
render();</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sys", action="append", required=True, help="label=run_id (exactly two)")
    ap.add_argument("--manifest", default="data/manifests/benchmark_v1.jsonl")
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    (la, ra), (lb, rb) = (s.split("=", 1) for s in args.sys)
    load = lambda run: {p["utterance_id"]: p["hypothesis"] for p in map(json.loads, open(f"reports/runs/{run}/predictions.jsonl", encoding="utf-8"))}  # noqa: E731
    pa, pb = load(ra), load(rb)
    meta = {r["utterance_id"]: r for r in csv.DictReader(open("reports/runs/voicecursor_e2e_benchmark_v1/errors.csv", encoding="utf-8"))}
    man = {m["utterance_id"]: m for m in map(json.loads, open(args.manifest, encoding="utf-8"))}
    rng = random.Random(args.seed)
    pool = [u for u in man if u in pa and u in pb and pa[u].strip() and pb[u].strip() and pa[u].strip() != pb[u].strip()]
    strata = {"en_long": [], "en": [], "zh": [], "zh-en": []}
    for u in pool:
        mix = meta.get(u, {}).get("language_mix", "")
        key = ("en_long" if man[u]["duration_sec"] > 30 else "en") if mix == "en_only" else ("zh" if mix == "zh_only" else "zh-en")
        strata[key].append(u)
    want = {"en_long": 15, "en": 40, "zh": 20, "zh-en": 25}
    picked = [u for k, n in want.items() for u in rng.sample(strata[k], min(n, len(strata[k])))][: args.n]
    rng.shuffle(picked)
    out = Path("data/private/ab_review") / args.name
    out.mkdir(parents=True, exist_ok=True)
    data, key, audio = [], {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, u in enumerate(picked, 1):
            item = f"{args.name}-{k:03d}"
            a_is = rng.choice([la, lb])
            texts = {la: pa[u], lb: pb[u]}
            data.append({"item": item, "A": texts[a_is], "B": texts[lb if a_is == la else la],
                         "duration": round(man[u]["duration_sec"], 1)})
            key[item] = {"utterance_id": u, "A": a_is, "B": lb if a_is == la else la,
                         "stratum": next(s for s, v in strata.items() if u in v)}
            dst = Path(tmp) / f"{k}.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"data/private/{man[u]['audio_path']}", "-ac", "1", "-c:a", "aac", "-b:a", "48k", str(dst)], check=True)
            audio[item] = base64.b64encode(dst.read_bytes()).decode()
    js = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")  # noqa: E731
    (out / f"{args.name}.html").write_text(HTML.replace("__NAME__", args.name).replace("__DATA__", js(data)).replace("__AUDIO__", json.dumps(audio)), encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"systems": {la: ra, lb: rb}, "items": key}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{out / (args.name + '.html')}: {len(data)} items; key → {out / 'key.json'} (do not open before reviewing)")
    print("strata:", {k: sum(1 for v in key.values() if v["stratum"] == k) for k in strata})


if __name__ == "__main__":
    main()
