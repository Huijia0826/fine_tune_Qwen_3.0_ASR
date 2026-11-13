# ruff: noqa: E501  (the HTML / JS template is kept on long lines, as in build_review_html.py)
"""Unified review page for the end-to-end dataset (owner, 2026-11-09): one card per clip with the audio, the context
the model will see (app / tone, vocabulary, text before / after the cursor, screen text) and TWO text boxes —
strict transcript (policy v1) and final output — so labels, context and targets are reviewed together.

Input: an items.jsonl (scripts/make_e2e_items.py), one object per clip:
  item_id, source, split, audio_src (local file), duration_s, language, app, bundle_id, tone, target_vocab [..],
  user_vocab [..], before, after, screen, old_label, transcript, final_output, ref_final (production output,
  shown for reference with a button that copies it into the final-output box), note; optional draft + draft_model (a
  model's transcript, shown as a diff against the strict-transcript box, for reference only)
Output: <out>/<name>-<k>/{<name>-<k>.html (audio embedded), README.txt} + .zip, packs of --size items.
The page exports review.csv: item_id, source, split, transcript, final_output, verdict (ok / fix / drop), note,
review_seconds — read back by scripts/make_e2e_items.py apply-review.

    uv run python scripts/build_e2e_review.py data/private/e2e/items_dict.jsonl data/private/e2e/review --name e2e-dict
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_review_benchmark_v2 import transcode_pcm  # noqa: E402

HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>审核 __NAME__</title>
<style>
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;display:flex;gap:12px;
 align-items:center;flex-wrap:wrap;z-index:2}
main{max-width:1020px;margin:18px auto;padding:0 20px}
.card{background:#fff;border-radius:12px;padding:18px 24px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.meta{color:#666;font-size:13px;margin-bottom:8px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;background:#eee;margin-right:6px;font-size:12px}
.tag.ok{background:#d9f5df}.tag.fix{background:#fff1c2}.tag.drop{background:#e5e5e5}
.ctx{background:#f7f9fc;border:1px solid #e3e8ef;border-radius:8px;padding:8px 12px;font-size:13px;line-height:1.55;margin:6px 0}
.ctx b{color:#667;font-weight:500;display:inline-block;width:76px;vertical-align:top}
.ctx .v{display:inline-block;width:calc(100% - 84px);white-space:pre-wrap;word-break:break-word}
.chip{display:inline-block;padding:0 7px;border-radius:9px;margin:1px 4px 1px 0;font-size:12px;border:1px solid #ccc}
.chip.hit{background:#d9f5df;border-color:#8bd19c}.chip.miss{background:#ffe0e0;border-color:#f0a0a0}
.lbl{font-size:13px;color:#555;margin:10px 0 3px}
.old{font-size:14px;line-height:1.6;color:#444;margin:4px 0}
del{background:#ffd6d6;text-decoration:none}ins{background:#c9f0d2;text-decoration:none}
textarea{width:100%;box-sizing:border-box;font-size:16px;line-height:1.55;padding:8px 10px;border:2px solid #0071e3;
 border-radius:8px;min-height:64px;font-family:inherit}
textarea.fin{border-color:#7a4dd6}
input.note{width:100%;box-sizing:border-box;padding:6px 10px;margin-top:8px;border:1px solid #ccc;border-radius:6px}
button{font-size:14px;padding:6px 13px;border-radius:8px;border:1px solid #ccc;background:#fff;cursor:pointer}
button.primary{background:#0071e3;color:#fff;border-color:#0071e3}button.danger{color:#c00}
.row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
audio{width:100%;margin:4px 0 6px}details{font-size:13px;color:#444}kbd{background:#eee;border-radius:4px;padding:0 5px}
</style></head><body>
<header><b>__NAME__</b><span id="prog"></span>
 <select id="filter"><option value="all">全部</option><option value="todo">只看未审</option>
  <option value="vmiss">目标词没出现在转写里</option><option value="changed">v1 改动过的</option></select>
 <button onclick="jump(0)">⇤</button><button onclick="jump(-1)">⇥</button>
 <label>跳到 <input id="goto" type="number" min="1" style="width:60px" onkeydown="if(event.key=='Enter')jump(this.value-1)"></label>
 <button onclick="nextTodo()">下一条未审</button><button onclick="exportCsv()">导出 review.csv</button>
 <label><button onclick="document.getElementById('imp').click()">导入 CSV</button>
  <input id="imp" type="file" accept=".csv" style="display:none" onchange="importCsv(this.files[0])"></label>
</header>
<main><div class="card" id="card"></div>
<details open style="margin-top:14px"><summary>怎么审 & 快捷键</summary>
<p><b>① 严格转写</b>（docs/annotation_policy.md v1）：说了什么写什么，改口、重复都保留；去掉 嗯/呃/um/uh，"啊"当口头禅时去掉、
当句末语气词时保留；数量 / 时间 / 日期用阿拉伯数字（一个、两三个、等一下这类保留汉字）；中文里的英文词前后各一个空格；
拼读字母写成 V-E-R-C-E-L。灰色一行是旧标注，红绿是 v1 自动改动。<br>
<b>注意</b>：听错的地方（错字、人名、专有名词）请改在 ① 严格转写里；② 只放最终整理（标点、去口头禅、精简、格式、按词表统一写法）。<br>
<b>② 最终输出</b>：用户最后应该看到的文字（预填 = 严格转写，请改成想要的样子：比如按语气处理句末句号、应用改口等）。<br>
<b>上下文</b>只是参考（模型训练时会看到它）。目标词绿色 = 转写里有，红色 = 没有（检查是不是听错 / 拼错）。<br>
<b>drop</b>：听不清、音频和文字对不上、含隐私。</p>
<p><kbd>⌘/Ctrl+Enter</kbd> 确认并下一条（两个框都没改 = ok，改过 = fix） · <kbd>⌥D</kbd> drop · <kbd>⌥P</kbd> 播放/暂停 ·
<kbd>⌥R</kbd> 从头播放 · <kbd>⌥B</kbd>/<kbd>⌥F</kbd> 后退/前进 5 秒 · <kbd>⌥J</kbd>/<kbd>⌥K</kbd> 上一条/下一条 ·
<kbd>⌥C</kbd> 把转写复制到最终输出 · <kbd>⌥U</kbd> 采用线上输出 · <kbd>⌥0</kbd> 恢复预填。进度保存在本浏览器；审完点"导出 review.csv"发给 Claude。</p>
</details></main>
__AUDIO__<script>
const NAME="__NAME__", DATA=__DATA__, KEY="wf_e2e_"+NAME;
let st=JSON.parse(localStorage.getItem(KEY)||"{}"), view=[], pos=0, t0=Date.now(), rate=1;
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const save=()=>localStorage.setItem(KEY,JSON.stringify(st)), S=r=>st[r.item_id]||{};
const txt=r=>S(r).transcript!=null?S(r).transcript:r.transcript, fin=r=>S(r).final_output!=null?S(r).final_output:r.final_output;
const norm=s=>(s||"").toLowerCase().replace(/[\s\-_.]/g,"");
const vmiss=r=>(r.target_vocab||[]).some(v=>!norm(txt(r)).includes(norm(v)));
function filt(){const f=document.getElementById("filter").value;view=DATA.map((r,i)=>i).filter(i=>{const r=DATA[i];
 return f=="all"||(f=="todo"&&!S(r).verdict)||(f=="vmiss"&&vmiss(r))||(f=="changed"&&r.old_label!=r.transcript)})}
function diff(a,b){a=[...(a||"")];b=[...(b||"")];if(a.length*b.length>4e6)return esc(b.join(""));
 const n=a.length,m=b.length,d=Array.from({length:n+1},()=>new Uint16Array(m+1));
 for(let i=n-1;i>=0;i--)for(let j=m-1;j>=0;j--)d[i][j]=a[i]==b[j]?d[i+1][j+1]+1:Math.max(d[i+1][j],d[i][j+1]);
 let i=0,j=0,y="";while(i<n||j<m){if(i<n&&j<m&&a[i]==b[j]){y+=esc(b[j]);i++;j++}
  else if(j<m&&(i==n||d[i][j+1]>=d[i+1][j])){y+="<ins>"+esc(b[j])+"</ins>";j++}else{y+="<del>"+esc(a[i])+"</del>";i++}}
 return y}
function chips(list,t){return (list||[]).map(v=>`<span class="chip ${norm(t).includes(norm(v))?"hit":"miss"}">${esc(v)}</span>`).join("")}
// long context: show the part next to the cursor (what the model sees), the full text behind a click
function clip(l,v,n){if(!v)return "";const part=n<0?v.slice(n):v.slice(0,n);if(part.length==v.length)return ctxRow(l,v);
 return `<div><b>${l}</b><span class="v"><details><summary>${n<0?"…":""}${esc(part)}${n>0?"…":""}</summary>${esc(v)}</details></span></div>`}
function ctxRow(l,v,html){return v&&(v.length||v.length===undefined)?`<div><b>${l}</b><span class="v">${html||esc(v)}</span></div>`:""}
function tick(){const r=DATA[view[pos]];if(!r)return;const s=st[r.item_id]=S(r);s.seconds=(s.seconds||0)+Math.min(300,(Date.now()-t0)/1000);t0=Date.now();save()}
function render(){filt();if(pos>=view.length)pos=Math.max(0,view.length-1);const card=document.getElementById("card");
 const done=DATA.filter(r=>S(r).verdict),c=v=>done.filter(r=>S(r).verdict==v).length,secs=done.reduce((a,r)=>a+(S(r).seconds||0),0);
 document.getElementById("prog").textContent=`已审 ${done.length}/${DATA.length}（ok ${c("ok")} · fix ${c("fix")} · drop ${c("drop")}）`+(done.length?` · ${(secs/done.length).toFixed(0)} 秒/条`:"");
 if(!view.length){card.innerHTML="这个筛选下没有条目。";return}
 const r=DATA[view[pos]],s=S(r),scr=r.screen||"";
 card.innerHTML=`<div class="meta">第 ${pos+1}/${view.length} 条 <span class="tag">${esc(r.source)}</span><span class="tag">${esc(r.split)}</span>
  <span class="tag">${r.duration_s}s</span>${r.language?`<span class="tag">${esc(r.language)}</span>`:""}${s.verdict?`<span class="tag ${s.verdict}">${s.verdict}</span>`:""}
  <span style="float:right">${esc(r.item_id)}</span></div>
 <audio id="au" controls src="${audioSrc(r.item_id)}"></audio>
 <div class="row" style="margin-top:0">速度 ${[1,1.25,1.5].map(v=>`<button onclick="setRate(${v})">${v}x</button>`).join("")}</div>
 <div class="ctx">${ctxRow("App / 语气",r.app||r.tone,esc((r.app||"无")+(r.bundle_id?" · "+r.bundle_id:""))+" → <b style='width:auto;color:#333'>"+esc(r.tone||"unknown")+"</b>")}
  ${ctxRow("目标词",r.target_vocab,chips(r.target_vocab,txt(r)))}${ctxRow("用户词典",r.user_vocab,chips(r.user_vocab,txt(r)))}
  ${clip("光标前",r.before,-200)}${clip("光标后",r.after,120)}
  ${scr?`<div><b>屏幕文字</b><span class="v"><details><summary>${esc(scr.slice(0,80).replace(/\n/g," "))}${scr.length>80?"…":""}</summary>${esc(scr)}</details></span></div>`:""}
  ${!(r.app||r.before||r.after||scr||(r.user_vocab||[]).length||(r.target_vocab||[]).length)?"<i>没有上下文</i>":""}</div>
 ${r.old_label&&r.old_label!=r.transcript?`<div class="old">旧标注 → v1：${diff(r.old_label,r.transcript)}</div>`:""}
 ${r.draft&&r.draft!=txt(r)?`<div class="old">模型草稿（${esc(r.draft_model||"")}，参考）：${diff(txt(r),r.draft)}</div>`:""}
 ${r.ref_final?`<div class="old">线上输出（参考）：${esc(r.ref_final)} <button style="font-size:12px;padding:1px 8px" onclick="useRefFinal()">⌥U 采用为最终输出</button></div>`:""}
 <div class="lbl">① 严格转写</div><textarea id="tx">${esc(txt(r))}</textarea>
 <div class="lbl">② 最终输出 <button style="font-size:12px;padding:1px 8px" onclick="copyTx()">⌥C 复制转写</button></div>
 <textarea id="fo" class="fin">${esc(fin(r))}</textarea>
 <input class="note" id="nt" placeholder="备注（可选）" value="${esc(s.note!=null?s.note:(r.note||""))}">
 <div class="row"><button class="primary" onclick="accept()">确认并下一条 ⌘↵</button><button class="danger" onclick="drop()">drop ⌥D</button>
  <button onclick="go(-1)">← 上一条</button><button onclick="go(1)">下一条 →</button><button onclick="resetText()">恢复预填</button></div>`;
 const au=document.getElementById("au");au.playbackRate=rate;au.play().catch(()=>{});
 for(const id of ["tx","fo"]){const el=document.getElementById(id);fit(el);el.oninput=()=>fit(el)}
 const tx=document.getElementById("tx");tx.focus({preventScroll:true});tx.setSelectionRange(tx.value.length,tx.value.length);window.scrollTo(0,0)}
function fit(el){el.style.height="auto";el.style.height=(el.scrollHeight+4)+"px"}
let blobUrl=null;function audioSrc(id){if(blobUrl)URL.revokeObjectURL(blobUrl);const b=atob(AUDIO[id]||""),u=new Uint8Array(b.length);
 for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);blobUrl=URL.createObjectURL(new Blob([u],{type:"audio/mp4"}));return blobUrl}
function setRate(v){rate=v;document.getElementById("au").playbackRate=v}
const cur=()=>DATA[view[pos]];
function record(verdict){const r=cur(),s=st[r.item_id]=S(r),tx=document.getElementById("tx").value.trim(),fo=document.getElementById("fo").value.trim();
 s.transcript=tx;s.final_output=fo;s.note=document.getElementById("nt").value;
 s.verdict=verdict||(tx==(r.transcript||"").trim()&&fo==(r.final_output||"").trim()?"ok":"fix");save()}
function go(d){tick();pos=Math.min(Math.max(0,pos+d),view.length-1);render()}
function advance(){tick();pos=document.getElementById("filter").value=="todo"?pos:pos+1;pos=Math.min(pos,view.length-1);render()}
function accept(){record();advance()}function drop(){record("drop");advance()}
function jump(i){tick();filt();i=+i;pos=i<0?view.length-1:Math.min(Math.max(0,i),view.length-1);render()}
function useRefFinal(){const fo=document.getElementById("fo"),r=cur();if(r.ref_final){fo.value=r.ref_final;fit(fo)}}
function copyTx(){const fo=document.getElementById("fo");fo.value=document.getElementById("tx").value;fit(fo)}
function resetText(){const r=cur();document.getElementById("tx").value=r.transcript||"";document.getElementById("fo").value=r.final_output||"";
 for(const id of ["tx","fo"])fit(document.getElementById(id))}
function nextTodo(){tick();filt();const k=view.findIndex((v,p)=>p>pos&&!S(DATA[v]).verdict);pos=k>=0?k:Math.max(0,view.findIndex(v=>!S(DATA[v]).verdict));render()}
document.getElementById("filter").onchange=()=>{tick();pos=0;render()};
document.addEventListener("keydown",e=>{const au=document.getElementById("au");
 if((e.metaKey||e.ctrlKey)&&e.key=="Enter"){e.preventDefault();accept();return}if(!e.altKey)return;
 const k={KeyD:drop,KeyJ:()=>go(-1),KeyK:()=>go(1),Digit0:resetText,KeyC:copyTx,KeyU:useRefFinal,KeyP:()=>au&&(au.paused?au.play():au.pause()),
  KeyR:()=>{if(au){au.currentTime=0;au.play()}},KeyB:()=>{if(au)au.currentTime=Math.max(0,au.currentTime-5)},
  KeyF:()=>{if(au)au.currentTime=Math.min(au.duration||1e9,au.currentTime+5)}}[e.code];if(k){e.preventDefault();k()}});
function cell(v){v=v==null?"":String(v);return /[",\n\r]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
const COLS=["item_id","source","split","transcript","final_output","verdict","note","review_seconds","prefill_transcript","prefill_final_output","old_label"];
function exportCsv(){tick();const lines=[COLS.join(",")].concat(DATA.map(r=>{const s=S(r);const o={item_id:r.item_id,source:r.source,split:r.split,
  transcript:txt(r),final_output:fin(r),verdict:s.verdict||"",note:s.note!=null?s.note:(r.note||""),review_seconds:s.seconds?Math.round(s.seconds):"",
  prefill_transcript:r.transcript,prefill_final_output:r.final_output,old_label:r.old_label};return COLS.map(c=>cell(o[c])).join(",")}));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+lines.join("\r\n")],{type:"text/csv"}));a.download=NAME+"-review.csv";a.click()}
function parseCsv(t){const rows=[];let row=[],f="",q=false;t=t.replace(/^﻿/,"");
 for(let i=0;i<t.length;i++){const c=t[i];if(q){if(c=='"'){if(t[i+1]=='"'){f+='"';i++}else q=false}else f+=c}
  else if(c=='"')q=true;else if(c==","){row.push(f);f=""}else if(c=="\n"||c=="\r"){if(c=="\r"&&t[i+1]=="\n")i++;row.push(f);rows.push(row);row=[];f=""}else f+=c}
 if(f||row.length){row.push(f);rows.push(row)}const h=rows.shift();return rows.map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])))}
function importCsv(file){file.text().then(t=>{let n=0;for(const r of parseCsv(t)){if(!r.item_id||!r.verdict)continue;
 st[r.item_id]={transcript:r.transcript,final_output:r.final_output,verdict:r.verdict,note:r.note||"",seconds:+r.review_seconds||0};n++}
 save();render();alert(`已导入 ${n} 条`)})}
window.addEventListener("beforeunload",tick);render();
</script></body></html>
"""

README = """{name}：{n} 条，音频约 {minutes:.0f} 分钟（单文件网页，音频已嵌入）。
1. 解压后用 Chrome 打开 {name}.html。
2. 每条：听音频 → 看上下文 → 改 ① 严格转写 → 改 ② 最终输出 → ⌘+Enter 确认（或 ⌥D drop）。规则写在页面底部。
3. 进度自动保存在浏览器里；审完点"导出 review.csv"，把文件交给 Claude。
注意：里面有线上用户的真实录音和上下文，请不要转发或上传到其他地方。
"""


def transcode(src: str, dst: Path) -> None:
    if src.endswith(".ogg"):
        transcode_pcm(src, dst)  # iOS Ogg timestamps can be wrong (see transcode_pcm)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-c:a", "aac", "-b:a", "64k",
                        str(dst)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", type=int, default=200)
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    js = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")  # noqa: E731
    for k in range(0, len(items), args.size):
        pk = items[k:k + args.size]
        name = f"{args.name}-{k // args.size + 1}" if len(items) > args.size else args.name
        pack = Path(args.out) / name
        pack.mkdir(parents=True, exist_ok=True)
        clips = {}
        with tempfile.TemporaryDirectory() as tmp:
            for r in pk:
                dst = Path(tmp) / f"{r['item_id']}.m4a"
                transcode(r["audio_src"], dst)
                clips[r["item_id"]] = base64.b64encode(dst.read_bytes()).decode()
        data = [{k2: v for k2, v in r.items() if k2 != "audio_src"} for r in pk]
        html = (HTML.replace("__NAME__", name).replace("__DATA__", js(data))
                .replace("__AUDIO__", "<script>const AUDIO=" + json.dumps(clips) + ";</script>\n"))
        (pack / f"{name}.html").write_text(html, encoding="utf-8")
        minutes = sum(float(r.get("duration_s") or 0) for r in pk) / 60
        (pack / "README.txt").write_text(README.format(name=name, n=len(pk), minutes=minutes), encoding="utf-8")
        with zipfile.ZipFile(pack / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pack / f"{name}.html", f"{name}/{name}.html")
            z.write(pack / "README.txt", f"{name}/README.txt")
        print(f"{pack}: {len(pk)} items, {minutes:.0f} min audio, {(pack / f'{name}.html').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
