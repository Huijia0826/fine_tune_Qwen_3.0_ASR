"""Build an offline review page (review.html + audio/*.m4a) from a bench.py review.csv.

The page keeps progress in the browser (localStorage) and exports a review.csv with the same columns, plus
review_seconds, that bench.py apply-review can read. Nothing is uploaded anywhere.
With --embed, every audio/<id>.m4a is stored inside review.html (base64), so the page is one self-contained file.
--only ids.txt keeps just those audio_ids; --quality q.jsonl (scripts/audio_quality.py) shows SNR in each row's tags.
--refs "col:label,col:label,…" chooses the reference texts shown (default: the three bench.py columns); the first two
are diff-highlighted against each other, and each has a 采用 button that copies it into the text box.
--long (benchmark v2 long, 30–180 s clips): the text box grows to fit its text, the diff highlight works up to ~5,000
characters per text (default limit 250,000 character pairs ≈ 500 × 500), and ⌥B / ⌥F seek the audio −5 / +5 s.
Usage: uv run python scripts/build_review_html.py <bench dir> <pack dir> [--embed] [--only ids.txt] [--quality q.jsonl]
       [--refs …] [--name NAME] [--long]
"""

import base64
import csv
import json
import sys
from pathlib import Path

HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>审核 __NAME__</title>
<style>
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;display:flex;gap:14px;
 align-items:center;flex-wrap:wrap;z-index:2}
main{max-width:980px;margin:20px auto;padding:0 20px}
.card{background:#fff;border-radius:12px;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.meta{color:#666;font-size:13px;margin-bottom:10px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;background:#eee;margin-right:6px;font-size:12px}
.tag.high{background:#ffe0e0}.tag.ok{background:#d9f5df}.tag.fix{background:#fff1c2}.tag.drop{background:#e5e5e5}
.ref{margin:8px 0;font-size:15px;line-height:1.6}
.ref b{display:inline-block;width:92px;color:#888;font-weight:500;font-size:13px;vertical-align:top}
.ref span.t{display:inline-block;width:calc(100% - 160px)}
del{background:#ffd6d6;text-decoration:none}ins{background:#c9f0d2;text-decoration:none}
textarea{width:100%;box-sizing:border-box;font-size:17px;line-height:1.6;padding:10px;border:2px solid #0071e3;
 border-radius:8px;min-height:90px;font-family:inherit}
input.note{width:100%;box-sizing:border-box;padding:6px 10px;margin-top:8px;border:1px solid #ccc;border-radius:6px}
button{font-size:14px;padding:7px 14px;border-radius:8px;border:1px solid #ccc;background:#fff;cursor:pointer}
button.primary{background:#0071e3;color:#fff;border-color:#0071e3}button.danger{color:#c00}
.row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;align-items:center}
audio{width:100%;margin:6px 0 10px}
details{margin-top:18px;font-size:13px;color:#444}kbd{background:#eee;border-radius:4px;padding:0 5px;font-size:12px}
</style></head><body>
<header>
 <b>__NAME__</b><span id="prog"></span>
 <select id="filter"><option value="all">全部</option><option value="todo">只看未审</option>
  <option value="high">只看 high</option><option value="CS">中英混杂</option><option value="ZH">纯中文</option>
  <option value="S4">短音频</option><option value="S5">改口</option></select>
 <button onclick="jump(0)">⇤ 第一条</button><button onclick="jump(-1)">最后一条 ⇥</button>
 <label>跳到 <input id="goto" type="number" min="1" style="width:64px" onkeydown="if(event.key=='Enter')jump(this.value-1)"></label>
 <label><input type="checkbox" id="rev" onchange="dir=this.checked?-1:1;localStorage.setItem(KEY+'_rev',dir)"> 倒序审核</label>
 <button onclick="nextTodo()">下一条未审</button>
 <button onclick="exportCsv()">导出 review.csv</button>
 <label><button onclick="document.getElementById('imp').click()">导入 CSV</button>
  <input id="imp" type="file" accept=".csv" style="display:none" onchange="importCsv(this.files[0])"></label>
</header>
<main><div class="card" id="card"></div>
<details open><summary>审核规则 & 快捷键</summary>
<p><b>写成处理过的文字</b>：加标点；删掉当口头禅用的 就是/那个/然后/对（有实际意思的保留）；改口只留最终说法
（"周三，不对，周四"→"周四"）；口述的列表可以写成 1. 2.；数量和日期用阿拉伯数字。<br>
<b>不允许</b>：改写措辞、补充没说的内容、改动数字/否定/人名/指令。英文专有名词按标准拼写（VoiceCursor、Hugging Face）。<br>
<b>drop</b>：听不清、音频和文字明显对不上、含私人电话/地址等隐私。</p>
<p><kbd>⌘/Ctrl+Enter</kbd> 确认并下一条（文字没改 = ok，改过 = fix） · <kbd>⌥D</kbd> drop 并下一条 ·
<kbd>⌥P</kbd> 播放/暂停 · <kbd>⌥R</kbd> 从头播放 · <kbd>⌥J</kbd>/<kbd>⌥K</kbd> 上一条/下一条 ·
<kbd>⌥0</kbd> 恢复预填文字__LONGKEYS__。勾选"倒序审核"后，确认 / drop / 下一条未审都会往前走。进度自动保存在本浏览器；审完点"导出 review.csv"发给 Claude。</p>
</details></main>
__AUDIO__<script>
const NAME="__NAME__", COLS=__COLS__, DATA=__DATA__, REFS=__REFS__, LONG=__LONG__;
const KEY="wf_review_"+NAME;
let st=JSON.parse(localStorage.getItem(KEY)||"{}"), view=[], pos=0, t0=Date.now(), rate=1;
let dir=+(localStorage.getItem(KEY+"_rev")||1);  // 1 = forward, -1 = review from the end
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const save=()=>localStorage.setItem(KEY,JSON.stringify(st));
const S=r=>st[r.audio_id]||{};
function group(r){return {S1_zh_en:"CS",S3_terms:"CS",S2_zh:"ZH",S4_short:"S4",S5_selfcorr:"S5"}[r.scenario]}
function filt(){const f=document.getElementById("filter").value;
 view=DATA.map((r,i)=>i).filter(i=>{const r=DATA[i];return f=="all"||(f=="todo"&&!S(r).verdict)||
  (f=="high"&&r.priority=="high")||group(r)==f});}
function diff(a,b){a=[...(a||"")];b=[...(b||"")];if(a.length*b.length>(LONG?2.5e7:250000))return [esc(a.join("")),esc(b.join(""))];
 const n=a.length,m=b.length,d=Array.from({length:n+1},()=>new Uint16Array(m+1));
 for(let i=n-1;i>=0;i--)for(let j=m-1;j>=0;j--)d[i][j]=a[i]==b[j]?d[i+1][j+1]+1:Math.max(d[i+1][j],d[i][j+1]);
 let i=0,j=0,x="",y="";while(i<n||j<m){if(i<n&&j<m&&a[i]==b[j]){x+=esc(a[i]);y+=esc(b[j]);i++;j++}
  else if(j<m&&(i==n||d[i][j+1]>=d[i+1][j])){y+="<ins>"+esc(b[j])+"</ins>";j++}else{x+="<del>"+esc(a[i])+"</del>";i++}}
 return [x,y]}
function tick(){const r=DATA[view[pos]];if(!r)return;const s=st[r.audio_id]=S(r);
 s.seconds=(s.seconds||0)+Math.min(300,(Date.now()-t0)/1000);t0=Date.now();save()}
function render(){filt();if(pos>=view.length)pos=Math.max(0,view.length-1);const card=document.getElementById("card");
 const done=DATA.filter(r=>S(r).verdict),c=v=>done.filter(r=>S(r).verdict==v).length;
 const secs=done.reduce((a,r)=>a+(S(r).seconds||0),0);
 document.getElementById("prog").textContent=`已审 ${done.length}/${DATA.length}（ok ${c("ok")} · fix ${c("fix")} · drop ${c("drop")}）`
  +(done.length?` · 平均 ${(secs/done.length).toFixed(0)} 秒/条`:"");
 if(!view.length){card.innerHTML="这个筛选下没有条目。";return}
 const r=DATA[view[pos]],s=S(r),[d0,d1]=diff(r[REFS[0][0]],r[REFS[1][0]]);
 const refHtml=REFS.map(([c,l],i)=>`<div class="ref"><b>${esc(l)}</b><span class="t">${(i==0?d0:i==1?d1:esc(r[c]))||"<i>（空）</i>"}</span>
  <button style="font-size:12px;padding:2px 8px" onclick="useRef('${c}')">采用</button></div>`).join("");
 card.innerHTML=`<div class="meta">第 ${pos+1}/${view.length} 条 · <span class="tag ${r.priority}">${r.priority}</span>
  <span class="tag">${r.scenario}</span><span class="tag">${r.split}</span><span class="tag">${r.duration_s}s</span>${r.snr_db!==undefined?`<span class="tag">SNR ${r.snr_db} dB</span>`:""}
  ${r.seen?`<span class="tag">${r.seen=="yes"?"老用户":"新用户"}</span>`:""}${s.verdict?`<span class="tag ${s.verdict}">${s.verdict}</span>`:""}
  <span style="float:right">${esc(r.audio_id)}</span></div>
 <audio id="au" controls src="${audioSrc(r.audio_id)}"></audio><div id="aumsg" style="color:#c00;font-size:13px"></div>
 <div class="row" style="margin-top:0">速度 ${[1,1.25,1.5].map(v=>`<button onclick="setRate(${v})">${v}x</button>`).join("")}</div>
 ${refHtml}
 <textarea id="tx">${esc(s.final_transcript!=null?s.final_transcript:r.final_transcript)}</textarea>
 <input class="note" id="nt" placeholder="备注（可选，比如 场景标错了 / 不是改口）" value="${esc(s.note!=null?s.note:(r.note||""))}">
 <div class="row"><button class="primary" onclick="accept()">确认并下一条 ⌘↵</button>
  <button class="danger" onclick="drop()">drop ⌥D</button><button onclick="go(-1)">← 上一条</button>
  <button onclick="go(1)">下一条 →</button><button onclick="resetText()">恢复预填</button></div>`;
 const au=document.getElementById("au"),msg=document.getElementById("aumsg");au.playbackRate=rate;
 au.onerror=()=>{msg.innerHTML="音频加载失败：请先<b>解压 zip</b>，并确认 audio 文件夹和 review.html 在同一目录。"
  +"用 Safari 打不开的话，请双击 <b>start_review.command</b>，或改用 Chrome。"};
 au.play().catch(e=>{if(e.name=="NotAllowedError")msg.textContent="浏览器拦截了自动播放：点一下播放键（▶）或按 ⌥P，之后就会自动播放。"});
 const tx=document.getElementById("tx");if(LONG){fit();tx.oninput=fit}tx.focus();
 tx.setSelectionRange(tx.value.length,tx.value.length)}
function fit(){const tx=document.getElementById("tx");tx.style.height="auto";tx.style.height=(tx.scrollHeight+4)+"px"}
let blobUrl=null;
function audioSrc(id){  // embedded audio → blob URL (one at a time); otherwise the file next to the page
 if(typeof AUDIO=="undefined")return "audio/"+id+".m4a";if(blobUrl)URL.revokeObjectURL(blobUrl);
 const b=atob(AUDIO[id]||""),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
 blobUrl=URL.createObjectURL(new Blob([u],{type:"audio/mp4"}));return blobUrl}
function setRate(v){rate=v;document.getElementById("au").playbackRate=v}
function cur(){return DATA[view[pos]]}
function record(verdict){const r=cur(),s=st[r.audio_id]=S(r),tx=document.getElementById("tx").value.trim();
 s.final_transcript=tx;s.note=document.getElementById("nt").value;
 s.verdict=verdict||(tx==(r.final_transcript||"").trim()?"ok":"fix");save()}
function go(d){tick();pos=Math.min(Math.max(0,pos+d),view.length-1);render()}
function advance(){tick();const todo=document.getElementById("filter").value=="todo";
 // in "todo" view the reviewed row disappears, so going forward means staying at the same index
 pos=todo?(dir>0?pos:pos-1):pos+dir;pos=Math.min(Math.max(0,pos),view.length-1);render()}
function accept(){record();advance()}
function drop(){record("drop");advance()}
function jump(i){tick();filt();i=+i;pos=i<0?view.length-1:Math.min(Math.max(0,i),view.length-1);render()}
function useRef(c){const tx=document.getElementById("tx");tx.value=cur()[c]||"";if(LONG)fit();tx.focus()}
function resetText(){document.getElementById("tx").value=cur().final_transcript||"";if(LONG)fit()}
function nextTodo(){tick();filt();const todo=p=>!S(DATA[view[p]]).verdict,idx=[...view.keys()];
 const ahead=dir>0?idx.filter(p=>p>pos):idx.filter(p=>p<pos).reverse();
 const all=dir>0?idx:idx.slice().reverse();const k=ahead.find(todo);pos=k!=null?k:(all.find(todo)??pos);render()}
document.getElementById("filter").onchange=()=>{tick();pos=0;render()};
document.addEventListener("keydown",e=>{const au=document.getElementById("au");
 if((e.metaKey||e.ctrlKey)&&e.key=="Enter"){e.preventDefault();accept();return}
 if(!e.altKey)return;const k={KeyD:drop,KeyJ:()=>go(-1),KeyK:()=>go(1),Digit0:resetText,
  KeyP:()=>au&&(au.paused?au.play():au.pause()),KeyR:()=>{if(au){au.currentTime=0;au.play()}},
  Enter:accept,...(LONG&&au?{KeyB:()=>{au.currentTime=Math.max(0,au.currentTime-5)},
  KeyF:()=>{au.currentTime=Math.min(au.duration||1e9,au.currentTime+5)}}:{})}[e.code];if(k){e.preventDefault();k()}});
function csvCell(v){v=v==null?"":String(v);return /[",\n\r]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
function exportCsv(){tick();const cols=COLS.concat(["review_seconds"]);
 const lines=[cols.join(",")].concat(DATA.map(r=>{const s=S(r);const o=Object.assign({},r,{
  final_transcript:s.final_transcript!=null?s.final_transcript:r.final_transcript,verdict:s.verdict||"",
  note:s.note!=null?s.note:(r.note||""),review_seconds:s.seconds?Math.round(s.seconds):""});return cols.map(c=>csvCell(o[c])).join(",")}));
 const b=new Blob(["﻿"+lines.join("\r\n")],{type:"text/csv"});const a=document.createElement("a");
 a.href=URL.createObjectURL(b);a.download="review.csv";a.click()}
function parseCsv(t){const rows=[];let row=[],f="",q=false;t=t.replace(/^﻿/,"");
 for(let i=0;i<t.length;i++){const c=t[i];if(q){if(c=='"'){if(t[i+1]=='"'){f+='"';i++}else q=false}else f+=c}
  else if(c=='"')q=true;else if(c==","){row.push(f);f=""}else if(c=="\n"||c=="\r"){if(c=="\r"&&t[i+1]=="\n")i++;
   row.push(f);rows.push(row);row=[];f=""}else f+=c}
 if(f||row.length){row.push(f);rows.push(row)}const h=rows.shift();return rows.map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])))}
function importCsv(file){file.text().then(t=>{let n=0;for(const r of parseCsv(t)){if(!r.audio_id||!r.verdict)continue;
 st[r.audio_id]={final_transcript:r.final_transcript,verdict:r.verdict,note:r.note||"",
  seconds:+r.review_seconds||0};n++}save();render();alert(`已导入 ${n} 条审核结果`)})}
window.addEventListener("beforeunload",tick);
document.getElementById("rev").checked=dir<0;
if(dir<0){filt();pos=view.length-1}
render();
</script></body></html>
"""


def main() -> None:
    bench, pack, embed, long = Path(sys.argv[1]), Path(sys.argv[2]), "--embed" in sys.argv[3:], "--long" in sys.argv[3:]
    opt = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None  # noqa: E731
    with open(bench / "review.csv", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    if opt("--only"):
        keep = set(Path(opt("--only")).read_text().split())
        rows = [r for r in rows if r["audio_id"] in keep]
    if opt("--quality"):
        q = {x["audio_id"]: x for x in map(json.loads, open(opt("--quality")))}
        for r in rows:
            r["snr_db"] = q.get(r["audio_id"], {}).get("snr_db", "")
    name = opt("--name") or bench.name
    refs = [c.split(":", 1) for c in (opt("--refs") or "draft_1:Whisper 草稿,online_raw:线上识别,online_output:线上输出").split(",")]
    js = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")  # noqa: E731  (safe inside <script>)
    audio = ""
    if embed:
        clips = {r["audio_id"]: base64.b64encode((pack / "audio" / f"{r['audio_id']}.m4a").read_bytes()).decode()
                 for r in rows if (pack / "audio" / f"{r['audio_id']}.m4a").exists()}
        audio = "<script>const AUDIO=" + json.dumps(clips) + ";</script>\n"
    html = (HTML.replace("__AUDIO__", audio).replace("__NAME__", name).replace("__COLS__", js(cols))
            .replace("__DATA__", js(rows)).replace("__REFS__", js(refs)).replace("__LONG__", js(long))
            .replace("__LONGKEYS__", " · <kbd>⌥B</kbd>/<kbd>⌥F</kbd> 后退/前进 5 秒" if long else ""))
    (pack / "review.html").write_text(html, encoding="utf-8")
    missing = [r["audio_id"] for r in rows if not (pack / "audio" / f"{r['audio_id']}.m4a").exists()]
    size = (pack / "review.html").stat().st_size / 1e6
    print(f"{pack / 'review.html'}: {len(rows)} rows, {len(missing)} missing audio, {size:.1f} MB"
          + (" (audio embedded)" if embed else ""))


if __name__ == "__main__":
    main()
