"""Error browser: every error span of one run, aligned to the reference, with an attribution the owner can confirm.

For each clip with errors: audio (embedded), the reference with inline diff against the run's output (red = reference
tokens missed or replaced, green = output tokens added or substituted), a comparison system's output for context,
and one row per error span with a pre-filled attribution. The page exports a CSV (one row per span) for the summary.

Spans come from our tokenizer's alignment (eval/cleaning.extract_spans), so their count can differ slightly from the
official scorer's error count, which is shown per clip as well.
Usage: uv run --with openpyxl python scripts/build_error_browser.py --run bv1_lora_large_v3_B_step_800 \
         --compare voicecursor_e2e_benchmark_v1
"""

import argparse
import base64
import csv
import json
import re
from pathlib import Path

import openpyxl

from whisperflow.eval.cleaning import classify_span, join_tokens
from whisperflow.eval.metrics import align_text
from whisperflow.eval.official import official_mer
from whisperflow.text.normalize import is_cjk

CATS = [("rec_entity", "专有名词/术语听错"), ("rec_common", "普通词听错"), ("drop_segment", "漏段（≥4 字）"),
        ("drop_word", "少量漏字"), ("extra", "多出/幻觉/重复"), ("format", "格式/数字"), ("filler", "口头禅/风格"),
        ("self_correction", "改口处理"), ("ref_error", "参考文字有误"), ("other", "其他")]
NUM = re.compile(r"[\d零一二三四五六七八九十百千万两点%]")


def proper_nouns(text: str) -> set[str]:
    """Lower-cased words written with a capital inside a sentence (not right after . ! ? or at the start)."""
    out = set()
    for m in re.finditer(r"[A-Za-z][\w'-]*", text):
        w, before = m.group(0), text[:m.start()].rstrip()
        mid_sentence_capital = w[0].isupper() and before and before[-1] not in ".!?。！？"
        if w != "I" and (mid_sentence_capital or re.search(r"[A-Z].*[A-Z]", w)):  # Marc, OpenAI, API
            out.add(w.lower())
    return out


def guess(kind: str, ref: list[str], hyp: list[str], cat: str, names: set[str]) -> str:
    """Rule-based first guess; the owner corrects it in the page. names = proper nouns of the reference."""
    if kind == "del":
        return "drop_segment" if len(ref) >= 4 else "drop_word"
    if kind == "ins":
        return "filler" if cat in ("filler", "repeat") else "extra"
    if any(NUM.search(t) for t in ref + hyp):
        return "format"
    if any(t in names for t in ref):
        return "rec_entity"
    return "rec_common"


def segments(ref: str, hyp: str) -> list[dict]:
    """Alignment as a list of {"eq": text} or {"span": i, "kind", "ref", "hyp"} segments."""
    from whisperflow.eval.cleaning import Span

    ops = align_text(ref, hyp).ops
    names = proper_nouns(ref)
    out, run, eq, hyp_seen, n = [], [], [], [], 0

    def flush_eq():
        if eq:
            out.append({"eq": join_tokens(eq)})
            eq.clear()

    def flush_run():
        nonlocal n
        if not run:
            return
        kinds = {op for op, _, _ in run}
        kind = "sub" if "sub" in kinds or kinds == {"ins", "del"} else kinds.pop()
        r = [x for _, x, _ in run if x is not None]
        h = [y for _, _, y in run if y is not None]
        cat = classify_span(Span(kind, r, h, hyp_seen[:len(hyp_seen) - len(h)], []))
        out.append({"span": n, "kind": kind, "ref": join_tokens(r), "hyp": join_tokens(h), "auto": cat,
                    "guess": guess(kind, r, h, cat, names), "n_ref": len(r), "n_hyp": len(h)})
        n += 1
        run.clear()

    for op, r, h in ops:
        if op == "eq":
            flush_run()
            eq.append(r)
        else:
            flush_eq()
            run.append((op, r, h))
        if h is not None:
            hyp_seen.append(h)
    flush_run()
    flush_eq()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", help="another run shown for context (e.g. VoiceCursor)")
    ap.add_argument("--manifest", default="data/manifests/benchmark_v1.jsonl")
    ap.add_argument("--tags", default="/Users/huijia/Downloads/01a194b2-2262-75ff-acca-abb5aae2cb95/ground_truth.xlsx")
    ap.add_argument("--audio", default="data/private/error_browser/audio")
    ap.add_argument("--out", default="data/private/error_browser/errors.html")
    args = ap.parse_args()

    man = {json.loads(l)["utterance_id"]: json.loads(l) for l in open(args.manifest, encoding="utf-8")}
    hyp = {json.loads(l)["utterance_id"]: json.loads(l)["hypothesis"]
           for l in open(f"reports/runs/{args.run}/predictions.jsonl", encoding="utf-8")}
    cmp_ = ({json.loads(l)["utterance_id"]: json.loads(l)["hypothesis"]
             for l in open(f"reports/runs/{args.compare}/predictions.jsonl", encoding="utf-8")} if args.compare else {})
    tags = {}
    if args.tags and Path(args.tags).exists():
        rows = [r for r in openpyxl.load_workbook(args.tags).active.iter_rows(values_only=True) if any(r)]
        head = list(rows[0])
        for r in rows[1:]:
            d = dict(zip(head, r))
            tags[str(d["file_name"]).rsplit(".", 1)[0]] = {"primary_tag": d.get("primary_tag") or "",
                                                            "labels": d.get("labels") or "", "notes": d.get("notes") or ""}
    clips = []
    for k, h in hyp.items():
        r = man[k]
        if r["split"] != "test":
            continue
        o = official_mer(r["raw_transcript"], h)
        if not o.included or o.errors == 0:
            continue
        sid = re.sub(r"\.\w+$", "", r["source_utterance_id"])
        lang = ("zh-en" if any(is_cjk(c) for c in r["raw_transcript"]) and re.search(r"[A-Za-z]{2,}", r["raw_transcript"])
                else "zh" if any(is_cjk(c) for c in r["raw_transcript"]) else "en")
        a = Path(args.audio) / f"{k}.m4a"
        clips.append({"id": k, "dur": round(r["duration_sec"], 1), "lang": lang, "errors": o.errors, "n_ref": o.n_ref,
                      "ref": r["raw_transcript"], "hyp": h, "cmp": cmp_.get(k, ""), **tags.get(sid, {}),
                      "segs": segments(r["raw_transcript"], h),
                      "audio": base64.b64encode(a.read_bytes()).decode() if a.exists() else ""})
    clips.sort(key=lambda c: -c["errors"])
    total = sum(c["errors"] for c in clips)
    html = HTML.replace("__DATA__", json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")) \
               .replace("__CATS__", json.dumps(CATS, ensure_ascii=False)).replace("__RUN__", args.run) \
               .replace("__CMP__", args.compare or "").replace("__TOTAL__", str(total))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{args.out}: {len(clips)} clips with errors, {total} official errors, "
          f"{sum(1 for c in clips for s in c['segs'] if 'span' in s)} spans, {Path(args.out).stat().st_size / 1e6:.1f} MB")


HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>错误浏览 __RUN__</title><style>
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;display:flex;gap:12px;
 align-items:center;flex-wrap:wrap;z-index:2;font-size:14px}
main{max-width:1100px;margin:16px auto;padding:0 20px}
.card{background:#fff;border-radius:12px;padding:16px 20px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.meta{font-size:13px;color:#555;margin-bottom:6px}.tag{display:inline-block;padding:1px 8px;border-radius:10px;
 background:#eee;margin-right:6px;font-size:12px}.tag.bad{background:#ffe0e0}
.diff{font-size:15px;line-height:1.9;margin:8px 0}
del{background:#ffd6d6;color:#a00;text-decoration:line-through}ins{background:#c9f0d2;color:#060;text-decoration:none}
sup{font-size:10px;color:#06c;cursor:pointer;margin:0 1px}.hl{outline:2px solid #06c;border-radius:3px}
.cmp{font-size:13px;color:#666;margin-top:4px}.cmp b{color:#999;font-weight:500}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}td,th{border-bottom:1px solid #eee;padding:4px 6px;
 text-align:left;vertical-align:top}th{color:#888;font-weight:500}select{font-size:12px}
audio{width:100%;height:32px;margin:4px 0}details summary{cursor:pointer;color:#06c;font-size:13px}
input.note{width:100%;box-sizing:border-box;font-size:12px;padding:3px 6px;border:1px solid #ddd;border-radius:4px}
</style></head><body>
<header><b>__RUN__</b><span id="sum"></span>
 <select id="fLang"><option value="">全部语言</option><option>zh</option><option>zh-en</option><option>en</option></select>
 <select id="fDur"><option value="">全部时长</option><option value="long">&gt;30 秒</option><option value="short">≤30 秒</option></select>
 <select id="fTag"><option value="">全部标签</option></select>
 <select id="fCat"><option value="">全部归因</option></select>
 <label><input type="checkbox" id="fTodo"> 只看未确认</label>
 <button onclick="exportCsv()">导出 CSV</button>
</header><main id="main"></main>
<script>
const DATA=__DATA__, CATS=__CATS__, TOTAL=__TOTAL__, CMP="__CMP__", KEY="wf_errors___RUN__";
let st=JSON.parse(localStorage.getItem(KEY)||"{}");const save=()=>localStorage.setItem(KEY,JSON.stringify(st));
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const CN=Object.fromEntries(CATS);
const cat=(c,s)=>(st[c.id]&&st[c.id][s.span])||s.guess;
const done=c=>c.segs.every(s=>!("span" in s)||(st[c.id]&&st[c.id][s.span]));
for(const t of [...new Set(DATA.map(c=>c.primary_tag).filter(Boolean))].sort())fTag.add(new Option(t,t));
for(const [k,l] of CATS)fCat.add(new Option(l,k));
function summary(){const cnt={};let spans=0,conf=0;for(const c of DATA)for(const s of c.segs)if("span" in s){
 spans++;const k=cat(c,s);cnt[k]=(cnt[k]||0)+Math.max(s.n_ref,s.n_hyp);if(st[c.id]&&st[c.id][s.span])conf++}
 const tot=Object.values(cnt).reduce((a,b)=>a+b,0);
 sum.innerHTML=`${DATA.length} 条有错误 · 官方错误 ${TOTAL} 个 · 差异 ${spans} 处（已确认 ${conf}）<br><small>`+
  CATS.filter(([k])=>cnt[k]).map(([k,l])=>`${l} ${(100*cnt[k]/tot).toFixed(0)}%`).join(" · ")+`</small>`}
function render(){const L=fLang.value,D=fDur.value,T=fTag.value,C=fCat.value,todo=fTodo.checked;
 const show=DATA.filter(c=>(!L||c.lang==L)&&(!D||(D=="long")==(c.dur>30))&&(!T||c.primary_tag==T)&&
  (!C||c.segs.some(s=>"span" in s&&cat(c,s)==C))&&(!todo||!done(c)));
 main.innerHTML=show.map(c=>{const segs=c.segs;
  const diff=segs.map(s=>"eq" in s?esc(s.eq)+" ":(s.ref?`<del>${esc(s.ref)}</del>`:"")+(s.hyp?`<ins>${esc(s.hyp)}</ins>`:"")+
   `<sup id="m_${c.id}_${s.span}" onclick="hl('${c.id}',${s.span})">${s.span+1}</sup> `).join("");
  const rows=segs.filter(s=>"span" in s).map(s=>`<tr id="r_${c.id}_${s.span}"><td>${s.span+1}</td><td><del>${esc(s.ref)||"∅"}</del>
   → <ins>${esc(s.hyp)||"∅"}</ins></td><td><select onchange="setCat('${c.id}',${s.span},this.value)">${CATS.map(([k,l])=>
   `<option value="${k}"${cat(c,s)==k?" selected":""}>${l}</option>`).join("")}</select>${st[c.id]&&st[c.id][s.span]?" ✓":""}</td></tr>`).join("");
  const note=(st[c.id]&&st[c.id].note)||"";
  return `<div class="card"><div class="meta"><span class="tag bad">错误 ${c.errors} / ${c.n_ref} 字（${(100*c.errors/c.n_ref).toFixed(0)}%）</span>
   <span class="tag">${c.lang}</span><span class="tag">${c.dur}s</span>${c.primary_tag?`<span class="tag">${esc(c.primary_tag)}</span>`:""}
   <span style="float:right;color:#aaa">${c.id}</span></div>
   ${c.audio?`<audio controls preload="none" src="data:audio/mp4;base64,${c.audio}"></audio>`:""}
   <div class="diff">${diff}</div>
   ${c.cmp?`<div class="cmp"><b>${esc(CMP)}：</b>${esc(c.cmp)}</div>`:""}
   ${c.notes?`<div class="cmp"><b>v1.1 备注：</b>${esc(c.notes)}</div>`:""}
   <details${c.segs.filter(s=>"span" in s).length<=12?" open":""}><summary>${c.segs.filter(s=>"span" in s).length} 处差异的归因 ·
    整条设为 <select onchange="setAll('${c.id}',this.value);this.value=''"><option value="">…</option>${CATS.map(([k,l])=>`<option value="${k}">${l}</option>`).join("")}</select>
    <button onclick="confirmAll('${c.id}')">全部按当前确认</button></summary>
   <table><tr><th>#</th><th>参考 → 模型</th><th>归因</th></tr>${rows}</table></details>
   <input class="note" placeholder="这条的备注（可选）" value="${esc(note)}" onchange="setNote('${c.id}',this.value)"></div>`}).join("")||"没有符合筛选的条目。";
 summary()}
function hl(id,i){document.querySelectorAll(".hl").forEach(e=>e.classList.remove("hl"));
 const r=document.getElementById(`r_${id}_${i}`);if(r){r.classList.add("hl");r.closest("details").open=true;r.scrollIntoView({block:"center"})}}
function setCat(id,i,v){(st[id]=st[id]||{})[i]=v;save();summary()}
function setAll(id,v){if(!v)return;const c=DATA.find(x=>x.id==id);for(const s of c.segs)if("span" in s)(st[id]=st[id]||{})[s.span]=v;save();render()}
function confirmAll(id){const c=DATA.find(x=>x.id==id);for(const s of c.segs)if("span" in s)(st[id]=st[id]||{})[s.span]=cat(c,s);save();render()}
function setNote(id,v){(st[id]=st[id]||{}).note=v;save()}
function exportCsv(){const q=v=>{v=v==null?"":String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
 const rows=[["clip_id","lang","duration_s","primary_tag","span","kind","ref","hyp","n_ref","n_hyp","auto_category","guess","attribution","confirmed","clip_note"]];
 for(const c of DATA)for(const s of c.segs)if("span" in s)rows.push([c.id,c.lang,c.dur,c.primary_tag||"",s.span+1,s.kind,s.ref,s.hyp,
  s.n_ref,s.n_hyp,s.auto,s.guess,cat(c,s),st[c.id]&&st[c.id][s.span]?"yes":"no",(st[c.id]&&st[c.id].note)||""]);
 const b=new Blob(["﻿"+rows.map(r=>r.map(q).join(",")).join("\r\n")],{type:"text/csv"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="error_attribution.csv";a.click()}
[fLang,fDur,fTag,fCat,fTodo].forEach(e=>e.onchange=render);render();
</script></body></html>"""


if __name__ == "__main__":
    main()
