"""Side-by-side listening page for the TTS vendor comparison (no API calls here).

    uv run python scripts/tts_compare/build_listen_page.py
    open data/private/tts_compare/listen.html

Reads every data/private/tts_compare/<vendor>/manifest.jsonl written by synth.py and builds one HTML page:
one block per test sentence, one row per vendor, one audio player per voice. Audio is referenced by relative
path (nothing is embedded), so keep the page inside data/private/tts_compare/.

Scoring: each player has two 1-5 selectors (P = 专名/英文词发音对不对, N = 自然度/口语感). Scores live in the
browser's localStorage; "导出 CSV" downloads them so they can be averaged per vendor / voice later.
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("data/private/tts_compare")
SENTENCES = Path(__file__).with_name("sentences.txt")


def load_sentences(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sid, lang, cat, text = line.split("\t", 3)
        rows.append({"id": sid, "lang": lang, "category": cat, "text": text})
    return rows


def load_manifests(root: Path) -> dict[str, list[dict]]:
    by_vendor: dict[str, list[dict]] = {}
    for mf in sorted(root.glob("*/manifest.jsonl")):
        rows = [json.loads(x) for x in mf.read_text(encoding="utf-8").splitlines() if x.strip()]
        by_vendor[mf.parent.name] = rows
    return by_vendor


CSS = """
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;color:#222}
h2{margin-top:36px;font-size:17px} .meta{color:#777;font-size:12px}
table{border-collapse:collapse;margin-top:6px} td{border-top:1px solid #eee;padding:6px 8px;vertical-align:top}
td.v{font-weight:600;white-space:nowrap} .cell{display:inline-block;margin:0 14px 8px 0;font-size:12px}
.cell audio{display:block;height:30px;width:230px} .err{color:#b00;font-size:12px}
select{font-size:11px} #bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd}
"""

JS = """
const KEY='tts_compare_scores';
const S=JSON.parse(localStorage.getItem(KEY)||'{}');
document.querySelectorAll('select[data-k]').forEach(el=>{
  if(S[el.dataset.k]) el.value=S[el.dataset.k];
  el.onchange=()=>{S[el.dataset.k]=el.value;localStorage.setItem(KEY,JSON.stringify(S));};
});
function exportCsv(){
  let out='vendor,voice,id,metric,score\\n';
  for(const [k,v] of Object.entries(S)){ if(v) out+=k.split('|').join(',')+','+v+'\\n'; }
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out],{type:'text/csv'}));
  a.download='tts_scores.csv';a.click();
}
function filterLang(l){document.querySelectorAll('section').forEach(s=>{
  s.style.display=(l==='all'||s.dataset.lang===l)?'':'none';});}
// play one sentence across all vendors/voices in order
async function playAll(sid){
  for(const a of document.querySelectorAll('section[data-id="'+sid+'"] audio')){
    a.currentTime=0; await a.play().catch(()=>{});
    await new Promise(r=>a.onended=r);
  }
}
"""


def score_select(vendor: str, voice: str, sid: str, metric: str) -> str:
    key = html.escape(f"{vendor}|{voice}|{sid}|{metric}")
    opts = "".join(f"<option>{i}</option>" for i in range(1, 6))
    return f'{metric}<select data-k="{key}"><option value=""></option>{opts}</select>'


def build(root: Path, sentences: list[dict]) -> str:
    vendors = load_manifests(root)
    # (vendor, id) -> list of rows, keeping the voice order of the manifest
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v, rows in vendors.items():
        for r in rows:
            idx[(v, r["id"])].append(r)

    parts = [
        "<!doctype html><meta charset='utf-8'><title>TTS 对比试听</title>",
        f"<style>{CSS}</style>",
        "<div id='bar'>语言：",
        *(f"<button onclick=\"filterLang('{x}')\">{x}</button> " for x in ["all", "mix", "zh", "en"]),
        " &nbsp; <button onclick='exportCsv()'>导出 CSV</button>",
        f" &nbsp; <span class='meta'>vendors: {html.escape(', '.join(vendors) or '（还没有生成任何音频）')}"
        " · P=发音(专名/英文词) N=自然度 · 1 差 5 好</span></div>",
    ]
    for s in sentences:
        parts.append(f"<section data-id='{s['id']}' data-lang='{s['lang']}'>")
        parts.append(
            f"<h2>{s['id']} · {html.escape(s['text'])}</h2><div class='meta'>{s['lang']} · "
            f"{html.escape(s['category'])} · <button onclick=\"playAll('{s['id']}')\">依次播放全部</button></div>"
        )
        parts.append("<table>")
        for v in vendors:
            cells = []
            for r in idx.get((v, s["id"]), []):
                label = html.escape(r.get("voice_label") or r["voice"])
                if r.get("error"):
                    cells.append(f"<span class='cell'>{label}<br><span class='err'>"
                                 f"{html.escape(r['error'][:160])}</span></span>")
                    continue
                rel = Path(r["path"]).relative_to(root).as_posix()
                cells.append(
                    f"<span class='cell'>{label} <span class='meta'>{r.get('duration_s', 0):.1f}s</span>"
                    f"<audio controls preload='none' src='{html.escape(rel)}'></audio>"
                    + score_select(v, r["voice"], s["id"], "P") + " "
                    + score_select(v, r["voice"], s["id"], "N") + "</span>"
                )
            if cells:
                parts.append(f"<tr><td class='v'>{html.escape(v)}</td><td>{''.join(cells)}</td></tr>")
        parts.append("</table></section>")
    parts.append(f"<script>{JS}</script>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--sentences", type=Path, default=SENTENCES)
    args = ap.parse_args()
    out = args.root / "listen.html"
    args.root.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args.root, load_sentences(args.sentences)), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
