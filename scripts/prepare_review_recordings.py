"""Review package for the owner's scripted recordings (2026-11-04): takes.jsonl → one review page.

The text box is pre-filled with the SCRIPT sentence (its terms are spelled right); the production ASR text of the take is
shown under it, differences highlighted. The owner fixes the wording to what was actually said (re-phrasings, two
sentences read as one) and drops fragments (a single word tried out, an aborted read). Most different first.
Writes data/private/recording_script/review/<name>/{bench/review.csv, pack/…} and <name>.zip.
Usage: uv run python scripts/prepare_review_recordings.py --name recording-v1
       uv run python scripts/prepare_review_recordings.py --freeform data/private/recording_script/freeform_2026-11-03 \
         --name freeform-2026-11-03
--freeform: free dictation (no script). Pre-filled with the production ASR text, the D step 600 draft (drafts_D600.jsonl
in that folder) shown under it; the owner writes what was said and drops clips naming private persons (PII policy).
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_review_pool import COLS, tier  # noqa: E402

ROOT = Path("data/private/recording_script")
README = """录制脚本审核 {name}（{n} 条录音，覆盖脚本 {k} 句）—— 单文件版，音频已嵌入网页

和训练池审核的操作一样：听音频 → 改文本框 → ⌘+Enter 确认并跳到下一条（没改 = ok，改过 = fix）；不要的按 ⌥D（drop）。
- 文本框预填的是【脚本原文】（专有名词写法是对的）；第二行【线上识别】是这条录音当时的识别结果，不同的地方会高亮。
- 请按【你实际说的】改：读的时候改了措辞、多说了几句、两句连在一起读，都照实写。
- 只读了一个词试发音、读到一半中断的，直接 drop。同一句读了好几遍、每遍都完整的，都可以保留（都是有效数据）。
审完点"导出 review.csv"，把文件交给 Claude。
"""
README_FREE = """自由口述审核 {name}（{n} 条录音）—— 单文件版，音频已嵌入网页

操作和训练池审核一样：听音频 → 改文本框 → ⌘+Enter 确认（没改 = ok，改过 = fix）；不要的按 ⌥D（drop）。
- 文本框预填的是【线上识别】；第二行【D 草稿】是我们当前的模型，不同的地方会高亮，觉得更好可以点"采用"。
- 专有名词（人名、公司、产品）请按正确写法写；没听清、太短（一两个词）的直接 drop。
- 提到私人信息（非公开人物的名字、电话、地址等）的录音请 drop，或者把那部分说明写在备注里。
审完点"导出 review.csv"，把文件交给 Claude。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="recording-v1")
    ap.add_argument("--freeform", help="folder written by fetch_recordings.py --freeform")
    ap.add_argument("--takes-dir", help="folder with takes.jsonl + audio/ (e.g. recording_script/v2); default: recording v1")
    args = ap.parse_args()
    src, refs, readme = ROOT, "online_raw:脚本原文,draft_1:线上识别", README
    if args.freeform:
        src = Path(args.freeform)
        drafts = {d["audio_id"]: d["hypothesis"] for d in map(json.loads, open(src / "drafts_D600.jsonl", encoding="utf-8"))}
        takes = [{**t, "script_id": t["stem"], "take": 1, "block": "freeform", "targets": "",
                  "script_text": t["transcript"], "transcript": drafts.get(t["audio_id"], "")}
                 for t in map(json.loads, open(src / "items.jsonl", encoding="utf-8"))]
        refs, readme = "online_raw:线上识别,draft_1:D 草稿", README_FREE
    else:
        src = Path(args.takes_dir) if args.takes_dir else ROOT
        takes = [json.loads(l) for l in open(src / "takes.jsonl", encoding="utf-8")]
    root = ROOT / "review" / args.name
    bench, pack = root / "bench", root / "pack"
    review = bench / "review.csv"
    if review.exists() and any(r["verdict"] for r in csv.DictReader(open(review, encoding="utf-8-sig"))):
        raise SystemExit(f"{review} already has verdicts; not overwriting")
    rows = []
    for t in takes:
        pri, rate = tier(t["script_text"], t["transcript"])
        unmatched = "" if t.get("matched", True) else " 未对上脚本（预填的是识别结果）"
        rows.append({"priority": pri, "split": t.get("split", "train"),
                     "scenario": f"{t['script_id']} take{t['take']} {t['block']}{unmatched}",
                     "seen": "", "duration_s": "", "draft_vs_online": round(rate, 3),
                     "audio_file": f"audio/{t['audio_id']}.{t['ext']}", "final_transcript": t["script_text"],
                     "verdict": "", "note": "", "draft_1": t["transcript"], "online_raw": t["script_text"],
                     "online_output": "", "audio_id": t["audio_id"], "mode": "dictate", "lang": t.get("configured_language"),
                     "disagreement": "", "flags": t["targets"], "trace": ""})
    rows.sort(key=lambda x: -x["draft_vs_online"])
    bench.mkdir(parents=True, exist_ok=True)
    with open(review, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    (pack / "audio").mkdir(parents=True, exist_ok=True)
    for t in takes:
        dst = pack / "audio" / f"{t['audio_id']}.m4a"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src / "audio" / f"{t['audio_id']}.{t['ext']}"),
                            "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(dst)], check=True)
    subprocess.run([sys.executable, "scripts/build_review_html.py", str(bench), str(pack), "--embed", "--name", args.name,
                    "--refs", refs], check=True)
    (pack / "README.txt").write_text(readme.format(name=args.name, n=len(rows), k=len({t["script_id"] for t in takes})),
                                     encoding="utf-8")
    zdir = root / args.name
    if zdir.exists():
        shutil.rmtree(zdir)
    zdir.mkdir()
    shutil.copy(pack / "review.html", zdir / "review.html")
    shutil.copy(pack / "README.txt", zdir / "README.txt")
    shutil.make_archive(str(root / args.name), "zip", root, args.name)
    tiers = {t: sum(r["priority"] == t for r in rows) for t in ("agree", "small", "medium", "large")}
    print(f"{len(rows)} rows {tiers} -> {zdir / 'review.html'}")


if __name__ == "__main__":
    main()
