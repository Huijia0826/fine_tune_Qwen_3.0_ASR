"""Build a review package for one batch of the online training pool (2026-11-02; batch 1 = CS high-quality, 528).

    uv run python scripts/prepare_review_pool.py --group CS \
        --drafts data/private/online_pool/drafts_C600_batch1.jsonl --name online-pool-batch1

- final_transcript is pre-filled with the PRODUCTION ASR text (`transcription`) minus 嗯/呃/um/uh: on the 287 clips
  the owner reviewed in phase 1 it was much closer to the final labels (MER 4.16%) than the Whisper large-v3 draft
  (8.36%) or the post-processed output (7.47%).
- draft_1 = our model's output (C step 600), shown next to it with differences highlighted: where the two disagree
  is where to listen. `draft_vs_online` = official MER of the draft against the production text; `priority` =
  triage tier (agree / small / medium / large); rows are sorted most different first, as in phase 1.
- seen = the user also has clips in a val set (online_phase1 val or benchmark_chinese val).
Writes data/private/online_pool/<name>/bench/review.csv, <name>/pack/{audio/*.m4a, review.html (embedded),
README.txt} and <name>.zip. Re-running never overwrites a review.csv that already has verdicts.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from whisperflow.eval.official import official_mer
from whisperflow.text.fillers import strip_fillers
from whisperflow.text.normalize import _t2s

POOL = Path("data/private/online_pool")
COLS = ["priority", "split", "scenario", "seen", "duration_s", "draft_vs_online", "audio_file", "final_transcript",
        "verdict", "note", "draft_1", "online_raw", "online_output", "audio_id", "mode", "lang", "disagreement",
        "flags", "trace"]
README = """线上训练池审核 {name}（{n} 条，{desc}）—— 单文件版，音频已嵌入网页

1. 解压后双击 review.html，用 Chrome 或 Safari 打开。不需要联网，音频不会上传到任何地方。
2. 第一次打开时浏览器可能拦截自动播放：点一下播放键（▶）就好。
3. 每条：听音频 → 改文本框 → ⌘+Enter 确认并跳到下一条（没改 = ok，改过 = fix）；不要的按 ⌥D（drop）。
   ⌥P 播放/暂停，⌥R 从头播放，⌥J / ⌥K 上一条 / 下一条，⌥0 恢复预填文字。审核规则在页面底部（和第一期相同）。
4. 进度自动保存在这个浏览器里。建议每审完一批就点"导出 review.csv"备份。
5. 审完后点"导出 review.csv"，把文件交给 Claude。

这一批和第一期的区别：
- 文本框预填的是【线上识别】（去掉了 嗯/呃/um/uh）。第一期你审过的 287 条里，线上识别和你的最终答案差距最小
  （MER 4.16%，Whisper 草稿 8.36%）。
- 第二行【C 草稿】是我们自己的模型（large-v3 + LoRA C）的输出，和线上识别不同的地方会高亮。
  两边一致的地方大概率是对的；高亮的地方请重点听。觉得 C 草稿更好可以点它旁边的"采用"。
- 标签里的 priority：agree = 两边一致（只需检查标点），small / medium / large = 差异由小到大。排序是差异大的在前。
- 有两位用户的录音特别多（190 条、118 条）。如果审到后面觉得同一个人的内容很重复，可以直接 drop 或先停下告诉我。

注意：这里是线上用户的真实录音，请不要转发或上传到其他地方。
"""


def pseudo(uid: str) -> str:
    return "online_phase1-u" + hashlib.sha256(uid.encode()).hexdigest()[:12]


def tier(online: str, draft: str) -> tuple[str, float]:
    t2s = _t2s().convert
    norm = lambda s: re.sub(r"[^\w]|_", "", t2s(s or "")).lower()  # noqa: E731
    if norm(online) == norm(draft):
        return "agree", 0.0
    o = official_mer(online, draft)
    r = o.errors / max(o.n_ref, 1)
    return ("small" if r <= 0.10 else "medium" if r <= 0.30 else "large"), r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--desc", default="中英混说、高音质")
    args = ap.parse_args()

    root = POOL / args.name
    bench, pack = root / "bench", root / "pack"
    review = bench / "review.csv"
    if review.exists() and any(r["verdict"] for r in csv.DictReader(open(review, encoding="utf-8-sig"))):
        raise SystemExit(f"{review} already has verdicts; not overwriting")
    drafts = {d["audio_id"]: d for d in map(json.loads, open(args.drafts, encoding="utf-8"))}
    pool = [r for r in map(json.loads, open(POOL / "pool.jsonl", encoding="utf-8"))
            if r["group"] == args.group and r["high_quality"] and r["audio_id"] in drafts]
    val_spk = {r["speaker_id"] for m in ("online_phase1", "benchmark_chinese")
               for r in map(json.loads, open(f"data/manifests/{m}.jsonl", encoding="utf-8"))
               if r["split"] == "val"}

    rows = []
    for r in pool:
        draft = drafts[r["audio_id"]]["hypothesis"]
        pri, rate = tier(r["transcription"], draft)
        rows.append({"priority": pri, "split": "train",
                     "scenario": " ".join(r["traits"]) or r["scenario"],
                     "seen": "yes" if pseudo(r["user_id"]) in val_spk else "no",
                     "duration_s": round(r["quality"]["duration_s"], 1), "draft_vs_online": round(rate, 3),
                     "audio_file": f"audio/{r['audio_id']}.ogg",
                     "final_transcript": strip_fillers(r["transcription"]),
                     "verdict": "", "note": "", "draft_1": draft, "online_raw": r["transcription"],
                     "online_output": r["processed"], "audio_id": r["audio_id"], "mode": r["mode"],
                     "lang": r["lang"], "disagreement": "",
                     "flags": " ".join(r["flags"]) if isinstance(r["flags"], list) else r["flags"],
                     "trace": r["langfuse"]})
    rows.sort(key=lambda x: -x["draft_vs_online"])
    bench.mkdir(parents=True, exist_ok=True)
    with open(review, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(bench / "quality.jsonl", "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r["quality"]) + "\n")

    (pack / "audio").mkdir(parents=True, exist_ok=True)
    for r in pool:  # same format as the phase 1 package: AAC ~64 kbit/s mono
        dst = pack / "audio" / f"{r['audio_id']}.m4a"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(POOL / "audio" / f"{r['audio_id']}.ogg"),
                            "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(dst)], check=True)
    subprocess.run(["python", "scripts/build_review_html.py", str(bench), str(pack), "--embed",
                    "--quality", str(bench / "quality.jsonl"), "--name", args.name,
                    "--refs", "online_raw:线上识别,draft_1:C 草稿,online_output:线上输出"], check=True)
    (pack / "README.txt").write_text(README.format(name=args.name, n=len(rows), desc=args.desc), encoding="utf-8")
    zdir = root / args.name
    if zdir.exists():
        shutil.rmtree(zdir)
    zdir.mkdir()
    shutil.copy(pack / "review.html", zdir / "review.html")
    shutil.copy(pack / "README.txt", zdir / "README.txt")
    shutil.make_archive(str(root / args.name), "zip", root, args.name)
    tiers = {t: sum(r["priority"] == t for r in rows) for t in ("agree", "small", "medium", "large")}
    print(f"{len(rows)} rows {tiers}; seen {sum(r['seen'] == 'yes' for r in rows)}; -> {root / args.name}.zip")


if __name__ == "__main__":
    main()
