"""Review packages for benchmark v2 (2026-11-04): data/private/benchmark_v2/items.jsonl → N packs of ~200 clips.

Same page and pre-fill as the training-pool batches (scripts/prepare_review_pool.py): the text box starts with the
production ASR text minus 嗯/呃/um/uh, the D step 600 draft is shown under it with the differences highlighted, and
priority = how much the two disagree (agree / small / medium / large; most different first within a pack).
Differences for a TEST set: every clip must be listened to in full (agree only means two systems agree, not that the
text is right); clips are spread over the packs language-stratified (zh / zh-en / en interleaved in a seeded order)
so each pack is a mini benchmark and a half-finished review is still balanced.
Caveat: pre-filling with the production text can anchor labels toward the production model; v2 results comparing our
model with VoiceCursor should mention it.
Writes data/private/benchmark_v2/review/<name>-<k>/{bench/review.csv, pack/…} and <name>-<k>.zip.
Usage: uv run python scripts/prepare_review_benchmark_v2.py --drafts data/private/benchmark_v2/drafts_D600.jsonl --packs 3
Long part (2026-11-09, 30–180 s, I2 step 600 drafts; --long = text box grows with the text, ⌥B / ⌥F seek 5 s):
       uv run python scripts/prepare_review_benchmark_v2.py --root data/private/benchmark_v2_long --long \
           --drafts data/private/benchmark_v2_long/drafts_I2_600.jsonl --packs 2 --name benchmark-v2-long \
           --draft-label "I2 草稿" --draft-desc "large-v3 + LoRA I2 step 600"
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_review_pool import COLS, tier  # noqa: E402
from whisperflow.text.fillers import strip_fillers  # noqa: E402

ROOT = Path("data/private/benchmark_v2")  # --root
SEED = 20261104
README = """benchmark v2 审核 {name}（{n} 条：中文 {zh} / 中英混说 {cs} / 英文 {en}）—— 测试集，单文件版，音频已嵌入网页

这是新的锁定测试集（benchmark v2）的第 {k}/{total} 包。和训练池审核操作完全一样，但因为是测试集，有两点不同：
- 每一条都请【完整听完】再确认。agree（两边一致）只说明两个系统认成一样，不代表一定对。
- 标注要忠实于说出的内容（专有名词、数字、中英文按实际说的写），不确定的地方宁可 drop。

1. 双击 review.html，用 Chrome 或 Safari 打开。不需要联网，音频不会上传到任何地方。
2. 每条：听音频 → 改文本框 → ⌘+Enter 确认并跳到下一条（没改 = ok，改过 = fix）；不要的按 ⌥D（drop）。
   ⌥P 播放/暂停，⌥R 从头播放，⌥J / ⌥K 上一条 / 下一条，⌥0 恢复预填文字。审核规则在页面底部（和训练池相同）。
3. 文本框预填的是【线上识别】（去掉了 嗯/呃/um/uh）；第二行【{draft}】是我们当前最好的模型（{draft_desc}），
   和线上识别不同的地方会高亮，觉得 {draft}更好可以点"采用"。
4. 进度自动保存在浏览器里；审完点"导出 review.csv"，把文件交给 Claude。{time}
{extra}
注意：这里是线上用户的真实录音，请不要转发或上传到其他地方。
"""


def transcode_pcm(src: str, dst: Path) -> None:
    """ogg → m4a through raw PCM at the native rate, ignoring container timestamps. Some iOS clips carry Ogg granule
    positions at 1/3 of the real timing: a direct transcode keeps only the first third (benchmark v2 long 47d777fc…:
    35.1 s of speech → 11.7 s m4a; short part a12be89c…: 11.0 → 3.7 s), while the PCM decode has every sample."""
    sr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                         "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout.strip()
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", sr, "-ac", "1", "-i", "-",
                    "-c:a", "aac", "-b:a", "64k", str(dst)], input=pcm, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--packs", type=int, default=3)
    ap.add_argument("--name", default="benchmark-v2")
    ap.add_argument("--root", default=str(ROOT), help="benchmark dir with items.jsonl / audio (long part: benchmark_v2_long)")
    ap.add_argument("--draft-label", default="D 草稿")
    ap.add_argument("--draft-desc", default="large-v3 + LoRA D step 600")
    ap.add_argument("--long", action="store_true", help="long clips: growing text box + 5 s seek keys, time estimate "
                    "from the audio length (owner's pool review: mean ≈ 4 s + 2.6 × audio per clip)")
    args = ap.parse_args()
    root_dir = Path(args.root)
    drafts = {d["audio_id"]: d for d in map(json.loads, open(args.drafts, encoding="utf-8"))}
    items = [r for r in map(json.loads, open(root_dir / "items.jsonl", encoding="utf-8")) if r["audio_id"] in drafts]
    missing = sum(1 for _ in open(root_dir / "items.jsonl", encoding="utf-8")) - len(items)
    rng = random.Random(SEED)
    by = {c: [r for r in items if r["lang_class"] == c] for c in ("zh", "zh-en", "en")}
    packs, dealt = [[] for _ in range(args.packs)], 0
    for c, rs in by.items():  # deal each language round-robin, continuing where the previous language stopped
        rng.shuffle(rs)
        for r in rs:
            packs[dealt % args.packs].append(r)
            dealt += 1

    for k, pk in enumerate(packs, 1):
        name = f"{args.name}-{k}"
        root = root_dir / "review" / name
        bench, pack = root / "bench", root / "pack"
        review = bench / "review.csv"
        if review.exists() and any(r["verdict"] for r in csv.DictReader(open(review, encoding="utf-8-sig"))):
            raise SystemExit(f"{review} already has verdicts; not overwriting")
        rows = []
        for r in pk:
            draft = drafts[r["audio_id"]]["hypothesis"]
            pri, rate = tier(r["transcription"], draft)
            rows.append({"priority": pri, "split": "test", "scenario": " ".join([r["lang_class"]] + r["traits"]),
                         "seen": "yes" if r["seen"] else "no", "duration_s": round(r["quality"]["duration_s"], 1),
                         "draft_vs_online": round(rate, 3), "audio_file": f"audio/{r['audio_id']}.ogg",
                         "final_transcript": strip_fillers(r["transcription"]), "verdict": "", "note": "",
                         "draft_1": draft, "online_raw": r["transcription"], "online_output": r["processed"],
                         "audio_id": r["audio_id"], "mode": r["mode"], "lang": r["lang"], "disagreement": "",
                         "flags": " ".join(r["flags"]) if isinstance(r["flags"], list) else r["flags"],
                         "trace": r["langfuse"]})
        rows.sort(key=lambda x: -x["draft_vs_online"])
        bench.mkdir(parents=True, exist_ok=True)
        with open(review, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        with open(bench / "quality.jsonl", "w", encoding="utf-8") as f:
            for r in pk:
                f.write(json.dumps(r["quality"]) + "\n")
        (pack / "audio").mkdir(parents=True, exist_ok=True)
        for r in pk:  # AAC ~64 kbit/s mono, as in the earlier packages
            dst = pack / "audio" / f"{r['audio_id']}.m4a"
            if not dst.exists():
                src = str(root_dir / "audio" / f"{r['audio_id']}.ogg")
                if args.long:
                    transcode_pcm(src, dst)
                else:
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                                    "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(dst)], check=True)
        subprocess.run([sys.executable, "scripts/build_review_html.py", str(bench), str(pack), "--embed",
                        "--quality", str(bench / "quality.jsonl"), "--name", name,
                        "--refs", f"online_raw:线上识别,draft_1:{args.draft_label},online_output:线上输出"]
                       + (["--long"] if args.long else []), check=True)
        n = {c: sum(r["lang_class"] == c for r in pk) for c in ("zh", "zh-en", "en")}
        time, extra = "每包大约 1 小时。", ""
        if args.long:
            audio_min = sum(r["quality"]["duration_s"] for r in pk) / 60
            time = (f"这包音频共 {audio_min:.0f} 分钟（每条 30 秒–3 分钟），按之前的审核速度估计"
                    f"大约 {(4 * len(pk) / 60 + 2.6 * audio_min) / 60:.1f} 小时，可以分几次审。")
            extra = ("5. 长音频：⌥B / ⌥F 后退 / 前进 5 秒（光标留在文本框里也能用），也可以拖进度条；文本框会随文字变高，"
                     "整段文字都在里面，不会截断。\n")
        (pack / "README.txt").write_text(README.format(name=name, n=len(rows), zh=n["zh"], cs=n["zh-en"], en=n["en"],
                                                       k=k, total=args.packs, draft=args.draft_label,
                                                       draft_desc=args.draft_desc, time=time, extra=extra), encoding="utf-8")
        zdir = root / name
        if zdir.exists():
            shutil.rmtree(zdir)
        zdir.mkdir()
        shutil.copy(pack / "review.html", zdir / "review.html")
        shutil.copy(pack / "README.txt", zdir / "README.txt")
        shutil.make_archive(str(root / name), "zip", root, name)
        tiers = {t: sum(r["priority"] == t for r in rows) for t in ("agree", "small", "medium", "large")}
        print(f"{name}: {len(rows)} rows {n} {tiers}; seen {sum(r['seen'] == 'yes' for r in rows)}; -> {root / name}.zip")
    if missing:
        print(f"WARNING: {missing} items have no draft and were left out")


if __name__ == "__main__":
    main()
