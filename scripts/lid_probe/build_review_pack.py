"""Step 5: owner review pack for the probe's Chinese-speech clips -> real (not proxy) error rates of the online service.

Selection (from data/private/lid_probe/clips.jsonl, test-like clips excluded, Qwen spoken class zh / zh-en):
  en-only ~100: every clip whose production text shows no Chinese (strata Z+R, the "hidden" failures), every H clip
                with a non-ok proxy category, and a seeded random fill of H ok clips up to 100
  control  30: 15 of the non-ok proxy clips + 15 of the ok ones
sel_prob (per group x category) is kept in selection.jsonl so the reviewed MER can be re-weighted back to the sample.
Same page and format as earlier packs (scripts/build_review_html.py, review.csv columns from prepare_review_pool):
final_transcript is pre-filled with QWEN3-ASR's transcript (auto language) minus 嗯/呃/um/uh -- NOT the production
text -- because the production text is what is being scored. Qwen's draft and the production raw text are shown with
differences highlighted. Pre-filling anchors labels toward Qwen; say so when reporting.
Writes data/private/lid_probe/review/<name>/{bench/review.csv, bench/selection.jsonl, pack/…} and <name>.zip.
Usage: uv run python scripts/lid_probe/build_review_pack.py [--name lid-probe-zh]
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_review_pool import COLS, tier  # noqa: E402
from whisperflow.text.fillers import strip_fillers  # noqa: E402

OUT = ROOT / "data/private/lid_probe"
SEED, N_EN, N_CTL_BAD, N_CTL_OK = 20261109, 100, 15, 15
README = """语种探针审核 {name}（{n} 条：仅英语设置 {n_en} 条 + 中文设置对照 {n_ctl} 条）—— 单文件版，音频已嵌入网页

目的：这些是设置为"仅英语"、但实际在说中文 / 中英混说的线上请求（另有 {n_ctl} 条中文设置用户的对照）。
请标出【实际说的话】，用来计算线上服务在这些请求上的真实错误率。

1. 双击 review.html，用 Chrome 或 Safari 打开。不需要联网，音频不会上传到任何地方。
2. 每条：听音频 → 改文本框 → ⌘+Enter 确认并跳到下一条（没改 = ok，改过 = fix）；不要的按 ⌥D（drop，例如听不清、不是人声、不是中文也不是英文）。
   ⌥P 播放/暂停，⌥R 从头播放，⌥J / ⌥K 上一条 / 下一条，⌥0 恢复预填文字。审核规则在页面底部（和之前的包相同）。
3. 注意：文本框预填的是【Qwen3-ASR 草稿】（我们集群上的开源模型，自动判断语种，去掉了 嗯/呃/um/uh），不是线上结果。
   第二行【线上识别】是线上服务的原始 ASR 输出，和 Qwen 不同的地方会高亮；觉得线上识别更对可以点"采用"。
   Qwen 不是标准答案：请以耳朵为准。中文按实际说的写，英文词保留英文，数字写法不用纠结（评分会归一化）。
4. 很多条只有 1–3 秒，线上输出是一个英文词（比如把"买一个吧"写成 "Maigoba"）——这正是要确认的情况，请认真听。
5. 进度自动保存在浏览器里；审完点"导出 review.csv"，把文件交给 Claude。预计 {minutes} 分钟左右。

注意：这里是线上用户的真实录音，请不要转发或上传到其他地方。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lid-probe-zh")
    args = ap.parse_args()
    rng = random.Random(SEED)
    clips = [json.loads(line) for line in open(OUT / "clips.jsonl", encoding="utf-8")]
    zh = [c for c in clips if c["spoken"] in ("zh", "zh-en") and not c["test_like"]]
    en_hidden = [c for c in zh if c["stratum"] in ("Z", "R")]
    en_h_bad = [c for c in zh if c["stratum"] == "H" and c["category"] != "ok"]
    en_h_ok = [c for c in zh if c["stratum"] == "H" and c["category"] == "ok"]
    ctl_bad = [c for c in zh if c["stratum"] == "C" and c["category"] != "ok"]
    ctl_ok = [c for c in zh if c["stratum"] == "C" and c["category"] == "ok"]
    k_ok = max(0, N_EN - len(en_hidden) - len(en_h_bad))
    picks = []
    for grp, pool, k in (("en_only_hidden", en_hidden, len(en_hidden)), ("en_only_H_nonok", en_h_bad, len(en_h_bad)),
                         ("en_only_H_ok", en_h_ok, k_ok), ("control_nonok", ctl_bad, N_CTL_BAD),
                         ("control_ok", ctl_ok, N_CTL_OK)):
        pool = sorted(pool, key=lambda c: c["audio_id"])
        k = min(k, len(pool))
        for c in rng.sample(pool, k):
            picks.append((grp, k / len(pool), c))
    rng.shuffle(picks)  # interleave groups so a half-finished review is still balanced

    name = args.name
    root = OUT / "review" / name
    bench, pack = root / "bench", root / "pack"
    review = bench / "review.csv"
    if review.exists() and any(r["verdict"] for r in csv.DictReader(open(review, encoding="utf-8-sig"))):
        raise SystemExit(f"{review} already has verdicts; not overwriting")
    bench.mkdir(parents=True, exist_ok=True)
    (pack / "audio").mkdir(parents=True, exist_ok=True)
    rows, sel = [], []
    for grp, prob, c in picks:
        pri, rate = tier(c["transcription"], c["qwen_text"])
        group = "仅英语设置" if grp.startswith("en_only") else "中文设置对照"
        rows.append({"priority": pri, "split": "probe", "scenario": f"{group} {c['spoken']} {c['platform']}",
                     "seen": "no", "duration_s": round(c["dur"], 1), "draft_vs_online": round(rate, 3),
                     "audio_file": f"audio/{c['audio_id']}.ogg", "final_transcript": strip_fillers(c["qwen_text"]),
                     "verdict": "", "note": "", "draft_1": c["qwen_text"], "online_raw": c["transcription"],
                     "online_output": c["processed"], "audio_id": c["audio_id"], "mode": "dictate",
                     "lang": c["lang"], "disagreement": "", "flags": "", "trace": c.get("langfuse") or ""})
        sel.append({"audio_id": c["audio_id"], "group": grp, "sel_prob": prob, "stratum": c["stratum"],
                    "spoken": c["spoken"], "proxy_category": c["category"], "model": c["model"],
                    "platform": c["platform"], "setting": c["setting"], "weight": c["weight"], "dur": c["dur"]})
        dst = pack / "audio" / f"{c['audio_id']}.m4a"
        if not dst.exists():  # AAC ~64 kbit/s mono, as in the earlier packages
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT / "audio" / f"{c['audio_id']}.ogg"),
                            "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(dst)], check=True)
    with open(review, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(bench / "selection.jsonl", "w", encoding="utf-8") as f:
        for s in sel:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/build_review_html.py"), str(bench), str(pack), "--embed",
                    "--name", name, "--refs", "draft_1:Qwen 草稿,online_raw:线上识别,online_output:线上输出"], check=True)
    n_en = sum(s["group"].startswith("en_only") for s in sel)
    audio_s = sum(s["dur"] for s in sel)
    minutes = round((audio_s * 2.5 + 20 * len(sel)) / 60 / 5) * 5  # listen ~2.5x + ~20 s handling per clip
    (pack / "README.txt").write_text(README.format(name=name, n=len(sel), n_en=n_en, n_ctl=len(sel) - n_en,
                                                   minutes=minutes), encoding="utf-8")
    zdir = root / name
    if zdir.exists():
        shutil.rmtree(zdir)
    zdir.mkdir()
    shutil.copy(pack / "review.html", zdir / "review.html")
    shutil.copy(pack / "README.txt", zdir / "README.txt")
    shutil.make_archive(str(root / name), "zip", root, name)
    from collections import Counter
    print(f"{name}: {len(sel)} clips {dict(Counter(s['group'] for s in sel))}; audio {audio_s / 60:.1f} min; "
          f"est. review {minutes} min; -> {root / name}.zip")


if __name__ == "__main__":
    main()
