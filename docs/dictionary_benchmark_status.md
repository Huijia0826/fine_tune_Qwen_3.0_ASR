# 词典（Dictionary / Vocabulary）Benchmark 状态

最后更新：2026-11-12

## 当前版本一览

| 项目 | 状态 |
|---|---|
| 当前版本 | **v3（2026-11-12）**：199 条，46.8 分钟 |
| Ground truth | 全部经过 owner 人工 review（2026-11-09 首轮 + 2026-11-12 补审） |
| ASR Arena | ✅ 2026-11-12 已上传 v3（数据集 id：`____`，请补填） |
| 本地评测 manifest | ✅ `data/manifests/dictionary_bench.jsonl` 与 Arena 版一致（199 条） |
| 未决问题 | 无 |

## 版本历史

| 版本 | 日期 | 条数 | 说明 | 文件 |
|---|---|---|---|---|
| v1 | 2026-10 | 207 | 最初标注版（从 287 条线上「用户手改专有名词」候选中筛出）。Arena 旧数据集 `01a177dc-05c8-7a6a-bd73-150130d18b5a` 用的就是这一版的 GT（无空格、无标点） | `~/Downloads/01a1bf4a-…/source_ground_truth/vocab_benchmark_upload.csv`；`data/manifests/dictionary_bench.pre_review.jsonl`（转成 policy v1 后） |
| v2 | 2026-11-09 | 201 | owner 首轮 review：ok 97 / fix 100 / drop 6 / 4 条漏审；39 条严格转写改动；另补 2 条听写修正（Raemon、经历） | `data/manifests/dictionary_bench.pre_recheck.jsonl` |
| **v3** | **2026-11-12** | **199** | 补审 4 条漏审 + 1 条拼写确认：6f919ffa、b7f1cbf8 ok；07a559af、526c9616 drop；2a176c9b stella → stalla（与词典词一致） | `data/manifests/dictionary_bench.jsonl`（当前） |

## 数据概况（v3）

| 维度 | 分布 |
|---|---|
| 时长 | 合计 46.8 分钟 |
| 语言（按文本） | en 125 · zh+en 33 · zh 6 · other（葡语等，不在 zh/en 范围，单独计分）35 |
| 词典 | 349 个词条，267 个不重复 |
| 两种参考 | 严格转写（`raw_transcript`，评 ASR）；最终输出（`formatted_transcript`，评成品）——74 条两者不同 |

## Ground truth 来源（按优先级）

1. `data/private/e2e/reviewed/e2e-dict-recheck-review.csv` —— 2026-11-12 补审（5 条）
2. `data/private/e2e/reviewed/e2e-dict_D_fixes.json` —— 2026-11-09 两条听写修正同步到严格转写
3. `data/private/e2e/reviewed/e2e-dict-review.csv` —— 2026-11-09 首轮 review（207 条）

规则：ASR 部分对 ①严格转写 打分，最终输出对 ②最终输出 打分；听错的修正写在 ①。

## Arena 上传包

- 目录：`/Users/huijia/Desktop/whisper training/data/private/dictionary_bench/upload/`
  - `dictionary_bench_reviewed_upload.xlsx` / `.csv`：列 `file / transcript / audio_context / audio_url / audio_id / vocabulary`
  - `audio/`：199 个 `<audio_id>.ogg`
- `transcript` = 严格转写；`audio_context`、`vocabulary` 沿用 v1 上传表，未改
- 音频 = 2026-11-12 用 audio-fetch 从线上重新下载的原始 ogg（`data/private/dictionary_bench/ogg_raw/`，207 条全部拿到，无空音频）
- **注意**：3 条 Android 录音线上 ogg 本身是 3 倍速（3a68ed10、3bb81e3f、af83babc），改用 review 时听的正常速度 wav 转成 ogg
- 重新生成：

  ```bash
  uv run --with openpyxl python scripts/dictionary_bench_upload.py --recheck data/private/e2e/reviewed/e2e-dict-recheck-review.csv
  ```

## 已有成绩（注意：都是 v2，201 条）

zh/en MER，带上下文（词典）解码；详见 `docs/decisions.md` 与 `reports/runs/dict_*`。

| 模型 | MER | 目标词召回 |
|---|---|---|
| Qwen3-ASR J4 step 1101 | 10.12%（最好） | — |
| Qwen3-ASR J3 step 600 | 10.79% | 80% |
| Qwen3-ASR zero-shot | 11.36% | 75% |
| Qwen3-ASR J1 step 400 | 11.38% | 63% |
| Whisper large-v3 I2 step 600 | 15.06% | — |

v3 删掉了 2 条 other 语言、改了 1 条英文拼写，zh/en 分数可能有小幅变化；之后的对比需要在 v3 上重跑，不要和上表直接比。

## 注意事项

- **不要重跑** `scripts/dictionary_bench_to_manifest.py`：它会从 review 前的标注重建 manifest，覆盖 review 结果。
- 这是纯测试集：词典目标词禁止出现在合成训练数据里。
- 里面是线上用户真实录音和上下文，不要外传。
