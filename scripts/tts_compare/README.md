# TTS 厂商试听对比工具（2026-11-09）

目的：在买量（试点 500 条 → 约 1 万条）之前，用**同一批 20 句话**、各家的**官方预置音色**（不做声音克隆）各合成一遍，
放在一个网页里逐句对比，听专有名词和中英混说读得对不对、自然不自然。

> 本目录下的脚本**会调用付费 API**。脚本只从环境变量读 key，不会把 key 写进任何文件。
> 生成的音频在 `data/private/tts_compare/`（已被 .gitignore 忽略，不进 git）。

## 文件

| 文件 | 作用 |
|---|---|
| `sentences.txt` | 20 句测试句（制表符分隔：id / lang / 类别 / 文本）。mix = 中英混说 10 句，zh = 纯中文 5 句，en = 纯英文 5 句；覆盖难读的英文名、品牌和地名、数字和日期、问句、说错改口 |
| `synth.py` | 合成脚本，用 `--vendor` 选厂商；每个厂商默认带 8 个中文音色和 2–4 个英文音色；输出 16 kHz 单声道 wav 和 manifest |
| `build_listen_page.py` | 生成试听网页 `data/private/tts_compare/listen.html`：每句一块，每家一行，可以一键依次播放，也可以打 1–5 分并导出 CSV |

支持的 `--vendor`：`minimax`、`aliyun_cosyvoice`、`aliyun_qwen`、`volcengine`、`azure`，或者 `all`（只跑已设置 key 的厂商）。

## 第 0 步：先读条款（这一步是硬门槛）

这次试听只是评估，不是训练。但正式买量之前请先确认：

- **MiniMax**：用户协议里没有找到禁止用输出训练模型、或禁止开发竞品的条款。仍建议发邮件要一份书面确认。
- **阿里百炼**：协议写的是"**未经我们书面许可**，不得利用模型输出训练、开发与其竞争的产品"。阿里自己卖 ASR，所以要先拿到书面许可。
- **火山引擎**：条款明确禁止"利用本服务开发、训练或改善与火山引擎存在竞争关系的模型"。**没有书面许可就不能用于训练**。
- **Azure**：Product Terms 禁止用服务产生的数据训练"同类或竞争产品"，结论偏禁止，需要微软确认。

详细引用见上层调研回复。被禁止的厂商生成的试听音频，**不要**混进训练集。

## 第 1 步：准备环境

```bash
cd "/Users/huijia/Desktop/whisper training"
uv pip install requests scipy        # 已在 .venv 里的话可以跳过
uv pip install dashscope             # 只有 aliyun_cosyvoice 需要（官方 SDK）
```

## 第 2 步：设置 key（只设你要试的那几家）

```bash
# MiniMax（国内站 platform.minimaxi.com 创建 API Key）
export MINIMAX_API_KEY=...
# export MINIMAX_API_BASE=https://api.minimax.io     # 用国际站账号时设置；默认是国内 https://api.minimax.cn
# export MINIMAX_MODEL=speech-2.8-turbo               # 默认 speech-2.8-hd

# 阿里云百炼（CosyVoice 和 Qwen-TTS 共用一个 key；默认北京地域）
export DASHSCOPE_API_KEY=sk-...
# 新加坡地域的 key 还要设置下面两个：
# export DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
# export DASHSCOPE_WS_URL=wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference

# 火山引擎 豆包语音合成 2.0（控制台开通"豆包语音合成模型2.0"）
export VOLC_TTS_API_KEY=...                 # 新版控制台的 API Key
# 旧版控制台改用下面两个：
# export VOLC_TTS_APP_ID=...  VOLC_TTS_ACCESS_KEY=...

# Azure AI Speech（Speech 资源的 Key 和区域，例如 eastasia / southeastasia）
export AZURE_SPEECH_KEY=...
export AZURE_SPEECH_REGION=eastasia
# Azure 中国（世纪互联）改用：
# export AZURE_SPEECH_ENDPOINT=https://chinanorth3.tts.speech.azure.cn/cognitiveservices/v1
```

## 第 3 步：先估费用，再合成

```bash
uv run python scripts/tts_compare/synth.py --vendor all --dry-run
```

`--dry-run` 只统计条数和计费字符，不调用 API。全部默认音色跑完，每家约 170–180 条、约 1 万计费字符。
按列表价估算：

| 厂商 | 估算费用 |
|---|---|
| MiniMax hd | ≈ ¥4 |
| 火山 | ≈ ¥3 |
| CosyVoice | ≈ ¥1.2 |
| Qwen-TTS | ≈ ¥1 |
| Azure | ≈ $0.2（免费层 F0 每月 50 万字符就够） |

```bash
# 先用 2 个音色、2 句话冒烟测试，确认 key、接口和音频都正常
uv run python scripts/tts_compare/synth.py --vendor minimax --max-voices 2 --only s01,s16
# 再全量跑
uv run python scripts/tts_compare/synth.py --vendor minimax
uv run python scripts/tts_compare/synth.py --vendor aliyun_cosyvoice
uv run python scripts/tts_compare/synth.py --vendor aliyun_qwen
uv run python scripts/tts_compare/synth.py --vendor volcengine
uv run python scripts/tts_compare/synth.py --vendor azure
```

常用参数：

- `--voices a,b,c`：换成音色列表里的任意预置音色 ID。音色列表链接在 synth.py 的 VENDORS 注释里。
- `--force`：已经存在的 wav 也重新合成。
- `--sleep 1`：遇到限流时把请求间隔加大。

失败的条目会以 `error` 字段写进 manifest，网页上显示成红字，不会中断整个运行。

输出结构：

```
data/private/tts_compare/<vendor>/<voice>/<id>.wav    # 16 kHz、单声道、PCM16
data/private/tts_compare/<vendor>/manifest.jsonl      # vendor, voice, voice_label, id, lang, text, path, duration_s, error...
```

## 第 4 步：试听、打分

```bash
uv run python scripts/tts_compare/build_listen_page.py
open data/private/tts_compare/listen.html
```

建议这样听：

1. 先听 **mix 10 句**：顶部点 `mix`，每句点"依次播放全部"。重点听：
   - 英文名有没有读成中文腔或者逐字母拼读（PostHog、Supabase、LibTV、Claude Code、Windsurf）
   - 缩写的读法对不对（DAU / ARPU / PR / CI 应该逐字母读）
   - 中英之间的停顿是否自然
2. 每个播放器下面有两个分数（1 差，5 好）：
   - **P**：发音对不对
   - **N**：像不像随口说话（不像新闻播报）
3. 再听 zh 和 en。重点是：
   - s13 的数字和日期
   - s15 的多音字：银行**行**长、**重**要、**重**新
   - s19 说错改口的语气
   - 中文音色读 en 句子的效果：这正好是"中国人说英文"的口音来源
4. 点"导出 CSV"保存分数。分数存在浏览器的 localStorage 里，换浏览器会丢。

可选的客观检查：用我们现有的 Qwen3-ASR zero-shot 解码这些 wav，和 sentences.txt 对比算 MER。专名读错的音色会直接显示出来。

## 发音修正（试听发现读错时）

| 厂商 | 方法 |
|---|---|
| MiniMax | `export MINIMAX_PRON='["Supabase/(ˈsuːpəbeɪs)","重庆/(chong2)(qing4)"]'`，对应接口的 `pronunciation_dict.tone`，格式为"原词/替换" |
| 火山 | `export VOLC_PRON='[...]'`，对应 `additions.pronunciation_dict`。也支持 SSML `<phoneme alphabet="py">`。英文用 IPA 是否生效**未验证** |
| CosyVoice | SSML `<phoneme alphabet="py"/"cmu">`，或者 `hot_fix` 参数。脚本暂未接入，需要时再加 |
| Azure | SSML `<phoneme>`（sapi/ipa）和自定义 lexicon。脚本暂未接入 |

## 未验证 / 已知风险

所有请求格式都按 2026-11-09 的官方文档写成，但**没有 key，一条都没有实际跑过**。第一次请先用 `--only s01` 冒烟测试。具体待确认的点：

- **MiniMax**：
  - 旧账号可能需要 `MINIMAX_GROUP_ID`
  - 国内站和国际站的音色表不同：国内中文音色 58 个、英文 16 个；国际站中文 34 个、英文 45 个
- **Qwen-TTS**：
  - 中英混说时 `language_type` 用 Auto 还是 Chinese 更好，未验证
  - 返回的是 24 小时内有效的 wav 链接，采样率可能是 24 kHz，脚本会重采样到 16 kHz
- **火山**：
  - 返回体是若干个 JSON 拼接成的流，脚本已兼容
  - 参数 `explicit_language` 设为 `zh-cn` 表示"中文为主，支持中英混读"
- **Azure**：
  - 输出格式用的是文档里列出的 `raw-16khz-16bit-mono-pcm`
  - DragonHD 音色不支持 `express-as` 风格
  - 世纪互联（Azure 中国）可用的音色较少
