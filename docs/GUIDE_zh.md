# Fine Tune Qwen 3.0 ASR：训练框架说明

这是一个对 Qwen3-ASR-1.7B 做全参数微调的通用框架：给它一份「音频 + 转写」清单和一个 YAML 配置，它负责准备数据、训练、在验证集上选 checkpoint、在测试集上打分。路径、字段名、语言和超参都在配置里，代码里没有写死的东西。

## 概览

一条命令 `qasr pipeline -c 配置.yaml` 依次做四件事，每一步也可以单独运行：

1. **prepare**：把清单（manifest）转成 Qwen 的训练格式——语言标签、上下文 prompt、目标文本；可选上下文增强。
2. **train**：全参数微调，每 N 步存一个 checkpoint。
3. **select**：每个 checkpoint 在验证集上解码、打分，选 MER 最低的那个。
4. **test**：只用选中的 checkpoint 解码测试集，输出总体和分组的 MER、置信区间、逐条结果。

## 快速上手

先用 73 条公开英文音频跑一遍小测试（一张 GPU 约 10 分钟），确认环境没问题：

```bash
git clone https://github.com/Huijia0826/fine_tune_Qwen_3.0_ASR.git && cd fine_tune_Qwen_3.0_ASR
bash scripts/setup_env.sh .venv && source .venv/bin/activate   # Linux + NVIDIA GPU
pytest                                                          # 不需要 GPU 的测试
python examples/make_toy_dataset.py --out examples/toy
qasr pipeline -c examples/toy.yaml                              # 训练 5 步、选择、测试
```

然后换成自己的数据：

```bash
# CSV / TSV / JSONL 表格，或「一个音频 + 同名 txt」的文件夹 → 清单，按说话人分 train / val / test
python scripts/table_to_manifest.py my_data.csv --audio-col path --text-col transcript --speaker-col user --out data/
cp configs/example.yaml configs/my_run.yaml     # 改路径、字段名、超参
qasr pipeline -c configs/my_run.yaml --with-base   # 顺便给原始模型打分，用来对比
```

在 Slurm 集群上：`sbatch scripts/slurm_pipeline.sbatch configs/my_run.yaml`。

## 代码地图

| 文件 | 做什么 |
| --- | --- |
| `qwen_asr_ft/cli.py` | `qasr` 命令：prepare / train / select / test / pipeline / transcribe / score / compare |
| `qwen_asr_ft/config.py` | 读 YAML，补默认值，展开 `${环境变量}`，接受 `--set key=value` 覆盖 |
| `qwen_asr_ft/data.py` | 读清单、解析音频路径、定语言标签、生成 SFT 数据、按 `weight` 过采样 |
| `qwen_asr_ft/context.py` | 拼上下文 prompt、防泄漏、上下文增强（干扰词、形近词、错配的光标文字） |
| `qwen_asr_ft/train.py` | 训练，改编自 Qwen 官方微调脚本（Apache-2.0） |
| `qwen_asr_ft/infer.py` | 批量解码：长音频分块、上下文、防照抄、转简体；支持断点续跑 |
| `qwen_asr_ft/evaluate.py`、`metrics.py`、`text.py` | MER、分组统计、按说话人 bootstrap 的置信区间、两个系统的配对对比 |
| `configs/example.yaml` | 所有可配置项，带注释 |
| `scripts/` | 建环境、数据转换、Slurm 提交脚本 |
| `docs/` | 数据格式、训练代码详解、我们 J1–J4 的经验 |

## 数据格式

清单是 JSON Lines，一行一条音频，只有 `audio` 和 `text` 是必填的：

```json
{"id": "c-001", "audio": "audio/c-001.wav", "text": "我们周三，不对，周四用 Vercel 部署", "speaker": "u42"}
{"id": "c-002", "audio": "audio/c-002.flac", "text": "Ship it to staging first.", "speaker": "u7",
 "vocabulary": ["Vercel", "PostHog"], "app": "Slack", "before_cursor": "Can you deploy today?"}
```

- `speaker` 用来按说话人分集和算置信区间；`language` 不填时，有中文字就算 Chinese，否则 English；`weight` 是训练时重复次数。
- 上下文可以是自由文本 `context`，也可以是结构化字段（App、词表、光标前后文字）。字段名和自己的数据不一样时，在配置 `data.context.fields` 里对应一下就行。
- 其他字段会保留，可以拿来分组看结果（`eval.slice_by`）。

`qasr prepare` 把每条变成训练行：`prompt` 放进 Qwen 的 system 轮，`text` 是 `language English<asr_text>Ship it to staging first.`。和标签有 8 个字以上重叠的光标文字会被丢掉，防止模型学会照抄答案。详见 `docs/DATA_FORMAT.md`。

## 关键代码：训练

以下摘自 `qwen_asr_ft/train.py`。训练算法和 Qwen 官方脚本一致，只是参数从配置里读。

### 1. 加载模型

`model.base` 可以是 Hugging Face 仓库名或本地文件夹（离线集群用），可用 `model.revision` 锁定版本。真正的前向在 `model.thinker` 里，所以要把外层 `forward` 转发过去，否则 `Trainer` 调不到。

```python
base = resolve_base(cfg["model"])          # 本地文件夹，或下载并缓存 HF 快照
wrapper = Qwen3ASRModel.from_pretrained(base, dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map=None)
model, processor = wrapper.model, wrapper.processor   # 音频编码器 + Qwen3 LLM，全部参数参与训练
patch_outer_forward(model)                             # model.forward(...) -> model.thinker.forward(...)
```

### 2. 样本 → 输入序列

用 chat 模板生成前缀：system = 上下文，user = 一段音频，然后是 assistant 开头；目标文本接在后面。

```python
def build_prefix_messages(prompt, audio_array):
    return [{"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}]}]

prefix_text = processor.apply_chat_template(
    [build_prefix_messages(prompt, None)], add_generation_prompt=True, tokenize=False)[0]
```

### 3. Collator：只对输出文字算 loss

同一批样本编码两次：「前缀 + 目标 + EOS」和「只有前缀」，用后者的长度把前缀和 padding 的 label 设成 -100。模型看得到上下文和音频，但只学「输出什么文字」，不会学着复述上下文。

```python
full_inputs   = self.processor(text=[p + t + eos for p, t in zip(prefix_texts, targets)], audio=audios,
                               return_tensors="pt", padding=True, truncation=False)
prefix_inputs = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)

labels = full_inputs["input_ids"].clone()
for i, n in enumerate(prefix_inputs["attention_mask"].sum(dim=1).tolist()):
    labels[i, :n] = -100                       # system prompt + 音频占位不算 loss
labels[labels == self.processor.tokenizer.pad_token_id] = -100
full_inputs["labels"] = labels
```

音频在 collator 里实时读取并重采样到 16 kHz，所以 `train.num_workers` 影响速度；长音频不截断，显存不够就把 `train.batch_size` 调小、`train.grad_acc` 调大。

### 4. Trainer

标准 Hugging Face `Trainer`，两处定制：浮点输入（音频特征）转成模型的 dtype；每次存 checkpoint 时把 tokenizer / processor 文件拷进去，任何 checkpoint 都能直接解码。参数全部来自配置 `train.*`。

```python
args = TrainingArguments(
    output_dir=str(work / "train"),
    per_device_train_batch_size=tcfg["batch_size"],      # 默认 4
    gradient_accumulation_steps=tcfg["grad_acc"],        # 默认 4 -> 有效 batch 16
    learning_rate=tcfg["lr"],                            # 2e-5
    num_train_epochs=tcfg["epochs"],                     # 3
    max_steps=tcfg["max_steps"],                         # -1；小测试时设 5
    lr_scheduler_type=tcfg["lr_scheduler_type"], warmup_ratio=tcfg["warmup_ratio"],
    save_steps=tcfg["save_steps"], save_only_model=tcfg["save_only_model"],
    eval_strategy="steps" if has_val else "no", seed=tcfg["seed"],
    bf16=use_bf16, fp16=not use_bf16, remove_unused_columns=False,
)
trainer = CastFloatInputsTrainer(model=model, args=args, train_dataset=ds["train"],
                                 eval_dataset=ds.get("validation"),
                                 data_collator=DataCollatorForQwen3ASRFinetuning(processor=processor),
                                 callbacks=[MakeEveryCheckpointInferable()])
trainer.train(resume_from_checkpoint=resume)           # train.resume: true 时从最后一个 checkpoint 续训
```

### 5. 选 checkpoint

`qasr select` 用每个 checkpoint 解码 `data.val`，算池化 MER（总错误数 / 总参考 token 数），最低者胜，平局取更早的步数；结果写进 `best.txt`。规则在看测试结果之前就定好，测试集只用选中的那个跑一次。

```python
def pick_best(scores_by_ckpt):          # {"checkpoint-200": 7.68, "checkpoint-400": 7.25, ...}
    step = lambda name: int(name.rsplit("-", 1)[-1])
    return min(scores_by_ckpt, key=lambda k: (round(scores_by_ckpt[k], 4), step(k)))
```

## 解码与评测

- **解码**：greedy；`decode.language: auto` 由模型自己判断语言。超过 `decode.max_chunk_sec`（25 秒）的音频在安静处切块，短于 5 秒的尾块并到前一块——整段解码多分钟音频会漂移或循环，很短的尾块会让模型编内容。可选 `decode.copy_guard` 切掉照抄光标文字的尾巴。
- **MER**：中文按字、英文按词，忽略标点和大小写，繁体转简体；纯中文等于 CER，纯英文等于 WER。按总错误数 / 总 token 数池化，不是逐条平均。
- **置信区间**：按说话人重采样（同一个人的录音是相关的）。`qasr compare` 给出两个系统在同一批音频上的 MER 差值和 95% 区间，区间跨过 0 就是「看不出差别」；说话人少于 2 个时不判断显著性。
- **输出**：`metrics.md`（总体 + 分组）、`metrics.json`、`per_clip.csv`（按错误数排序，带参考答案和识别结果）。

## 注意事项

- **环境**：`qwen-asr==0.0.6` 锁定了 `transformers==4.57.6`，要放在单独的环境里（`scripts/setup_env.sh` 会建好）。训练和解码需要支持 bf16 的 NVIDIA GPU；数据准备、打分和测试在任何机器上都能跑。
- **标签风格要统一**：模型会学会你标签的写法（数字、空格、填充词…）。来源不一致时，写一个函数并设 `data.normalize_label: "模块:函数"`。我们的经验是 ASR 只做逐字转写，整理交给后处理。
- **测试集纪律**：只用验证集选模型；按说话人划分数据。
- **结果有噪声**：一个实验只跑一个 seed 时，1–2 个点的差异要看区间再下结论。
- **数据不在仓库里**：`data/`、`runs/` 和音频文件都在 `.gitignore` 里。我们 J1–J4 的成绩和经验见 `docs/LESSONS.md`。
- **状态**：框架的 CPU 部分已测试；GPU 训练和解码请先用上面的小测试跑一遍。
