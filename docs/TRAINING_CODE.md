# ASR 训练代码说明（Qwen3-ASR J 系列）

2026-11-13 · Mira

## 概览

我们对开源 Qwen3-ASR-1.7B 做全参数微调，训练脚本基本沿用 Qwen 官方，工作量主要在数据构建和评测上。一次训练（如 J3）由一个 shell 脚本从头跑到尾：

1. **合成数据 QC**：用零样本 Qwen 回读合成音频，把 TTS 读错的加入丢弃表。
2. **构建训练数据**：把各来源的 manifest、线上上下文、合成文本组装成 `train.jsonl` / `val.jsonl`（音频路径 + 上下文 prompt + 目标文本）。
3. **训练**：`qwen3_asr_sft.py` 在 1 张 H100 上训，每 200 步存一个 checkpoint。
4. **选 checkpoint**：每个 checkpoint 在 users' val 上解码和打分，按事先定好的规则选最低 MER 的那个。
5. **测试**：只对选中的 checkpoint 跑锁定测试集（benchmark v1、词典 benchmark、负责人测试集），带 / 不带上下文各一遍。
6. **对比**：用按说话人的配对 bootstrap 判断和线上 / 其他模型的差距是否显著，写进 `reports/`。

## 代码地图

| 环节 | 文件 | 做什么 |
| --- | --- | --- |
| 总入口 | `scripts/run_qwen3_asr_J3.sh` | 一次训练的全部步骤：QC → 建数据 → 训练 → 选 checkpoint → 测试；J1/J2/J4 各有一份 |
| 数据 | `scripts/make_qwen_e2e_data.py` | 组装 SFT 数据：标签转规则 v1、拼上下文、防泄漏、上下文增强、加长音频和合成数据 |
| 数据 | `src/whisperflow/text/label_style.py` | `to_label_v1()`：把任意标签统一成严格转写规则 v1 |
| 合成 | `scripts/synth/` | TTS、增强、回读 QC、拼接长音频 |
| 训练 | `scripts/qwen3_asr_sft.py` | Qwen 官方微调脚本的副本，只加了 seed、只存模型等小参数 |
| 解码 | `src/whisperflow/eval/transcribe.py` + `qwen_asr_backend.py` | `wf-transcribe`：按 yaml 配置解码一个数据集，支持断点续跑 |
| 打分 | `src/whisperflow/eval/score.py` + `official.py` | `wf-score`：官方 MER、分片统计、置信区间、逐条 `errors.csv` |
| 选择 | `scripts/pick_checkpoint.py` | 在 users' val 上按池化 MER 选 checkpoint |
| 显著性 | `scripts/paired_bootstrap.py` | 两个系统按说话人配对 bootstrap，给出差值的 95% 区间 |
| 上下文指标 | `scripts/context_metrics.py` | 目标词召回、干扰词误插入、照抄率 |
| 配置 | `configs/experiments/qwen3asr_*.yaml` | 模型版本、解码参数、数据和打分规则 |
| 同步 | `scripts/cloud_sync.sh` | 把代码和数据 rsync 到集群（不传 `.venv`、`runs`） |

## 训练数据格式

`make_qwen_e2e_data.py` 把每条音频变成一行 JSON，这是训练脚本唯一的输入：

```json
{"audio": "/abs/path/clip.wav",
 "prompt": "App: 微信 · casual\nVocabulary: Vercel, PostHog\nBefore cursor: …",
 "text": "language Chinese<asr_text>我们周三，不对，周四用 Vercel 部署"}
```

- `prompt`：线上记录的上下文，作为 Qwen 的 system prompt（App 与语气、用户词典 ≤ 50 词、光标前 200 字、光标后 120 字）；没有上下文时为空字符串。
- `text`：语言标签 + `<asr_text>` + 严格转写（经 `to_label_v1()` 统一规则）。

## 训练

`qwen3_asr_sft.py` 是 Qwen 官方微调脚本（QwenLM/Qwen3-ASR `finetuning/`，Apache-2.0，2026-11-09 拷贝）的副本，训练算法没有改。

- **只对输出文本算 loss**：前缀（system prompt + 音频）的 label 设为 -100。
- **全参数微调**：音频编码器和 LLM 解码器都更新，没有用 LoRA。bf16，HuggingFace `Trainer`。
- **我们加的小改动**（代码里标了 `# whisperflow:`）：`--seed`、`--save_only_model`、`--max_steps`、`--eval_steps`。
- 每个 checkpoint 保存时会把 tokenizer / processor 配置一起拷进去，可以直接拿来解码。

J3 的参数（有效 batch 16，963 步，1 张 H100 约 21 分钟）：

```bash
python scripts/qwen3_asr_sft.py --model_path <Qwen3-ASR-1.7B 快照> \
  --train_file runs/qwen_sft/J3/train.jsonl --eval_file runs/qwen_sft/J3/val.jsonl \
  --output_dir runs/qwen3asr_J3 --lr 2e-5 --lr_scheduler_type linear --warmup_ratio 0.05 \
  --batch_size 4 --grad_acc 4 --epochs 3 --save_steps 200 --seed 1234 --save_only_model 1
```

## 关键代码：训练

以下摘自 `scripts/qwen3_asr_sft.py` 和启动脚本，省略了参数解析和续跑等细节。带 `# whisperflow` 的行是我们改的。

### 1. 加载模型

`Qwen3ASRModel` 是 `qwen-asr` 包的封装：`.model` 是可训练的 HF 模型，`.processor` 负责音频特征和分词。真正的前向在 `model.thinker` 里，所以先把外层 `forward` 打补丁转发过去。

```python
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
asr_wrapper = Qwen3ASRModel.from_pretrained(
    args_cli.model_path,
    dtype=torch.bfloat16 if use_bf16 else torch.float16,
    device_map=None,
)
model = asr_wrapper.model          # 音频编码器 + Qwen3 LLM，全部参数参与训练
processor = asr_wrapper.processor

patch_outer_forward(model)         # model.forward(...) -> model.thinker.forward(...)
model.generation_config = GenerationConfig.from_model_config(model.config)
```

### 2. 样本 → 输入序列

用 chat 模板生成前缀：system = 上下文，user = 一段音频，然后是 assistant 开头。音频在 collator 里才读。

```python
def build_prefix_messages(prompt, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]

def _preprocess(ex):
    prefix_text = processor.apply_chat_template(
        [build_prefix_messages(ex.get("prompt", ""), None)],
        add_generation_prompt=True, tokenize=False)[0]
    return {"prompt": ex.get("prompt", ""), "audio": ex["audio"],
            "target": ex["text"],          # "language Chinese<asr_text>…"
            "prefix_text": prefix_text}
```

### 3. Collator：只对输出文本算 loss

同一批样本编码两次：一次是「前缀 + 目标 + EOS」，一次只有前缀，用后者的长度把前缀部分的 label 设成 -100。模型看到上下文和音频，但只学「输出什么文字」。

```python
@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features):
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]
        audios = [librosa.load(f["audio"], sr=self.sampling_rate, mono=True)[0] for f in features]

        full_inputs = self.processor(text=full_texts, audio=audios,
                                     return_tensors="pt", padding=True, truncation=False)
        prefix_inputs = self.processor(text=prefix_texts, audio=audios,
                                       return_tensors="pt", padding=True, truncation=False)

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100                      # 不对 system prompt + 音频占位算 loss
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100            # padding 也不算
        full_inputs["labels"] = labels
        return full_inputs
```

音频在 collator 里实时读取并重采样到 16 kHz，所以 `--num_workers` 开大一些（J3 用 8）。`truncation=False`：长音频不截断，显存不够时减小 batch。

### 4. Trainer

标准 HuggingFace `Trainer`，两处定制：浮点输入转成模型 dtype；每次存 checkpoint 时拷入 tokenizer / processor 配置。

```python
class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=dtype)
        return inputs

training_args = TrainingArguments(
    output_dir=args_cli.output_dir,
    per_device_train_batch_size=args_cli.batch_size,     # J3: 4
    gradient_accumulation_steps=args_cli.grad_acc,       # J3: 4  -> 有效 batch 16
    learning_rate=args_cli.lr,                           # 2e-5
    num_train_epochs=args_cli.epochs,                    # 3
    lr_scheduler_type=args_cli.lr_scheduler_type,        # linear
    warmup_ratio=args_cli.warmup_ratio,                  # 0.05
    save_steps=args_cli.save_steps,                      # 200
    save_only_model=(args_cli.save_only_model == 1),     # whisperflow: 不存优化器状态
    max_steps=args_cli.max_steps,                        # whisperflow: 冒烟测试用
    seed=args_cli.seed,                                  # whisperflow: 1234
    eval_strategy="steps" if args_cli.eval_file else "no",  # whisperflow
    bf16=use_bf16, fp16=not use_bf16,
    remove_unused_columns=False,                         # 保留 audio / prefix_text 给 collator
    report_to="none",
)

trainer = CastFloatInputsTrainer(
    model=model, args=training_args,
    train_dataset=ds["train"], eval_dataset=ds.get("validation"),
    data_collator=DataCollatorForQwen3ASRFinetuning(processor=processor),
    tokenizer=processor.tokenizer,
    callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
)
trainer.train()
```

### 5. 启动脚本里的训练和选择

摘自 `scripts/run_qwen3_asr_J3.sh`。训完写 `done` 标记，重跑时跳过训练；每个 checkpoint 在 users' val 两个子集上解码，交给 `pick_checkpoint.py` 选步数。

```bash
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-qwen/bin; RUN=runs/qwen3asr_J3; DATA=runs/qwen_sft/J3

[ -e $RUN/done ] || { $PY/python scripts/qwen3_asr_sft.py --model_path $BASE \
  --train_file $DATA/train.jsonl --eval_file $DATA/val.jsonl --output_dir $RUN \
  --lr 2e-5 --lr_scheduler_type linear --warmup_ratio 0.05 \
  --batch_size 4 --grad_acc 4 --epochs 3 --save_steps 200 --save_total_limit 50 \
  --num_workers 8 --seed 1234 --log_steps 10 --save_only_model 1 > $RUN.out 2>&1 \
  || { tail -30 $RUN.out; exit 1; }; touch $RUN/done; }

for ck in $(ls -d $RUN/checkpoint-* | sort -t- -k2 -n); do
  step=step_${ck##*-}; cks="$cks $step"
  for set in bczh:benchmark_chinese online:online_phase1; do
    decode sel_qwen3asr_J3_${step}_${set%%:*} ${set#*:} val $ck ctx   # wf-transcribe + wf-score
  done
done
best=$($PY/python scripts/pick_checkpoint.py qwen3asr_J3 $cks --refs v1)
```

`pick_checkpoint.py` 的核心：两个 val 子集合并算池化官方 MER（总错误数 / 总 token 数），最低者胜，平局取更早的步数。

```python
def pooled(prefix, v1=False):
    err = tok = 0
    for tag, man in SETS.items():          # benchmark_chinese val + online_phase1 val
        refs = {… utterance_id: raw_transcript for val rows …}
        for p in predictions(f"reports/runs/{prefix}_{tag}"):
            ref = refs[p["utterance_id"]]
            o = official_mer(to_label_v1(ref) if v1 else ref, p["hypothesis"])
            if o.included:
                err, tok = err + o.errors, tok + o.n_ref
    return 100 * err / tok

scores = {s: pooled(f"sel_{run}_{s}", v1) for s in steps}
print(min(steps, key=lambda s: (round(scores[s], 4), int(s.split("_")[1]))))
```

## 评测与 checkpoint 选择

- **解码 `wf-transcribe`**：读 yaml 配置，输出 `predictions.jsonl` 和 `run_meta.json`。Qwen 后端：greedy；超过 25 秒的音频在低能量处切块，短于 5 秒的尾块合并到前一块；上下文按片段从 `context_file` 读；输出转简体；可选 `copy_guard`。测试集必须加 `--allow-test`，并记入 `reports/test_access_log.jsonl`。
- **打分 `wf-score`**：主指标是官方 MER（`eval/vendor/benchmark_metrics.py`，按 sha256 锁定），与 ASR Arena 可比。注意 `errors.csv` 里的 `mer` 列是诊断口径，不是官方 MER。
- **选 checkpoint**：users' val（benchmark_chinese val 68 + online_phase1 val 57）上带上下文解码，池化官方 MER 最低者胜。
- **显著性**：`paired_bootstrap.py A B` 按说话人重采样 2,000 次，给出 B − A 的 95% 区间；跨过 0 视为不显著。

## 怎么复现一次训练

1. 同步代码和数据到集群（在 Mac 上）：`scripts/cloud_sync.sh - h100-coding '~/whisperflow'`
2. 建 Qwen 环境（只需一次）：`.venv-qwen`（`qwen-asr==0.0.6` 锁定 transformers 4.57.6）
   ```bash
   ssh h100-coding 'cd ~/whisperflow && srun -p debug --cpus-per-task=8 --mem=32G -t 01:00:00 bash scripts/qwen_asr_setup.sh'
   ```
3. 模型权重：Qwen/Qwen3-ASR-1.7B，revision `7278e1e7…`，放在 `~/whisperflow/hf_cache`；计算节点离线运行（`HF_HUB_OFFLINE=1`）。
4. 提交任务：
   ```bash
   sbatch -J qwen3asr-J3 -p batch --gres=gpu:h100:1 --cpus-per-task=16 --mem=128G -t 16:00:00 \
     -o runs/qwen3asr_J3.log --wrap "bash scripts/run_qwen3_asr_J3.sh"
   ```
5. 看结果：`runs/qwen3asr_J3.log`；细节在 `reports/runs/*/metrics.md`。

## 注意事项

- **数据不在仓库里**：`data/`、`reports/`、`runs/` 都没有上传（含线上用户音频和文本）；要跑的同事需要单独申请访问。
- **测试集纪律**：只用 users' val 选模型；测试集每个候选只跑一次。
- **标签规则要一致**：新数据先过 `to_label_v1()`；规则在 `docs/annotation_policy.md`。
- **结果的噪声**：每个实验只跑一个 seed，1–2 个点的差异要看 bootstrap 区间；benchmark v1 的参考答案偏向线上输出。
- **合成数据**：重新生成音频需要 `MINIMAX_API_KEY`。
