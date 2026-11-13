# 训练代码逐段讲解（用于复习和逐行重写）

对应代码：`src/whisperflow/training/data.py`、`src/whisperflow/training/train.py`
对应测试：`tests/test_training_data.py`

**重写练习的方法：** 新建一个空文件（比如 `practice/train_mine.py`），按本文的顺序，不看源码，一段一段写出来；
每写完一段，跑对应的测试或用小数据验证。卡住了再回来看本文，最后才看源码。

标 ⭐ 的是**必须能独立写出来**的核心块。

---

## 全景：一条数据从磁盘到梯度

```
manifest 一行 ──► 读 wav (16 kHz float32)
               ──► feature_extractor ──► log-mel 特征 [80, 3000]        ← 模型的"输入"
               ──► make_labels       ──► 标签 token id 列表              ← 模型要学会输出的"答案"
多条 ──► WhisperCollator ──► 一个 batch：features [B, 80, 3000]，labels [B, 最长长度]（补 -100）
batch ──► model(input_features, labels) ──► loss（交叉熵）
loss ──► backward ──► 梯度只流进 LoRA 参数 ──► optimizer.step()
```

---

## 1. 输入：log-mel 特征

- Whisper 不直接吃波形，吃的是 **log-mel 频谱图**：把音频切成每 10 ms 一帧，每帧算 80 个频带的能量（取对数）。
- 30 秒 × 每秒 100 帧 = **3000 帧** → 形状 `[80, 3000]`。不足 30 秒的**补零到 30 秒**，所以每条特征形状都一样。
  这就是为什么 collator 里特征可以直接 `np.stack`，不用补齐。
- 代码：`processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]`

---

## 2. ⭐ 标签：`make_labels`

Whisper 的解码器是**一个 token 一个 token 往外写**的。训练时要告诉它完整的"正确写法"：

```
<|startoftranscript|> <|zh|> <|transcribe|> <|notimestamps|> 我 们 先 跑 ... <|endoftext|>
```

| token | 作用 |
|---|---|
| `<|startoftranscript|>` | 开始信号 |
| `<|zh|>` / `<|en|>` | 语言。纯英文的条目用 `en`，中文和中英混说用 `zh`（`clip_language`） |
| `<|transcribe|>` | 任务是转写（不是翻译） |
| `<|notimestamps|>` | 不输出时间戳 |
| 文字 token | 真正的内容 |
| `<|endoftext|>` | 结束信号：模型必须学会"什么时候停"，否则会无限输出（幻觉） |

**关键细节：标签里要去掉开头的 `<|startoftranscript|>`。**

训练用的是 **teacher forcing**：解码器在第 t 步看到的是"正确答案的前 t-1 个 token"，要预测第 t 个。
模型内部用 `shift_tokens_right` 把标签**右移一位、开头补 `<|startoftranscript|>`**，作为解码器输入：

```
解码器输入:  <|sot|>  <|zh|>  <|transcribe|>  <|notimestamps|>  我   们  ...
要预测的:     <|zh|>   <|transcribe|>  <|notimestamps|>   我    们  ... <|eot|>
```

如果标签里自己也带了 `<|sot|>`，右移后开头就会出现两个 `<|sot|>`，模型学到的是错位的东西，**而且不会报错**。

重写要点：
```python
prefix = tokenizer.convert_tokens_to_ids([...四个特殊 token...])
ids = prefix + tokenizer.encode(文字, add_special_tokens=False) + [tokenizer.eos_token_id]
return ids[1:]          # 去掉 <|startoftranscript|>
```
英文条目文字前面加一个空格（`" " + text`）：Whisper 原本输出英文时，第一个词前面就带空格 token，保持一致。

验证：`tests/test_training_data.py::test_labels_layout_zh`

---

## 3. ⭐ 补齐：`WhisperCollator` 和 `-100`

一个 batch 里每条标签长度不一样，张量要求是矩形 → **短的要补齐**。

```
labels:  [1, 2, 3]          →  [[ 1,  2,    3],
         [4, 5]                 [ 4,  5, -100]]
```

**为什么补 `-100` 而不是补 0 或 `<|endoftext|>`？**
PyTorch 的 `CrossEntropyLoss` 默认 `ignore_index=-100`：标签是 -100 的位置**不计入 loss**。
- 补 `<|endoftext|>`（Whisper 的 pad id 就是它）会让模型在每条短句后面学"连续输出很多个结束符"，
  还会让短句在 loss 里被过度加权。
- 补 0 会让模型学着输出 token 0（一个感叹号）。

这两种错误**都不会报错**，只会让模型悄悄变差。这就是为什么 collator 是"最容易写错又不会报错"的地方。

重写要点：
```python
feats = torch.from_numpy(np.stack([...]))                       # [B, 80, 3000]
labels = torch.full((B, 最长长度), -100, dtype=torch.long)       # 先全部填 -100
for row, b in enumerate(batch):
    labels[row, :len(b["labels"])] = torch.tensor(b["labels"])   # 再把真实标签填到左边
```
验证：`tests/test_training_data.py::test_collator_pads_labels_with_ignore_index`

---

## 4. ⭐ LoRA：只训练"小补丁"

**原理：** 原模型的一个权重矩阵 W（比如 768×768）保持冻结不动，旁边加两个小矩阵 A（768×r）和 B（r×768）：

```
输出 = W·x  +  (alpha / r) · B·A·x
```

- r = 32 时，A+B 只有 2×768×32 ≈ 4.9 万个参数，而 W 有 59 万个 → 每层只训练约 8%。
- B 初始化为 0 → 训练开始时 LoRA 不改变任何输出，模型从原版 Whisper 出发。
- `alpha / r` 是缩放系数（这里 64/32 = 2），控制补丁的影响强度。

挂在哪里（`target_modules`）：注意力层的 `q_proj, k_proj, v_proj, out_proj`，
包括编码器自注意力、解码器自注意力、解码器交叉注意力（看音频的那一层）。

重写要点：
```python
from peft import LoraConfig, get_peft_model
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
**自检：** 可训练参数应该约 708 万（总参数的 2.9%）。如果是 100%，说明 LoRA 没挂上，在训练整个模型。

---

## 5. ⭐ 训练循环（`train()` 里的 `while` 循环）

```python
for batch in train_loader:
    loss = model(**batch).loss / accum      # ① 前向，算 loss
    loss.backward()                          # ② 反向，梯度累加到 .grad 里
    micro += 1
    if micro % accum:                        # ③ 还没攒够 accum 个小 batch：继续攒
        continue
    clip_grad_norm_(params, 1.0)             # ④ 梯度裁剪
    optimizer.step()                         # ⑤ 用梯度更新参数
    scheduler.step()                         # ⑥ 更新学习率
    optimizer.zero_grad()                    # ⑦ 清空梯度，准备下一轮
    step += 1
```

| 概念 | 为什么 |
|---|---|
| **梯度累加**（`accum=2`） | 内存只放得下 8 条，但想要 16 条的稳定性 → 跑 2 个小 batch 再更新一次。loss 要 `/ accum`，这样累加的梯度等于大 batch 的平均 |
| **`zero_grad`** | PyTorch 的梯度默认是**累加**的，不清空就会混进上一轮的梯度 |
| **梯度裁剪** | 偶尔某个 batch 梯度特别大（比如一条标注错误的数据），裁剪到 1.0 防止一步把模型带歪 |
| **warmup + 线性衰减** | 前 5% 的步数学习率从 0 慢慢升到 1e-4（刚开始梯度不稳定，步子先小一点），之后线性降到 0 |
| **AdamW 只传可训练参数** | 冻结的参数不需要优化器状态，省内存 |

**`model.train()` vs `model.eval()`：** 训练模式下 dropout 生效；评估时要切到 eval 模式关掉 dropout，
评估完再切回 train。`@torch.no_grad()` 让评估时不记录计算图，省内存、更快。

---

## 6. 验证 loss 和"选最好的检查点"

- 每 200 步在验证集上算一次 loss（`evaluate`），保存当前 adapter；如果是目前最低的，另存为 `best_adapter`。
- **验证 loss 是便宜的代理指标**，真正的成绩要用 `wf-transcribe` + `wf-score` 算 MER。
- 这就是"用验证集做选择"：挑哪个检查点。**测试集（benchmark v1）不能参与这个选择。**

---

## 7. 训练前的两个检查（顺序很重要）

| 检查 | 做法 | 通过标准 | 没通过说明 |
|---|---|---|---|
| **冒烟测试** | 跑 20 步 | 不报错；loss 是正常数字（不是 nan）；记录每步秒数和内存 | 代码或环境有问题 |
| **过拟合测试** | 16 条数据反复训练 150 步 | 训练 loss 降到接近 0 | 标签/补齐/LoRA 有 bug：连 16 条都背不下来，大数据更学不会 |

**过拟合测试是找"不报错的 bug"最有效的办法。** 标签错位、-100 写错、LoRA 没挂上，都会让 loss 降不下去。
