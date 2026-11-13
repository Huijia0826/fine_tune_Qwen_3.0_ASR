# 学习笔记

做练习时遇到的概念和 debug 经验，按出现顺序记录。练习列表见 `exercises.md`。

---

## 1. 写数据代码之前，先把一行数据拆开看

**习惯：** 写任何处理数据的代码之前，先在交互式 Python 里把一行数据拆开看清楚：有哪些列、每列是什么类型、长什么样。这一步省下的 debug 时间远比它花的多。

在项目根目录启动交互式 Python：

```bash
uv run python
```

> 为什么用 `uv run python` 而不是 `python`？`uv run` 会使用项目自己的环境（装好了 pandas 等依赖）；
> 直接 `python` 可能用的是系统 Python，找不到这些包。

出现 `>>>` 后逐行输入，每行回车看结果：

```python
import pandas as pd
df = pd.read_parquet("data/private/ascend/raw/validation-00000-of-00001.parquet")

df.columns          # 有哪些列
len(df)             # 有多少行
row = df.iloc[0]    # 取第 0 行
row                 # 整行看一眼

a = row["audio"]    # 取出 audio 这一格
type(a)             # 它是什么类型？→ dict
a.keys()            # 字典里有哪些键 → 'bytes', 'path'
a["path"]           # 文件名
type(a["bytes"])    # → bytes
len(a["bytes"])     # 有多少字节
a["bytes"][:12]     # 只看开头 12 个字节
```

`exit()` 退出。

ASCEND 一行数据的真实样子：

```
id                  '00000'            ← 每个 split 都从 00000 开始 → 不唯一
audio               {'bytes': b'RIFF....WAVE...', 'path': 'ses4_spk20_L1593_13.1900_0.8200.wav'}
transcription       '嗯'               ← 列名是 transcription，不是 text
original_speaker_id 20
session_id          4                  ← 每个说话人都是 1~4 → 不是全局的对话 ID
```

---

## 2. wav 文件就是一串字节；只搬运时不要解码再编码

**概念：** 磁盘上的 `.wav` 文件本质上就是一串字节。开头是格式说明（`RIFF`、`WAVE`、采样率 16000、单声道…），
后面是声音的采样数据。

`a["bytes"][:12]` 的结果：

```
b'RIFF\xa4f\x00\x00WAVE'
```

`b'...'` 表示字节串，以 `RIFF...WAVE` 开头，正是 wav 文件的开头。也就是说 **`a["bytes"]` 就是一个完整 wav
文件的全部内容**，数据集发布者把整个文件塞进了表格的一格里。

所以不需要"转换成 wav"，原样写到磁盘就行：

```python
from pathlib import Path
Path("/tmp/test.wav").write_bytes(a["bytes"])
```

`write_bytes` 是 `Path` 自带的方法：打开文件 → 写入字节 → 关闭，一行完成。写完在终端运行 `open /tmp/test.wav`
就能听到录音。

**两种做法的对比：**

| | 解码再编码 | 直接写字节 |
|---|---|---|
| 步骤 | 字节 → 解码成数字数组（load）→ 再编码成 wav（save） | 字节 → 原样写入文件 |
| 风险 | save 时采样率、位深设错，音质就变了 | 和原始数据完全一样 |
| 速度 | 慢 | 快 |

**什么时候才需要 load / save？** 音频要被**修改**时，比如重采样、转单声道、剪切。只是搬运就不要动它。
原则是：**能不动原始数据，就不动。**
测试 `test_audio_is_written_unchanged` 检查的就是写出来的文件和原始字节是否完全一样。

---

## 3. 读 Python 报错（traceback）

例子：

```
ERROR collecting tests/test_ascend_adapter.py           ← ① 发生在哪个阶段
tests/test_ascend_adapter.py:11: in <module>             ← ② 调用链：测试文件第 11 行…
    from whisperflow.data.adapters.ascend import export_ascend
src/whisperflow/data/adapters/ascend.py:12: in <module>  ← …进到了你的文件第 12 行
    def export_ascend(parquet:list[Path], ...
                                   ^^^^                  ← ③ 箭头指着出错的位置
E   NameError: name 'Path' is not defined                ← ④ 错误类型 + 原因
```

**阅读顺序：** 先看最后一行（是什么错）→ 往上找**自己的文件**（错在哪一行）→ 最上面（哪个阶段）。

- `collecting` 阶段出错 = pytest 在**导入**你的文件时就失败了，一个测试都还没跑。
- `ERROR collecting` → 文件导入就失败了；`FAILED test_xxx` → 文件能导入，测试**跑起来了**，只是某一行出错。
  从 ERROR 走到 FAILED 是进展。
- pytest 输出里以 `>` 开头的那一行是出错的行，`^^^^` 指向出错的具体片段。
- 用 `pytest -x` 遇到第一个失败就停下。一次只盯一个问题，修一处，再跑。

**常见错误类型：**

| 错误 | 含义 | 例子 |
|---|---|---|
| `NameError` | 用了一个 Python 不认识的名字 | 用了 `Path` 但没 `from pathlib import Path` |
| `ModuleNotFoundError` | `import` 的模块（文件）不存在 | `from ...adapters.utils import ...`，但没有 `utils.py` |
| `AttributeError` | 对一个对象取了它没有的属性/方法 | `f"{name}".wav`：`.` 在引号外 = "取字符串的 wav 属性" |
| `TypeError` | 给函数传了错误类型的参数 | `write_bytes(row['audio'])`：要 `bytes`，给了 `dict` |

- `import` 只能导入**已经存在**的东西，不会凭空生成。
- 为什么 `def` 那一行就报 `NameError`？定义函数时 Python 会立刻读取参数的类型标注（`list[Path]`）；
  函数体里的代码要到**调用时**才执行，所以函数体里没导入的名字暂时不会报错。

**报错很长时：** 路径里带 `site-packages` 的是第三方库（pandas、pyarrow…）内部的调用链，只是在把你给的参数往下传，
一般不用看。只看两处：**最后一行**（结论）和**属于你的那一层**（你的文件，或交互式环境里的 `<stdin>`）。
库几乎从来不是出错的那一方，出问题的是你传进去的东西。

```
File "<stdin>", line 1, in <module>          ← 你敲的那一行
File ".../site-packages/pandas/..."          ┐
File ".../site-packages/pyarrow/..."         ┘ 库内部，跳过
FileNotFoundError: data/private/ascend/raw/validation-00000-of-00001.parquet   ← 结论
```

**缺一个函数时的三条路：** ① 自己写 ② 找项目里现成的（如 `src/whisperflow/data/audio.py`）
③ 先想想是不是根本不需要。

---

## 4. "file not found" 先看当前目录

```
➜  src git:(main) ✗ uv run pytest -q tests/test_ascend_adapter.py -x
ERROR: file or directory not found: tests/test_ascend_adapter.py
```

`tests/...` 是**相对路径**，意思是"从当前目录出发去找"。人在 `src/` 里，就会去找 `src/tests/...`，所以找不到。

```
whisper training/          ← 项目根目录：所有命令都在这里运行
├── src/
├── tests/
└── pyproject.toml
```

**习惯：** 这个项目的命令和配置里的路径都以项目根目录为起点。遇到 "file not found"，先运行 `pwd`
看看自己在哪个目录，用 `cd ..` 往上走一层。

交互式 Python 里也一样：相对路径从**启动 `uv run python` 时所在的目录**算起。在 `>>>` 里查当前目录：

```python
import os
os.getcwd()
```

所以要先 `cd` 到项目根目录再启动 `uv run python`；或者临时用绝对路径（以 `/Users/...` 开头）。

---

## 5. 嵌套数据：一层一层往里取

```
parquet          文件路径的列表
 └─ line         一个文件路径（只是路径，不是数据）
     └─ df       pd.read_parquet(line)，一张表
         └─ row  df.iloc[i]，一行
             ├─ row["transcription"]          文字 (str)
             └─ row["audio"]                  字典 (dict)
                 ├─ row["audio"]["bytes"]     音频内容 (bytes) ← 交给 write_bytes
                 └─ row["audio"]["path"]      文件名 (str)
```

方括号从左往右，一层一层拆，像拆快递：`row` 是箱子，`['audio']` 拿出盒子，`['bytes']` 才是里面的东西。
每次取东西前先问：**我手里现在是哪一层？** 不确定就 `type(...)` 看一下：`dict` 说明还是盒子，要继续拆。

常见混淆：
- `for i in range(len(df))` 里的 `i` 是**行号**（整数），要 `df.iloc[i]` 才拿到那一行。
- `'[UNK]' in 字符串` 是"包含这一段"；`'[UNK]' in 数组` 是"有没有某个元素**恰好等于**它"。
- `Path` 用 `/` 拼路径：`audio_out / name` 结果还是 `Path`，不需要 f-string。

---

## 6. 写文件：文件可以不存在，目录必须存在

```
E   FileNotFoundError: [Errno 2] No such file or directory: '.../test_xxx0/audio/ses1_spk1_L1_0.0_1.0.wav'
```

写文件时，**文件本身**不存在没关系，Python 会创建；但**它所在的目录**必须已经存在，Python 不会自动建目录。
报错只说"路径找不到"，不说是哪一级缺了，所以要**沿着路径一级一级往上查**：

```
.../test_xxx0/                ← 存在（pytest 建的）
             audio/           ← 不存在 ← 真正缺的是它
                   xxx.wav    ← 想新建的文件
```

修法：在写任何文件之前（函数开头、循环外面，只做一次）先建目录：

```python
audio_out.mkdir(parents=True, exist_ok=True)
```

| 参数 | 作用 | 不加会怎样 |
|---|---|---|
| `parents=True` | 中间缺的上级目录一起建 | 上级目录不存在时报错 |
| `exist_ok=True` | 目录已存在也不报错 | 第二次运行时报 `FileExistsError` |

**pytest 的 `tmp_path`：** 每个测试分到一个全新的空临时目录（`/private/var/folders/...`），在里面随便写，
不会弄脏项目。测试传进来的子目录（如 `tmp_path / "audio"`）是故意不预先建好的，因为真实运行时目标目录一开始也不存在。

---

## 7. `AssertionError`：代码没崩，但结果不对；没有 `return` 就返回 `None`

```
>       assert stats == {"n_total": 3, "n_skipped_unk": 1, "n_written": 2}
E       AssertionError: assert None == {'n_total': 3, 'n_skipped_unk': 1, 'n_written': 2}
```

- 前面那些错误（`FileNotFoundError`、`AttributeError`…）是代码**跑到一半崩了**；`AssertionError` 是代码**跑完了**，
  但结果不符合测试的要求。从"崩溃"走到"断言失败"是进展。
- pytest 会把 `==` **左右两边的实际值**都打印出来：左边是你的结果，右边是期望值。
- **函数执行到末尾都没遇到 `return`，自动返回 `None`。**

**计数器写法：** 先准备 → 边走边加 → 最后交回去。

```python
stats = {"a": 0, "b": 0}      # 循环之前
for ...:
    stats["a"] += 1           # 每一行都加：放在 if 之前
    if 条件:
        stats["b"] += 1       # 必须在 continue 之前，continue 之后的代码这次不会执行
        continue
    ...
return stats                  # 所有循环之外；缩进进循环 = 处理完第一行就返回
```

**为什么要报告计数：** 每一步过滤都要报告过滤掉了多少。数字突然异常（比如跳过的条数暴增），说明过滤条件或数据出了问题。
没有计数，就不知道自己悄悄丢掉了多少数据。

---

## 8. `with` 块：缩进回退 = 文件关闭

```
E       ValueError: I/O operation on closed file.
```

`with open(...) as f:` 进入时打开文件，**缩进一回退，块就结束，文件自动关闭**。在块外面再用它写文件，就会报这个错。

```python
with open(meta_out, "w", ...) as f:   ← 打开
    writer = csv.DictWriter(f, ...)
    writer.writeheader()
                                      ← 缩进回退：文件在这里关闭了
for line in parquet:                  ← 和 with 同一层 = 不在 with 里面
    writer.writerow(...)              ← 写已关闭的文件 → 报错
```

修法：所有要写这个文件的代码（整个循环）都要缩进到 `with` 里面；`return` 放在 `with` 外面（写完、关好再返回）。

- 离开 `with` 后变量（`writer`）还在，但它连着的文件已经关了，就像笔还在手上，本子已经合上了。
- VS Code 批量缩进：选中多行按 `Tab`（往右）/ `Shift+Tab`（往左）。

**写 TSV 用 `csv` 模块，不要自己 `"\t".join(...)`：** 转写里本身就可能有 Tab 或引号，`csv` 会自动转义。

```python
with open(path, "w", encoding="utf-8", newline="") as f:   # 中文必须 utf-8；newline="" 交给 csv 处理换行
    writer = csv.DictWriter(f, fieldnames=[...], delimiter="\t")
    writer.writeheader()          # 第一行：列名
    writer.writerow({...})        # 每条记录一行
```

---

## 9. 改了代码，报错却一字不差 → 先怀疑改动没生效

先别琢磨报错，确认改动真的进了**被运行的那个文件**：

- 文件没保存（VS Code 标签上有白点 ●，`Cmd+S` 保存）
- 改了另一个同名文件
- 有两处相似代码，只改了一处

**读测试失败时，先读测试在检查什么：** 测试函数名就是它的意图（如 `test_session_ids_do_not_merge_speakers`），
再看注释和 `assert`。pytest 输出第一行每个字符代表一个测试：`.` 通过，`F` 失败。

**ID 陷阱（ASCEND）：** `session_id` 只有 1~4，每个说话人都有 → 直接用会把十几个人捆成一组，
后面按 session 分组切分时整个数据集只剩 4 组，而且**不会报任何错**。要拼成唯一的：

```python
f"spk{row['original_speaker_id']}_ses{row['session_id']}"   # 外双引号，里单引号
```

这类"不报错但悄悄毁掉实验"的 bug，只能靠专门的测试抓。

---

## 10. 报错成千上万条时：先分类计数，再看分布

`wf-validate` 第一次在 ASCEND 验证集上报了 1075 个错误。不要逐条翻，先按错误代码数一下：

| 级别 | 代码 | 数量 |
|---|---|---:|
| error | `split_not_allowed` | 1067（每一条都有） |
| error | `audio_too_short` | 8 |
| warning | `clipping` | 251 |

**规律 1：每一条数据都有同一个错误 → 问题多半在配置或代码，不在数据。** 这次是 `configs/sources.yaml` 里
ASCEND 的 `allowed_splits` 漏了 `unassigned`（ingest 后所有记录都是 `unassigned`）。改一行配置，1067 个错误全部消失。

**规律 2：再按"谁"分组看分布，常常一眼看出根源。** 251 条削波里有 245 条来自同一个说话人 `ascend-12`
（他 41% 的录音削波）→ 是这个人的录音条件（麦克风增益/距离）问题，不是数据集整体问题。

**几个音频质量概念：**
- **削波（clipping）**：声音超过设备能记录的上限，波形顶端被削平 → 破音。轻微削波可以保留，真实用户录音也有。
- **过短音频（< 0.3 s）**：多是"嗯/对/啊"，信息太少、学习价值低，而且极短音频最容易让 Whisper 出幻觉 → 去掉。
- **error vs warning**：error = 这条不能用（命令退出码 1，后续步骤不该继续）；warning = 能用但值得人看一眼。

---

## 11. 正则表达式入门：`strip_fillers` 逐步拆解

```python
FILLERS = re.compile(r"[呃嗯]+")   # [..] 任意一个字符；+ 连续一次或多次
SPACES  = re.compile(r"\s+")      # \s 任意空白；\s+ 一串连续空白

text = FILLERS.sub("", text)             # 1. 删掉语气词
return SPACES.sub(" ", text).strip()     # 2. 连续空白合成一个，再去掉首尾空白
```

用 `"我觉得 呃 还可以"` 一步步看（`repr` 能把空格显示清楚）：

```
原文              '我觉得 呃 还可以'
第 1 步删"呃"后    '我觉得  还可以'     ← "呃"两边的空格还在，变成了两个空格
\s+ → " "         '我觉得 还可以'
.strip()          '我觉得 还可以'
```

`"呃 so"` 第 1 步后是 `' so'`（开头留了空格），所以还需要 `.strip()`。

- **为什么分两步：** 删字符会留下"孤儿空格"，第 2 步专门清理。每一步只做一件事，更容易检查。
- **`re.compile`：** 把模式预先编译好存成变量。函数要对上万条标注调用，不用每次重新解析模式，名字也更好读。
- **`r"..."`：** 原样字符串，让 `\s` 的反斜杠原样交给正则，写正则一律加 `r`。
- **`[呃嗯]` 只删这两个字，不会误伤"额度"**，因为"额"不在方括号里。规则越窄，误伤越少。

**教训：** 写清洗规则前，先去目标数据里看这个词实际怎么用（"额度"、句尾"啊"、benchmark 里的 um/uh），
不要凭印象定规则。

---

## 12. 训练"卡住"不动：先查内存（swap）

**现象：** 冒烟测试跑了 4 分钟，连第 1 步都没打出来；CPU 只用了 10%。

**排查步骤：**
1. 进程还活着吗？在干什么？→ `sample <pid>`（macOS 自带）看调用栈：停在 `run_backward` 等 GPU，说明不是死循环，是慢。
2. 为什么慢？→ 看内存：
   ```bash
   footprint <pid>          # 这个进程用了多少内存 → 27 GB
   sysctl vm.swapusage      # 交换区用了多少 → 39 GB
   ```
   机器只有 24 GB 内存，进程要 27 GB → 系统把内存"借"到硬盘上（swap），硬盘比内存慢几百倍 → 极慢。

**为什么训练这么吃内存：** 反向传播要用到前向传播时每一层的中间结果（激活值）。Whisper 编码器每条 1500 个位置 × 12 层，
每层注意力矩阵 1500×1500，batch 8、fp32 → 光激活值就十几 GB。

**解决：梯度检查点（gradient checkpointing）**：前向时不存每一层的激活值，反向需要时重新算一遍。
用时间换内存。实测（Apple M5，whisper-small + LoRA，fp32）：

| batch | 检查点 | 峰值内存 | 每条耗时 |
|---:|:---:|---:|---:|
| 8 | 关 | 27 GB（swap） | 极慢 |
| 4 | 关 | 15.8 GB | 0.69 s |
| 8 | **开** | **6.5 GB** | **0.78 s** |

**另一个办法：** 减小 batch + 增加梯度累加（batch 2 × 累加 8 = 等效 16），训练效果相同，同一时刻占用的内存更少。

**教训：** 先用冒烟测试（20 步）实测内存和速度，再估算完整训练要多久。不要凭感觉直接开跑几个小时的训练。
另外 macOS 没有 `timeout` 命令；zsh 里 `for a in "2 1"` 的 `$a` 不会自动拆成两个参数（bash 会）。

---

## 13. 长时间任务开跑前：把所有数据完整空跑一遍

**事故（2026-10-29）：** 正式训练跑到第 200 步（约 42 分钟），第一次算验证 loss 时崩溃：

```
ValueError: label has 1040 tokens > 448: 所以这个文章就是几件事情一直在重复讲吗？…
```

验证集里有超过 30 秒的长录音（留给转写评测用），但"训练中算验证 loss"和训练一样，只能处理 ≤30 秒、标签 ≤448 token 的数据。
冒烟测试只用了 5 个验证 batch，正好没碰到这些长录音。

**修法：**
1. 训练开始前用 `usable_for_loss` 把所有训练/验证数据检查一遍，不符合的筛掉并**打印数量**；
2. 正式开跑前，把训练集和验证集**全部**过一遍数据加载（读音频 + 特征 + 标签 + 补齐），不跑模型，几分钟就够。

**教训：** 冒烟测试只验证"代码能跑"，不能保证"每一条数据都能跑"。几个小时的任务开跑前，
要让所有数据在**第 0 秒**就被检查一遍：问题越早暴露，损失越小。

---

## 14. 读懂 pytest 的输出（练习 10）

运行 `uv run pytest -q tests/test_cleaning.py -k step1`（`-k step1` 只跑名字里带 step1 的测试）。

**第一行**：每个符号是一个测试，`.` 通过，`F` 失败。

```
.FF.F.F......
```

**每个失败的详情：**

```
hyp = ['然', '后']                                  ← 这次输入的是什么
>       assert classify_span(ins(hyp)) == "filler"  ← 出错的那一行（> 指向它）
E       AssertionError: assert 'other' == 'filler'  ← 实际返回 'other'，期望 'filler'
```

**测试名里的 `[hyp0]`、`[hyp1]`…**：同一个测试用 `@pytest.mark.parametrize` 换不同输入跑了几次，方括号里是第几个
输入（从 0 开始）。`test_step1_filler[hyp1]` = 列表里第 2 个例子 `["然","后"]` 没通过。

**报错是 `NotImplementedError: exercise 10: write me`**：说明跑的还是占位符，你的代码还没放进文件（或没保存）。
这不是代码写错，是代码根本没运行。

**看失败要找共同点**：练习 10 第一版失败的 4 个（然后、就是、you know、就是那个）都是"一个词由多个 token 拼成"，
通过的（对、um、对对、so um）都是"每个 token 本身就是一个词"。找到共同点，就找到了 bug 的原因。

---

## 15. `span.hyp` 是怎么取出来的（练习 10）

**字段在哪定义**：`cleaning.py` 的 `Span` 类（`@dataclass` 自动生成构造函数，`Span("ins", [], ["然","后"])` 就是
按顺序给 `kind`、`ref`、`hyp` 赋值）。

**值从哪来**：`extract_spans` 填进去，三步：

1. **对齐**：得到一串三元组 `(操作, 参考文字的字, 模型输出的字)`。参考文字 "之类的我们" vs 输出 "然后之类的对我们"：

   ```
   ('ins', None, '然')   ← 模型多写了"然"，参考文字这边没有 → None
   ('ins', None, '后')
   ('eq',  '之', '之')   ← 两边一样
   ('eq',  '类', '类')
   ('eq',  '的', '的')
   ('ins', None, '对')
   ('eq',  '我', '我')
   ('eq',  '们', '们')
   ```

2. **从头扫，把连续的"不一致"收进一个篮子 `run`**，遇到 `eq` 就把篮子打包成一个 Span。
   这里得到两个篮子：`[然, 后]` 和 `[对]`。

3. **打包时取每个三元组的第 3 个位置**（模型输出的字）：

   ```python
   hyp = [h for _, _, h in run if h is not None]
   #        ↑ 拆三元组，前两个用不上写成 _   ↑ del 时模型侧是 None，跳过
   ```

**一句话**：`hyp` = 一段连续对不上的地方里，模型那一侧写了哪些字。`ref` 同理，取第 2 个位置。

---

## 16. "从前往后剥"：判断一段文字是不是全由口头禅组成（练习 10 第 1 步）

**为什么不能逐个 token 查词表**：中文 token 是单个字，词表里存的是 "然后"、"就是"、"you know" 这样的完整词。
逐个查 "然"、"后" 都不在词表 → 误判。要先用 `join_tokens` 拼回文字，再按词去匹配。

**两个小动作**（可以先在 Python 里亲手试）：

```python
'就是那个'.startswith('就是')   # True：是不是以这个词开头
'就是那个'[2:]                  # '那个'：去掉开头 2 个字；通用写法 text[len(w):]
```

**完整函数**（写在 `classify_span` 上面）：

```python
FILLER_WORDS = sorted(FILLERS_ZH | FILLERS_EN, key=len, reverse=True)  # 长的排前面："就是" 先于 "就"


def is_all_fillers(text: str) -> bool:
    while text != "":                      # 还没剥完，就继续
        found = False                      # 这一轮有没有剥掉一个词？先假设没有
        for w in FILLER_WORDS:             # 一个词一个词地试
            if text.startswith(w):         # 开头是这个词吗？
                text = text[len(w):].strip()   # 是：剥掉它（strip 去掉英文词之间的空格）
                found = True
                break                      # 剥掉一个就够了，回到 while 再来一轮
        if not found:                      # 所有词都试过，一个都没剥掉
            return False                   # 卡住了 → 不是全由口头禅组成
    return True                            # text 剥空了 → 全是口头禅
```

**手动走一遍：**

| 输入 | 轮次 | text | 匹配到 | 结果 |
|---|---|---|---|---|
| 就是那个 | 1 | 就是那个 | "就是" | 剥掉 → "那个" |
| | 2 | 那个 | "那个" | 剥掉 → "" → **True** |
| 就是狗 | 1 | 就是狗 | "就是" | 剥掉 → "狗" |
| | 2 | 狗 | 无 | **False** |

**为什么要从长到短排**：如果先试 "就"，"就是那个" 会被剥成 "是那个"，而 "是" 不在词表 → 卡住，误判成不是 filler。

**接到 `classify_span`**：`if is_all_fillers(join_tokens(span.hyp)): return "filler"`。

**练习**：不看笔记，自己用 "so um"、"you know"、"后" 各走一遍表格，再对照测试结果。

---

## 17. 两个英文词是不是"同一个词的不同形式"（练习 10 第 3 步）

**概念**：英文词按语法变形叫**屈折变化**（复数、过去式、进行时……）。"是不是同一个词"= 还原成**原形（lemma）**后是否相同。

| 原形 | 变化形式 |
|---|---|
| intention | intention**s** |
| use | use**s** / use**d** / us**ing** |
| have | **has** / **had**（不规则） |

**三种做法：**

| 做法 | 怎么做 | 优点 | 缺点 |
|---|---|---|---|
| ① 规则 + 查表（我们用的） | 长的 = 短的 + 后缀；不规则的查 `IRREGULAR` | 简单透明，无依赖 | 只覆盖常见情况 |
| ② 词干提取（Porter stemmer） | 机械砍词尾 | 快、覆盖广 | 粗糙，会砍出非词、误合并 |
| ③ 词形还原（spaCy / WordNet） | 词典 + 语法还原原形 | 最准，含不规则 | 要装大库，有时需上下文 |

**我们的规则漏掉的**：studies/study（y→ies）、running/run（双写）、made/make、children/child。**误判的**：as/a。
**为什么够用**：诊断只比较系统之间的相对差异，三个系统漏的一样多；非母语常见错误（have/has、单复数）已覆盖。
以后需要更准，只换 `is_same_word` 内部实现，接口不变。

**两个小动作：**

```python
'intention' + 's' == 'intentions'                         # True：拼接再比较
frozenset(('have', 'has')) == frozenset(('has', 'have'))  # True：不分顺序的集合，查表不用写两遍
```

**补充规则**：以 e 结尾的词过去式只加 d（use → used）。只对以 e 结尾的词允许 "+d"，否则 "and" = "an" + "d" 会被误判。

**函数：**

```python
SUFFIXES = ("s", "es", "ed", "ing")


def is_same_word(a: str, b: str) -> bool:
    if not (a.isascii() and a.isalpha() and b.isascii() and b.isalpha()):
        return False                                  # 不是英文词（如 "然"）
    if frozenset((a, b)) in IRREGULAR:
        return True                                   # 不规则变化，查表
    for long, short in ((a, b), (b, a)):              # 两个方向都试
        for suf in SUFFIXES:
            if long == short + suf:
                return True                           # 长的 = 短的 + 后缀
        if short.endswith("e") and long == short + "d":
            return True                               # use → used
    return False
```

**手动走一遍：**

| a, b | 结果 | 原因 |
|---|---|---|
| intentions, intention | True | intention + s |
| has, have | True | 在 IRREGULAR |
| used, use | True | use 以 e 结尾 + d |
| has, had | False | 两个方向加任何后缀都对不上 |
| on, one | False | e 不是后缀；on 不以 e 结尾，不能 +d |

**接到 `classify_span`**：保留 `sub` + 两边各一个词的条件，只把 `ref.lower() == hyp.lower()` 换成 `is_same_word(ref, hyp)`。
和第 1 步一样的思路：**判断的东西换对了**（完全相同 → 同一个词的不同形式）。
