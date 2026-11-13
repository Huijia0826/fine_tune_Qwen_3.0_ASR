"""Bring training / test labels to annotation policy v1 (docs/annotation_policy.md, owner 2026-11-09).

Unlike text/numerals.py (scoring: applied to both sides, so it may be aggressive), these rewrite LABELS, so every
rule fires only where the reading is unambiguous; anything doubtful is left as written.

  1. 啊 as a hesitation filler is removed, 啊 as a sentence-final particle is kept   (strip_a_filler)
  2. Chinese numerals → Arabic for quantities, money, time, dates, ordinals, digit strings   (zh_numbers_to_digits)
     kept: fixed words and idioms (一下 一起 一般 十分 万一 …), approximate numbers (两三个 十几个 几百), and
     一 + any measure word (一个 一篇 一次: read as an article; owner to confirm on the change list)
  3. one space around every Latin / digit token inside Chinese text, none next to Chinese punctuation
     (fix_spacing)
English number words keep the existing house style (text/numbers.py: ≥ 10 as digits).
"""

from __future__ import annotations

import re

from whisperflow.eval.vendor import benchmark_metrics as _bm
from whisperflow.text.fillers import strip_fillers

CJK = r"一-鿿"
ZH_PUNCT = "，。？！、；：“”‘’（）《》【】…"
_DIG = "零〇一二三四五六七八九两幺"
_SINGLE = {c: str(i) for i, c in enumerate("零一二三四五六七八九")} | {"两": "2", "〇": "0", "幺": "1"}
_NUMERAL = rf"[{_DIG}十百千万亿]+"

# measure words / units after which a numeral is a quantity (longest first so 分钟 wins over 分)
MEASURES = sorted("""
    个 位 名 人 口 块 元 毛 角 美金 美元 刀 次 遍 趟 天 周 星期 礼拜 个月 月 年 号 日 分钟 小时 秒 岁 页 张 条
    篇 本 家 台 部 件 款 种 版 层 楼 室 期 届 场 轮 步 章 节 倍 公里 千米 米 厘米 斤 公斤 克 份 套 道 题 行 列
    集 首 杯 瓶 箱 单 票 分 点 寸 栋 间 辆 座 所 支 只 双 对 组 班 项 处 回 声 封 下单 毫秒 毫米 微秒 帧 像素 G K W
    """.split(), key=len, reverse=True)
IDIOMS = ("一下", "一起", "一般", "一样", "一定", "一直", "一些", "一点", "一切", "一旦", "一边", "一共", "一致",
          "一会", "一方面", "十分", "万一", "千万", "万万", "百分百", "一心", "一向", "统一", "唯一", "第一时间",
          "一五一十", "三心二意", "七上八下", "乱七八糟", "一模一样", "一时", "一番", "一阵", "两者", "两边",
          "两人世界", "第三方", "第一次", "第一手", "一两", "五道口", "七天酒店", "七天", "四季酒店",
          "三只松鼠", "七匹狼", "六神", "九阳", "一点点", "八达岭", "九寨沟", "三里屯", "十三行",
          "四周", "二次", "四面八方", "三番五次", "一五一十")
# not 早 / 晚 (早一点 = a bit earlier)
TIME_WORDS = ("上午", "下午", "晚上", "早上", "凌晨", "中午", "傍晚", "今天", "明天", "昨天", "后天")
CONNECTORS = ("然后", "就是", "那个", "这个", "所以", "但是", "因为", "而且", "其实", "反正", "那", "就")


def _to_int(s: str) -> str | None:
    """'五十' → '50', '一八零六' → '1806', '两' → '2', '一万五' → '15000'; None if s is not a clean numeral."""
    m = re.fullmatch(r"(.+?)([万千百])([一二三四五六七八九两])", s)  # colloquial: the next unit is left out
    if m and (head := _to_int(m[1])) and head.isdigit() and "万千百".find(m[1][-1]) < 0:
        unit = {"万": 10000, "千": 1000, "百": 100}[m[2]]
        return str(int(head) * unit + int(_SINGLE[m[3]]) * unit // 10)
    if all(c in _SINGLE for c in s):
        return "".join(_SINGLE[c] for c in s) if len(s) > 1 else _SINGLE[s]
    v = _bm._parse_zh_numeral(s.replace("两", "二"))
    return v or None


def _num_with_unit(s: str) -> str | None:
    """Keep 万 / 亿 as written units: 十五万 → 15万, 三千万 → 3000万, 一万五 → 15000 (no trailing unit)."""
    m = re.fullmatch(rf"([{_DIG}十百千]+)([万亿])", s)
    if m and (v := _to_int(m[1])):
        return v + m[2]
    return _to_int(s)


def approximate(n: str, prev: str, nxt: str) -> bool:
    """Estimates and non-numbers stay as characters: 好几个 十几个 二十来个 三十多块 (几 / 多 / 来 / 余 around),
    上万条 成千 近百 (上 / 成 / 近 / 数 before), 两三个 一两天 三四十个 (two adjacent digits), and a bare
    magnitude without a digit (茶百道; 万一 is an idiom)."""
    if (prev and prev in "几好数上成近") or (nxt and nxt in "几多来余"):
        return True
    if re.fullmatch(r"[十百千万亿]+", n):
        return True
    a, b = (_SINGLE.get(c) for c in (n[:2] + "  ")[:2])
    return bool(a and b and int(b) == int(a) + 1)


def strip_a_filler(text: str) -> str:
    """Remove 啊 used as a hesitation filler; keep it as a sentence-final particle.

    Filler = at the start of the text or of a clause (after punctuation), or right after a connector word, AND
    followed by a pause mark or more speech: "啊，那个" → "那个", "然后啊，就是" → "然后，就是".
    Particle = before a sentence end or at the very end: "好啊" "对啊。" "快点啊！" stay.
    """
    t = re.sub(rf"(^|[{ZH_PUNCT},.?!]\s*)啊+[，,]?\s*(?=[{CJK}A-Za-z0-9])", r"\1", text)
    conn = "|".join(CONNECTORS)
    t = re.sub(rf"({conn})啊+(?=[，,])", r"\1", t)
    return t


def zh_numbers_to_digits(text: str) -> str:
    def protect(t: str) -> tuple[str, dict]:
        keep = {}
        for i, w in enumerate(sorted(IDIOMS, key=len, reverse=True)):
            if w in t:
                tok = f"\x00{i}\x00"
                keep[tok] = w
                t = t.replace(w, tok)
        return t, keep

    t = text
    # percent: 百分之五十 → 50%
    t = re.sub(rf"百分之({_NUMERAL})", lambda m: (_to_int(m[1]) or m[1]) + "%" if _to_int(m[1]) else m[0], t)
    # times, before idiom protection (it would lock 一点 in 下午一点); 一点 only after a time word ("a little")
    def time_sub(m: re.Match) -> str:
        pre, n, rest = m[1] or "", m[2], m[3] or ""
        if n == "一" and not pre:
            return m[0]
        v = _to_int(n)
        if not v or int(v) > 24:
            return m[0]
        rest_d = rest
        if rest and rest not in ("半", "整", "钟"):
            mm = re.fullmatch(rf"({_NUMERAL})(分?)", rest)
            rest_d = ((_to_int(mm[1]) or mm[1]) + mm[2]) if mm else rest
        return f"{pre}{v}点{rest_d}"
    t = re.sub(rf"({'|'.join(TIME_WORDS)})?({_NUMERAL})点(半|整|钟|{_NUMERAL}分?)?", time_sub, t)
    t, keep = protect(t)
    # ordinals: 第二版 → 第2版 (第一时间 etc. are protected idioms)
    t = re.sub(rf"第({_NUMERAL})", lambda m: "第" + (_to_int(m[1]) or m[1]), t)
    # digit-by-digit strings of 3+ (room numbers, codes, years): 一八零六 → 1806
    t = re.sub(r"[零〇一二三四五六七八九幺]{3,}", lambda m: "".join(_SINGLE[c] for c in m[0]), t)
    # quantities: numeral + measure word
    meas = "|".join(map(re.escape, MEASURES))

    def qty(m: re.Match) -> str:
        n, unit = m[1], m[2]
        prev = t[max(0, m.start() - 1):m.start()]
        nxt = t[m.end():m.end() + 1]
        if approximate(n, prev, nxt):
            return m[0]
        if n == "一":                                                 # 一个 / 一篇 / 一次: article-like
            return m[0]
        v = _num_with_unit(n)
        return f"{v}{unit}" if v else m[0]
    t = re.sub(rf"({_NUMERAL})({meas})", qty, t)

    # standalone multi-character numerals with a magnitude (预算十五万, 三十了): unambiguous numbers
    def standalone(m: re.Match) -> str:
        prev, nxt = t[max(0, m.start() - 1):m.start()], t[m.end():m.end() + 1]
        if (prev and prev == "第") or approximate(m[0], prev, nxt):
            return m[0]
        return _num_with_unit(m[0]) or m[0]
    t = re.sub(rf"[{_DIG}]*[十百千万亿][{_DIG}十百千万亿]*", lambda m: standalone(m) if len(m[0]) > 1 else m[0], t)
    for tok, w in keep.items():
        t = t.replace(tok, w)
    return t


def fix_spacing(text: str) -> str:
    """One space between Chinese and Latin / digit tokens, none next to Chinese punctuation."""
    t = re.sub(rf"([{CJK}])([A-Za-z0-9])", r"\1 \2", text)
    t = re.sub(rf"([A-Za-z0-9%])([{CJK}])", r"\1 \2", t)
    t = re.sub(rf"\s+([{ZH_PUNCT}])", r"\1", t)
    t = re.sub(rf"([{ZH_PUNCT}])\s+", r"\1", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def to_label_v1(text: str) -> str:
    """Full policy-v1 pass for a strict transcript: fillers, 啊, numerals, spacing (in that order)."""
    t = fix_spacing(zh_numbers_to_digits(strip_a_filler(strip_fillers(text))))
    return re.sub(r"^[，,、。\s]+", "", t)  # a removed leading filler leaves its pause mark behind
