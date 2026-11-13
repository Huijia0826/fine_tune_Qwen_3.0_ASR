"""Tests for exercise 2 (strip_fillers). Cases come from real ASCEND / benchmark transcripts."""

import pytest

from whisperflow.text.fillers import strip_fillers


@pytest.mark.parametrize("text, expected", [
    ("嗯", ""),                                  # filler-only clip -> empty (caller drops it)
    ("嗯嗯", ""),                                # runs of fillers
    ("呃 就是那个", "就是那个"),                   # leading filler + its space
    ("我觉得 呃 还可以", "我觉得 还可以"),          # middle: no double space left behind
    ("还可以 嗯", "还可以"),                       # trailing
    ("我呃觉得", "我觉得"),                        # glued to Chinese characters
    ("嗯对", "对"),
    ("呃 so we just start", "so we just start"),  # mixed zh/en
    ("um I in fact I have found one", "I in fact I have found one"),   # English fillers (decision 2026-10-30)
    ("you know uh uh you know", "you know you know"),
    ("a lot of uh companies正在做", "a lot of companies正在做"),     # glued to Chinese
    ("Umm yeah", "yeah"),
])
def test_removes_fillers(text, expected):
    assert strip_fillers(text) == expected


@pytest.mark.parametrize("text", [
    "免费额度也基本够用",          # 额 is part of 额度: must not be touched
    "感觉有点猛啊",               # sentence-final 啊 is a real particle
    "哦 对了",                    # 哦 is kept
    "take an umbrella",           # um/uh only as whole words
    "book an Uber",
    "跟parents联系",              # no spaces added between zh and en
])
def test_keeps_everything_else(text):
    assert strip_fillers(text) == text
