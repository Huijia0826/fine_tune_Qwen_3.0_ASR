"""English number words -> digits, matching the benchmark v1 house style (1-9 spelled out, >= 10 digits)."""

import pytest

from whisperflow.text.numbers import en_numbers_to_digits


@pytest.mark.parametrize("text, expected", [
    ("we have like forty forty-five students", "we have like 40 45 students"),
    ("we all live until one hundred years old", "we all live until 100 years old"),
    ("three hundred and sixty dollars", "360 dollars"),
    ("就是will you kill ten people to save one", "就是will you kill 10 people to save one"),
    ("我们有一个 ten minutes 的 break", "我们有一个 10 minutes 的 break"),
])
def test_converts_numbers_ten_and_above(text, expected):
    assert en_numbers_to_digits(text) == expected


@pytest.mark.parametrize("text", [
    "there are four of us",          # 1-9 stay spelled out (benchmark v1: "four of us", "two kids")
    "one of the hottest topics",
    "for one thing",
    "the first",
    "我们有一个想法 第一个 一些",       # Chinese numerals are words here; never touched
])
def test_leaves_small_numbers_and_chinese_alone(text):
    assert en_numbers_to_digits(text) == text
