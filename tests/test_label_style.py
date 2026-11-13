import pytest

from whisperflow.text.label_style import fix_spacing, strip_a_filler, to_label_v1, zh_numbers_to_digits


@pytest.mark.parametrize("src, want", [
    ("啊，那个我们明天开会", "那个我们明天开会"),
    ("然后啊，就是这样", "然后，就是这样"),
    ("好啊", "好啊"),
    ("对啊。", "对啊。"),
    ("快点啊！我们走", "快点啊！我们走"),
    ("我觉得，啊，可以", "我觉得，可以"),
])
def test_a_filler(src, want):
    assert strip_a_filler(src) == want


@pytest.mark.parametrize("src, want", [
    ("三个人", "3个人"),
    ("十五块", "15块"),
    ("下午三点", "下午3点"),
    ("下午三点半", "下午3点半"),
    ("三点五十分", "3点50分"),
    ("第二版", "第2版"),
    ("我住在一八零六", "我住在1806"),
    ("百分之五十", "50%"),
    ("十五万", "15万"),
    ("预算三十万左右", "预算30万左右"),
    ("我三十了", "我30了"),
    ("十几万", "十几万"),
    ("两个人", "2个人"),
    ("预算一万五", "预算15000"),
    ("三千二", "3200"),
    ("两千块", "2000块"),
    ("第三季度", "第3季度"),
    # kept (reported by the synthetic-text agent, 2026-11-09)
    ("一两天", "一两天"),
    ("三四十个", "三四十个"),
    ("上万条", "上万条"),
    ("第三方", "第三方"),
    ("第一次", "第一次"),
    ("茶百道", "茶百道"),
    ("五道口", "五道口"),
    ("七天酒店", "七天酒店"),
    ("看看四周有没有人", "看看四周有没有人"),
    ("二次确认一下", "二次确认一下"),
    ("做二次销售", "做二次销售"),
    # kept
    ("等一下", "等一下"),
    ("我们一起", "我们一起"),
    ("十分重要", "十分重要"),
    ("万一不行", "万一不行"),
    ("两三个", "两三个"),
    ("十几个", "十几个"),
    ("好几百", "好几百"),
    ("二十来个", "二十来个"),
    ("一个问题", "一个问题"),
    ("发一篇博客", "发一篇博客"),
    ("快一点", "快一点"),
    ("有一点难", "有一点难"),
    ("下午一点", "下午1点"),
    ("第一时间", "第一时间"),
])
def test_zh_numbers(src, want):
    assert zh_numbers_to_digits(src) == want


@pytest.mark.parametrize("src, want", [
    ("用Cursor写", "用 Cursor 写"),
    ("下午sync一下", "下午 sync 一下"),
    ("在 Figma 里 ，F-I-G-M-A 。", "在 Figma 里，F-I-G-M-A。"),
    ("3个人", "3 个人"),
    ("50%的人", "50% 的人"),
    ("This is fine.", "This is fine."),
])
def test_spacing(src, want):
    assert fix_spacing(src) == want


def test_full_pass():
    src = "嗯，啊，我们下午三点用Cursor sync一下，两三个人就够"
    assert to_label_v1(src) == "我们下午 3 点用 Cursor sync 一下，两三个人就够"
