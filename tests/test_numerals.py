import pytest

from whisperflow.eval.official import official_mer_num
from whisperflow.text.numerals import normalize_numerals


@pytest.mark.parametrize("a,b", [("下午三点五十开会", "下午3:50开会"), ("下午三点五十分开会", "下午3点50开会"),
                                 ("三点半", "3:30"), ("下午三点", "下午3:00"), ("百分之五十", "50%"),
                                 ("一千万个", "10000000个"), ("两个", "2个"), ("第一个", "第1个"),
                                 ("二零二四年", "2024年")])
def test_same_after_normalization(a, b):
    assert normalize_numerals(a) == normalize_numerals(b)


def test_scores_zero_errors_across_styles():
    assert official_mer_num("把三点改成三点五十", "把3点改成3:50").errors == 0


def test_does_not_hide_real_errors():
    assert official_mer_num("下午三点开会", "下午四点开会").errors == 1


def test_words_with_yi_are_symmetric():
    assert official_mer_num("我们一起看一下", "我们一起看一下").errors == 0
