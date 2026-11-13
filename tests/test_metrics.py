import random

import jiwer
import pytest

from whisperflow.eval.metrics import (
    align,
    align_text,
    cluster_bootstrap_ci,
    corpus_rate,
    score_alignment,
)


def test_identity():
    a = align_text("我们跑 benchmark", "我們跑 Benchmark。")
    s = score_alignment(a)
    assert s.errors == 0 and s.n_ref == 4 and s.en_correct == 1


def test_sub_del_ins_counts():
    a = align(list("abcd"), list("axcde"))
    assert (a.count("sub"), a.count("del"), a.count("ins")) == (1, 0, 1)
    a = align(list("abcd"), list("ad"))
    assert a.count("del") == 2 and a.errors == 2


def test_empty_hypothesis():
    s = score_alignment(align_text("你好世界", ""))
    assert s.dele == 4 and s.mer == 1.0


def test_language_attribution():
    # ref: 看 cer ; hyp: 看 sir 好  -> en sub on 'cer', zh insertion '好'
    s = score_alignment(align_text("看 CER", "看 sir 好"))
    assert s.err_by_lang["en"] == 1 and s.err_by_lang["zh"] == 1
    assert s.n_ref_by_lang == {"zh": 1, "en": 1, "num": 0}
    assert s.en_correct == 0


def test_render():
    assert align(["a", "b"], ["a", "c", "d"]).render() == "a [b→c] [+d]"


@pytest.mark.parametrize("seed", range(20))
def test_edit_distance_matches_jiwer(seed):
    """Independent cross-check of our DP against jiwer's word-level counts."""
    rng = random.Random(seed)
    vocab = list("甲乙丙丁戊") + ["gpu", "cer"]
    ref = [rng.choice(vocab) for _ in range(rng.randint(1, 15))]
    hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 15))]
    ours = align(ref, hyp).errors
    out = jiwer.process_words(" ".join(ref), " ".join(hyp) if hyp else "")
    assert ours == out.substitutions + out.deletions + out.insertions


def test_corpus_rate_is_micro_averaged():
    s1 = score_alignment(align(list("ab"), list("ax")))  # 1/2
    s2 = score_alignment(align(list("abcdefgh"), list("abcdefgh")))  # 0/8
    assert corpus_rate([s1, s2]) == pytest.approx(0.1)  # not mean(0.5, 0) = 0.25


def test_bootstrap_ci():
    scores = [score_alignment(align(list("abcd"), list("abxd" if i % 3 else "abcd"))) for i in range(30)]
    spk = [f"s{i % 6}" for i in range(30)]
    ci = cluster_bootstrap_ci(scores, spk, n_boot=500, seed=1)
    point = corpus_rate(scores)
    assert ci[0] <= point <= ci[1]
    assert ci == cluster_bootstrap_ci(scores, spk, n_boot=500, seed=1)  # deterministic
    assert cluster_bootstrap_ci(scores, ["s0"] * 30) is None
