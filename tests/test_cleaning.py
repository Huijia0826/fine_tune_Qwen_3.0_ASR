"""Exercise 10 tests. Run one step at a time: uv run pytest -q tests/test_cleaning.py -k step1"""

import pytest

from whisperflow.eval.cleaning import Span, classify_span, extract_spans, join_tokens


def ins(hyp, before="", after=""):
    return Span("ins", [], hyp, before.split() if before else [], after.split() if after else [])


# --- given (already passing): span extraction ------------------------------------------------------------

def test_extract_groups_insertions():
    spans = extract_spans("之类的我们", "然后之类的对我们")
    assert [(s.kind, s.hyp) for s in spans] == [("ins", ["然", "后"]), ("ins", ["对"])]
    assert spans[1].before == list("然后之类的") and spans[1].after == list("我们")


def test_extract_substitution():
    (s,) = extract_spans("the instruction has", "the instruction have")
    assert (s.kind, s.ref, s.hyp, s.before) == ("sub", ["has"], ["have"], ["the", "instruction"])


def test_join_tokens():
    assert join_tokens(["然", "后", "user", "feedback", "的"]) == "然后user feedback的"


# --- step 1: filler ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("hyp", [["对"], ["然", "后"], ["就", "是"], ["um"], ["you", "know"],
                                 ["对", "对"], ["就", "是", "那", "个"], ["so", "um"]])
def test_step1_filler(hyp):
    assert classify_span(ins(hyp)) == "filler"


@pytest.mark.parametrize("hyp", [["是"], ["后"], ["dog"], ["you"], ["就", "是", "狗"]])
def test_step1_not_filler(hyp):
    assert classify_span(ins(hyp)) != "filler"


# --- step 2: repeat ---------------------------------------------------------------------------------------

def test_step2_repeat_after():  # 他们[最近]最近的: the inserted copy comes first
    assert classify_span(Span("ins", [], ["最", "近"], list("他们"), list("最近的"))) == "repeat"


def test_step2_repeat_before():
    assert classify_span(Span("ins", [], ["the"], ["use", "the"], ["model"])) == "repeat"


def test_step2_filler_wins_over_repeat():  # 对[对]: both a filler and a repeat -> filler (rule order)
    assert classify_span(Span("ins", [], ["对"], ["对"], [])) == "filler"


def test_step2_not_repeat():
    assert classify_span(Span("ins", [], ["最", "近"], list("他们"), list("的融资"))) == "other"


# --- step 3: grammar --------------------------------------------------------------------------------------

@pytest.mark.parametrize("ref,hyp", [("has", "have"), ("intentions", "intention"), ("uses", "use"),
                                     ("used", "use"), ("going", "go"), ("are", "is")])
def test_step3_grammar(ref, hyp):
    assert classify_span(Span("sub", [ref], [hyp])) == "grammar"


@pytest.mark.parametrize("ref,hyp", [("cursor", "person"), ("has", "had"), ("on", "one"), ("然", "关")])
def test_step3_not_grammar(ref, hyp):
    assert classify_span(Span("sub", [ref], [hyp])) == "other"


def test_step3_only_single_word():
    assert classify_span(Span("sub", ["has", "it"], ["have", "its"])) == "other"


# --- step 4: everything else ------------------------------------------------------------------------------

def test_step4_deletion_is_other():
    assert classify_span(Span("del", ["对"], [])) == "other"
