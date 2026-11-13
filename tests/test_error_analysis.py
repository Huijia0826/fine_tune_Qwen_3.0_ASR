from tests.conftest import make_record
from whisperflow.data.schema import parse_record
from whisperflow.eval.error_analysis import analyze_system, tokens_formatted


def _rec(uid, text, scen="en_general", tags=("en",)):
    return parse_record(make_record(uid=uid, raw_transcript=text, primary_scenario=scen,
                                    language_tags=list(tags), hotwords_present=[]))[0]


def test_formatting_layer_handles_numbers_and_contractions():
    assert tokens_formatted("I'm buying four apples") == tokens_formatted("I am buying 4 apples")
    assert tokens_formatted("I’m here") == tokens_formatted("I am here")  # curly apostrophe


def test_layers_attribute_script_formatting_recognition():
    recs = [_rec("a", "我们先跑一下测试", "zh_general", ("zh",)), _rec("b", "I am buying four apples")]
    hyps = {"a": "我們先跑一下測試",          # script only: 們 in official table, 測/試 are not
            "b": "I'm buying 4 oranges"}     # formatting (I'm, 4) + 1 real error (oranges)
    r = analyze_system(recs, hyps)
    c = r["layer_contrib"]
    n = 8 + 5
    assert abs(c["recognition"] - 1 / n) < 1e-9          # only apples->oranges remains
    assert c["script"] > 0 and c["formatting"] > 0
    assert r["recognition_errors"] == {"en_sub": 1}
    assert r["top_en_substitutions"] == [["apples→oranges", 1]]


def test_bursts_and_language_id_failure():
    recs = [_rec("z", "今天我们讨论一下这个项目的进度安排", "zh_general", ("zh",)),
            _rec("e", "please send the report")]
    hyps = {"z": "Today we discuss the progress of this project",   # translated -> LID failure
            "e": "please send the report thank you thank you thank you"}  # inserted span
    r = analyze_system(recs, hyps)
    assert r["language_id_failures"] == 1
    assert r["burst_insertions"]["count"] >= 1
