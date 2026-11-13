from whisperflow.text.normalize import contains_term, detect_languages, normalize, token_lang, tokenize


def test_mixed_tokenization_and_punctuation_removed():
    assert tokenize("我们先跑一下 benchmark，然后看 CER。") == [
        "我", "们", "先", "跑", "一", "下", "benchmark", "然", "后", "看", "cer"]


def test_no_space_between_scripts_still_splits():
    assert tokenize("用GPU训练") == ["用", "gpu", "训", "练"]


def test_traditional_to_simplified():
    assert normalize("我們測試") == normalize("我们测试")


def test_fullwidth_ascii_folded():
    assert normalize("ＧＰＵ１２３") == "gpu123"


def test_technical_tokens_preserved():
    assert tokenize("用 Node.js 和 C++ 还有 C#，版本 3.5") == [
        "用", "node.js", "和", "c++", "还", "有", "c#", "版", "本", "3.5"]


def test_apostrophe_and_trailing_dot():
    assert tokenize("don't stop. OK.") == ["don't", "stop", "ok"]


def test_numerals_not_normalized_in_v1():
    # Documented limitation: Chinese numerals and digits are different tokens.
    assert normalize("二零二四") != normalize("2024")


def test_token_lang_and_detect_languages():
    assert token_lang("我") == "zh" and token_lang("gpu") == "en" and token_lang("2024") == "num"
    assert detect_languages("在2024年") == {"zh"}
    assert detect_languages("跑 benchmark") == {"zh", "en"}


def test_contains_term_boundaries():
    assert contains_term("然后看 CER 吧", "cer")
    assert not contains_term("certain things", "CER")
    assert contains_term("这个大模型很好", "模型")
    assert contains_term("我们用 Node.js", "node.js")
    assert not contains_term("anything", "")
