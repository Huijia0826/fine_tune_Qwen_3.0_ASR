"""merge_compounds / official_mer_compound: English spacing and compounding are forgiven, nothing else."""

from whisperflow.eval.official import merge_compounds, official_mer, official_mer_compound


def test_split_hypothesis_merged():
    assert merge_compounds("我用TopstepX下单", "我用top step X下单") == (
        "我用TopstepX下单", [("top step X", "TopstepX")])
    assert official_mer("TopstepX", "top step X").errors == 3
    assert official_mer_compound("TopstepX", "top step X").errors == 0


def test_joined_hypothesis_split_and_partial_spacing():
    assert merge_compounds("open inside entry now", "open InsideEntry now")[0] == "open inside entry now"
    assert merge_compounds("Topstep X", "topstepX")[0] == "Topstep X"


def test_recognition_errors_still_count():
    assert official_mer_compound("ProjectX", "project Y").errors == official_mer("ProjectX", "project Y").errors
    assert official_mer_compound("我用TopstepX", "我要top step X").errors == 1  # 用→要 still wrong


def test_same_word_count_untouched_and_punctuation_breaks_runs():
    assert merge_compounds("Project X", "project x") == ("project x", [])  # same split: scorer lowercases anyway
    assert merge_compounds("TopstepX", "top, step X")[1] == []
