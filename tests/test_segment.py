"""E3 long-clip segmentation: alignment and reference cutting (pure functions, no model)."""

from whisperflow.data.segment import Segment, align, edit_rate, plan_pieces, tokens


def test_tokens_keep_spans_and_normalise():
    t = tokens("我們先跑 Benchmark，然后看 CER。")
    assert [x.key for x in t] == ["我", "们", "先", "跑", "benchmark", "然", "后", "看", "cer"]
    assert "我們先跑 Benchmark，然后看 CER。"[t[4].start:t[4].end] == "Benchmark"


def test_align_maps_hyp_positions_to_ref_positions():
    ref = list("abcdef")
    hyp = list("abXdef")             # one substitution
    assert align(ref, hyp) == [0, 1, 2, 3, 4, 5, 6]
    ref_at = align(list("abcdef"), list("abdef"))   # one deletion (c)
    assert ref_at[0] == 0 and ref_at[-1] == 6


def test_edit_rate():
    assert edit_rate(list("abcd"), list("abcd")) == 0.0
    assert edit_rate(list("abcd"), list("abxd")) == 0.25


def test_plan_pieces_cuts_reference_at_segment_boundaries():
    ref = "今天我们先开会。然后讨论 deadline 的安排，最后总结一下。"
    segs = [Segment(0.0, 12.0, "今天我们先开会"),
            Segment(12.5, 25.0, "然后讨论deadline的安排"),     # Whisper glues the English word on
            Segment(25.5, 40.0, "最后总结一下")]
    pieces, dropped = plan_pieces(ref, segs, duration=41.0)
    assert dropped == 0
    # segments 1+2 fit in 28 s; segment 3 starts a new piece
    assert [p.reference for p in pieces] == ["今天我们先开会。然后讨论 deadline 的安排，", "最后总结一下。"]
    assert pieces[0].start == 0.0 and pieces[0].end == 25.15
    assert pieces[1].start == 25.35 and pieces[1].end == 40.15
    assert all(p.end - p.start <= 30.0 for p in pieces)


def test_plan_pieces_drops_piece_whose_text_disagrees():
    ref = "今天我们先开会。最后总结一下。"
    segs = [Segment(0.0, 20.0, "今天我们先开会"), Segment(20.5, 40.0, "完全不一样的内容啊啊啊")]
    pieces, dropped = plan_pieces(ref, segs, duration=41.0)
    assert [p.reference for p in pieces] == ["今天我们先开会。"] and dropped == 1
