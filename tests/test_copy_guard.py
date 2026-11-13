from whisperflow.eval.qwen_asr_backend import cut_copied_tail

CTX = "App: Signal · casual\nBefore cursor: 我觉得突破口就在弗洛瓦了。把弗洛瓦的那个朋友搞定。让他给我们多推一些"


def test_cuts_appended_context():
    said = "其实我认为现在我们最强的是韩国，韩语贼溜。"
    out = cut_copied_tail(said + "我觉得突破口就在弗洛瓦了。把弗洛瓦的那个朋友搞定", CTX)
    assert out == "其实我认为现在我们最强的是韩国，韩语贼溜"


def test_keeps_text_that_only_matches_context():  # re-dictating the same sentence: nothing left to keep → no cut
    assert cut_copied_tail("我觉得突破口就在弗洛瓦了", CTX) == "我觉得突破口就在弗洛瓦了"


def test_short_overlap_kept():
    text = "我们去弗洛瓦吧，明天见面聊一下这个事情"
    assert cut_copied_tail(text, CTX) == text


def test_english_tail_and_no_cursor_text():
    assert cut_copied_tail("左侧和右侧分别都是 0 分钟。The editor is",
                           "Before cursor: The editor is not accessible") == "左侧和右侧分别都是 0 分钟"
    assert cut_copied_tail("abc def ghi jkl", "App: X · casual\nVocabulary: def ghi jkl") == "abc def ghi jkl"


def test_merge_short_tail():
    import numpy as np

    from whisperflow.eval.qwen_asr_backend import merge_short_tail

    def fake_split(wav, sr, max_chunk_sec):  # cuts every max_chunk_sec seconds
        n = int(max_chunk_sec * sr)
        return [(wav[i:i + n], i / sr) for i in range(0, len(wav), n)]

    split = merge_short_tail(fake_split, 5.0)
    parts = split(np.zeros(16000 * 26), 16000, 25)          # 25 s + 1 s → one 26 s chunk
    assert [len(p) / 16000 for p, _ in parts] == [26.0]
    parts = split(np.zeros(16000 * 60), 16000, 25)          # 25 + 25 + 10 → unchanged
    assert [len(p) / 16000 for p, _ in parts] == [25.0, 25.0, 10.0]


def test_unglue_chunk_boundaries():
    from whisperflow.eval.qwen_asr_backend import unglue

    glued = "how about the frontend?Well, the ticket.One of"
    assert unglue(glued) == "how about the frontend? Well, the ticket. One of"
    assert unglue("mail lisa@gmail.com, version 3.5, Node.js") == "mail lisa@gmail.com, version 3.5, Node.js"
