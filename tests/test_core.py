"""CPU-only tests: no model download, no GPU. Run with `pytest`."""

import json

import numpy as np
import pytest
import soundfile as sf

from qwen_asr_ft.config import load_config
from qwen_asr_ft.context import Augmenter, build_context, label_terms, leaks
from qwen_asr_ft.data import build_sft, language_name, read_manifest
from qwen_asr_ft.evaluate import compare, evaluate, pick_best
from qwen_asr_ft.infer import cut_copied_tail, unglue
from qwen_asr_ft.metrics import edit_ops, paired_bootstrap, pooled_mer, score
from qwen_asr_ft.text import language_mix, tokenize


# ---- text / metrics ------------------------------------------------------------------------------------------
def test_tokenize_mixed():
    assert tokenize("用 Cursor 写，Node.js 3.5！") == ["用", "cursor", "写", "node.js", "3.5"]


def test_language_mix():
    assert language_mix("你好") == "zh"
    assert language_mix("hello there") == "en"
    assert language_mix("用 GitHub 部署") == "zh-en"


def test_edit_ops_counts():
    assert edit_ops(list("abc"), list("abc")) == (0, 0, 0)
    assert edit_ops(list("abc"), list("axc")) == (1, 0, 0)
    assert edit_ops(list("abc"), list("ac")) == (0, 1, 0)
    assert edit_ops(list("ac"), list("abc")) == (0, 0, 1)


def test_score_and_pooled():
    s1, s2 = score("我们 明天 开会", "我们后天开会"), score("hello world", "hello word")
    assert (s1.n_ref, s1.errors) == (6, 1)
    assert (s2.n_ref, s2.errors) == (2, 1)
    assert pooled_mer([s1, s2]) == pytest.approx(100 * 2 / 8)  # pooled, not the mean of 16.7% and 50%


def test_punctuation_and_case_ignored():
    assert score("Hello, World!", "hello world").errors == 0


def test_paired_bootstrap_detects_clear_difference():
    a = [score("a b c d", "a b c d")] * 20
    b = [score("a b c d", "x y c d")] * 20
    diff, lo, hi = paired_bootstrap(a, b, [f"s{i}" for i in range(20)], n_boot=200)
    assert diff == pytest.approx(50.0) and lo > 0


# ---- context ---------------------------------------------------------------------------------------------------
CCFG = load_config()["data"]["context"]


def test_leaks():
    assert leaks("我们明天下午三点开会讨论", "上次说的明天下午三点开会")
    assert not leaks("我们明天开会", "完全无关的文字")


def test_build_context_drops_leaking_cursor_text():
    row = {"app": "Slack", "tone": "casual", "vocabulary": ["Vercel", "PostHog"],
           "before_cursor": "please deploy to Vercel tomorrow morning", "after_cursor": "thanks"}
    ctx = build_context(row, CCFG, label="please deploy to Vercel tomorrow morning")
    assert "App: Slack · casual" in ctx and "Vocabulary: Vercel, PostHog" in ctx
    assert "Before cursor" not in ctx and "After cursor: thanks" in ctx


def test_free_text_context_wins():
    assert build_context({"context": "会议纪要", "app": "x"}, CCFG) == "会议纪要"


def test_label_terms():
    assert label_terms("我们用 PostHog 和 Claude Code 看数据") == ["PostHog", "Claude Code"]
    assert label_terms("The cat sat") == []


def test_augmenter_is_seeded_and_mixes():
    acfg = load_config()["data"]["augment"]
    rows = [{"text": f"用 Vercel 部署第{i}个", "vocabulary": ["Vercel"], "before_cursor": "x" * 40} for i in range(200)]
    def run():
        aug = Augmenter(acfg, CCFG, ["Versal", "Notion", "Figma", "Slack", "Linear"], set())
        return [aug.context(r, r["text"]) for r in rows]
    a, b = run(), run()
    assert a == b                                   # seeded
    assert len(set(a)) > 10                         # varied
    assert any("Vocabulary" not in c for c in a) or any("Versal" in c for c in a)


# ---- data ------------------------------------------------------------------------------------------------------
@pytest.fixture
def toy(tmp_path):
    for i in range(4):
        sf.write(tmp_path / f"a{i}.wav", np.zeros(16000, dtype="float32"), 16000)
    rows = [{"audio": "a0.wav", "text": "你好世界", "speaker": "s1", "weight": 3},
            {"audio": "a1.wav", "text": "hello world", "speaker": "s2", "vocabulary": "Vercel, Notion"},
            {"audio": "a2.wav", "text": "", "speaker": "s2"},
            {"audio": "missing.wav", "text": "x", "speaker": "s3"},
            {"audio": "a3.wav", "text": "用 GitHub 部署", "speaker": "s3", "language": "Cantonese"}]
    m = tmp_path / "train.jsonl"
    m.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path, m


def test_read_manifest_resolves_paths(toy):
    d, m = toy
    rows = read_manifest(m)
    assert rows[0]["audio"] == str((d / "a0.wav").resolve()) and rows[0]["id"] == "train-000000"


def test_build_sft(toy):
    _, m = toy
    cfg = load_config()
    out, stats = build_sft(read_manifest(m), cfg, train=True)
    assert stats["missing_audio"] == 1 and stats["empty_label"] == 1 and stats["clips"] == 3
    assert len(out) == 3 + 1 + 1                     # weight 3 repeats the first clip
    assert out[0]["text"] == "language Chinese<asr_text>你好世界"
    assert out[3]["text"].startswith("language English<asr_text>") and "Vocabulary: Vercel, Notion" in out[3]["prompt"]
    assert out[4]["text"].startswith("language Cantonese<asr_text>")


def test_language_name_auto():
    dcfg = load_config()["data"]
    assert language_name({"text": "用 GitHub"}, dcfg) == "Chinese"
    assert language_name({"text": "use GitHub"}, dcfg) == "English"
    assert language_name({"text": "x", "language": "zh"}, dcfg) == "Chinese"


# ---- config ----------------------------------------------------------------------------------------------------
def test_config_overrides_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", "/scratch/me")
    p = tmp_path / "c.yaml"
    p.write_text("run_name: r1\nwork_dir: ${WORK}/${run_name}\ndata:\n  train: a.jsonl\n", encoding="utf-8")
    cfg = load_config(str(p), ["train.lr=1e-5", "decode.language=Chinese"])
    assert cfg["work_dir"] == "/scratch/me/r1"
    assert cfg["data"]["train"] == ["a.jsonl"]
    assert cfg["train"]["lr"] == 1e-5 and cfg["train"]["batch_size"] == 4
    assert cfg["decode"]["language"] == "Chinese"


# ---- evaluate / select -------------------------------------------------------------------------------------------
def test_evaluate_and_compare(toy, tmp_path):
    _, m = toy
    cfg = load_config()
    cfg["eval"]["bootstrap"] = 50
    rows = read_manifest(m)
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("\n".join(json.dumps({"id": r["id"], "hypothesis": r["text"]}) for r in rows), encoding="utf-8")
    pb.write_text("\n".join(json.dumps({"id": r["id"], "hypothesis": "错"}) for r in rows), encoding="utf-8")
    rep = evaluate(m, pa, cfg, tmp_path / "eval")
    assert rep["overall"]["mer"] == 0.0 and (tmp_path / "eval" / "per_clip.csv").exists()
    assert "language_mix" in rep["slices"]
    cmp = compare(m, pa, pb, cfg, n_boot=50)
    assert cmp["mer_b"] > cmp["mer_a"] and cmp["clips_a_better"] >= 1


def test_pick_best_prefers_lower_then_earlier():
    assert pick_best({"checkpoint-200": 7.2, "checkpoint-400": 6.9, "checkpoint-600": 6.9}) == "checkpoint-400"


# ---- decoding post-processing ------------------------------------------------------------------------------------
def test_unglue():
    assert unglue("the frontend?Well then") == "the frontend? Well then"
    assert unglue("mail me at a@gmail.com 3.5") == "mail me at a@gmail.com 3.5"


def test_copy_guard():
    ctx = "Before cursor: see you at the standup tomorrow"
    assert cut_copied_tail("sounds good see you at the standup tomorrow", ctx) == "sounds good"
    assert cut_copied_tail("sounds good", ctx) == "sounds good"


def test_duplicate_ids_rejected(tmp_path):
    from qwen_asr_ft.data import read_manifests
    m = tmp_path / "m.jsonl"
    m.write_text('{"id": "x", "audio": "a.wav", "text": "a"}\n{"id": "x", "audio": "b.wav", "text": "b"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        read_manifests([m])
