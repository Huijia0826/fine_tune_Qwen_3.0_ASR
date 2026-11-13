"""End-to-end eval workflow with a FAKE transcriber (no model download).

The fake returns scripted hypotheses so we can assert exact metric values; these numbers
test the plumbing only and say nothing about Whisper's accuracy."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from tests.conftest import make_record, write_jsonl
from whisperflow.data.schema import load_manifest
from whisperflow.eval.score import score_run
from whisperflow.eval.transcribe import main as transcribe_main
from whisperflow.eval.transcribe import run


class FakeTranscriber:
    def __init__(self, scripted: dict[str, str], order: list[str]):
        self.scripted, self.order, self.calls = scripted, order, 0

    def transcribe(self, audios):
        out = [self.scripted[self.order[self.calls + i]] for i in range(len(audios))]
        self.calls += len(audios)
        return out

    def describe(self):
        return {"backend": "fake", "model_commit": None}


@pytest.fixture
def eval_setup(repo):
    rows = [
        make_record(repo, "v1", seed=1, split="val", speaker_id="s1",
                    raw_transcript="我们先跑一下 benchmark 然后看 CER", hotwords_present=["benchmark", "CER"]),
        make_record(repo, "v2", seed=2, split="val", speaker_id="s2",
                    raw_transcript="今天天气很好", primary_scenario="zh_general",
                    language_tags=["zh"], hotwords_present=[]),
        make_record(repo, "t1", seed=3, split="test", speaker_id="s3"),
    ]
    m = write_jsonl(repo / "data/manifests/pilot.jsonl", rows)
    cfg = yaml.safe_load((repo / "configs/eval.yaml").read_text())
    return repo, m, cfg


def test_run_and_score_exact_values(eval_setup):
    repo, m, cfg = eval_setup
    recs = [r for r in load_manifest(m) if r.split == "val"]
    scripted = {"v1": "我們先跑一下 benchmark，然后看 sir。",  # trad. script (free) + 1 en sub
                "v2": "今天天氣很好"}                          # trad. script only -> 0 errors
    fake = FakeTranscriber(scripted, [r.utterance_id for r in recs])
    out = repo / "reports/runs/fake"
    meta = run(cfg, recs, fake, out, data_root=repo / "data/private", batch_size=1,
               manifest_path=str(m), split="val")
    assert meta["n_decoded_this_invocation"] == 2 and meta["config_hash"]

    res = score_run(out, str(m), cfg["scoring"])
    o = res["overall"]
    assert o["n"] == 2 and o["n_ref_tokens"] == 11 + 6
    assert o["mer"] == round(1 / 17, 4)
    assert o["en_word_error_rate"] == 0.5 and o["zh_char_error_rate"] == 0.0
    assert o["hotword_recall"] == 0.5
    assert o["n_hyp_traditional_script"] == 2
    assert res["slices"]["language_mix"]["zh_only"]["mer"] == 0.0
    assert res["slices"]["language_mix"]["zh_en_mixed"]["insufficient_n"] is True
    with open(out / "errors.csv", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    assert first["utterance_id"] == "v1" and "[cer→sir]" in first["alignment"]
    # Official scorer: 們->们 is in its 40-char T2S table but 氣 is not, so v2 costs 1 error there.
    assert o["official_mer"] == round(2 / 17, 4)
    assert o["official_t2s_mer"] == round(1 / 17, 4)  # full t2s: only the real error (cer->sir) remains
    assert "Official MER" in (out / "metrics.md").read_text()


def test_score_skips_predictions_no_longer_in_split(eval_setup):
    repo, m, cfg = eval_setup
    recs = [r for r in load_manifest(m) if r.split == "val"]
    out = repo / "reports/runs/excl"
    fake = FakeTranscriber({"v1": "我们先跑一下 benchmark 然后看 CER", "v2": "完全不对"},
                           [r.utterance_id for r in recs])
    run(cfg, recs, fake, out, data_root=repo / "data/private", batch_size=2,
        manifest_path=str(m), split="val")
    # v2 is excluded from the benchmark after decoding (split -> unassigned)
    lines = Path(m).read_text(encoding="utf-8").splitlines()
    Path(m).write_text("\n".join(ln.replace('"split": "val"', '"split": "unassigned"')
                                 if '"v2"' in ln else ln for ln in lines) + "\n", encoding="utf-8")
    res = score_run(out, str(m), cfg["scoring"])
    assert res["overall"]["n"] == 1 and res["overall"]["mer"] == 0.0
    assert res["n_predictions_out_of_split_skipped"] == 1


def test_resume_skips_done_and_rejects_config_change(eval_setup):
    repo, m, cfg = eval_setup
    recs = [r for r in load_manifest(m) if r.split == "val"]
    out = repo / "reports/runs/resume"
    fake = FakeTranscriber({"v1": "x", "v2": "y"}, ["v1"])
    run(cfg, recs[:1], fake, out, data_root=repo / "data/private", batch_size=4,
        manifest_path=str(m), split="val")
    fake2 = FakeTranscriber({"v1": "x", "v2": "y"}, ["v2"])
    meta = run(cfg, recs, fake2, out, data_root=repo / "data/private", batch_size=4,
               manifest_path=str(m), split="val")
    assert meta["n_decoded_this_invocation"] == 1
    lines = (out / "predictions.jsonl").read_text().splitlines()
    assert [json.loads(x)["utterance_id"] for x in lines] == ["v1", "v2"]

    cfg2 = json.loads(json.dumps(cfg))
    cfg2["decoding"]["num_beams"] = 1
    with pytest.raises(RuntimeError, match="different config"):
        run(cfg2, recs, fake2, out, data_root=repo / "data/private", batch_size=4,
            manifest_path=str(m), split="val")


def test_test_split_requires_flag(eval_setup):
    repo, m, _ = eval_setup
    assert transcribe_main(["--split", "test", "--manifest", str(m)]) == 2
    assert not (repo / "reports/test_access_log.jsonl").exists()
