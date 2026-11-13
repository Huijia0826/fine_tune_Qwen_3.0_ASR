import hashlib
import random

import pytest

from tests.conftest import make_record, write_jsonl
from whisperflow.data.audit import check_leakage, dataset_stats
from whisperflow.data.audit import main as audit_main
from whisperflow.data.common import IssueLog
from whisperflow.data.schema import parse_record, write_manifest
from whisperflow.data.split import (
    SplitError,
    assign_splits,
    build_groups,
    check_lock_integrity,
    make_lock,
)


def R(uid, spk, dur=10.0, scen="zh_general", sess=None, **over):
    base = dict(uid=uid, speaker_id=spk, session_id=sess or f"{spk}-sess", duration_sec=dur,
                sha256=hashlib.sha256(uid.encode()).hexdigest(), split="unassigned",
                primary_scenario=scen, raw_transcript=f"第{uid}句话" if scen == "zh_general"
                else f"第{uid}句 code switch", language_tags=["zh"] if scen == "zh_general"
                else ["zh", "en"], hotwords_present=[])
    base.update(over)
    rec, errs = parse_record(make_record(**base))
    assert not errs, errs
    return rec


def pool(n_spk=30, per_spk=12, seed=0):
    rng = random.Random(seed)
    out = []
    for s in range(n_spk):
        scen = "zh_en_switch" if s % 3 == 0 else "zh_general"
        for u in range(per_spk):
            out.append(R(f"s{s}u{u}", f"spk{s:02d}", dur=rng.uniform(3, 15), scen=scen))
    return out


def split_of(recs):
    return {r.utterance_id: r.split for r in recs}


def test_speaker_disjoint_and_proportional(cfg):
    out, rep = assign_splits(pool(), cfg)
    log = IssueLog()
    check_leakage(out, log)
    assert log.errors == []
    total = rep["available_hours"]
    for s, frac in (("train", 10 / 15), ("val", 2 / 15), ("test", 3 / 15)):
        assert abs(rep["achieved_hours"][s] - frac * total) < 0.1 * total
    # stratification: code-switch present in every split
    stats = dataset_stats(out)
    for s in ("train", "val", "test"):
        assert "zh_en_switch" in stats[s]["primary_scenario"]
    assert rep["insufficient_data"] is True  # ~1h available vs 15h target: reported, not hidden


def test_deterministic_and_order_independent(cfg):
    p = pool()
    a, _ = assign_splits(p, cfg)
    b, _ = assign_splits(list(reversed(p)), cfg)
    assert split_of(a) == split_of(b)


def test_grouping_merges_sessions_and_augmentations(cfg):
    recs = [
        R("a1", "spkA", sess="shared"), R("b1", "spkB", sess="shared"),  # 2 speakers, 1 session
        R("c1", "spkC"), R("c1_aug", "spkC_aug_label", parent_utterance_id="c1"),
        R("d1", "spkD", sha256="d" * 64), R("e1", "spkE", sha256="d" * 64),  # identical audio
    ]
    groups = build_groups(recs, cfg["split"]["group_by"])
    members = sorted(sorted(r.utterance_id for r in g.records) for g in groups)
    assert members == [["a1", "b1"], ["c1", "c1_aug"], ["d1", "e1"]]


def test_draft_and_rejected_stay_unassigned(cfg):
    p = pool(n_spk=6, per_spk=3)
    p.append(R("draft", "spkX", label_status="machine_draft", reviewer_id=None))
    out, rep = assign_splits(p, cfg)
    assert split_of(out)["draft"] == "unassigned" and rep["n_held_out_unassigned"] == 1


def test_locked_test_is_stable_when_data_grows(cfg):
    first, _ = assign_splits(pool(), cfg)
    lock = make_lock(first)
    locked_ids = set(lock["test_utterance_ids"])
    grown = pool() + [R(f"new{i}", f"newspk{i}", dur=20) for i in range(20)]
    # new utterance from an already-locked test speaker
    tspk = lock["test_speakers"][0]
    grown.append(R("late", tspk))
    out, _ = assign_splits(grown, cfg, lock=lock)
    s = split_of(out)
    assert check_lock_integrity(out, lock) == []
    assert all(s[f"new{i}"] in ("train", "val") for i in range(20))
    assert s["late"] == "test"
    assert {u for u, v in s.items() if v == "test"} == locked_ids | {"late"}


def test_respect_existing_conflict_raises(cfg):
    recs = [R("x1", "spk1", split="train"), R("x2", "spk1", split="test")]
    with pytest.raises(SplitError, match="conflicting"):
        assign_splits(recs, cfg, respect_existing=True)


def test_audit_detects_injected_leakage(repo, cfg):
    out, _ = assign_splits(pool(n_spk=12, per_spk=4), cfg)
    victim = next(r for r in out if r.split == "test")
    leaked = R("leak", victim.speaker_id, split="train", sha256=victim.sha256)
    m = repo / "data/manifests/leaky.jsonl"
    write_manifest([*out, leaked], m)
    assert audit_main([str(m), "--out-dir", str(repo / "audit")]) == 1
    md = (repo / "audit/audit.md").read_text()
    assert "leak_speaker" in md and "leak_audio" in md

    clean = repo / "data/manifests/clean.jsonl"
    write_manifest(out, clean)
    assert audit_main([str(clean), "--out-dir", str(repo / "audit2")]) == 0


def test_split_cli_lock_roundtrip(repo):
    from whisperflow.data.split import main as split_main
    rows = [r.model_dump(mode="json") for r in pool(n_spk=9, per_spk=3)]
    src = write_jsonl(repo / "data/manifests/pool.jsonl", rows)
    out = repo / "data/manifests/pilot.jsonl"
    assert split_main([str(src), "--out", str(out), "--lock-test"]) == 0
    assert (repo / "data/manifests/locked_test_groups.json").exists()
    assert split_main([str(src), "--out", str(out), "--lock-test"]) == 1  # refuses overwrite
    assert split_main([str(src), "--out", str(out)]) == 0  # re-split honours lock
