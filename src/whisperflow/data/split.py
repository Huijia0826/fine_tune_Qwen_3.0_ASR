"""Leakage-safe, stratified, deterministic train/val/test assignment.

Algorithm
---------
1. **Grouping (union-find).** Records sharing any value of the configured keys
   (speaker_id, session_id, parent_utterance_id, sha256, optionally normalized transcript)
   are merged into one connected group. A group is the atomic unit of assignment, so a
   speaker, a multi-speaker session, or an augmented variant can never straddle splits.
2. **Test lock.** If a lock file exists, every group touching a locked speaker/session is
   forced into test, and no *new* groups enter test unless explicitly allowed. Once the
   locked test set is frozen, adding data can only grow train/val.
3. **Stratified greedy assignment.** Remaining groups are sorted largest-first (with a
   seeded hash as tie-break) and each is placed in the split whose per-scenario hour
   deficit it reduces most. Targets come from configs/dataset.yaml and are scaled
   proportionally when less data exists than the targets require.

Only records with label_status in {source_provided, human_verified} are assigned;
machine_draft and rejected records stay `unassigned`.

Usage:
    wf-split data/manifests/pool.jsonl --out data/manifests/pilot.jsonl \
        [--respect-existing] [--lock-test] [--allow-new-test-groups]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from whisperflow.data.common import load_yaml
from whisperflow.data.schema import Utterance, load_manifest, write_manifest
from whisperflow.text.normalize import normalize

SPLITS = ("train", "val", "test")
ASSIGNABLE_STATUS = {"source_provided", "human_verified"}


class SplitError(Exception):
    pass


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root choice
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass
class Group:
    gid: str
    records: list[Utterance] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return sum(r.duration_sec for r in self.records) / 3600

    def hours_by_scenario(self) -> dict[str, float]:
        d: dict[str, float] = defaultdict(float)
        for r in self.records:
            d[r.primary_scenario] += r.duration_sec / 3600
        return d

    @property
    def speakers(self) -> set[str]:
        return {r.speaker_id for r in self.records}

    @property
    def sessions(self) -> set[str]:
        return {r.session_id for r in self.records}


def build_groups(records: list[Utterance], group_by: list[str],
                 group_identical_transcripts: bool = False) -> list[Group]:
    uf = _UnionFind()
    for r in records:
        node = f"utt={r.utterance_id}"
        uf.find(node)
        for key in group_by:
            if key == "parent_utterance_id":
                if r.parent_utterance_id:
                    uf.union(node, f"utt={r.parent_utterance_id}")
                continue
            val = getattr(r, key)
            if val:
                uf.union(node, f"{key}={val}")
        if group_identical_transcripts:
            uf.union(node, f"text={normalize(r.raw_transcript)}")

    groups: dict[str, Group] = {}
    for r in records:
        root = uf.find(f"utt={r.utterance_id}")
        groups.setdefault(root, Group(gid=root)).records.append(r)
    return sorted(groups.values(), key=lambda g: g.gid)


def _seeded_key(seed: int, gid: str) -> str:
    return hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()


def load_lock(path: Path) -> dict | None:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def assign_splits(records: list[Utterance], cfg: dict, *, lock: dict | None = None,
                  respect_existing: bool = False,
                  allow_new_test_groups: bool = False) -> tuple[list[Utterance], dict]:
    scfg = cfg["split"]
    seed = int(scfg["seed"])
    targets = {s: float(scfg["target_hours"][s]) for s in SPLITS}

    eligible = [r for r in records if r.label_status in ASSIGNABLE_STATUS]
    held_out = [r for r in records if r.label_status not in ASSIGNABLE_STATUS]
    groups = build_groups(eligible, scfg["group_by"], scfg.get("group_identical_transcripts", False))

    assignment: dict[str, str] = {}
    reasons: dict[str, str] = {}

    # (a) pre-assigned splits from the source corpus (e.g. official speaker-disjoint splits)
    if respect_existing:
        for g in groups:
            pre = {r.split for r in g.records} - {"unassigned"}
            if len(pre) > 1:
                raise SplitError(f"group {g.gid} has conflicting pre-assigned splits {sorted(pre)}; "
                                 "the source's official split leaks under our grouping keys")
            if pre:
                assignment[g.gid] = pre.pop()
                reasons[g.gid] = "pre-assigned"

    # (b) locked test set
    if lock:
        lk_spk, lk_ses = set(lock["test_speakers"]), set(lock["test_sessions"])
        for g in groups:
            if g.speakers & lk_spk or g.sessions & lk_ses:
                if assignment.get(g.gid, "test") != "test":
                    raise SplitError(f"group {g.gid} contains locked-test speakers/sessions but "
                                     f"was pre-assigned to {assignment[g.gid]!r}")
                assignment[g.gid] = "test"
                reasons[g.gid] = "locked-test"

    # (c) stratified greedy for the rest
    free = [g for g in groups if g.gid not in assignment]
    total_by_sc: dict[str, float] = defaultdict(float)
    for g in groups:
        for sc, h in g.hours_by_scenario().items():
            total_by_sc[sc] += h
    total_hours = sum(total_by_sc.values())
    candidate_splits = list(SPLITS) if (not lock or allow_new_test_groups) else ["train", "val"]
    tsum = sum(targets[s] for s in SPLITS)
    frac = {s: targets[s] / tsum for s in SPLITS}
    # per (split, scenario) target hours, scaled to available data
    tgt = {s: {sc: frac[s] * h for sc, h in total_by_sc.items()} for s in SPLITS}
    cur: dict[str, dict[str, float]] = {s: defaultdict(float) for s in SPLITS}
    for g in groups:
        if g.gid in assignment:
            for sc, h in g.hours_by_scenario().items():
                cur[assignment[g.gid]][sc] += h

    free.sort(key=lambda g: (-round(g.hours, 6), _seeded_key(seed, g.gid)))
    for g in free:
        ghs = g.hours_by_scenario()
        best, best_gain = None, None
        for s in candidate_splits:
            # reduction in squared deviation from target if g goes to split s
            gain = sum((tgt[s][sc] - cur[s][sc]) ** 2 - (tgt[s][sc] - cur[s][sc] - h) ** 2
                       for sc, h in ghs.items())
            if best_gain is None or gain > best_gain + 1e-12:
                best, best_gain = s, gain
        assignment[g.gid] = best
        reasons[g.gid] = "greedy"
        for sc, h in ghs.items():
            cur[best][sc] += h

    # materialize
    out: list[Utterance] = []
    gid_of = {r.utterance_id: g.gid for g in groups for r in g.records}
    for r in records:
        if r.label_status in ASSIGNABLE_STATUS:
            out.append(r.model_copy(update={"split": assignment[gid_of[r.utterance_id]]}))
        else:
            out.append(r.model_copy(update={"split": "unassigned"}))

    achieved = {s: sum(cur[s].values()) for s in SPLITS}
    report = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_records": len(records),
        "n_held_out_unassigned": len(held_out),
        "n_groups": len(groups),
        "available_hours": round(total_hours, 4),
        "target_hours_config": targets,
        "target_hours_scaled": {s: round(frac[s] * total_hours, 4) for s in SPLITS},
        "achieved_hours": {s: round(achieved[s], 4) for s in SPLITS},
        "groups_per_split": {s: sum(1 for v in assignment.values() if v == s) for s in SPLITS},
        "assignment_reasons": dict(sorted(
            {k: sum(1 for v in reasons.values() if v == k) for k in set(reasons.values())}.items())),
        "insufficient_data": total_hours < tsum,
    }
    return out, report


def make_lock(records: list[Utterance]) -> dict:
    test = [r for r in records if r.split == "test"]
    ids = sorted(r.utterance_id for r in test)
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_speakers": sorted({r.speaker_id for r in test}),
        "test_sessions": sorted({r.session_id for r in test}),
        "test_utterance_ids": ids,
        "test_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def check_lock_integrity(records: list[Utterance], lock: dict) -> list[str]:
    """Locked test utterances must remain in test (they may not be removed or moved)."""
    split_of = {r.utterance_id: r.split for r in records}
    problems = []
    for uid in lock["test_utterance_ids"]:
        if uid not in split_of:
            problems.append(f"locked test utterance {uid} missing from manifest")
        elif split_of[uid] != "test":
            problems.append(f"locked test utterance {uid} now in {split_of[uid]!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--report", help="JSON split report path (default: <out>.split_report.json)")
    ap.add_argument("--respect-existing", action="store_true",
                    help="keep non-'unassigned' splits already present in the manifest")
    ap.add_argument("--lock-test", action="store_true",
                    help="write the lock file from the resulting test split (refuses to overwrite)")
    ap.add_argument("--allow-new-test-groups", action="store_true",
                    help="with an existing lock, allow new speakers into test (deliberate growth)")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    lock_path = Path(cfg["split"]["locked_test_file"])
    lock = load_lock(lock_path)
    records = load_manifest(args.manifest)
    try:
        out, report = assign_splits(records, cfg, lock=lock, respect_existing=args.respect_existing,
                                    allow_new_test_groups=args.allow_new_test_groups)
    except SplitError as e:
        print(f"ERROR: {e}")
        return 1
    if lock:
        problems = check_lock_integrity(out, lock)
        if problems:
            print("ERROR: locked test set would change:\n  " + "\n  ".join(problems))
            return 1
    write_manifest(out, args.out)
    report_path = Path(args.report or f"{args.out}.split_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.lock_test:
        if lock:
            print(f"ERROR: lock {lock_path} already exists; refusing to overwrite")
            return 1
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(make_lock(out), indent=2), encoding="utf-8")
        print(f"Locked test set written to {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
