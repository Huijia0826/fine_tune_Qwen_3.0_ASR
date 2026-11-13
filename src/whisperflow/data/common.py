"""Shared helpers for data-pipeline CLIs: config loading and issue reporting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Issue:
    level: str  # "error" | "warning"
    code: str  # stable machine-readable code, e.g. "audio_missing"
    message: str
    utterance_id: str | None = None
    line: int | None = None


class IssueLog:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def error(self, code: str, msg: str, uid: str | None = None, line: int | None = None):
        self.issues.append(Issue("error", code, msg, uid, line))

    def warn(self, code: str, msg: str, uid: str | None = None, line: int | None = None):
        self.issues.append(Issue("warning", code, msg, uid, line))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def summary(self) -> dict:
        return {
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "by_code": dict(Counter(f"{i.level}:{i.code}" for i in self.issues)),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "issues": [asdict(i) for i in self.issues]},
                      f, ensure_ascii=False, indent=2)

    def print(self, max_lines: int = 50) -> None:
        for i in self.issues[:max_lines]:
            where = f"line {i.line}" if i.line else ""
            uid = i.utterance_id or ""
            print(f"[{i.level.upper():7}] {i.code:28} {uid:16} {where:9} {i.message}")
        if len(self.issues) > max_lines:
            print(f"... {len(self.issues) - max_lines} more (see JSON report)")
        s = self.summary()
        print(f"\n{s['n_errors']} error(s), {s['n_warnings']} warning(s)")
