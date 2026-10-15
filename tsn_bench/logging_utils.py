"""Structured JSONL logging, keyed by instance ID, so Phase 5 analysis
(report.py) can be fully scripted from files under results/ instead of
hand-copied numbers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path, mode: str = "a") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, mode, encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
