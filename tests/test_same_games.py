"""`tools/same_games.py`: two measurements of a suite held to each other,
game for game."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def measured(path: Path, outcomes: list[str]) -> Path:
    suite = {"mirror": {"games": len(outcomes), "outcomes": outcomes}}
    path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}))
    return path


def told(first: Path, second: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/same_games.py"), str(first), str(second)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_same_games_are_told_so_and_others_are_refused(tmp_path):
    one = measured(tmp_path / "one.json", ["win", "loss", "draw"])
    same = told(one, measured(tmp_path / "same.json", ["win", "loss", "draw"]))
    assert same.returncode == 0 and "the same 3 games" in same.stdout
    other = told(one, measured(tmp_path / "other.json", ["win", "draw", "loss"]))
    assert other.returncode != 0 and "are other games" in other.stderr
