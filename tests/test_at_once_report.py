"""`tools/at_once_report.py`: what measurements played so many at a time
come to, on logs written as `experiments/commands/fourteen-decks.sh` leaves
them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("at_once_report", ROOT / "tools/at_once_report.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def measured(path: Path, games: int, turns: float) -> str:
    """A measurement of two cases of a suite, half of the games each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    told = {"games": games // 2, "mean_turns": turns}
    suite = {"one": told, "other": told}
    path.write_text(json.dumps({"records": [{"evaluation": told, "evaluation_suite": suite}]}))
    return str(path)


def test_each_count_at_a_time_is_told_beside_one_alone(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    groups = [
        {
            "part": "mirrors",
            "at_once": 3,
            "from": 100,
            "to": 200,
            "files": [measured(tmp_path / "a.json", 30, 10)],
        },
        {
            "part": "alone",
            "at_once": 1,
            "from": 200,
            "to": 300,
            "files": [measured(tmp_path / "b.json", 10, 10)],
        },
        {
            "part": "mirrors",
            "at_once": 3,
            "from": 300,
            "to": 400,
            "files": [measured(tmp_path / "c.json", 30, 10)],
        },
        {
            "part": "rest",
            "at_once": 3,
            "from": 400,
            "to": 500,
            "files": [measured(tmp_path / "d.json", 50, 10)],
        },
    ]
    (logs / "levels.jsonl").write_text("".join(json.dumps(group) + "\n" for group in groups))
    # The processor half busy from 100 to 200, the card at 40% then 60%.
    (logs / "load.txt").write_text(
        "100 40 1000 10 0 0 10 0 0 0 0 0 0\n"
        "200 60 1500 60 0 0 60 0 0 0 0 0 0\n"
        "250 - - 60 0 0 160 0 0 0 0 0 0\n"
    )
    found = tool.report(tmp_path)
    assert [(row["part"], row["at once"]) for row in found] == [
        ("alone", 1),
        ("mirrors", 3),
        ("rest", 3),
    ]
    alone, mirrors, rest = found
    assert mirrors["games"] == 60 and mirrors["seconds"] == 200
    assert mirrors["turns a second"] == pytest.approx(3.0)
    assert mirrors["times alone"] == pytest.approx(3.0)
    assert alone["times alone"] == pytest.approx(1.0)
    assert rest["games"] == 50 and rest["times alone"] is None
    assert mirrors["card"] == pytest.approx(50.0) and mirrors["memory"] == 1500
    assert mirrors["processor"] == pytest.approx(0.5)
    assert alone["processor"] == pytest.approx(0.0)
    assert "times alone" in tool.shown(found)
