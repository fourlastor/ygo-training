"""`tools/first_batch_report.py`: the readings of the first batch test."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "first_batch_report", ROOT / "tools/first_batch_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

POINTS = [540, 1080, 2160, 4320]


def written(path: Path, wins: dict[str, float], games: int = 400) -> None:
    """A file of cases that each win a share of their games, spread evenly."""
    suite = {}
    for name, share in wins.items():
        won = round(share * games)
        outcomes = ["win"] * won + ["loss"] * (games - won)
        suite[name] = {"outcomes": outcomes}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}), encoding="utf8")


def run(tmp_path: Path, gains: list[float], old: float) -> dict:
    written(tmp_path / "0/pairings.json", {"a-vs-each": 0.2, "b-vs-each": 0.3})
    for point, gain in zip(POINTS, gains, strict=True):
        wins = {"a-vs-each": 0.2 + gain, "b-vs-each": 0.3 + gain}
        written(tmp_path / f"{point}/pairings.json", wins)
        written(tmp_path / f"{point}/old.json", {"old-vs-each": old}, games=4000)
    return tool.report(tmp_path, POINTS)


def test_a_deck_that_gains_learns_and_the_budget_is_where_the_gain_flattens(tmp_path):
    found = run(tmp_path, [0.04, 0.09, 0.10, 0.10], 0.52)
    assert found["learns"] == "learns"
    assert found["gains"][2160]["gap"] == pytest.approx(0.10)
    assert found["each"]["a-vs-each"][1080] == pytest.approx(0.09)
    # Identical games are only so far apart: 1,080 is 1 point under the last.
    assert found["budget"] == 1080
    assert found["holds"] == "hold"


def test_decks_that_do_not_gain_do_not_learn(tmp_path):
    assert run(tmp_path, [0.0, 0.0, 0.01, 0.01], 0.50)["learns"] == "does not"


def test_decks_still_rising_set_no_budget_and_old_decks_can_slip(tmp_path):
    found = run(tmp_path, [0.02, 0.05, 0.10, 0.20], 0.47)
    assert found["budget"] is None
    assert found["holds"] == "slip"


@pytest.mark.parametrize(
    ("gain", "error", "told"),
    [(0.08, 0.01, "learns"), (0.06, 0.01, "not decided"), (0.0, 0.01, "does not")],
)
def test_reading_one_asks_two_standard_errors(gain, error, told):
    assert tool.learns(gain, error) == told


@pytest.mark.parametrize(
    ("share", "error", "told"),
    [(0.50, 0.003, "hold"), (0.49, 0.003, "not decided"), (0.48, 0.003, "slip")],
)
def test_reading_three_asks_two_standard_errors(share, error, told):
    assert tool.holds(share, error) == told


def test_what_is_told_beside_is_told_where_it_was_measured(tmp_path):
    found = run(tmp_path, [0.04, 0.09, 0.10, 0.10], 0.50)
    assert "held out" not in found
    written(tmp_path / "0/held-out.json", {"c-vs-each": 0.2})
    written(tmp_path / "4320/held-out.json", {"c-vs-each": 0.21})
    written(tmp_path / "0/pilots.json", {"a-vs-pilots": 0.2})
    written(tmp_path / "4320/pilots.json", {"a-vs-pilots": 0.3})
    found = tool.report(tmp_path, POINTS)
    assert found["held out"]["gap"] == pytest.approx(0.01)
    assert found["pilots"]["gap"] == pytest.approx(0.10)
    shown = tool.shown(found, POINTS)
    assert "1. A new deck learns" in shown
    assert "The next twelve decks" in shown
