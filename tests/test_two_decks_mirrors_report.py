"""`tools/two_decks_mirrors_report.py`: the readings of the network of two
decks with its duels against itself in the mirrors only, on measurements
written as `PARTS=measure` of `experiments/commands/two-decks-mirrors.sh`
leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "two_decks_mirrors_report", ROOT / "tools/two_decks_mirrors_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 100
OURS = ("all71-blackwing", "all71-cyber", "cross", "blackwing", "cyber")
OURS += ("both-blackwing", "both-cyber")
BOTH = ("all71-blackwing", "all71-cyber", "cross", "all71-alone")


def played(path: Path, outcomes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {"games": len(outcomes), "outcomes": outcomes}
    path.write_text(json.dumps({"records": [{"evaluation_suite": {"some-vs-checkpoint": case}}]}))


def pair(folder: Path, stem: int, first_wins: int, second_wins: int) -> None:
    """The first network wins `first_wins` deals in the learner's seat, the
    second `second_wins` in it."""
    played(folder / f"{stem}.plays.json", ["win"] * first_wins + ["loss"] * (DEALS - first_wins))
    played(folder / f"{stem}.held.json", ["win"] * second_wins + ["loss"] * (DEALS - second_wins))


def measured(root: Path, folders: tuple[str, ...], shares: dict[str, tuple[int, int]]) -> Path:
    for folder in folders:
        first, second = shares.get(folder, (50, 50))
        for stem in (960, 1200, 1440):
            pair(root / folder, stem, first, second)
    return root


def test_the_cross_games_bring_cost_or_are_not_told_apart(tmp_path: Path) -> None:
    ours = measured(tmp_path / "mirrors", OURS, {"all71-cyber": (70, 30)})
    both = measured(tmp_path / "both", BOTH, {"all71-blackwing": (70, 30)})
    result = tool.report(ours, both)
    blackwing, cyber = result["cross games"]["Blackwing"], result["cross games"]["Cyber"]
    assert (blackwing["both"], blackwing["mirrors"]) == (pytest.approx(0.7), pytest.approx(0.5))
    assert blackwing["gap"] == pytest.approx(0.2)
    assert blackwing["verdict"] == "the cross games bring strength"
    assert cyber["gap"] == pytest.approx(-0.2)
    assert cyber["verdict"] == "the cross games cost strength"
    assert tool.brings(0.02, 0.01) == "not told apart"
    assert tool.brings(0.05, 0.01) == "the cross games bring strength"
    assert tool.brings(-0.05, 0.01) == "the cross games cost strength"


def test_the_champions_head_to_head_as_the_first_readings(tmp_path: Path) -> None:
    ours = measured(tmp_path / "mirrors", OURS, {"cyber": (20, 90)})
    result = tool.report(ours, measured(tmp_path / "both", BOTH, {}))
    assert result["champions"]["Blackwing"]["verdict"] == "matches"
    assert result["champions"]["Cyber"]["share"] == pytest.approx((0.2 + 0.1) / 2)
    assert result["champions"]["Cyber"]["verdict"] == "weaker"
    assert result["champions"]["Cyber"]["champion"] == "cyber-old"


def test_beside_the_champions_against_all71_and_the_cross_matchup(tmp_path: Path) -> None:
    ours = measured(tmp_path / "mirrors", OURS, {"cross": (40, 60), "all71-blackwing": (60, 40)})
    both = measured(tmp_path / "both", BOTH, {"cross": (70, 30), "all71-alone": (50, 50)})
    cyber_alone = measured(tmp_path / "cyber-alone", (".",), {".": (80, 20)})
    result = tool.report(ours, both, cyber_alone)
    blackwing = result["beside"]["Blackwing"]["against all71, over the champion"]
    assert blackwing["gap"] == pytest.approx(0.6 - 0.5)
    cyber = result["beside"]["Cyber"]["against all71, over the champion"]
    assert cyber["gap"] == pytest.approx(0.5 - 0.8)
    cross = result["beside"]["Blackwing against Cyber, against all71"]
    assert cross["paired"]["gap"] == pytest.approx(0.4 - 0.7)
    assert "points over the champion" in tool.shown(result)
    # Without cyber-old's own games, Cyber's line is the head to head alone.
    alone = tool.report(ours, both)
    assert "against all71, over the champion" not in alone["beside"]["Cyber"]
    assert "against all71, over the champion" in alone["beside"]["Blackwing"]


def test_a_folder_never_measured_is_refused(tmp_path: Path) -> None:
    ours = measured(tmp_path / "mirrors", OURS, {})
    both = measured(tmp_path / "both", BOTH, {})
    for path in (ours / "both-cyber").iterdir():
        path.unlink()
    with pytest.raises(SystemExit, match="Nothing is measured"):
        tool.report(ours, both)


def test_the_hours_are_the_runs_own(tmp_path: Path) -> None:
    ours = measured(tmp_path / "mirrors", OURS, {})
    rows = [{"wall_seconds": 3600.0}, {"wall_seconds": 1800.0}]
    (ours / "mirrors.metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = tool.report(ours, measured(tmp_path / "both", BOTH, {}))
    assert result["hours"] == {"mirrors": pytest.approx(1.5)}
