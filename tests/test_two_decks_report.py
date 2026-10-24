"""`tools/two_decks_report.py`: the readings of a network of two decks, on
measurements written as `PARTS=measure` of `experiments/commands/two-
decks.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "two_decks_report", ROOT / "tools/two_decks_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 100
FOLDERS = (
    "blackwing",
    "cyber",
    "all71-blackwing",
    "all71-cyber",
    "all71-alone",
    "decks-01",
    "cross",
)


def played(path: Path, outcomes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {"games": len(outcomes), "outcomes": outcomes}
    record = {"evaluation_suite": {"some-vs-checkpoint": case}}
    path.write_text(json.dumps({"records": [record]}))


def pair(folder: Path, stem: int, first_wins: int, second_wins: int, deals: int = DEALS) -> None:
    """The first network wins `first_wins` deals in the learner's seat, the
    second `second_wins` in it; the first wins what the second loses."""
    played(folder / f"{stem}.plays.json", ["win"] * first_wins + ["loss"] * (deals - first_wins))
    played(folder / f"{stem}.held.json", ["win"] * second_wins + ["loss"] * (deals - second_wins))


def measured(tmp_path: Path, shares: dict[str, tuple[int, int]]) -> Path:
    for folder in FOLDERS:
        first, second = shares.get(folder, (50, 50))
        for stem in (1920, 2400, 2880):
            pair(tmp_path / folder, stem, first, second)
    return tmp_path


def test_a_network_level_with_both_champions_matches(tmp_path: Path) -> None:
    result = tool.report(measured(tmp_path, {}))
    for deck in ("Blackwing", "Cyber"):
        assert result["readings"][deck]["share"] == pytest.approx(0.5)
        assert result["readings"][deck]["verdict"] == "matches"


def test_the_first_wins_what_the_second_loses(tmp_path: Path) -> None:
    # 70 of 100 won in the learner's seat, 40 lost by the champion in it.
    result = tool.report(measured(tmp_path, {"blackwing": (70, 60), "cyber": (20, 90)}))
    assert result["readings"]["Blackwing"]["share"] == pytest.approx((0.7 + 0.4) / 2)
    assert result["readings"]["Cyber"]["share"] == pytest.approx((0.2 + 0.1) / 2)
    assert result["readings"]["Cyber"]["verdict"] == "weaker"


def test_a_wide_share_near_the_bar_is_not_decided() -> None:
    assert tool.verdict(0.49, 0.01) == "not decided"
    assert tool.verdict(0.50, 0.01) == "matches"
    assert tool.verdict(0.45, 0.01) == "weaker"


def test_beside_the_networks_against_all71_are_paired_deal_by_deal(tmp_path: Path) -> None:
    directory = measured(tmp_path, {"all71-blackwing": (60, 50), "all71-alone": (70, 40)})
    cyber_alone = tmp_path / "cyber-alone"
    for stem in (480, 600, 720):
        pair(cyber_alone, stem, 80, 20)
    result = tool.report(directory, cyber_alone=cyber_alone)
    blackwing = result["beside"]["Blackwing mirror against all71"]
    assert blackwing["paired"]["gap"] == pytest.approx(0.55 - 0.65)
    cyber = result["beside"]["Cyber mirror against all71"]
    assert cyber["alone"]["share"] == pytest.approx(0.8)
    assert cyber["paired"]["gap"] == pytest.approx(0.5 - 0.8)
    assert "cyber-new against cyber-old" not in result["beside"]
    assert "the deals paired" in tool.shown(result)


def test_files_of_other_deals_are_refused(tmp_path: Path) -> None:
    directory = measured(tmp_path, {})
    played(directory / "cross" / "1920.held.json", ["win"] * 10)
    with pytest.raises(SystemExit, match="other deals"):
        tool.report(directory)


def test_a_folder_never_measured_is_refused(tmp_path: Path) -> None:
    directory = measured(tmp_path, {})
    for path in (directory / "decks-01").iterdir():
        path.unlink()
    with pytest.raises(SystemExit, match="Nothing is measured"):
        tool.report(directory)


def test_the_hours_are_the_runs_own(tmp_path: Path) -> None:
    directory = measured(tmp_path, {})
    rows = [{"wall_seconds": 1800.0}, {"wall_seconds": 1800.0}]
    (directory / "both.metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = tool.report(directory)
    assert result["hours"] == {"both": pytest.approx(1.0)}
