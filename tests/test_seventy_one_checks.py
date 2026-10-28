"""`tools/seventy_one_checks.py`: the checks of the seventy-one decks' PPO
as it trains, on measurements written as
`experiments/commands/seventy-one-champions.sh` leaves them (three decks
here, for brevity)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.test_four_decks_report import DEALS, DECKS, pair, played

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "seventy_one_checks", ROOT / "tools/seventy_one_checks.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def checked(folder: Path, mirrors: int, pairings: int) -> None:
    """A check: head to head against each teacher at `mirrors`, against all71
    at `pairings`, of a hundred."""
    for deck in DECKS:
        pair(folder / deck, mirrors)
    played(
        folder / "pairings.json",
        {
            f"{deck}-vs-{other}-checkpoint": ["win"] * pairings + ["loss"] * (DEALS - pairings)
            for deck, other in zip(DECKS, DECKS[1:] + DECKS[:1], strict=True)
        },
    )


def checks(directory: Path, *made: tuple[int, int, int]) -> Path:
    """The copy at 50 and 50, then each check at its updates."""
    checked(directory / "copy", 50, 50)
    for point, mirrors, pairings in made:
        checked(directory / str(point), mirrors, pairings)
    return directory


def test_the_suites_are_each_mirror_on_so_many_deals(tmp_path):
    for deck in DECKS:
        case = {"name": f"{deck}-vs-checkpoint", "learner": deck, "opponent": deck, "games": 2048}
        (tmp_path / f"{deck}-against-checkpoint.json").write_text(
            json.dumps({"name": f"{deck}-against-checkpoint", "cases": [case]})
        )

    tool.suites(DECKS, tmp_path, 128, tmp_path / "suites")

    made = json.loads((tmp_path / "suites" / "cyber.json").read_text())
    assert made["name"] == "cyber-check"
    assert [case["games"] for case in made["cases"]] == [128]


def test_two_checks_in_a_row_under_the_copy_stop_it(tmp_path):
    result = tool.read(checks(tmp_path, (2880, 50, 52), (5760, 40, 52), (8640, 40, 52)), DECKS)

    assert result["checks"][2880]["mirrors"]["gap"] == pytest.approx(0.0)
    assert result["checks"][5760]["mirrors"]["gap"] == pytest.approx(-0.10)
    assert result["checks"][5760]["mirrors"]["under"]
    assert not result["checks"][5760]["pairings"]["under"]
    assert result["met"] == [{"after": 8640, "by": "mirrors"}]
    assert result["stop"]
    assert tool.shown(result).endswith("STOP.")


def test_one_check_under_or_the_two_in_turn_do_not_stop_it(tmp_path):
    # The mirrors under, then the pairings, then the mirrors again: never the
    # same twice in a row.
    result = tool.read(checks(tmp_path, (2880, 40, 50), (5760, 50, 40), (8640, 40, 50)), DECKS)

    assert [found["mirrors"]["under"] for found in result["checks"].values()] == [True, False, True]
    assert result["checks"][5760]["pairings"]["under"]
    assert result["met"] == [] and not result["stop"]
    assert tool.shown(result).endswith("Go on.")


def test_the_pairings_stop_it_and_a_stop_once_passed_is_told(tmp_path):
    result = tool.read(checks(tmp_path, (2880, 50, 40), (5760, 50, 40), (8640, 50, 50)), DECKS)

    # Passed by a run told to go on: the rule met at the second check, not at the last.
    assert result["met"] == [{"after": 5760, "by": "pairings"}]
    assert not result["stop"]
    assert "The rule met after 5,760 updates, by the pairings" in tool.shown(result)


def test_only_the_copy_goes_on(tmp_path):
    result = tool.read(checks(tmp_path), DECKS)

    assert result["checks"] == {} and not result["stop"]
    assert result["copy"]["mirrors"] == pytest.approx(0.5)
