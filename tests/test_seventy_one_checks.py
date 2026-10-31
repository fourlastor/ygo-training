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


def test_the_games_against_all71_are_told_by_how_many_of_the_two_decks_were_met(tmp_path):
    suite = {
        "cases": [
            {
                "name": f"{deck}-vs-each-checkpoint",
                "learner": deck,
                "opponent": ",".join(DECKS),
                "games": 2,
            }
            for deck in DECKS
        ]
    }
    win, loss = "win", "loss"
    games = {
        "blackwing-vs-each-checkpoint": [win, win, win, loss, loss, loss],
        "cyber-vs-each-checkpoint": [loss, loss, win, win, loss, loss],
        "lightsworn-vs-each-checkpoint": [loss, loss, win, win, win, win],
    }
    played(tmp_path / "copy" / "pairings.json", games)
    played(tmp_path / "2880" / "pairings.json", {name: [win] * 6 for name in games})

    result = tool.pairings(tmp_path, suite, {"cyber"})

    assert [name for name in result] == ["copy", "2880"]
    copy = result["copy"]
    assert (copy["mirrors"]["share"], copy["mirrors"]["games"]) == (1.0, 6)
    # Both seats of each pairing: blackwing and lightsworn with cyber, and the
    # two without it.
    assert (copy["one"]["share"], copy["one"]["games"]) == (0.375, 8)
    assert (copy["neither"]["share"], copy["neither"]["games"]) == (0.0, 4)
    assert "both" not in copy
    assert result["2880"]["neither"]["share"] == 1.0
    told = tool.shown_pairings(result)
    assert "The copy: the mirrors 100.0%" in told and "neither met 0.0%" in told
    assert "After 2,880 updates:" in told

    copy = tool.pairings(tmp_path, suite, {"cyber", "lightsworn"})["copy"]
    assert (copy["both"]["share"], copy["both"]["games"]) == (0.5, 4)
    assert (copy["one"]["share"], copy["one"]["games"]) == (0.125, 8)
    assert "neither" not in copy

    played(tmp_path / "copy" / "pairings.json", {name: [win] * 4 for name in games})
    with pytest.raises(SystemExit, match="4 games of 6"):
        tool.pairings(tmp_path, suite, {"cyber"})


def test_a_pairing_is_told_the_same_from_either_seat():
    for one, other in (
        ("blackwing", "cyber"),
        ("cyber", "lightsworn"),
        ("blackwing", "lightsworn"),
    ):
        for met in ({"cyber"}, {"cyber", "lightsworn"}, set()):
            assert tool.met_of(one, other, met) == tool.met_of(other, one, met)
    assert tool.met_of("cyber", "cyber", set()) == "mirrors"


def test_only_the_copy_goes_on(tmp_path):
    result = tool.read(checks(tmp_path), DECKS)

    assert result["checks"] == {} and not result["stop"]
    assert result["copy"]["mirrors"] == pytest.approx(0.5)


def test_the_mirrors_are_told_along_each_run_against_its_copy(tmp_path):
    directory = tmp_path / "ours"
    checks(directory / "checks", (2880, 44, 52))
    for point, mirrors in ((640, 46), (160, 49)):
        for deck in DECKS:
            pair(directory / "along" / "ppo" / str(point) / deck, mirrors)
    fourteen = tmp_path / "fourteen"
    two = DECKS[:2]
    for deck in two:
        pair(fourteen / "heads-taught" / deck, 50)
        for arm in tool.ARMS:
            pair(fourteen / f"heads-{arm}" / deck, 51)
            pair(directory / "along" / arm / "640" / deck, 45)

    runs = tool.runs_along(directory, fourteen, DECKS, two)

    assert list(runs["PPO"]) == [160, 640, 2880]
    assert runs["PPO"][640]["gap"] == pytest.approx(-0.04)
    assert runs["PPO, the fourteen decks"][2880]["gap"] == pytest.approx(-0.06)
    assert list(runs["quarter"]) == [640, tool.ARM_UPDATES]
    assert runs["quarter"][640]["gap"] == pytest.approx(-0.05)
    assert runs["quarter"][tool.ARM_UPDATES]["gap"] == pytest.approx(0.01)
    assert "    After 640 updates: 45.0% (-5.0 ± " in tool.shown_along(runs)


@pytest.mark.parametrize(
    ("mirrors", "told"), [(60, "raised"), (50, "not raised"), (52, "not told")]
)
def test_the_mean_is_read_against_the_check_it_is_of(tmp_path, mirrors, told):
    directory = checks(tmp_path, (5760, 50, 50))
    checked(directory / "mean", mirrors, 50)

    found = tool.mean_of_weights(directory, 5760, DECKS)

    assert found["mirrors"]["gap"] == pytest.approx((mirrors - 50) / 100)
    assert found["told"] == told
    assert tool.shown_mean(found, 5760).endswith(f"The mirrors: {told}.")


def test_a_mean_that_broke_the_network_says_nothing_of_the_mirrors(tmp_path):
    directory = checks(tmp_path, (5760, 50, 50))
    checked(directory / "mean", 60, 40)

    found = tool.mean_of_weights(directory, 5760, DECKS)

    assert found["pairings"]["under"]
    assert tool.shown_mean(found, 5760).endswith("the mean has broken the network.")


def test_the_mean_is_not_taken_for_a_check(tmp_path):
    directory = checks(tmp_path, (2880, 50, 50))
    checked(directory / "mean", 60, 60)
    checked(directory / "mean-160-1360", 60, 60)

    assert tool.checks_of(directory) == [2880]


@pytest.mark.parametrize(("last", "rising"), [(60, True), (50, False), (40, False)])
def test_the_means_are_rising_when_the_last_is_over_the_second_by_two_errors(
    tmp_path, last, rising
):
    directory = checks(tmp_path)
    checked(directory / "mean", 50, 50)
    windows = ["160-1360", "1600-2800", "3040-4240", "4480-5680"]
    # The first over them all: it is left out of the rule.
    for window, mirrors in zip(windows, (70, 50, 55, last), strict=True):
        checked(directory / f"mean-{window}", mirrors, 52)

    result = tool.means(directory, windows, DECKS)

    assert list(result["along"]) == windows
    assert result["along"]["160-1360"]["mirrors"]["gap"] == pytest.approx(0.20)
    assert result["rise"]["mirrors"]["gap"] == pytest.approx((last - 50) / 100)
    assert result["rising"] == rising
    assert result["against_the_mean"]["mirrors"]["gap"] == pytest.approx((last - 50) / 100)
    told = tool.shown_means(result)
    assert "  Updates 1600-2800: the mirrors 50.0% (+0.0 ± " in told
    assert told.endswith(f"{'rising' if rising else 'not rising'}.")
