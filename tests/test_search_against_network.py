"""`tools/search_against_network.py`: the reading of the staged search
against the network, its score by deals, and the search's seat of each arm."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "search_against_network", ROOT / "tools/search_against_network.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


@pytest.mark.parametrize(
    ("rate", "error", "reading"),
    [
        (0.60, 0.02, "the network is stronger"),
        (0.55, 0.02, "not told apart"),
        (0.40, 0.03, "the network is weaker"),
        (0.45, 0.02, "not told apart"),
        (0.50, 0.01, "not told apart"),
    ],
)
def test_the_reading(rate: float, error: float, reading: str) -> None:
    assert tool.reading(rate, error) == reading


def test_a_deals_two_games_count_together() -> None:
    # Three deals: both won, one of two, none; a draw is not a win.
    games = [
        {"deal": 0, "result": "win"},
        {"deal": 0, "result": "win"},
        {"deal": 1, "result": "win"},
        {"deal": 1, "result": "draw"},
        {"deal": 2, "result": "loss"},
        {"deal": 2, "result": "loss"},
    ]
    rate, error, deals = tool.scored(games)
    assert deals == 3
    assert rate == pytest.approx(0.5)
    assert error == pytest.approx(0.5 / 3**0.5)


def test_each_arms_seat_for_the_search() -> None:
    assert tool.search_seat("generic", "blackwing", "monarch") == {
        "control": "search",
        "alone": True,
        "policy": "generic",
    }
    assert tool.search_seat("specialist", "blackwing", "monarch") == {
        "control": "search",
        "alone": True,
        "policy": "blackwing",
        "model": "monarch",
    }
    with pytest.raises(SystemExit):
        tool.search_seat("oracle", "blackwing", "monarch")


def _found(results: list[str]) -> dict:
    """A play's file: deals of two games, the network's results in order."""
    games = [
        {"deal": index // 2, "seed": index // 2, "decks": ["cyber", "zombie"], "result": result}
        for index, result in enumerate(results)
    ]
    return {"arm": "specialist", "games": games, "seconds": 1.0, "threads": 1}


def test_two_searches_are_compared_deal_by_deal():
    # The network wins every game against the search as pinned, half against
    # the improved one: 50 points less.
    before = _found(["win"] * 40)
    after = _found(["win", "loss"] * 20)
    result = tool.compared(before, after)
    assert result["deals"] == 20
    assert result["drop"] == pytest.approx(0.5)
    assert result["reading"] == "the improved pilot makes the search stronger"
    assert tool.compared(before, before)["reading"] == "not shown"


def test_two_searches_are_compared_on_the_same_deals_alone():
    other = _found(["win"] * 40)
    other["games"][0]["seed"] = 99
    with pytest.raises(SystemExit, match="other deals"):
        tool.compared(_found(["win"] * 40), other)
