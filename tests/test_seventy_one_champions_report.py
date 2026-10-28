"""`tools/seventy_one_champions_report.py`: the readings of the network of
the seventy-one decks, on measurements written as
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
    "seventy_one_champions_report", ROOT / "tools/seventy_one_champions_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

PILOTS = ["blackwing", "burn"]


def won(wins: int) -> list[str]:
    return ["win"] * wins + ["loss"] * (DEALS - wins)


def measured(out: Path, *, heads: dict[str, int] | None = None) -> Path:
    """The copy measured as the script measures it: head to head against
    each teacher at `heads` (48 where not named), against all71 in each
    mirror at 44 and then 46, and against the pilots at 70 to all71's 60."""
    heads = heads or {}
    for deck in DECKS:
        pair(out / "heads-taught" / deck, heads.get(deck, 48))
        for point, share in ((4, 44), (8, 46)):
            pair(out / f"taught-{deck}" / str(point), share)
    for who, wins in (("taught", 70), ("all71", 60)):
        played(
            out / "pilots" / f"{who}.json",
            {f"{deck}-vs-{pilot}": won(wins) for deck in DECKS for pilot in PILOTS},
        )
    return out


def test_the_teachers_are_kept_lost_or_not_decided(tmp_path):
    def read(name: str, heads: dict[str, int]) -> dict:
        out = measured(tmp_path / name, heads=heads)
        return tool.report(out, DECKS, tmp_path / "none", tmp_path / "none")["teachers"]

    assert read("kept", dict.fromkeys(DECKS, 50))["verdict"] == "kept"
    # One mirror under the floor is lost, whatever the mean.
    under = read("under", {"cyber": 40})
    assert under["under the floor"] == ["cyber"] and under["verdict"] == "lost"
    level = read("level", {})
    assert level["mean"] == pytest.approx(0.48) and level["verdict"] == "not decided"


def test_each_mirror_is_told_beside_its_teachers_figure_along_the_copy(tmp_path):
    champions, teachers = tmp_path / "champions", tmp_path / "teachers"
    (champions / "blackwing").mkdir(parents=True)
    figure = {"share": 0.6, "error": 0.005, "deals": 2048, "verdict": "a champion"}
    (champions / "blackwing" / "report.json").write_text(json.dumps({"blackwing": figure}))
    # Of the champions' release, as the fourteen decks measured it.
    pair(teachers / "cyber", 55)

    result = tool.report(measured(tmp_path / "out"), DECKS, champions, teachers)

    against = result["against all71"]
    assert against["each"]["blackwing"]["teacher"]["from"] == "its making"
    assert against["each"]["blackwing"]["gap"] == pytest.approx(0.46 - 0.60)
    assert against["each"]["cyber"]["teacher"]["from"] == "the fourteen decks"
    assert against["each"]["cyber"]["gap"] == pytest.approx(0.46 - 0.55)
    assert against["unknown"] == ["lightsworn"]
    assert against["gap"] == pytest.approx((0.46 - 0.60 + 0.46 - 0.55) / 2)
    assert [point["updates"] for point in result["curve"]] == [4, 8]
    assert result["curve"][0]["mean"] == pytest.approx(0.44)
    assert result["pilots"]["mean"] == pytest.approx(0.70)
    assert result["pilots"]["gap"] == pytest.approx(0.10)
    assert result["hours"] is None

    shown = tool.shown(result, DECKS)
    assert "1. The teachers: not decided" in shown
    assert "No teacher's figure: lightsworn" in shown
    assert "The copy took not measured." in shown
