"""`tools/seventy_one_decks_report.py`: the readings of seventy-one decks, on
measurements written as `PARTS=measure` of
`experiments/commands/seventy-one-decks.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "seventy_one_decks_report", ROOT / "tools/seventy_one_decks_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

THIRTEEN = ["blackwing", "monarch"]
# Points of PPO 3, 6 and 9, a checkpoint every update; copies of 2 updates a
# deck: the network of four decks, half of the two, all of three.
SHAPE = {"copied": 8, "half_copied": 4, "all_copied": 6, "control_point": 3, "step": 1}
POINTS = [3, 6, 9]


def cases(deck: str, won: int, games: int = 100) -> dict:
    """A deck's games against blackwing and monarch, the first so many won."""
    told = ["win"] * won + ["loss"] * (games - won)
    half = games // 2
    return {
        f"{deck}-vs-blackwing": {"games": half, "outcomes": told[:half]},
        f"{deck}-vs-monarch": {"games": half, "outcomes": told[half:]},
    }


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": 2012, "records": records}))


def checkpoints(run: str, cases_: dict) -> list[dict]:
    """A run's three checkpoints at its point, each winning the same games."""
    before, point = tool.runs_of(POINTS, 8, 4, 6, 3)[run]
    return [
        {"checkpoint_step": before + at, "evaluation_suite": cases_}
        for at in (point - 2, point - 1, point)
    ]


SCREENS = {"all71-3": 40, "all71-6": 50, "all71-9": 60, "half": 60, "all": 58}
OLD = {"all71-3": 30, "all71-6": 40, "all71-9": 55, "half": 35, "all": 50}
NEW = {"all71-3": 30, "all71-6": 40, "all71-9": 60, "all": 20}


def measurements(folder: Path) -> None:
    for run, won in SCREENS.items():
        for deck in THIRTEEN:
            write(folder / "screens" / f"{run}.{deck}.json", checkpoints(run, cases(deck, won)))
    for kind, deck, wins, copy, pilot in (
        ("old", "arcana", OLD, 20, 45),
        ("new", "agents", NEW, 35, 45),
    ):
        for run, won in wins.items():
            write(folder / kind / f"{run}.json", checkpoints(run, cases(deck, won)))
        write(
            folder / kind / "all71-copy.json",
            [{"checkpoint_step": 8, "evaluation_suite": cases(deck, copy)}],
        )
        write(
            folder / kind / "pilot.json",
            [{"checkpoint_label": "pilot", "evaluation_suite": cases(deck, pilot)}],
        )


def test_the_readings_at_their_points(tmp_path):
    measurements(tmp_path)
    result = tool.report(tmp_path, THIRTEEN, POINTS, **SHAPE)
    readings = result["readings"]
    assert readings["thirteen"]["gap"] == pytest.approx(0.0)
    assert readings["thirteen"]["verdict"] == "equivalent"
    assert readings["thirteen"]["games"] == 200
    # At R2 against all on the eighteen: 40 against 50.
    assert readings["eighteen"]["gap"] == pytest.approx(-0.10)
    assert readings["eighteen"]["verdict"] == "worse"
    # At R3 against the forty's pilots: 60 against 45.
    assert readings["forty"]["gap"] == pytest.approx(0.15)
    assert readings["forty"]["verdict"] == "over"
    beside = result["beside"]
    assert beside["the screens at all71-3 against half"]["gap"] == pytest.approx(-0.20)
    assert beside["the forty at all71-9 against all, which never trained on them"][
        "gap"
    ] == pytest.approx(0.40)
    assert beside["the copy on the forty against their pilots"]["gap"] == pytest.approx(-0.10)
    assert result["by_deck"]["agents"]["kind"] == "new"
    assert result["by_deck"]["agents"]["all71-9"] == pytest.approx(0.60)
    assert result["by_deck"]["arcana"]["pilot"] == pytest.approx(0.45)
    shown = tool.show(result)
    assert "1. the thirteen screens, all71-9 against half" in shown and "equivalent" in shown


def test_a_point_not_reached_leaves_its_reading_unmeasured(tmp_path):
    measurements(tmp_path)
    for path in tmp_path.rglob("all71-9*.json"):
        path.unlink()
    result = tool.report(tmp_path, THIRTEEN, POINTS, **SHAPE)
    assert "thirteen" not in result["readings"] and "forty" not in result["readings"]
    assert result["readings"]["eighteen"]["verdict"] == "worse"
    assert "not measured" in tool.show(result)
