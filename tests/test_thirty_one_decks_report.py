"""`tools/thirty_one_decks_report.py`: the readings of thirty-one decks, on
measurements written as `PARTS=measure` of
`experiments/commands/thirty-one-decks.sh` leaves them: a deck outside the
decks of training read by the arm that held it out, and by the one that
trained on it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "thirty_one_decks_report", ROOT / "tools/thirty_one_decks_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DECKS = ["blackwing", "monarch"]
HALF_A, HALF_B = ["arcana"], ["harpie"]
# A copy of 2 updates a deck and arms of 3 of PPO, a checkpoint every one.
SHAPE = {"copy": 2, "updates": 3, "step": 1}
COPIED = {"plus-a": 6, "plus-b": 6, "all": 8, "control": 4}
END = [1, 2, 3]


def cases(counts: dict[str, int], games: int = 100) -> dict:
    """Each deck's games against blackwing and monarch, the first so many
    of its games won."""
    found = {}
    for deck, count in counts.items():
        told = ["win"] * count + ["loss"] * (games - count)
        half = games // 2
        found[f"{deck}-vs-blackwing"] = {"games": half, "outcomes": told[:half]}
        found[f"{deck}-vs-monarch"] = {"games": half, "outcomes": told[half:]}
    return found


def measured(path: Path, wins: dict[int, dict[str, int]]) -> None:
    """A file of evaluate-suite: each checkpoint of `wins`, by its updates
    in all, winning so many of each deck's games."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"checkpoint_step": step, "evaluation_suite": cases(counts)}
        for step, counts in wins.items()
    ]
    path.write_text(json.dumps({"seed": 2012, "records": records}))


def measurements(folder: Path, outside: dict[str, dict[str, int]], screens: dict[str, int]) -> None:
    """Every run's checkpoints on the decks outside training and on the
    screens, its copy winning half as many outside, and the pilots 45 of
    each deck's games."""
    for run in (*tool.ARMS, tool.CONTROL):
        measured(
            folder / "unseen" / f"{run}.json",
            {COPIED[run] + point: outside[run] for point in END},
        )
        measured(
            folder / "unseen" / f"{run}-copy.json",
            {COPIED[run]: {deck: count // 2 for deck, count in outside[run].items()}},
        )
        for deck in DECKS:
            measured(
                folder / "screens" / f"{run}.{deck}.json",
                {COPIED[run] + point: {deck: screens[run]} for point in END},
            )
    pilots = dict.fromkeys(next(iter(outside.values())), 45)
    path = folder / "unseen" / "pilot.json"
    path.write_text(
        json.dumps(
            {
                "seed": 2012,
                "records": [{"checkpoint_label": "pilot", "evaluation_suite": cases(pilots)}],
            }
        )
    )


def test_a_deck_outside_training_is_read_by_the_arm_that_held_it_out(tmp_path):
    measurements(
        tmp_path,
        outside={
            "plus-a": {"arcana": 50, "harpie": 40},
            "plus-b": {"arcana": 36, "harpie": 55},
            "all": {"arcana": 60, "harpie": 60},
            "control": {"arcana": 30, "harpie": 30},
        },
        screens={"plus-a": 60, "plus-b": 60, "all": 50, "control": 60},
    )
    result = tool.report(tmp_path, DECKS, HALF_A, HALF_B, **SHAPE)
    readings = result["readings"]
    # Arcana by plus-b (36), Harpie by plus-a (40), against 30 and 30.
    assert readings["held"]["gap"] == pytest.approx(0.08)
    assert readings["held"]["games"] == 200
    assert readings["held"]["verdict"] == "better"
    assert readings["kept"]["plus-a"]["verdict"] == "equivalent"
    assert readings["kept"]["all"]["gap"] == pytest.approx(-0.10)
    assert readings["kept"]["all"]["verdict"] == "worse"
    assert readings["all"]["gap"] == pytest.approx(0.15)
    assert readings["all"]["verdict"] == "over"
    # Trained on: Arcana by plus-a (50), Harpie by plus-b (55).
    assert result["unseen"]["trained"] == pytest.approx(0.525)
    assert result["unseen"]["held-copy"] == pytest.approx((18 + 20) / 200)
    assert result["beside"]["trained against the pilots"]["gap"] == pytest.approx(0.075)
    assert result["by_deck"]["arcana"] == {
        "half": "A",
        "pilot": pytest.approx(0.45),
        "control": pytest.approx(0.30),
        "held": pytest.approx(0.36),
        "trained": pytest.approx(0.50),
        "all": pytest.approx(0.60),
    }
    shown = tool.show(result)
    assert "1. the arm that held each deck out against the control" in shown
    assert "+8.0" in shown


def test_a_deck_in_neither_half_is_refused(tmp_path):
    played = {"arcana": 30, "harpie": 30, "ojama": 30}
    measurements(
        tmp_path,
        outside={run: played for run in (*tool.ARMS, tool.CONTROL)},
        screens=dict.fromkeys((*tool.ARMS, tool.CONTROL), 50),
    )
    with pytest.raises(SystemExit, match="in neither half: ojama"):
        tool.report(tmp_path, DECKS, HALF_A, HALF_B, **SHAPE)
