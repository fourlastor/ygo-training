"""`tools/card_descriptions_report.py`: the readings of card descriptions
and hidden identities, on measurements written as `PARTS=measure` of
`experiments/commands/card-descriptions.sh` leaves them, and what the decks
outside training share with those of training."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "card_descriptions_report", ROOT / "tools/card_descriptions_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DECKS = ["blackwing", "monarch"]
OUTSIDE = ["arcana", "harpie"]
# A copy of 2 updates a deck and arms of 3 of PPO, a checkpoint every one.
SHAPE = {"copy": 2, "updates": 3, "step": 1}
COPIED = len(DECKS) * SHAPE["copy"]
END = [1, 2, 3]


def won(count: int, of: int) -> list[str]:
    return ["win"] * count + ["loss"] * (of - count)


def measured(path: Path, wins: dict[int, int] | None, players: list[str], games: int) -> None:
    """A file of evaluate-suite of a deck against two others, or of the
    decks outside training: the checkpoints of `wins` (updates in all ->
    games won of each deck's), or the pilot where it is None, winning a
    quarter."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def cases(count: int) -> dict:
        told = won(count, games)
        half = games // 2
        return {
            f"{deck}-vs-{other}": {"games": half, "outcomes": told[:half] if first else told[half:]}
            for deck in players
            for first, other in ((True, "blackwing"), (False, "monarch"))
        }

    if wins is None:
        records = [{"checkpoint_label": "pilot", "evaluation_suite": cases(games // 4)}]
    else:
        records = [
            {"checkpoint_step": step, "evaluation_suite": cases(count)}
            for step, count in wins.items()
        ]
    path.write_text(json.dumps({"seed": 2012, "records": records}))


def measurements(folder: Path, outside: dict[str, int], screens: dict[str, int]) -> None:
    """Every run's checkpoints winning so many of each deck's 100 games
    outside training and of each screen's, its copy half as many."""
    for run in (*tool.ARMS, tool.CONTROL):
        steps = {COPIED + point: outside[run] for point in END}
        measured(folder / "unseen" / f"{run}.json", steps, OUTSIDE, 100)
        measured(folder / "unseen" / f"{run}-copy.json", {COPIED: outside[run] // 2}, OUTSIDE, 100)
        for deck in DECKS:
            measured(
                folder / "screens" / f"{run}.{deck}.json",
                {COPIED + point: screens[run] for point in END},
                [deck],
                100,
            )
    measured(folder / "unseen" / "pilot.json", None, OUTSIDE, 100)


def table(path: Path) -> Path:
    """A table of four codes: two cards of the same text, one near them and
    one far, and a Normal Monster without one."""
    vectors = np.array([[1.0, 0.0], [0.9, np.sqrt(1 - 0.81)], [0.0, 1.0]])
    np.savez(
        path,
        codes=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        rows=np.array([1, 1, 2, 3, 0], dtype=np.int32),
        descriptions=np.vstack([np.zeros((1, 2)), vectors]).astype(np.float16),
        mean=np.zeros(2, dtype=np.float32),
        spread=np.ones(2, dtype=np.float32),
    )
    return path


def test_the_readings_read_the_arms_against_the_control(tmp_path):
    measurements(
        tmp_path,
        outside={"both": 40, "descriptions": 31, "hidden": 30, "control": 30},
        screens={"both": 60, "descriptions": 60, "hidden": 60, "control": 60},
    )
    result = tool.report(tmp_path, DECKS, **SHAPE)
    readings = result["readings"]
    assert readings["outside"]["gap"] == pytest.approx(0.10)
    assert readings["outside"]["games"] == 200
    assert readings["outside"]["verdict"] == "better"
    assert readings["descriptions"]["verdict"] == "not told apart"
    assert readings["hidden"]["gap"] == 0.0
    assert readings["kept"]["gap"] == 0.0
    assert readings["kept"]["verdict"] == "equivalent"
    assert result["unseen"]["both-copy"] == pytest.approx(0.20)
    assert result["unseen"]["pilot"] == pytest.approx(0.25)
    assert result["from_pilots"]["both"]["gap"] == pytest.approx(0.15)
    # Each deck outside training, and the share of its pilot's wins kept.
    assert result["by_deck"]["arcana"]["both"] == pytest.approx(0.40)
    assert result["by_deck"]["arcana"]["kept"]["both"] == pytest.approx(0.40 / 0.25)
    assert "screens" in result and "readings" in result
    assert "both against the control on the decks outside training: +10.0" in tool.show(result)


def test_the_second_reading_is_decided_only_away_from_its_mark():
    assert tool.kept_rule(-0.005, 0.005) == "equivalent"
    assert tool.kept_rule(-0.05, 0.01) == "worse"
    assert tool.kept_rule(-0.02, 0.01) == "not decided"


def test_what_a_deck_shares_is_told_card_by_card_with_their_copies(tmp_path):
    codes, rows, unit = tool.described(table(tmp_path / "table.npz"))
    decks = {
        "blackwing": [10, 10],
        "monarch": [10],
        # A card trained on, one of its text, two copies of one near it, and
        # a Normal Monster.
        "arcana": [10, 20, 30, 30, 50],
        "harpie": [40],
    }
    found = tool.coverage(decks, DECKS, codes, rows, unit)
    assert set(found) == {"arcana", "harpie"}
    arcana = found["arcana"]
    assert arcana["same_cards"] == pytest.approx(1 / 5)
    assert arcana["same_effect"] == pytest.approx(4 / 5)
    # The Normal Monster has no text to be near by.
    assert arcana["nearest"] == pytest.approx((1.0 + 1.0 + 0.9 + 0.9) / 4, abs=1e-3)
    assert found["harpie"] == {
        "cards": 1,
        "same_cards": 0.0,
        "same_effect": 0.0,
        "nearest": pytest.approx(0.0, abs=1e-3),
    }
