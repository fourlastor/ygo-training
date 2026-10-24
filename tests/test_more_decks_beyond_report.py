"""`tools/more_decks_beyond_report.py`: the four readings of the networks of
more decks beyond their own decks, on measurements written as
`PARTS=beyond` of `experiments/commands/more-decks.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "more_decks_beyond_report", ROOT / "tools/more_decks_beyond_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

SIZES = [1, 13]
# A copy of 2 updates a deck, then 6 of PPO with a checkpoint every 2.
SHAPE = {"copy": 2, "updates": 6, "step": 2}
CHECKPOINT = "runs/more-decks/{}/checkpoints/model-{:05d}.pt"


def won(count: int, of: int) -> list[str]:
    """Games of which the first so many are won."""
    return ["win"] * count + ["loss"] * (of - count)


def cases(count: int) -> dict:
    """Two decks outside training, each against two pilots, 100 games each:
    the first `count` of the 400 won."""
    games = won(count, 400)
    names = ["arcana-vs-burn", "arcana-vs-monarch", "toon-vs-burn", "toon-vs-monarch"]
    return {
        name: {"games": 100, "outcomes": games[100 * index : 100 * (index + 1)]}
        for index, name in enumerate(names)
    }


def played(folder: Path, size: int, copy: int, three: int, seed: int = 2012) -> None:
    (folder / "unseen").mkdir(parents=True, exist_ok=True)
    steps = tool.steps(size, SHAPE["copy"], SHAPE["updates"], SHAPE["step"])
    wins = {tool.COPY: copy} | {name: three for name in steps if name != tool.COPY}
    (folder / "unseen" / f"{tool.network(size)}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "records": [
                    {"checkpoint_step": step, "evaluation_suite": cases(wins[name])}
                    for name, step in steps.items()
                ],
            }
        )
    )


def pilots(folder: Path, count: int) -> None:
    (folder / "unseen").mkdir(parents=True, exist_ok=True)
    (folder / "unseen" / "pilot.json").write_text(
        json.dumps({"seed": 2012, "records": [{"evaluation_suite": cases(count)}]})
    )


def mirror(folder: Path, point: int, many: int, deals: int = 400) -> None:
    """The network of the most decks against that of the fewest after
    `point` updates of PPO: in its own seat it wins the first `many` deals,
    and held, it loses the first `many` the other plays."""
    (folder / "head-to-head").mkdir(parents=True, exist_ok=True)
    big = CHECKPOINT.format("decks-13", 13 * SHAPE["copy"] + point - 1)
    small = CHECKPOINT.format("decks-01", SHAPE["copy"] + point - 1)
    for suffix, measured, held, outcomes in (
        ("", big, small, won(many, deals)),
        (".back", small, big, ["loss"] * many + ["win"] * (deals - many)),
    ):
        (folder / "head-to-head" / f"mirror-{point}{suffix}.json").write_text(
            json.dumps(
                {
                    "seed": 2012,
                    "suite": "eval_suites/blackwing-against-checkpoint.json",
                    "opponent_checkpoint": {"path": held},
                    "records": [
                        {
                            "checkpoint": measured,
                            "checkpoint_step": int(measured[-8:-3]) + 1,
                            "evaluation_suite": {
                                "blackwing-vs-checkpoint": {"games": deals, "outcomes": outcomes}
                            },
                        }
                    ],
                }
            )
        )


def read(folder: Path) -> dict:
    return tool.report(folder, SIZES, SHAPE["copy"], SHAPE["updates"], SHAPE["step"])


def test_the_four_readings(tmp_path):
    pilots(tmp_path, 200)
    played(tmp_path, 1, copy=100, three=120)
    played(tmp_path, 13, copy=180, three=240)
    for point in (2, 4, 6):
        mirror(tmp_path, point, many=240)
    result = read(tmp_path)
    readings = result["readings"]
    assert readings["pilots"]["gap"] == pytest.approx(0.1)
    assert readings["pilots"]["verdict"] == "over"
    assert readings["more decks"]["gap"] == pytest.approx(0.3)
    assert readings["more decks"]["verdict"] == "more decks help"
    assert readings["copy"]["gap"] == pytest.approx(0.15)
    assert readings["copy"]["verdict"] == "PPO carries over"
    assert readings["mirror"]["share"] == pytest.approx(0.6)
    assert readings["mirror"]["verdict"] == "better"
    assert readings["mirror"]["games"] == 2400
    assert result["by_deck"]["arcana"] == {"pilot": 1.0, "three": 1.0}
    assert result["by_deck"]["toon"]["three"] == pytest.approx(0.2)
    text = tool.show(result)
    assert "1. against the decks' own pilots: +10.0" in text
    assert "4. in the mirror: 60.0%" in text


def test_a_network_under_the_pilots_and_the_mirror_not_told_apart(tmp_path):
    pilots(tmp_path, 260)
    played(tmp_path, 1, copy=100, three=200)
    played(tmp_path, 13, copy=180, three=200)
    for point in (2, 4, 6):
        mirror(tmp_path, point, many=202)
    readings = read(tmp_path)["readings"]
    assert readings["pilots"]["verdict"] == "under"
    assert readings["more decks"]["verdict"] == "not shown"
    assert readings["mirror"]["verdict"] == "not told apart"


def test_other_games_than_the_pilots_are_refused(tmp_path):
    pilots(tmp_path, 200)
    played(tmp_path, 13, copy=180, three=240, seed=7)
    with pytest.raises(SystemExit, match="other games than the pilots"):
        read(tmp_path)
