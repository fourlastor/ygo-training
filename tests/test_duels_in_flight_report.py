"""`tools/duels_in_flight_report.py`: the readings of duels in flight and
larger minibatches, on measurements written as `PARTS=measure` of
`experiments/commands/duels-in-flight.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "duels_in_flight_report", ROOT / "tools/duels_in_flight_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

THIRTEEN = ["blackwing", "monarch"]
# A copy of 8 updates and the point of PPO 3, a checkpoint every update; the
# speed read over the first two updates.
SHAPE = {"copied": 8, "point": 3, "step": 1, "first": 2}


def cases(deck: str, won: int, games: int = 200) -> dict:
    """A deck's games against blackwing and monarch, the first so many won."""
    told = ["win"] * won + ["loss"] * (games - won)
    half = games // 2
    return {
        f"{deck}-vs-blackwing": {"games": half, "outcomes": told[:half]},
        f"{deck}-vs-monarch": {"games": half, "outcomes": told[half:]},
    }


def write(path: Path, cases_: dict, seed: int = 2012) -> None:
    """A run's three checkpoints at the point, each winning the same games."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"checkpoint_step": 8 + at, "evaluation_suite": cases_} for at in (1, 2, 3)]
    path.write_text(json.dumps({"seed": seed, "records": records}))


def figures(path: Path, seconds: float, updates: int = 3, **more: float) -> None:
    rows = [
        {"wall_seconds": seconds, "duels": 520, "seconds": seconds / 2, "inference_seconds": 1.0,
         "ppo_update_seconds": seconds / 3, "other_seat_win_rate": {"pilot": 0.5}, **more}
        for _ in range(updates)
    ]  # fmt: skip
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def measured(directory: Path, ours: int, theirs: int, fast: float = 6.0) -> None:
    for run, won in (("fast", ours), ("all71-3", theirs)):
        for deck in THIRTEEN:
            write(directory / "screens" / f"{run}.{deck}.json", cases(deck, won))
        write(directory / "old" / f"{run}.json", cases("arcana", won))
        write(directory / "new" / f"{run}.json", cases("agents", won))
    figures(directory / "fast.metrics.jsonl", fast, stream_age=0.6, stream_oldest=2.0)
    figures(directory / "plain.metrics.jsonl", 9.0)
    figures(directory / "all71.metrics.jsonl", 9.0)


@pytest.mark.parametrize(
    ("ours", "theirs", "verdict"),
    [(100, 100, "equivalent"), (96, 100, "not decided"), (80, 100, "worse")],
)
def test_the_first_reading_over_the_three_suites_together(tmp_path, ours, theirs, verdict):
    measured(tmp_path, ours, theirs)
    result = tool.report(tmp_path, THIRTEEN, **SHAPE)
    reading = result["reading"]
    assert reading["verdict"] == verdict
    assert reading["games"] == 4 * 200
    assert reading["gap"] == pytest.approx((ours - theirs) / 200)
    assert "1. It learns as well: " + verdict in tool.show(result)


def test_the_mark_of_the_first_reading_is_two_points_beyond_two_standard_errors():
    assert tool.kept_rule(-0.01, 0.004) == "equivalent"
    assert tool.kept_rule(-0.015, 0.004) == "not decided"
    assert tool.kept_rule(-0.03, 0.004) == "worse"


@pytest.mark.parametrize(("fast", "verdict"), [(6.0, "faster"), (8.0, "not faster")])
def test_the_second_reading_sets_the_first_updates_beside_the_trials(tmp_path, fast, verdict):
    measured(tmp_path, 100, 100, fast=fast)
    result = tool.report(tmp_path, THIRTEEN, **SHAPE)
    speed = result["speed"]
    assert speed["verdict"] == verdict
    assert speed["ratio"] == pytest.approx(9.0 / fast)
    assert speed["fast"] == pytest.approx(3600 * 520 / fast)
    shown = tool.show(result)
    assert "2. It is faster: " + verdict in shown
    assert "begun 0.60 updates before on average, 2 at most" in shown


def test_the_second_reading_wants_as_many_updates_of_each(tmp_path):
    measured(tmp_path, 100, 100)
    figures(tmp_path / "plain.metrics.jsonl", 9.0, updates=1)
    with pytest.raises(SystemExit, match="2 updates of each"):
        tool.report(tmp_path, THIRTEEN, **SHAPE)
