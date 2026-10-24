"""`tools/overlap_report.py`: the readings of the pass beside the next
update's games, on measurements written as `PARTS=measure` of
`experiments/commands/overlap.sh` leaves them, `fast`'s where duels in
flight left them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("overlap_report", ROOT / "tools/overlap_report.py")
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
         "ppo_update_seconds": seconds / 3, "other_seat_win_rate": {"pilot": 0.5},
         "duel_clip_fraction": 0.1, **more}
        for _ in range(updates)
    ]  # fmt: skip
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def measured(directory: Path, base: Path, ours: int, theirs: int, seconds: float = 4.0) -> None:
    for root, run, won in ((directory, "overlap", ours), (base, "fast", theirs)):
        for deck in THIRTEEN:
            write(root / "screens" / f"{run}.{deck}.json", cases(deck, won))
        write(root / "old" / f"{run}.json", cases("arcana", won))
        write(root / "new" / f"{run}.json", cases("agents", won))
    figures(
        directory / "overlap.metrics.jsonl",
        seconds,
        stream_age=0.6,
        stream_oldest=2.0,
        overlap_wait_seconds=0.5,
        overlap_beside_seconds=2.5,
    )
    figures(directory / "fast.metrics.jsonl", 6.8, stream_age=0.6, stream_oldest=2.0)
    figures(directory / "trial-fast.metrics.jsonl", 6.0)


@pytest.mark.parametrize(
    ("ours", "theirs", "verdict"),
    [(100, 100, "equivalent"), (96, 100, "not decided"), (80, 100, "worse")],
)
def test_the_first_reading_sets_the_arm_beside_fast(tmp_path, ours, theirs, verdict):
    base = tmp_path / "fast"
    measured(tmp_path, base, ours, theirs)
    result = tool.report(tmp_path, base, THIRTEEN, **SHAPE)
    reading = result["reading"]
    assert reading["verdict"] == verdict
    assert reading["games"] == 4 * 200
    assert reading["gap"] == pytest.approx((ours - theirs) / 200)
    assert result["by_screen"]["blackwing"]["fast"] == pytest.approx(theirs / 200)
    assert "1. It learns as well: " + verdict in tool.show(result)


@pytest.mark.parametrize(("seconds", "verdict"), [(4.8, "faster"), (5.0, "not faster")])
def test_the_second_reading_is_a_quarter_more_duels_an_hour(tmp_path, seconds, verdict):
    base = tmp_path / "fast"
    measured(tmp_path, base, 100, 100, seconds=seconds)
    result = tool.report(tmp_path, base, THIRTEEN, **SHAPE)
    speed = result["speed"]
    assert speed["verdict"] == verdict
    assert speed["ratio"] == pytest.approx(6.0 / seconds)
    shown = tool.show(result)
    assert "2. It is faster: " + verdict in shown
    assert "waited for the games 0.50 s; the games beside the pass 2.50 s" in shown
    assert "PPO's clip bound at 0.100 of its decisions" in shown


def test_the_second_reading_wants_as_many_updates_of_each(tmp_path):
    base = tmp_path / "fast"
    measured(tmp_path, base, 100, 100)
    figures(tmp_path / "trial-fast.metrics.jsonl", 6.0, updates=1)
    with pytest.raises(SystemExit, match="2 updates of each"):
        tool.report(tmp_path, base, THIRTEEN, **SHAPE)
