"""`tools/privileged_critic_run_report.py`: the third phase's reading, on
measurements written as `PARTS=measure` of
`experiments/commands/privileged-critic.sh` leaves them beside those of
seventy-one decks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "privileged_critic_run_report", ROOT / "tools/privileged_critic_run_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

THIRTEEN = ["blackwing", "monarch"]
# A copy of 8 updates and the point of PPO 3, a checkpoint every update.
SHAPE = {"copied": 8, "point": 3, "step": 1}


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


def measurements(arm: Path, base: Path, ours: int, theirs: int) -> None:
    for folder, run, won in ((arm, "critic", ours), (base, "all71-3", theirs)):
        for deck in THIRTEEN:
            write(folder / "screens" / f"{run}.{deck}.json", cases(deck, won))
        write(folder / "old" / f"{run}.json", cases("arcana", won))
        write(folder / "new" / f"{run}.json", cases("agents", won))


@pytest.mark.parametrize(
    ("ours", "theirs", "verdict"),
    [(120, 100, "faster"), (104, 100, "not told apart"), (100, 120, "slower")],
)
def test_the_reading_over_the_three_suites_together(tmp_path, ours, theirs, verdict):
    arm, base = tmp_path / "arm", tmp_path / "base"
    measurements(arm, base, ours, theirs)
    result = tool.report(arm, base, THIRTEEN, **SHAPE)
    reading = result["reading"]
    assert reading["verdict"] == verdict
    assert reading["games"] == 4 * 200
    assert reading["gap"] == pytest.approx((ours - theirs) / 200)
    for kind in ("screens", "old", "new"):
        assert result["by_kind"][kind]["gap"] == pytest.approx((ours - theirs) / 200)
    assert result["by_screen"]["monarch"]["critic"] == pytest.approx(ours / 200)
    assert "It learns faster: " + verdict in tool.show(result)


def test_the_mark_is_two_points_beyond_two_standard_errors():
    assert tool.faster(0.03, 0.004) == "faster"
    assert tool.faster(0.025, 0.004) == "not told apart"
    assert tool.faster(-0.03, 0.004) == "slower"
    assert tool.faster(-0.025, 0.004) == "not told apart"


def test_runs_are_read_on_the_same_games_only(tmp_path):
    arm, base = tmp_path / "arm", tmp_path / "base"
    measurements(arm, base, 110, 100)
    write(base / "old" / "all71-3.json", cases("arcana", 100), seed=7)
    with pytest.raises(SystemExit, match="other games"):
        tool.report(arm, base, THIRTEEN, **SHAPE)
    (base / "old" / "all71-3.json").unlink()
    with pytest.raises(SystemExit, match="not measured"):
        tool.report(arm, base, THIRTEEN, **SHAPE)


def test_the_runs_figures_along_the_run(tmp_path):
    arm, base = tmp_path / "arm", tmp_path / "base"
    measurements(arm, base, 110, 100)
    rows = [
        {
            "wall_seconds": 10.0,
            "duels": 520,
            "other_seat_win_rate": {"pilot": 0.5 + 0.01 * index},
            "critic_outcome_explained_variance": 0.4,
            "critic_head_outcome_explained_variance": 0.3,
        }
        for index in range(4)
    ]
    (arm / "critic.metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = tool.report(arm, base, THIRTEEN, **SHAPE)
    blocks = result["along"]["critic"]
    assert [(block["first"], block["last"]) for block in blocks] == [(1, 2), (3, 4)]
    assert blocks[1]["pilots"] == pytest.approx(0.525)
    assert blocks[0]["critic"] == pytest.approx(0.4) and blocks[0]["head"] == pytest.approx(0.3)
    assert result["along"]["all71"] == []
    assert "updates     3-    4" in tool.show(result)


def test_another_arm_is_read_the_same_way_with_its_first_reading_beside(tmp_path):
    arm, base = tmp_path / "arm", tmp_path / "base"
    measurements(arm, base, 120, 100)
    for path in sorted(arm.rglob("critic*.json")):
        path.rename(path.with_name(path.name.replace("critic", "cards", 1)))
    result = tool.report(arm, base, THIRTEEN, **SHAPE, arm="cards")
    assert result["arm"] == "cards" and result["reading"]["verdict"] == "faster"
    held = {"cards": 1000, "share": 0.7, "error": 0.05, "head_log_loss": 0.06,
            "frequencies_log_loss": 0.2}  # fmt: skip
    shown = tool.show(result, {"held": held})
    assert "1. It learns how cards move: it predicts" in shown
    assert "2. It learns faster: faster" in shown and "3. It learns" not in shown
    assert "The run cards against all71" in shown
