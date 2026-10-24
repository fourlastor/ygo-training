"""`tools/training_differences_report.py`: the readings of three of the
ways ygo-agent's trainer differs from this one, on measurements written as
`PARTS=measure` of `experiments/commands/training-differences.sh` leaves
them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "training_differences_report", ROOT / "tools/training_differences_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

BEGAN, READ, SMALLER = 2, [2, 4, 6], 4


def won(count: int, of: int = 400) -> list[str]:
    return ["win"] * count + ["loss"] * (of - count)


def held(path: Path, scale: int, wins: int, fingerprint: str = "deals") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "games": 400,
                "checkpoints": [
                    {"checkpoint_step": BEGAN + scale * point, "deals": {"outcomes": won(wins)}}
                    for point in READ
                ],
            }
        )
    )


def metrics(path: Path, rows: int, upgo: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "wall_seconds": 3.0,
        "duel_approximate_kl": 0.004,
        "duel_clip_fraction": 0.1,
        "duel_entropy": 0.2,
    }
    if upgo:
        row["upgo_advantage_mean"] = 0.05
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(rows)))


def test_each_run_is_read_against_the_control(tmp_path):
    held(tmp_path / "heldout" / "control.json", 1, 240)
    held(tmp_path / "heldout" / "small-updates.json", SMALLER, 280)
    held(tmp_path / "heldout" / "their-step.json", 1, 236)
    held(tmp_path / "heldout" / "upgo.json", 1, 180)
    held(tmp_path / "before.json", 1, 240)
    metrics(tmp_path / "runs" / "small-updates.metrics.jsonl", 24)
    # The control went on past the updates its checkpoints read had made.
    metrics(tmp_path / "runs" / "control.metrics.jsonl", 10)
    metrics(tmp_path / "runs" / "upgo.metrics.jsonl", 6, upgo=True)
    result = tool.report(tmp_path, BEGAN, READ, SMALLER, tmp_path / "before.json")
    readings = result["readings"]
    assert readings["small-updates"]["gap"] == pytest.approx(0.1)
    assert readings["small-updates"]["verdict"] == "better"
    assert readings["their-step"]["verdict"] == "not told apart"
    assert readings["upgo"]["verdict"] == "worse"
    assert result["control_before"]["gap"] == pytest.approx(0.0)
    # The seconds of each of the control's updates' worth: four of its own.
    assert result["runs"]["small-updates"]["figures"]["seconds"] == pytest.approx(12.0)
    assert result["runs"]["upgo"]["figures"]["upgo_advantage_mean"] == pytest.approx(0.05)
    assert result["control"]["figures"]["updates"] == 6
    text = tool.show(result)
    assert "1. Smaller updates: +10.0" in text
    assert "3. UPGO: -15.0" in text


def test_other_deals_than_the_controls_are_refused(tmp_path):
    held(tmp_path / "heldout" / "control.json", 1, 240)
    held(tmp_path / "heldout" / "upgo.json", 1, 240, fingerprint="other")
    with pytest.raises(SystemExit, match="other deals than the control"):
        tool.report(tmp_path, BEGAN, READ, SMALLER, None)


def test_a_checkpoint_not_measured_is_named(tmp_path):
    held(tmp_path / "heldout" / "control.json", 1, 240)
    # Read at the control's own updates, not four times as many.
    held(tmp_path / "heldout" / "small-updates.json", 1, 240)
    with pytest.raises(SystemExit, match="no checkpoint after 10 updates"):
        tool.report(tmp_path, BEGAN, READ, SMALLER, None)
