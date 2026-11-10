"""`tools/network_screen_report.py`: the readings of the screen of the
network under PPO, on measurements written as
`experiments/commands/champions.sh` leaves them for runs of a shape."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "network_screen_report", ROOT / "tools/network_screen_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 100


def played(path: Path, outcomes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {"games": len(outcomes), "outcomes": outcomes}
    path.write_text(json.dumps({"records": [{"evaluation_suite": {"cyber-vs-checkpoint": case}}]}))


def measured(folder: Path, share: int) -> None:
    """The first network wins `share` of the deals' games, at three points."""
    for point in (480, 600, 720):
        played(folder / f"{point}.plays.json", ["win"] * share + ["loss"] * (DEALS - share))
        played(folder / f"{point}.held.json", ["win"] * (DEALS - share) + ["loss"] * share)


def timed(directory: Path, name: str, seconds: list[float]) -> None:
    rows = [json.dumps({"wall_seconds": each}) for each in seconds]
    (directory / f"{name}.metrics.jsonl").write_text("\n".join(rows) + "\n")


def test_each_variant_is_read_against_its_control_by_its_own_rule(tmp_path):
    for name, share in (("cyber", 76), ("cyber-tokens", 80), ("cyber-no-history", 70)):
        measured(tmp_path / name, share)
    measured(tmp_path / "cyber-tokens-against-cyber", 60)
    measured(tmp_path / "cyber-no-history-against-cyber", 40)
    timed(tmp_path, "cyber-copy", [1800.0, 1800.0])
    timed(tmp_path, "cyber", [10.0, 14.0])
    result = tool.report(tmp_path, "cyber", ["cyber-tokens", "cyber-no-history"])
    tokens = result["variants"]["cyber-tokens"]
    assert tokens["share"] == pytest.approx(0.6) and tokens["reading"] == "stronger"
    assert tokens["against all71"]["share"] == pytest.approx(0.8)
    assert tokens["hours"] is None and tokens["seconds an update"] is None
    history = result["variants"]["cyber-no-history"]
    assert history["reading"] == "the history needed"
    control = result["control"]
    assert control["against all71"]["share"] == pytest.approx(0.76)
    assert control["seconds an update"] == pytest.approx(12.0)
    assert control["hours"] == pytest.approx(1800 * 2 / 3600 + 24 / 3600)
    text = tool.shown(result)
    assert "cyber-tokens: stronger. 60.0%" in text
    assert "cyber-no-history: the history needed. 40.0%" in text
    assert "cyber: 76.0%" in text and "12.0 s an update of PPO" in text


def test_the_rules_at_their_edges():
    # The board as its cards: 54% and 46%, by two standard errors.
    assert tool.reading("tokens", 0.55, 0.004) == "stronger"
    assert tool.reading("tokens", 0.55, 0.006) == "not a large lever for one deck"
    assert tool.reading("tokens", 0.45, 0.004) == "weaker"
    assert tool.reading("tokens", 0.50, 0.01) == "not a large lever for one deck"
    # The history: needed at 46% by two standard errors, and nothing on the other side.
    assert tool.reading("no-history", 0.45, 0.004) == "the history needed"
    assert tool.reading("no-history", 0.45, 0.006) == "the history not shown to be needed"
    assert tool.reading("no-history", 0.60, 0.004) == "the history not shown to be needed"
    # The switch of a trained network: 54% and 46%, by two standard errors.
    assert tool.reading("switched", 0.55, 0.004) == "the switch pays"
    assert tool.reading("switched", 0.55, 0.006) == "the switch not shown"
    assert tool.reading("switched", 0.45, 0.004) == "the switch harms"


def test_a_switched_run_is_read_without_a_copy_of_its_own(tmp_path):
    for name, share in (("cyber", 76), ("cyber-switched", 78)):
        measured(tmp_path / name, share)
    measured(tmp_path / "cyber-switched-against-cyber", 60)
    timed(tmp_path, "cyber-copy", [1800.0])
    timed(tmp_path, "cyber", [10.0, 14.0])
    timed(tmp_path, "cyber-switched", [12.0, 12.0])
    result = tool.report(tmp_path, "cyber", ["cyber-switched"])
    switched = result["variants"]["cyber-switched"]
    assert switched["reading"] == "the switch pays"
    assert switched["hours"] is None and switched["seconds an update"] == pytest.approx(12.0)
    assert "cyber-switched: the switch pays. 60.0%" in tool.shown(result)


def test_a_variant_is_named_after_its_control(tmp_path):
    measured(tmp_path / "cyber", 76)
    with pytest.raises(SystemExit):
        tool.report(tmp_path, "cyber", ["blackwing-tokens"])
    with pytest.raises(SystemExit):
        tool.report(tmp_path, "cyber", ["cyber-wide"])
