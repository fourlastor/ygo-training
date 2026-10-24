"""`tools/champions_report.py`: the readings of champions of single decks,
on measurements written as `experiments/commands/champions.sh` leaves
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
    "champions_report", ROOT / "tools/champions_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 100


def played(path: Path, outcomes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {"games": len(outcomes), "outcomes": outcomes}
    path.write_text(json.dumps({"records": [{"evaluation_suite": {"some-vs-checkpoint": case}}]}))


def measured(folder: Path, share: int) -> None:
    """The first network wins `share` of the deals' games, at three points."""
    for point in (480, 600, 720):
        played(folder / f"{point}.plays.json", ["win"] * share + ["loss"] * (DEALS - share))
        lost = DEALS - share
        played(folder / f"{point}.held.json", ["win"] * lost + ["loss"] * share)


def spent(directory: Path, name: str, hours: list[float]) -> None:
    rows = [json.dumps({"wall_seconds": 3600 * each}) for each in hours]
    (directory / f"{name}.metrics.jsonl").write_text("\n".join(rows) + "\n")


def test_a_champion_is_weaker_or_not_told_apart_by_its_mirror_against_all71(tmp_path):
    measured(tmp_path / "lightsworn", 70)
    measured(tmp_path / "infernity", 30)
    spent(tmp_path, "lightsworn-copy", [0.25])
    spent(tmp_path, "lightsworn", [0.5, 1.0])
    result = tool.report(tmp_path, ["lightsworn", "infernity"])
    assert result["lightsworn"]["share"] == pytest.approx(0.7)
    assert result["lightsworn"]["verdict"] == "a champion"
    assert result["lightsworn"]["hours"] == 1.75
    assert result["infernity"]["verdict"] == "weaker" and result["infernity"]["hours"] is None
    # The rule at its edges: 52% and 48%, by two standard errors.
    assert tool.verdict(0.53, 0.004) == "a champion"
    assert tool.verdict(0.53, 0.006) == "not told apart"
    assert tool.verdict(0.47, 0.004) == "weaker"
    assert "lightsworn: a champion. 70.0%" in tool.shown(result)


def test_a_run_of_another_seed_is_set_beside_the_decks_first_run(tmp_path):
    measured(tmp_path / "cyber-2013", 74)
    measured(tmp_path / "cyber-old", 76)
    measured(tmp_path / "cyber-2013-against-cyber-old", 47)
    result = tool.report(tmp_path, ["cyber-2013=cyber-old"])
    first = result["cyber-2013"]["first"]
    assert first["name"] == "cyber-old"
    assert first["against all71"]["share"] == pytest.approx(0.76)
    assert first["over it, against all71"]["gap"] == pytest.approx(-0.02)
    assert first["head to head"]["share"] == pytest.approx(0.47)
    assert "beside cyber-old, the deck's first run" in tool.shown(result)
