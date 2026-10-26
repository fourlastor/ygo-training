"""`tools/shorter_copy_report.py`: the short arm against the uniform arm, on
measurements written as `PARTS=short-measure` of
`experiments/commands/four-decks.sh` leaves them (three decks here)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.test_four_decks_report import DECKS, pair, pairings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "shorter_copy_report", ROOT / "tools/shorter_copy_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def read(tmp_path: Path, short: int, uniform: int = 60, weaker: dict | None = None) -> dict:
    """Both arms on every pairing at their last three points, the short arm
    winning `short` of each case's games (or what `weaker` says of one)."""
    out = tmp_path / "out"
    for point in (20, 22, 24):
        pairings(out / "short-pairings" / f"{point}.json", short, weaker)
        pairings(out / "uniform-pairings" / f"{point}.json", uniform)
    for arm in ("short", "uniform"):
        for point in (2, 4):
            pairings(out / f"{arm}-quick" / f"{point}.json", 55)
        for deck in DECKS:
            pair(out / f"heads-{arm}" / deck, 49)
    pairings(out / "weighted-weights" / "0.json", 50)
    return tool.report(out, DECKS)


def test_the_shorter_copy_is_as_good_worse_or_not_decided(tmp_path):
    level = read(tmp_path / "level", short=60)["shorter"]
    assert level["gap"] == 0 and level["verdict"] == "as good"
    worse = read(tmp_path / "worse", short=50)["shorter"]
    assert worse["gap"] == pytest.approx(-0.1) and worse["verdict"] == "worse"
    # Eight deals of one seating of one pairing lost: two thirds of a point
    # under, too near the mark for either.
    near = read(tmp_path / "near", short=60, weaker={"cyber-vs-lightsworn-checkpoint": 52})
    assert near["shorter"]["gap"] == pytest.approx(-0.04 / 6)
    assert near["shorter"]["verdict"] == "not decided"
    assert near["shorter"]["by pairing"]["cyber|lightsworn"]["short"] == pytest.approx(0.56)
    # The rule at its edges: a point either way, by two standard errors.
    assert tool.verdict(-0.005, 0.002) == "as good"
    assert tool.verdict(-0.006, 0.0021) == "not decided"
    assert tool.verdict(0.005, 0.002) == "as good"
    assert tool.verdict(-0.016, 0.0025) == "worse"


def test_beside_the_curves_start_from_the_copy_and_the_hours_are_told(tmp_path):
    result = read(tmp_path, short=60)
    curve = result["beside"]["curves"]["short"]
    assert [point["updates"] for point in curve] == [0, 2, 4]
    assert curve[0]["mean"] == pytest.approx(0.5)
    assert result["beside"]["head to head"]["short"]["cyber"] == pytest.approx(0.49)
    assert result["hours"]["short"] is None
    shown = tool.shown(result)
    assert "The shorter copy: as good." in shown and "Hours: the copy to 640 not measured" in shown
