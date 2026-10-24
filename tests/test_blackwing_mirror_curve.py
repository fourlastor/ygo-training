"""`tools/blackwing_mirror_curve.py`: where a network stands in the Blackwing
mirror on the curve of decks-01, on measurements written as
`experiments/commands/blackwing-mirror-curve.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "blackwing_mirror_curve", ROOT / "tools/blackwing_mirror_curve.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 200


def write(path: Path, outcomes: list[str]) -> None:
    case = {tool.CASE: {"games": len(outcomes), "outcomes": outcomes}}
    path.write_text(json.dumps({"records": [{"evaluation_suite": case}]}))


def point(directory: Path, stem: str, at: int, won: int, won_held: int) -> None:
    """The network's checkpoint STEM against decks-01 at AT: it wins the first
    WON deals it plays and the first WON_HELD of those decks-01 plays."""
    write(directory / f"{stem}.plays-{at}.json", ["win"] * won + ["loss"] * (DEALS - won))
    # decks-01 plays these: what it loses the network wins, and a draw is
    # neither's.
    held = ["loss"] * won_held + ["draw"] * 10 + ["win"] * (DEALS - won_held - 10)
    write(directory / f"{stem}.held-{at}.json", held)


def test_the_share_at_each_point_and_where_it_stands(tmp_path):
    point(tmp_path, "model-1", 0, 150, 130)
    point(tmp_path, "model-1", 120, 102, 98)
    point(tmp_path, "model-1", 240, 60, 50)
    found = tool.shares(tmp_path)
    assert found[0]["share"] == pytest.approx(0.70)
    assert found[0]["verdict"] == "over"
    assert found[0]["games"] == 2 * DEALS
    assert found[120]["share"] == pytest.approx(0.50)
    assert found[120]["verdict"] == "level"
    assert found[240]["share"] == pytest.approx(0.275)
    assert found[240]["verdict"] == "under"
    assert tool.stands(found) == "level with decks-01 after 120 updates of its PPO"
    assert "It stands level" in tool.show(found, "network")


def test_between_the_last_point_over_and_the_first_under(tmp_path):
    for stem in ("model-1", "model-2"):
        point(tmp_path, stem, 0, 150, 130)
        point(tmp_path, stem, 120, 60, 50)
    found = tool.shares(tmp_path)
    assert found[0]["checkpoints"] == 2 and found[0]["games"] == 4 * DEALS
    assert tool.stands(found) == (
        "between decks-01 after 0 and after 120 updates of its PPO"
        " (0 and 33,600 games of the mirror)"
    )


def test_a_point_with_one_seat_played_is_not_read(tmp_path):
    point(tmp_path, "model-1", 0, 150, 130)
    write(tmp_path / "model-1.plays-120.json", ["win"] * DEALS)
    assert list(tool.shares(tmp_path)) == [0]


def test_the_reading_of_a_copy_of_decks_01(tmp_path):
    point(tmp_path, "model-1", 120, 150, 130)
    point(tmp_path, "model-1", 720, 102, 98)
    found = tool.shares(tmp_path)
    assert tool.kept(found, 720, 120) == "it keeps its teacher's strength in the mirror"
    point(tmp_path, "model-1", 720, 60, 50)
    found = tool.shares(tmp_path)
    assert tool.kept(found, 720, 120) == "it keeps some of its teacher's strength in the mirror"
    point(tmp_path, "model-1", 120, 60, 50)
    found = tool.shares(tmp_path)
    assert tool.kept(found, 720, 120) == "it keeps little of its teacher's strength in the mirror"
    assert tool.kept(found, 960, 120).startswith("not measured")
    assert "The reading: it keeps little" in tool.show(found, "copy", tool.kept(found, 720, 120))
