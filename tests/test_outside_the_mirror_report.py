"""`tools/outside_the_mirror_report.py`: a single-deck champion outside its
mirror, on measurements written as `experiments/commands/outside-the-
mirror.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "outside_the_mirror_report", ROOT / "tools/outside_the_mirror_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

POINTS = [480, 600, 720]
DEALS = 100


def played(path: Path, cases: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = {name: {"games": len(told), "outcomes": told} for name, told in cases.items()}
    path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}))


def wins(won: int, games: int = DEALS) -> list[str]:
    return ["win"] * won + ["loss"] * (games - won)


def measured(tmp_path: Path, alone: int, both_blackwing: int, all71_against_cyber: int) -> Path:
    """Blackwing alone wins `alone` deals, the network of both `both_blackwing`
    as Blackwing; all71 wins `all71_against_cyber` against cyber-old and 50
    against the network of both playing Cyber."""
    for point in POINTS:
        played(tmp_path / "out/cross" / f"blackwing-{point}.json", {"c": wins(alone)})
        played(tmp_path / "out/cross" / f"cyber-{point}.json", {"c": wins(all71_against_cyber)})
        played(tmp_path / "both/cross" / f"{4 * point}.plays.json", {"c": wins(both_blackwing)})
        played(tmp_path / "both/cross" / f"{4 * point}.held.json", {"c": wins(50)})
        for part in ("blackwing", "cyber"):
            panel = {"x-vs-a": wins(4, 10), "x-vs-b": wins(6, 10)}
            played(tmp_path / "out/panel" / f"{part}-{point}.json", panel)
    for part in ("blackwing", "cyber"):
        played(
            tmp_path / "out/panel" / f"{part}-all71.json",
            {"x-vs-a": wins(3, 10), "x-vs-b": wins(5, 10)},
        )
    for point in POINTS:
        played(tmp_path / "both/all71-alone" / f"{point}.plays.json", {"c": wins(70)})
        played(tmp_path / "both/all71-alone" / f"{point}.held.json", {"c": wins(40)})
    return tmp_path


def test_each_side_of_blackwing_against_cyber(tmp_path: Path) -> None:
    root = measured(tmp_path, alone=60, both_blackwing=70, all71_against_cyber=50)
    result = tool.report(root / "out", root / "both", POINTS, 2)
    blackwing, cyber = result["readings"]["Blackwing"], result["readings"]["Cyber"]
    assert (blackwing["ours"], blackwing["theirs"]) == (pytest.approx(0.6), pytest.approx(0.7))
    assert blackwing["verdict"] == "worse"
    # cyber-old wins what all71 loses, as the network of both does.
    assert (cyber["ours"], cyber["gap"]) == (pytest.approx(0.5), pytest.approx(0.0))
    assert cyber["verdict"] == "as well" and cyber["champion"] == "cyber-old"
    assert tool.verdict(-0.01, 0.002) == "as well"
    assert tool.verdict(-0.05, 0.01) == "worse"
    assert tool.verdict(-0.02, 0.01) == "not decided"


def test_the_panel_is_the_champions_edge_over_all71(tmp_path: Path) -> None:
    root = measured(tmp_path, alone=60, both_blackwing=60, all71_against_cyber=50)
    result = tool.report(root / "out", root / "both", POINTS, 2)
    panel = result["panel"]["Blackwing"]
    assert panel["ours"] == pytest.approx(10 / 20) and panel["theirs"] == pytest.approx(8 / 20)
    assert panel["gap"] == pytest.approx(0.1) and panel["decks"] == 2
    # Blackwing alone wins 70 deals in the learner's seat and 60 in the other.
    assert panel["mirror edge"] == pytest.approx(0.65 - 0.5)
    assert "mirror edge" not in result["panel"]["Cyber"]
    assert "in its own mirror +15.0" in tool.shown(result)


def test_games_played_again_must_end_as_they_did(tmp_path: Path, capsys) -> None:
    played(tmp_path / "here.json", {"c": wins(7, 10)})
    played(tmp_path / "there.json", {"c": wins(7, 10)})
    tool.check(tmp_path / "here.json", tmp_path / "there.json")
    assert "10 games end as they did" in capsys.readouterr().out
    played(tmp_path / "other.json", {"c": wins(6, 10)})
    with pytest.raises(SystemExit, match="does not end its games"):
        tool.check(tmp_path / "here.json", tmp_path / "other.json")
