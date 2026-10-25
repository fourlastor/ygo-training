"""`tools/improved_pilots.py`: the improved pilots on `all71`'s yardstick, a
deck the branch left as it was told apart from one whose games changed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("improved_pilots", ROOT / "tools/improved_pilots.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def write(path: Path, won: dict[str, list[int]], records: int = 1) -> None:
    """A suite's file: each case's games, won (1) or lost (0), the same in
    each of so many records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = {
        name: {"games": len(games), "outcomes": ["win" if game else "loss" for game in games]}
        for name, games in won.items()
    }
    path.write_text(json.dumps({"seed": 2012, "records": [{"evaluation_suite": cases}] * records}))


def measured(tmp_path: Path, cyber: list[int]) -> tuple[Path, Path]:
    out, network = tmp_path / "out", tmp_path / "network"
    old_cyber = [0, 0, 0, 0, 1, 0, 0, 0]
    zombie = [1, 0, 1, 0, 1, 0, 1, 0]
    for folder, games in (("pinned", old_cyber), ("branch", cyber)):
        write(out / folder / "new.json", {"cyber-vs-burn": games, "zombie-vs-burn": zombie})
        write(out / folder / "old.json", {"chaos-vs-burn": [1, 1, 0, 0]})
    write(
        network / "new" / "all71-10560.json",
        {"cyber-vs-burn": [1, 1, 1, 1, 1, 0, 0, 0], "zombie-vs-burn": [1] * 8},
        records=3,
    )
    write(network / "old" / "all71-10560.json", {"chaos-vs-burn": [1, 1, 1, 0]}, records=3)
    return out, network


def test_a_deck_whose_pilot_changed_is_set_beside_all71(tmp_path):
    out, network = measured(tmp_path, cyber=[1, 1, 0, 0, 1, 0, 0, 0])
    result = tool.report(out / "pinned", out / "branch", network)
    assert result["decks"] == 3 and result["unchanged"] == 2
    cyber = result["changed"]["cyber"]
    assert cyber["pinned"] == pytest.approx(1 / 8)
    assert cyber["branch"] == pytest.approx(3 / 8)
    assert cyber["network"] == pytest.approx(5 / 8)
    assert cyber["gain"] == pytest.approx(2 / 8)
    assert cyber["gap"] == pytest.approx(4 / 8)
    assert cyber["closed"] == pytest.approx(0.5)
    assert "cyber: 37.5% against 12.5% as pinned" in tool.shown(result)


def test_decks_the_branch_left_as_they_were_are_not_measured(tmp_path):
    out, network = measured(tmp_path, cyber=[0, 0, 0, 0, 1, 0, 0, 0])
    result = tool.report(out / "pinned", out / "branch", network)
    assert result["changed"] == {} and result["unchanged"] == 3


def test_games_against_an_opponent_whose_pilot_changed_are_told_apart(tmp_path):
    """Where both folders hold the thirteen's screens they are read beside
    the two suites; where the branch says whose pilots it changed, games
    against those of the thirteen are told apart, those of every deck whose
    own pilot is as it was together."""
    out, network = measured(tmp_path, cyber=[1, 1, 0, 0, 1, 0, 0, 0])
    for folder, mine, theirs, mirror in (
        ("pinned", [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]),
        ("branch", [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]),
    ):
        write(
            out / folder / "screen-dragunity.json",
            {"dragunity-vs-heroes": mine, "dragunity-vs-dragunity": mirror},
        )
        write(out / folder / "screen-heroes.json", {"heroes-vs-dragunity": theirs})
    write(network / "screens" / "all71-10560.dragunity.json", {"dragunity-vs-heroes": [1] * 4}, 3)
    write(network / "screens" / "all71-10560.heroes.json", {"heroes-vs-dragunity": [1] * 4}, 3)
    (out / "branch" / "changed.txt").write_text(
        " crates/ygo-policies/src/decks/cyber.rs     | 52 ++-\n"
        " crates/ygo-policies/src/decks/dragunity.rs |  5 +-\n"
    )
    result = tool.report(out / "pinned", out / "branch", network)
    assert result["screens"] == ["dragunity", "heroes"] and result["moved"] == ["dragunity"]
    assert sorted(result["changed"]) == ["cyber", "dragunity"]
    dragunity = result["changed"]["dragunity"]
    assert (dragunity["pinned"], dragunity["branch"], dragunity["network"]) == (0.5, 1.0, 1.0)
    assert dragunity["closed"] == pytest.approx(1.0)
    # Heroes' only games are against Dragunity's changed pilot: told apart,
    # without the mirror, where Dragunity's own pilot changed too.
    assert "heroes" not in result["changed"]
    against = result["against the moved"]["dragunity"]
    assert against["games"] == 4 and against["gain"] == pytest.approx(-0.25)
    assert "Against dragunity's pilot as the branch has it" in tool.shown(result)
