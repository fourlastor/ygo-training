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
