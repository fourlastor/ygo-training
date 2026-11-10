"""`tools/pool_arms_report.py`: batch six's two arms side by side."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def loaded(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = loaded("pool_arms_report")
POINTS = [3840, 7680]


def written(path: Path, suite: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    told = {name: {"outcomes": games, "by_opponent": {}} for name, games in suite.items()}
    path.write_text(json.dumps({"records": [{"evaluation_suite": told}]}), encoding="utf8")


def head_to_head(folder: Path, share: float, deals: int = 20000) -> None:
    """switched wins a share of the deals from both seats, over two cases."""
    won = round(share * deals)
    plays = ["win"] * won + ["loss"] * (deals - won)
    # pooled plays the held games in the learner's seat: it loses what switched wins.
    held = ["loss"] * won + ["win"] * (deals - won)
    half = deals // 2
    written(folder / "arms.plays.json", {"a-vs-each": plays[:half], "b-vs-each": plays[half:]})
    written(folder / "arms.held.json", {"a-vs-each": held[:half], "b-vs-each": held[half:]})


def arms(tmp_path: Path, share: float) -> tuple[Path, Path, Path]:
    pooled, switched, folder = tmp_path / "pooled", tmp_path / "switched", tmp_path / "arms"
    for point in POINTS:
        head_to_head(folder / str(point), share)
    return pooled, switched, folder


@pytest.mark.parametrize(
    ("share", "verdict"), [(0.53, "pays"), (0.50, "not shown"), (0.47, "harms")]
)
def test_the_switch_is_read_head_to_head_at_the_last_point(tmp_path, share, verdict):
    pooled, switched, folder = arms(tmp_path, share)
    found = tool.report(pooled, switched, folder, POINTS)
    assert found["switch"] == verdict
    assert found["head to head"][7680]["share"] == pytest.approx(share)
    assert found["head to head"][7680]["deals"] == 20000
    assert f"4. The switch {verdict}" in tool.shown(found, POINTS)


def test_a_draw_is_a_deal_neither_wins(tmp_path):
    folder = tmp_path / "7680"
    written(folder / "arms.plays.json", {"a-vs-each": ["draw", "win"]})
    written(folder / "arms.held.json", {"a-vs-each": ["draw", "loss"]})
    assert tool.deals(folder) == [0.0, 1.0]


def test_the_measures_the_arms_share_are_told_deal_by_deal(tmp_path):
    pooled, switched, folder = arms(tmp_path, 0.50)
    # The same deals, switched winning one game in ten more against the start.
    written(pooled / "7680/pairings.json", {"x-vs-each": ["win", "loss"] * 50})
    written(switched / "7680/pairings.json", {"x-vs-each": ["win", "loss"] * 45 + ["win"] * 10})
    found = tool.report(pooled, switched, folder, POINTS)
    assert set(found["differences"]) == {"pairings"}
    assert found["differences"]["pairings"]["gap"] == pytest.approx(0.05)
    assert "the new decks against the start: +5.0" in tool.shown(found, POINTS)


def test_the_switched_start_plays_the_pilots_as_the_pooled_one(tmp_path):
    pooled, switched, folder = arms(tmp_path, 0.50)
    written(pooled / "0/pilots.json", {"x-vs-pilots": ["win", "loss", "draw"]})
    written(switched / "0/pilots.json", {"x-vs-pilots": ["win", "loss", "draw"]})
    found = tool.report(pooled, switched, folder, POINTS)
    assert found["start"] == {"same": 3, "games": 3}
    written(switched / "0/pilots.json", {"x-vs-pilots": ["win", "win", "draw"]})
    assert tool.report(pooled, switched, folder, POINTS)["start"] == {"same": 2, "games": 3}
