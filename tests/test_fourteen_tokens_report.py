"""`tools/fourteen_tokens_report.py`: the board as its cards from scratch at fourteen decks."""

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


tool = loaded("fourteen_tokens_report")
LAST = [4000, 4240, 4320]


def written(path: Path, suite: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    told = {name: {"outcomes": games, "by_opponent": {}} for name, games in suite.items()}
    path.write_text(json.dumps({"records": [{"evaluation_suite": told}]}), encoding="utf8")


def split(games: list[str]) -> dict[str, list[str]]:
    """The games over two cases, as a suite of pairings holds them."""
    half = len(games) // 2
    return {"b-vs-a-checkpoint": games[half:], "a-vs-b-checkpoint": games[:half]}


def head_to_head(folder: Path, points: list[int], share: float, deals: int = 20000) -> None:
    """tokens wins a share of the deals at each point."""
    won = round(share * deals)
    for point in points:
        written(folder / f"{point}.plays.json", split(["win"] * won + ["loss"] * (deals - won)))
        # pooled plays these in the learner's seat: it loses what tokens wins.
        written(folder / f"{point}.held.json", split(["loss"] * won + ["win"] * (deals - won)))


def against_all71(folder: Path, share: float, deals: int = 1000) -> None:
    won = round(share * deals)
    for point in LAST:
        written(folder / f"{point}.json", split(["win"] * won + ["loss"] * (deals - won)))


def measured(tmp_path: Path, share: float) -> Path:
    head_to_head(tmp_path / "heads/last", LAST, share)
    head_to_head(tmp_path / "heads/copy", [2240], 0.50)
    head_to_head(tmp_path / "heads/1360", [1360], 0.52)
    against_all71(tmp_path / "pooled-all71", 0.60)
    against_all71(tmp_path / "tokens-all71", 0.62)
    return tmp_path


@pytest.mark.parametrize(
    ("share", "verdict"), [(0.53, "stronger"), (0.505, "not decided"), (0.47, "weaker")]
)
def test_tokens_is_read_head_to_head_at_the_last_three_checkpoints(tmp_path, share, verdict):
    found = tool.report(measured(tmp_path, share), LAST, ["copy", "1360"])
    assert found["reading"] == verdict
    assert found["head to head"]["share"] == pytest.approx(share)
    # A deal's games at the three checkpoints are one deal.
    assert found["head to head"]["deals"] == 20000
    assert f"1. The board as its cards from scratch {verdict}" in tool.shown(found, LAST)


def test_a_deal_is_read_over_its_points_and_a_draw_is_won_by_neither(tmp_path):
    folder = tmp_path / "heads/last"
    written(folder / "4000.plays.json", {"a": ["win", "draw"]})
    written(folder / "4000.held.json", {"a": ["loss", "draw"]})
    written(folder / "4240.plays.json", {"a": ["loss", "win"]})
    written(folder / "4240.held.json", {"a": ["win", "loss"]})
    assert tool.heads(folder) == [0.5, 0.5]


def test_the_copies_and_the_points_along_are_told_beside(tmp_path):
    found = tool.report(measured(tmp_path, 0.53), LAST, ["copy", "1360"])
    assert found["along"]["copy"]["share"] == pytest.approx(0.50)
    assert found["along"]["1360"]["share"] == pytest.approx(0.52)
    text = tool.shown(found, LAST)
    assert "Head to head at the copies: 50.0%" in text
    assert "Head to head after 1,360 updates of PPO: 52.0%" in text


def test_each_arm_against_all71_and_their_gap_deal_by_deal(tmp_path):
    found = tool.report(measured(tmp_path, 0.53), LAST, [])
    assert found["against all71"]["pooled"]["share"] == pytest.approx(0.60)
    assert found["tokens less pooled against all71"]["gap"] == pytest.approx(0.02)
    assert "tokens less pooled against all71, deal by deal: +2.0" in tool.shown(found, LAST)


def test_pooled_is_told_beside_the_quarter_arm_on_the_same_deals(tmp_path):
    against_all71(tmp_path / "quarter", 0.59)
    found = tool.report(measured(tmp_path, 0.53), LAST, [], tmp_path / "quarter")
    assert found["pooled less quarter against all71"]["gap"] == pytest.approx(0.01)
    assert "pooled less it, deal by deal: +1.0" in tool.shown(found, LAST)


def test_the_points_of_a_measure_must_play_the_same_deals(tmp_path):
    folder = tmp_path / "heads/last"
    written(folder / "4000.plays.json", {"a": ["win", "loss"]})
    written(folder / "4000.held.json", {"a": ["loss", "win"]})
    written(folder / "4240.plays.json", {"a": ["win"]})
    written(folder / "4240.held.json", {"a": ["loss"]})
    with pytest.raises(SystemExit, match="other deals"):
        tool.heads(folder)


def test_the_copies_imitation_and_the_hours_are_told(tmp_path):
    directory = measured(tmp_path, 0.53)
    lines = [
        {"wall_seconds": 10.0, "imitation_loss": 0.2, "imitation_model_teacher_agreement": 0.9}
    ]
    lines += [
        {"wall_seconds": 14.0, "imitation_loss": 0.1, "imitation_model_teacher_agreement": 0.95}
    ] * 80
    (directory / "tokens-copy.metrics.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf8"
    )
    found = tool.report(directory, LAST, [])
    copy = found["runs"]["tokens-copy"]
    # The imitation is told over the last 80 updates, the hours over all.
    assert copy["imitation_loss"] == pytest.approx(0.1)
    assert copy["hours"] == pytest.approx((10.0 + 14.0 * 80) / 3600)
    assert found["runs"]["pooled"]["hours"] is None
    assert "tokens-copy: imitation loss 0.100, agreement 95.0%" in tool.shown(found, LAST)
