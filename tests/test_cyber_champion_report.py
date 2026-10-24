"""`tools/cyber_champion_report.py`: the readings of a Cyber champion, on
measurements written as `PARTS=measure` of `experiments/commands/cyber-
champion.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cyber_champion_report", ROOT / "tools/cyber_champion_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

COPIED = 4


def outcomes(won: int, games: int) -> list[str]:
    return ["win"] * won + ["loss"] * (games - won)


def suite(path: Path, won_by_step: dict[int | None, int], games: int = 100) -> None:
    """A file of the yardstick: Cyber against two decks, so many games won by
    each checkpoint, and another deck's case beside them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for step, won in won_by_step.items():
        told = outcomes(won, games)
        half = games // 2
        cases = {
            "cyber-vs-burn": {"games": half, "outcomes": told[:half]},
            "cyber-vs-monarch": {"games": half, "outcomes": told[half:]},
            "zombie-vs-burn": {"games": 2, "outcomes": ["win", "win"]},
        }
        records.append({"checkpoint_step": step, "evaluation_suite": cases})
    path.write_text(json.dumps({"seed": 2012, "records": records}))


def mirrored(path: Path, won: int, deals: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {"cyber-vs-checkpoint": {"games": deals, "outcomes": outcomes(won, deals)}}
    path.write_text(json.dumps({"seed": 2012, "records": [{"evaluation_suite": case}]}))


def measured(tmp_path: Path, mirror: int, new: dict[int, int], old: dict[int, int]) -> Path:
    out = tmp_path / "out"
    suite(out / "yardstick" / "cyber-new.json", {COPIED + point: won for point, won in new.items()})
    suite(out / "yardstick" / "cyber-old.json", {COPIED + point: won for point, won in old.items()})
    for stem in ("model-00483", "model-00603", "model-00723"):
        mirrored(out / "mirror" / f"{stem}.plays.json", mirror)
        # all71 in the learner's seat loses what the checkpoint wins.
        mirrored(out / "mirror" / f"{stem}.held.json", 400 - mirror)
    suite(tmp_path / "references" / "new" / "all71-10560.json", {27360: 57, 27480: 58, 27600: 57})
    suite(tmp_path / "references" / "new" / "pilot.json", {None: 19})
    return out


@pytest.mark.parametrize(
    ("won", "verdict"), [(240, "a champion"), (205, "not told apart"), (160, "weaker")]
)
def test_the_first_reading_takes_every_deal_from_both_seats(tmp_path, won, verdict):
    out = measured(tmp_path, won, {120: 50, 720: 60}, {120: 40, 720: 50})
    result = tool.report(out, tmp_path / "references", copied=COPIED)
    assert result["mirror"]["verdict"] == verdict
    assert result["mirror"]["share"] == pytest.approx(won / 400)
    assert result["mirror"]["checkpoints"] == 3
    assert f"1. A champion: {verdict}." in tool.shown(result)


def test_the_second_reading_pairs_the_same_games_at_both_points(tmp_path):
    out = measured(tmp_path, 200, {120: 60, 240: 62, 720: 52}, {120: 40, 240: 41, 720: 50})
    result = tool.report(out, tmp_path / "references", copied=COPIED)
    assert result["matters"][120]["verdict"] == "matters"
    assert result["matters"][120]["gain"] == pytest.approx(0.2)
    assert result["matters"][720]["verdict"] == "not shown"
    assert result["curves"]["cyber-new"] == {120: 0.6, 240: 0.62, 720: 0.52}
    assert result["references"]["all71"] == pytest.approx(172 / 300)
    assert result["references"]["pilot as pinned"] == pytest.approx(0.19)


def test_both_arms_are_read_at_both_points(tmp_path):
    out = measured(tmp_path, 200, {120: 60}, {120: 40, 720: 50})
    with pytest.raises(SystemExit, match="after 720 updates"):
        tool.report(out, tmp_path / "references", copied=COPIED)
