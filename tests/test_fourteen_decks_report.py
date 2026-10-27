"""`tools/fourteen_decks_report.py`: the readings of the network of fourteen
decks, on measurements written as `experiments/commands/fourteen-decks.sh`
leaves them (three decks here, for brevity)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.test_four_decks_report import DEALS, DECKS, pair, pairings, played

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "fourteen_decks_report", ROOT / "tools/fourteen_decks_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

PILOTS = ["blackwing", "burn"]


def won(wins: int) -> list[str]:
    return ["win"] * wins + ["loss"] * (DEALS - wins)


def measured(
    out: Path,
    arm: str = "selfplay",
    *,
    heads: dict[str, int] | None = None,
    share: int = 65,
    weaker: dict[str, int] | None = None,
) -> Path:
    """An arm measured as the script measures it: head to head against each
    champion at `heads` (48 where not named), every pairing against all71
    at `share` (or what `weaker` says of a case)."""
    heads = heads or {}
    for deck in DECKS:
        pair(out / f"heads-{arm}" / deck, heads.get(deck, 48))
        pair(out / "teachers" / deck, 50)
        for point in (4, 8):
            pair(out / f"taught-{deck}" / str(point), 46)
    for point in (16, 20, 22):
        pairings(out / f"{arm}-pairings" / f"{point}.json", share, weaker)
    pairings(out / "copy-quick" / "0.json", 55)
    for point in (2, 22):
        pairings(out / f"{arm}-quick" / f"{point}.json", 60)
    for who, wins in ((arm, 70), ("all71", 60)):
        played(
            out / "pilots" / f"{who}.json",
            {f"{deck}-vs-{pilot}": won(wins) for deck in DECKS for pilot in PILOTS},
        )
    four = {f"{one}-vs-{other}-checkpoint": won(50) for one in tool.FOUR for other in tool.FOUR}
    played(out / "four-decks" / f"{arm}.plays.json", four)
    played(out / "four-decks" / f"{arm}.held.json", four)
    return out


def test_the_mirrors_are_kept_lost_or_not_decided(tmp_path):
    kept = tool.report(measured(tmp_path / "kept", heads=dict.fromkeys(DECKS, 50)), DECKS)
    assert kept["arms"]["selfplay"]["mirrors"]["verdict"] == "kept"
    # One mirror under the floor is lost, whatever the mean.
    under = tool.report(measured(tmp_path / "under", heads={"cyber": 40}), DECKS)
    mirrors = under["arms"]["selfplay"]["mirrors"]
    assert mirrors["under the floor"] == ["cyber"] and mirrors["verdict"] == "lost"
    # Each mirror 2 points under the mark, none under the floor: the mean decides.
    low = tool.report(measured(tmp_path / "low", heads=dict.fromkeys(DECKS, 46)), DECKS)
    assert low["arms"]["selfplay"]["mirrors"]["verdict"] == "lost"
    level = tool.report(measured(tmp_path / "level"), DECKS)["arms"]["selfplay"]["mirrors"]
    assert level["mean"] == pytest.approx(0.48) and level["verdict"] == "not decided"


def test_every_pairing_beats_all71_falls_short_or_is_not_decided(tmp_path):
    beats = tool.report(measured(tmp_path / "beats"), DECKS)["arms"]["selfplay"]["pairings"]
    assert beats["verdict"] == "beats all71" and beats["mean"] == pytest.approx(0.65)
    # Two decks, one seating at 40%: 52.5% over the two, too near the mark.
    near = tool.report(
        measured(tmp_path / "near", weaker={"cyber-vs-lightsworn-checkpoint": 40}), DECKS
    )["arms"]["selfplay"]["pairings"]
    assert near["weakest"] == "cyber|lightsworn"
    assert near["each"]["cyber|lightsworn"]["share"] == pytest.approx(0.525)
    assert near["verdict"] == "not decided"
    short = tool.report(
        measured(tmp_path / "short", weaker={"lightsworn-vs-lightsworn-checkpoint": 30}), DECKS
    )["arms"]["selfplay"]["pairings"]
    assert short["verdict"] == "falls short"


def test_the_two_arms_are_compared_where_both_are_made(tmp_path):
    alone = tool.report(measured(tmp_path / "alone"), DECKS)
    assert "selfplay against recipe" not in alone
    level = measured(tmp_path / "level")
    measured(level, "recipe")
    assert tool.report(level, DECKS)["selfplay against recipe"]["verdict"] == "as good"
    worse = measured(tmp_path / "worse", share=55)
    measured(worse, "recipe")
    found = tool.report(worse, DECKS)["selfplay against recipe"]
    assert found["gap"] == pytest.approx(-0.1) and found["verdict"] == "worse"


def test_beside_the_readings(tmp_path):
    result = tool.report(measured(tmp_path), DECKS)
    arm = result["arms"]["selfplay"]
    assert arm["points"] == [16, 20, 22]
    assert [point["updates"] for point in arm["curve"]] == [0, 2, 22]
    assert arm["curve"][0]["mean"] == pytest.approx(0.55)
    assert arm["pilots"]["mean"] == pytest.approx(0.7)
    assert arm["pilots"]["all71"] == pytest.approx(0.6)
    assert arm["pilots"]["gap"] == pytest.approx(0.1)
    # The same games both ways: each network wins what the other loses.
    assert arm["four decks"]["mean"] == pytest.approx(0.5)
    assert len(arm["four decks"]["by pairing"]) == 10
    copy = result["copy"]["cyber"]
    assert [point["updates"] for point in copy] == [4, 8]
    assert copy[-1]["gap"] == pytest.approx(-0.04) and copy[-1]["teacher"] == pytest.approx(0.5)
    shown = tool.shown(result, DECKS)
    assert "1. The mirrors: not decided." in shown and "2. Against all71: beats all71." in shown
    assert "Hours: the copy not measured; selfplay not measured" in shown
    assert "copy head to head" not in result


def test_the_copy_head_to_head_beside_the_arm(tmp_path):
    out = measured(tmp_path)
    for deck in DECKS:
        pair(out / "heads-taught" / deck, 46)
    result = tool.report(out, DECKS)
    found = result["copy head to head"]
    assert found["mean"] == pytest.approx(0.46)
    # The arm's 48% and the copy's 46% on the same deals.
    assert found["selfplay less the copy"]["gap"] == pytest.approx(0.02)
    assert found["each"]["cyber"]["selfplay less the copy"]["gap"] == pytest.approx(0.02)
    shown = tool.shown(result, DECKS)
    assert "The copy head to head against each champion, the mean 46.0%" in shown
    assert "the selfplay arm less the copy +2.0" in shown
