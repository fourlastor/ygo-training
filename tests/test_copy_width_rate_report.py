"""`tools/copy_width_rate_report.py`: the copy's width and rate at seventy-one decks."""

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


tool = loaded("copy_width_rate_report")
DECKS = ["blackwing", "zombie"]


def written(
    path: Path, rows: list[dict[str, float]], first: int = 0, seconds: float = 10.0
) -> None:
    """A run's metrics: each row the decks' shares of an update, from version ``first``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "version": first + index,
                "imitation_model_teacher_agreement_by_deck": shares,
                "imitation_samples_by_deck": {deck: 100 for deck in shares},
                "wall_seconds": seconds,
            }
        )
        for index, shares in enumerate(rows)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def runs(tmp_path: Path, gaps: dict[str, float], updates: int = 40) -> Path:
    """control at 0.90 on each deck; each arm so many points off it on every update."""
    written(tmp_path / "control.metrics.jsonl", [dict.fromkeys(DECKS, 0.90)] * updates)
    for arm in ("wide", "narrow"):
        share = 0.90 + gaps.get(arm, 0.0)
        written(tmp_path / f"{arm}.metrics.jsonl", [dict.fromkeys(DECKS, share)] * updates)
    # slow begins halfway, from control's checkpoint.
    share = 0.90 + gaps.get("slow", 0.0)
    written(
        tmp_path / "slow.metrics.jsonl",
        [dict.fromkeys(DECKS, share)] * (updates // 2),
        first=updates // 2,
    )
    return tmp_path


@pytest.mark.parametrize(
    ("gap", "verdict"),
    [(0.005, "fits better"), (-0.005, "fits worse"), (0.001, "not a lever at this length")],
)
def test_each_arm_is_read_against_control_over_the_window(tmp_path, gap, verdict):
    directory = runs(tmp_path, {"wide": gap, "narrow": gap, "slow": gap})
    found = tool.report(directory, 21, 40, 5, DECKS)
    for arm in ("wide", "narrow", "slow"):
        assert found["arms"][arm]["reading"] == verdict
        assert found["arms"][arm]["gap"] == pytest.approx(gap)
        assert found["arms"][arm]["stretches"] == 4
    assert f"1. wide {verdict}" in tool.shown(found)


def test_the_error_comes_from_the_stretches(tmp_path):
    written(tmp_path / "control.metrics.jsonl", [dict.fromkeys(DECKS, 0.90)] * 20)
    # wide's lead swings between stretches of five: +0.5 points in two, +0.1 in two.
    swings = [0.905] * 5 + [0.901] * 5 + [0.905] * 5 + [0.901] * 5
    written(tmp_path / "wide.metrics.jsonl", [dict.fromkeys(DECKS, s) for s in swings])
    for arm in ("narrow", "slow"):
        written(tmp_path / f"{arm}.metrics.jsonl", [dict.fromkeys(DECKS, 0.90)] * 20)
    found = tool.report(tmp_path, 1, 20, 5, DECKS)
    told = found["arms"]["wide"]
    assert told["gap"] == pytest.approx(0.003)
    assert told["error"] == pytest.approx(0.002 / 2 / 3**0.5 * 2, rel=1e-6)
    # +0.30 ± 0.12 points: under the mark of 0.25 by two errors.
    assert told["reading"] == "not a lever at this length"


def test_each_deck_counts_alike_and_a_deck_without_decisions_not_at_all(tmp_path):
    rows = [{"blackwing": 0.80, "zombie": 1.00}] * 20
    written(tmp_path / "control.metrics.jsonl", rows)
    for arm in ("wide", "narrow", "slow"):
        written(tmp_path / f"{arm}.metrics.jsonl", rows)
    path = tmp_path / "control.metrics.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    for line in lines:
        line["imitation_samples_by_deck"] = {"blackwing": 1000, "zombie": 1}
        line["imitation_model_teacher_agreement_by_deck"]["cyber"] = 0.0
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf8")
    found = tool.report(tmp_path, 1, 20, 5, None)
    assert found["control"] == pytest.approx(0.90)


def test_the_fourteen_decks_are_told_beside_their_copy_at_as_many_updates_a_deck(tmp_path):
    directory = runs(tmp_path, {})
    # The fourteen decks' copy at 0.80 to its 140th update and at 0.95 after: two decks read
    # from their 21st update to their 40th are ten to twenty updates a deck, where that
    # copy is from its 141st to its 280th.
    written(tmp_path / "fourteen.jsonl", [{"blackwing": 0.80}] * 140 + [{"blackwing": 0.95}] * 140)
    found = tool.report(directory, 21, 40, 5, DECKS, fourteen=tmp_path / "fourteen.jsonl")
    beside = found["the fourteen decks"]
    assert beside["control"] == pytest.approx(0.90)
    assert beside["fourteen decks' copy"] == pytest.approx(0.95)


def test_control_is_told_against_the_copy_it_plays_again(tmp_path):
    directory = runs(tmp_path, {})
    written(tmp_path / "reference.jsonl", [dict.fromkeys(DECKS, 0.899)] * 40)
    found = tool.report(directory, 21, 40, 5, DECKS, reference=tmp_path / "reference.jsonl")
    told = found["control against the reference"]
    assert told["gap"] == pytest.approx(0.001)
    assert told["updates"] == 40
    assert "control less seventy-one-champions' copy, update by update: +0.10" in tool.shown(found)


def test_a_run_short_of_the_window_is_refused(tmp_path):
    directory = runs(tmp_path, {}, updates=30)
    with pytest.raises(SystemExit, match="missing"):
        tool.report(directory, 21, 40, 5, DECKS)


def test_a_reading_after_forty_updates_a_deck_takes_its_own_mark(tmp_path):
    # +0.30 points: a lever at the first mark (0.25), not at the second (0.38).
    directory = runs(tmp_path, {"wide": 0.003, "narrow": 0.003, "slow": 0.003})
    assert tool.report(directory, 21, 40, 5, DECKS)["arms"]["wide"]["reading"] == "fits better"
    found = tool.report(directory, 21, 40, 5, DECKS, mark=0.0038)
    assert found["mark"] == 0.0038
    assert found["arms"]["wide"]["reading"] == "not a lever at this length"


def suite(path: Path, games: list[str]) -> None:
    """A suite of one case, its games' outcomes for the network in the learner's seat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    told = {"case": {"outcomes": games, "by_opponent": {}}}
    path.write_text(json.dumps({"records": [{"evaluation_suite": told}]}), encoding="utf8")


def heads(folder: Path, won: dict[str, int], deals: int = 10, decks: list[str] = DECKS) -> Path:
    """Each run wins the first so many deals of each deck from both seats, and loses the rest."""
    for deck in decks:
        for run, count in won.items():
            outcomes = ["win"] * count + ["loss"] * (deals - count)
            suite(folder / deck / f"{run}.plays.json", outcomes)
            # The teacher plays these in the learner's seat: it loses what the run wins.
            flipped = ["loss" if game == "win" else "win" for game in outcomes]
            suite(folder / deck / f"{run}.held.json", flipped)
    return folder


def test_each_run_is_told_head_to_head_against_the_teachers_and_each_arm_less_control(tmp_path):
    folder = heads(tmp_path / "heads", {"control": 4, "wide": 6, "narrow": 3, "slow": 4})
    each = tool.heads(folder, DECKS)
    assert each["control"]["share"] == pytest.approx(0.4)
    assert each["control"]["deals"] == 20
    assert each["wide"]["share"] == pytest.approx(0.6)
    # wide wins every deal control wins and two more of each deck's ten.
    assert each["wide"]["less control"]["gap"] == pytest.approx(0.2)
    assert each["narrow"]["less control"]["gap"] == pytest.approx(-0.1)
    assert each["slow"]["less control"] == {"gap": 0.0, "error": 0.0}


def test_the_head_to_head_is_told_beside_the_readings(tmp_path):
    directory = runs(tmp_path, {"wide": 0.003})
    won = {"control": 4, "wide": 6, "narrow": 3, "slow": 4}
    folder = heads(tmp_path / "heads", won, decks=tool.FOURTEEN)
    found = tool.report(directory, 21, 40, 5, DECKS, against_teachers=folder)
    assert found["against the teachers"]["wide"]["share"] == pytest.approx(0.6)
    shown = tool.shown(found)
    assert "Head to head against the fourteen decks' teachers in their mirrors, 140 deals" in shown
    assert "less control, deal by deal: wide +20.00 ± 3.39 points" in shown


def test_a_deck_without_its_head_to_head_is_refused(tmp_path):
    folder = heads(tmp_path / "heads", {"control": 4, "wide": 6, "narrow": 3, "slow": 4})
    with pytest.raises(SystemExit, match="Missing"):
        tool.heads(folder, [*DECKS, "cyber"])
