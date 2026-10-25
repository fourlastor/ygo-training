"""`tools/four_decks_report.py`: the readings of the network of four decks,
on measurements written as `experiments/commands/four-decks.sh` leaves
them (three decks here, for brevity)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "four_decks_report", ROOT / "tools/four_decks_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DECKS = ["blackwing", "cyber", "lightsworn"]
DEALS = 100


def played(path: Path, cases: dict[str, list[str]], seconds: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = {
        name: {"games": len(outcomes), "outcomes": outcomes} for name, outcomes in cases.items()
    }
    path.write_text(json.dumps({"seconds": seconds, "records": [{"evaluation_suite": suite}]}))


def pair(stem: Path, share: int) -> None:
    """The first network wins `share` of the deals' games in all."""
    played(
        stem.with_name(f"{stem.name}.plays.json"),
        {"one": ["win"] * share + ["loss"] * (DEALS - share)},
    )
    played(
        stem.with_name(f"{stem.name}.held.json"),
        {"one": ["win"] * (DEALS - share) + ["loss"] * share},
    )


def pairings(
    path: Path, share: int, weaker: dict[str, int] | None = None, seconds: float = 0.0
) -> None:
    """Every ordered pairing, the learner winning `share` of each case's
    games, or what `weaker` says for a case."""
    weaker = weaker or {}
    cases = {}
    for one in DECKS:
        for other in DECKS:
            name = f"{one}-vs-{other}-checkpoint"
            wins = weaker.get(name, share)
            cases[name] = ["win"] * wins + ["loss"] * (DEALS - wins)
    played(path, cases, seconds)


def measured(
    tmp_path: Path, *, copy: int = 50, weighted: int = 60, uniform: int = 60, four: int = 61
) -> tuple[Path, Path]:
    out, distilled = tmp_path / "out", tmp_path / "distilled"
    for deck in DECKS:
        pair(out / "teachers" / deck, 50)
        for point in (8, 9, 10):
            pair(out / f"taught-{deck}" / str(point), copy)
        pair(out / f"taught-curve-{deck}" / "4", 40)
        for who in ("taught", "uniform", "weighted"):
            pair(out / f"heads-{who}" / deck, 48)
    for point in (20, 22, 24):
        pairings(out / "uniform-pairings" / f"{point}.json", uniform)
        pairings(out / "weighted-pairings" / f"{point}.json", weighted)
    for point in (2, 4):
        pairings(
            out / "uniform-quick" / f"{point}.json", 55, {"cyber-vs-lightsworn-checkpoint": 30}
        )
    for point in (0, 2, 4):
        pairings(out / "weighted-weights" / f"{point}.json", 52, seconds=360.0)
        (out / "weighted-weights" / f"{point}.shares.json").write_text(
            json.dumps({"weights": {f"{one}>{other}": 1 / 9 for one in DECKS for other in DECKS}})
        )
    pair(out / "uniform-cross" / "16", four)
    pair(distilled / "240", 61)
    return out, distilled


def read(tmp_path: Path, **made) -> dict:
    out, distilled = measured(tmp_path, **made)
    return tool.report(out, DECKS, distilled, 16)


def test_the_copy_keeps_each_champions_mirror_or_not(tmp_path):
    result = read(tmp_path, copy=40)
    blackwing = result["mirrors"]["blackwing"]
    assert (blackwing["copy"], blackwing["its teacher"]) == (pytest.approx(0.4), 0.5)
    assert blackwing["verdict"] == "loses strength"
    assert read(tmp_path, copy=50)["mirrors"]["cyber"]["verdict"] == "keeps"


def test_weighting_pays_costs_or_is_not_told_apart_by_the_mean_over_pairings(tmp_path):
    paid = read(tmp_path, weighted=70, uniform=60)["weighting"]
    assert (paid["weighted"], paid["uniform"]) == (pytest.approx(0.7), pytest.approx(0.6))
    assert paid["gap"] == pytest.approx(0.1) and paid["verdict"] == "pays"
    assert set(paid["by pairing"]) == {f"{one}|{other}" for one, other in tool.pairings(DECKS)}
    cost = read(tmp_path, weighted=50, uniform=60)["weighting"]
    assert cost["verdict"] == "costs"
    level = read(tmp_path, weighted=60, uniform=60)["weighting"]
    assert level["gap"] == 0 and level["verdict"] == "not told apart"
    # The rule at its edges: a point, by two standard errors.
    assert tool.verdict(0.015, 0.002, tool.PAYS, ("pays", "costs", "no")) == "pays"
    assert tool.verdict(0.015, 0.003, tool.PAYS, ("pays", "costs", "no")) == "no"


def test_four_decks_against_two_for_one_pairing(tmp_path):
    helped = read(tmp_path, four=70)["sharing"]
    assert (helped["four decks"], helped["two decks"]) == (pytest.approx(0.7), pytest.approx(0.61))
    assert helped["verdict"] == "helps"
    assert read(tmp_path, four=50)["sharing"]["verdict"] == "hurts"
    assert read(tmp_path, four=61)["sharing"]["verdict"] == "not told apart"


def test_beside_the_curves_the_weights_the_copy_head_to_head_and_the_hours(tmp_path):
    result = read(tmp_path)
    uniform = result["beside"]["curves"]["uniform"]
    # The copy's measurement starts both curves.
    assert [point["updates"] for point in uniform] == [0, 2, 4]
    assert uniform[1]["weakest"] == "cyber|lightsworn"
    assert [point["updates"] for point in result["beside"]["curves"]["weighted"]] == [0, 2, 4]
    assert sorted(result["beside"]["weights"]) == [0, 2, 4]
    copy = result["beside"]["the copy along its updates"]["lightsworn"]
    assert [point["updates"] for point in copy] == [4, 8, 9, 10]
    assert copy[0]["over its teacher"]["gap"] == pytest.approx(-0.1)
    assert result["beside"]["head to head"]["weighted"]["cyber"] == pytest.approx(0.48)
    assert result["hours"]["the weighted arm measuring"] == pytest.approx(0.3)
    assert result["hours"]["taught"] is None
    shown = tool.shown(result)
    assert (
        "2. Weighting by weakness: not told apart" in shown and "Weights before update 0" in shown
    )
