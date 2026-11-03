"""`tools/pool_batch_report.py` and `tools/pool_batch_suites.py`: a batch of the
expanded pool, its suites and its readings."""

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


tool = loaded("pool_batch_report")
suites = loaded("pool_batch_suites")

POINTS = [4020, 8040]


def written(path: Path, wins: dict[str, float], games: int = 400) -> None:
    """A file of cases that each win a share of their games, spread evenly."""
    suite = {}
    for name, share in wins.items():
        won = round(share * games)
        suite[name] = {"outcomes": ["win"] * won + ["loss"] * (games - won)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}), encoding="utf8")


def mirror(stem: Path, share: float, games: int = 400) -> None:
    """A mirror played both ways, the network winning a share of the deals."""
    won = round(share * games)
    # The champion plays the held games in the learner's seat: it loses what the
    # network wins.
    for kind, first, second in (("plays", "win", "loss"), ("held", "loss", "win")):
        outcomes = [first] * won + [second] * (games - won)
        path = stem.with_name(f"{stem.name}.{kind}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        suite = {"mirror": {"outcomes": outcomes}}
        path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}), encoding="utf8")


def run(tmp_path: Path, gain: float, old: float, mirrors: dict[str, float] | None = None) -> dict:
    written(tmp_path / "0/pairings.json", {"a-vs-each": 0.2, "b-vs-each": 0.3})
    for point in POINTS:
        written(
            tmp_path / f"{point}/pairings.json", {"a-vs-each": 0.2 + gain, "b-vs-each": 0.3 + gain}
        )
        written(tmp_path / f"{point}/old.json", {"old-vs-each": old}, games=4000)
    for deck, share in (mirrors or {}).items():
        mirror(tmp_path / f"{POINTS[-1]}/mirrors/{deck}", share)
    return tool.report(tmp_path, POINTS, "batch-one")


def test_new_decks_that_gain_learn_and_old_decks_that_stay_even_hold(tmp_path):
    found = run(tmp_path, 0.10, 0.51)
    assert found["learns"] == "learns"
    assert found["gains"][8040]["gap"] == pytest.approx(0.10)
    assert found["each"]["a-vs-each"] == pytest.approx(0.10)
    assert found["holds"] == "hold"
    assert "mirrors" not in found


def test_decks_that_do_not_gain_do_not_learn_and_old_decks_can_slip(tmp_path):
    found = run(tmp_path, 0.0, 0.47)
    assert found["learns"] == "does not"
    assert found["holds"] == "slip"


@pytest.mark.parametrize(
    ("mirrors", "verdict"),
    [
        ({"x": 0.53, "y": 0.52}, "kept"),
        ({"x": 0.42, "y": 0.43}, "lost"),
        # A mean over 48% does not keep a mirror at most 45% by two standard errors.
        ({"x": 0.58, "y": 0.38}, "lost"),
        ({"x": 0.49, "y": 0.48}, "not decided"),
    ],
)
def test_the_mirrors_are_read_by_their_mean_and_each_ones_floor(tmp_path, mirrors, verdict):
    assert run(tmp_path, 0.10, 0.51, mirrors)["mirrors"]["verdict"] == verdict


def test_what_is_told_beside_is_told_where_it_was_measured(tmp_path):
    run(tmp_path, 0.10, 0.51, {"x": 0.53, "y": 0.52})
    written(tmp_path / "0/held-out.json", {"c-vs-each": 0.2})
    written(tmp_path / "8040/held-out.json", {"c-vs-each": 0.21})
    written(tmp_path / "0/pilots.json", {"a-vs-pilots": 0.2})
    written(tmp_path / "8040/pilots.json", {"a-vs-pilots": 0.3})
    found = tool.report(tmp_path, POINTS, "batch-one")
    assert found["held-out"]["gap"] == pytest.approx(0.01)
    assert found["pilots"]["gap"] == pytest.approx(0.10)
    shown = tool.shown(found, POINTS)
    assert "1. A new deck learns" in shown
    assert "3. The mirrors kept" in shown
    assert "The next decks, trained on nothing" in shown


def test_a_batch_meets_the_seventy_one_and_the_game_decks_before_it():
    found = suites.suites("one", suites.numbered("game-0013:game-0014"), ["game-0015"])
    pairings = found["pool-batch-one-pairings-checkpoint"]["cases"]
    assert [case["learner"] for case in pairings] == ["game-0013", "game-0014"]
    opponents = pairings[0]["opponent"].split(",")
    assert len(opponents) == len(suites.seventy_one()) + 12
    assert opponents[-12:] == suites.numbered("game-0001:game-0012")
    assert pairings[0]["opponent_policy"] == "checkpoint"
    assert len(found["pool-batch-one-old-pairings-checkpoint"]["cases"]) == len(opponents)
    assert found["pool-batch-one-against-pilots"]["cases"][0]["opponent"] == suites.THIRTEEN


def test_the_suites_of_batch_one_are_the_tools():
    # game-0037 and game-0065 each hold a card that names a card.
    left = {"game-0037", "game-0065"}
    new = [deck for deck in suites.numbered("game-0013:game-0063") if deck not in left]
    held_out = [deck for deck in suites.numbered("game-0064:game-0076") if deck not in left]
    assert len(new) == 50 and len(held_out) == 12
    made = suites.suites("one", new, held_out)
    for name, told in made.items():
        assert (
            json.loads((ROOT / "eval_suites" / f"{name}.json").read_text(encoding="utf8")) == told
        )
    for case in made["pool-batch-one-pairings-checkpoint"]["cases"]:
        assert (ROOT / "data/decks/games" / f"{case['learner']}.ydk").is_file()
