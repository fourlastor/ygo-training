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
        suite[name] = {"outcomes": ["win"] * won + ["loss"] * (games - won), "by_opponent": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}), encoding="utf8")


def pairings(path: Path, table: dict[str, dict[str, float]], games: int = 400) -> None:
    """A file of old pairings: each case's share of its games against each opponent."""
    suite = {}
    for name, against in table.items():
        outcomes, by_opponent = [], {}
        for opponent, share in against.items():
            won = round(share * games)
            outcomes += ["win"] * won + ["loss"] * (games - won)
            by_opponent[opponent] = {"wins": won, "games": games}
        suite[name] = {"outcomes": outcomes, "by_opponent": by_opponent}
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
        record = {"evaluation_suite": suite, "evaluation": {"outcomes": outcomes}}
        path.write_text(json.dumps({"records": [record]}), encoding="utf8")


def run(
    tmp_path: Path,
    gain: float,
    old: float,
    mirrors: dict[str, float] | None = None,
    old_points: list[int] = POINTS,
    **readings,
) -> dict:
    written(tmp_path / "0/pairings.json", {"a-vs-each": 0.2, "b-vs-each": 0.3})
    for point in POINTS:
        written(
            tmp_path / f"{point}/pairings.json", {"a-vs-each": 0.2 + gain, "b-vs-each": 0.3 + gain}
        )
    for point in old_points:
        written(tmp_path / f"{point}/old.json", {"old-vs-each": old}, games=4000)
    for deck, share in (mirrors or {}).items():
        mirror(tmp_path / f"{POINTS[-1]}/mirrors/{deck}", share)
    return tool.report(tmp_path, POINTS, "batch-one", **readings)


def network(path: Path, mirrors: dict[str, float]) -> Path:
    """Another network's mirrors, as a check leaves them."""
    for deck, share in mirrors.items():
        mirror(path / deck, share)
    return path


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


@pytest.mark.parametrize(
    ("mirrors", "verdict"),
    [
        # A mirror under 45% does not lose them where the reference plays it as low.
        ({"x": 0.38, "y": 0.60}, "kept"),
        ({"x": 0.37, "y": 0.60}, "kept"),
        ({"x": 0.36, "y": 0.58}, "not decided"),
        ({"x": 0.30, "y": 0.52}, "lost"),
    ],
)
def test_from_batch_three_the_mirrors_are_read_against_a_reference_deal_by_deal(
    tmp_path, mirrors, verdict
):
    reference = network(tmp_path / "reference", {"x": 0.38, "y": 0.60})
    found = run(tmp_path, 0.10, 0.51, mirrors, reference=reference)
    moved = found["mirrors"]["against"]
    expected = (mirrors["x"] - 0.38 + mirrors["y"] - 0.60) / 2
    assert moved["difference"] == pytest.approx(expected)
    assert moved["decks"] == 2
    assert found["mirrors"]["verdict"] == verdict
    assert "under the floor" not in found["mirrors"]
    assert f"3. The mirrors {verdict}: {100 * expected:+.1f}" in tool.shown(found, POINTS)


def test_the_mirrors_are_told_beside_against_another_network(tmp_path):
    reference = network(tmp_path / "reference", {"x": 0.50, "y": 0.50})
    before = network(tmp_path / "before", {"x": 0.55, "y": 0.55})
    found = run(tmp_path, 0.10, 0.51, {"x": 0.50, "y": 0.51}, reference=reference, beside=before)
    assert found["mirrors"]["verdict"] == "kept"
    assert found["mirrors"]["beside"]["difference"] == pytest.approx(-0.045)
    shown = tool.shown(found, POINTS)
    assert f"deal by deal against {before}: -4.5" in shown
    assert "The mirrors against their champions: 50.5%" in shown


def test_the_old_decks_can_be_measured_at_the_last_point_alone(tmp_path):
    found = run(tmp_path, 0.10, 0.51, old_points=POINTS[-1:])
    assert list(found["old"]) == POINTS[-1:]
    assert found["holds"] == "hold"
    assert "The old decks along it" not in tool.shown(found, POINTS)
    assert "The old decks along it" in tool.shown(run(tmp_path, 0.10, 0.51), POINTS)


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


def test_the_suites_of_the_replay_are_the_tools():
    # From seventy-one-champions: the first batch test's and the first batch's decks new,
    # the seventy-one alone old.
    left = {"game-0037", "game-0065"}
    new = [deck for deck in suites.numbered("game-0001:game-0063") if deck not in left]
    held_out = [deck for deck in suites.numbered("game-0064:game-0076") if deck not in left]
    assert len(new) == 62 and len(held_out) == 12
    made = suites.suites("replay", new, held_out)
    for name, told in made.items():
        assert (
            json.loads((ROOT / "eval_suites" / f"{name}.json").read_text(encoding="utf8")) == told
        )
    old = made["pool-batch-replay-pairings-checkpoint"]["cases"][0]["opponent"].split(",")
    assert old == suites.seventy_one()


def test_the_suites_of_batch_two_are_the_tools():
    # From distilled: the batch again's sixty-two game Decks old with the seventy-one;
    # game-0037, game-0065 and game-0093 each hold a card that names a card.
    left = frozenset({"game-0037", "game-0065", "game-0093"})
    new = [deck for deck in suites.numbered("game-0064:game-0115") if deck not in left]
    held_out = suites.numbered("game-0116:game-0127")
    assert len(new) == 50 and len(held_out) == 12
    made = suites.suites("two", new, held_out, left)
    for name, told in made.items():
        assert (
            json.loads((ROOT / "eval_suites" / f"{name}.json").read_text(encoding="utf8")) == told
        )
    old = made["pool-batch-two-old-pairings-checkpoint"]["cases"]
    assert len(old) == 133
    assert [case["learner"] for case in old][71:] == [
        deck for deck in suites.numbered("game-0001:game-0063") if deck not in left
    ]
    for case in (
        made["pool-batch-two-pairings-checkpoint"]["cases"]
        + made["pool-batch-two-held-out-checkpoint"]["cases"]
    ):
        assert (ROOT / "data/decks/games" / f"{case['learner']}.ydk").is_file()


def test_the_suites_of_batch_three_are_the_tools():
    # From batch two: its fifty game Decks old too; game-0167 and game-0171 among the
    # held out hold a card that names a card, as three among the old do.
    left = frozenset({"game-0037", "game-0065", "game-0093", "game-0167", "game-0171"})
    new = suites.numbered("game-0116:game-0165")
    held_out = [deck for deck in suites.numbered("game-0166:game-0179") if deck not in left]
    assert len(new) == 50 and len(held_out) == 12
    made = suites.suites("three", new, held_out, left)
    for name, told in made.items():
        assert (
            json.loads((ROOT / "eval_suites" / f"{name}.json").read_text(encoding="utf8")) == told
        )
    old = [case["learner"] for case in made["pool-batch-three-old-pairings-checkpoint"]["cases"]]
    assert len(old) == 183
    assert old[71:] == [deck for deck in suites.numbered("game-0001:game-0115") if deck not in left]
    for case in (
        made["pool-batch-three-pairings-checkpoint"]["cases"]
        + made["pool-batch-three-held-out-checkpoint"]["cases"]
    ):
        assert (ROOT / "data/decks/games" / f"{case['learner']}.ydk").is_file()


def test_the_old_decks_are_told_apart_by_the_decks_of_each_pairing(tmp_path):
    found = run(tmp_path, 0.10, 0.51)
    assert "old apart" not in found
    # The seventy-one win most of their games against a game Deck whichever network
    # plays them, which a deck's own share against all the old decks carries; the
    # network gains four points where one of each meets, from either side.
    pairings(
        tmp_path / f"{POINTS[-1]}/old.json",
        {
            "agents-vs-each": {"agents": 0.50, "zombie": 0.50, "game-0001": 0.74},
            "zombie-vs-each": {"agents": 0.50, "zombie": 0.50, "game-0001": 0.74},
            "game-0001-vs-each": {"agents": 0.34, "zombie": 0.34, "game-0001": 0.47},
        },
    )
    found = tool.report(tmp_path, POINTS, "batch-one")
    apart = found["old apart"]
    assert apart["seventy-one"]["share"] == pytest.approx(0.50)
    assert apart["game decks"]["share"] == pytest.approx(0.47)
    assert apart["one of each"]["share"] == pytest.approx(0.54)
    assert apart["one of each"]["deals"] == 4 * 400
    shown = tool.shown(found, POINTS)
    assert "both of the seventy-one 50.0%" in shown
    assert "both game Decks 47.0%" in shown
    assert "one of each 54.0%" in shown
