"""Duels through the environment's shared library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ygo_training.environment import DuelError, Environment, library_path

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]

DUELS = [(7, ["blackwing", "monarch"]), (8, ["monarch", "blackwing"]), (11, ["heroes", "spellcaster"])]


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


def play(environment: Environment, setup: dict[str, Any], decide=None):
    """Run a duel to its end: observed decisions, the caller's own, the end."""
    samples, decisions = [], []
    with environment.start(setup) as episode:
        while True:
            event = episode.advance()
            samples += event["samples"]
            if event["decision"] is not None:
                decisions.append(event["decision"])
                episode.choose(decide(event["decision"]))
            if event["over"] is not None:
                return samples, decisions, event["over"]


def pilots(seed: int, decks: list[str], **more: Any) -> dict[str, Any]:
    return {"seed": seed, "decks": decks, "seats": [{"control": "pilot"}, {"control": "pilot"}], **more}


def outcome(over: dict[str, Any]) -> tuple:
    return over["digest"], over["winner"], over["turns"], over["decisions"]


def test_the_pool_names_each_policy_with_its_deck(environment):
    blackwing = environment.decks["blackwing"]
    assert blackwing["deck"] == "Blackwing Assassin"
    assert len(blackwing["main"]) >= 40
    assert environment.decklist("blackwing") == [*blackwing["main"], *blackwing["extra"]]
    assert environment.vocabulary["decision"]["Idle"] == 1
    assert environment.vocabulary["choice"]["Cancel"] == 23


@pytest.mark.parametrize("seed,decks", DUELS)
def test_a_duel_is_the_same_every_time(environment, seed, decks):
    first = play(environment, pilots(seed, decks))[2]
    second = play(environment, pilots(seed, decks))[2]
    assert first["winner"] in (0, 1, 2)
    assert outcome(first) == outcome(second)
    assert outcome(play(environment, pilots(seed + 1, decks))[2]) != outcome(first)


@pytest.mark.parametrize("seed,decks", DUELS)
@pytest.mark.parametrize("caller", [0, 1])
def test_a_caller_that_follows_its_pilot_plays_the_pilots_duel(environment, seed, decks, caller):
    expected = play(environment, pilots(seed, decks))[2]
    seats = [{"control": "pilot"}, {"control": "pilot"}]
    seats[caller] = {"control": "caller"}
    _, decisions, over = play(
        environment, {"seed": seed, "decks": decks, "seats": seats}, lambda decision: decision["label"]
    )
    assert outcome(over) == outcome(expected)
    place = environment.vocabulary["decision"]["Place"]
    cancel = environment.vocabulary["choice"]["Cancel"]
    shuffle = environment.vocabulary["choice"]["ShuffleHand"]
    assert decisions
    for decision in decisions:
        assert decision["player"] == caller
        assert len(decision["choices"]) > 1
        assert decision["question"]["kind"] != place
        assert all(choice["kind"] not in (cancel, shuffle) for choice in decision["choices"])
        assert decision["label"] == decision["pilot"]
        assert decision["chosen"] is None


@pytest.mark.parametrize("seed,decks", DUELS)
def test_a_record_plays_again_with_the_same_decisions(environment, seed, decks):
    setup = pilots(seed, decks, keep_record=True)
    setup["seats"][1]["observe"] = True
    samples, _, over = play(environment, setup)
    assert samples and all(sample["player"] == 1 for sample in samples)
    again = {
        "seed": seed,
        "decks": decks,
        "seats": [{"control": "record", "alone": True}, {"control": "record", "observe": True}],
        "record": over["record"],
    }
    replayed, _, end = play(environment, again)
    assert outcome(end) == outcome(over)
    assert end["departures"] == 0
    assert replayed == samples


@pytest.mark.parametrize("seed,decks", DUELS)
def test_a_face_down_card_of_the_other_player_has_no_identity(environment, seed, decks):
    setup = pilots(seed, decks)
    for seat in setup["seats"]:
        seat["observe"] = True
    samples, _, _ = play(environment, setup)
    face_down = {
        environment.vocabulary["position"]["FaceDownAttack"],
        environment.vocabulary["position"]["FaceDownDefense"],
    }
    hand = environment.vocabulary["location"]["Hand"]
    hidden = 0
    for sample in samples:
        for card in sample["cards"]:
            if card["controller"] == 1 and card["position"] in face_down:
                hidden += 1
                assert (card["code"], card["attack"], card["defense"], card["level"]) == (0, 0, 0, 0)
            if card["controller"] == 0 and (card["position"] or card["location"] == hand):
                assert card["code"] != 0
    assert hidden > 0


def test_a_choice_that_is_not_offered_is_refused(environment):
    setup = {
        "seed": 7,
        "decks": ["blackwing", "monarch"],
        "seats": [{"control": "caller"}, {"control": "pilot"}],
    }
    with environment.start(setup) as episode:
        decision = episode.advance()["decision"]
        with pytest.raises(DuelError, match="offers"):
            episode.choose(len(decision["choices"]))
        with pytest.raises(DuelError, match="not been answered"):
            episode.advance()
        episode.choose(decision["label"])
        assert episode.advance()["over"] is None


def test_an_unknown_deck_is_refused(environment):
    with pytest.raises(DuelError, match="no deck"):
        environment.start(pilots(7, ["blackwing", "no-such-deck"]))
