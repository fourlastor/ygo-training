"""A duel the web client saved, played again with the checkpoint in the
trainer's own seat."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ygo_training.edopro_review import checkpoint_asked
from ygo_training.environment import Environment, library_path
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "web_client_duels", ROOT / "tools/web_client_duels.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

DECKS = ["blackwing", "monarch"]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


needs_the_environment = pytest.mark.skipif(
    not _built(), reason="the environment is not built: cargo build --release"
)


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


def network(seed: int) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet(embedding_dim=8, hidden_dim=16).eval()


def saved_by_a_page(
    environment: Environment, model: PolicyValueNet, *, seat: int, seed: int = 31
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A duel as the page plays and saves one, and the record it is made
    from: the network in `seat` with its deck's pilot alongside, a pilot in
    the other, stopped part of the way as a save is taken."""
    decks = DECKS if seat == 0 else DECKS[::-1]
    asked = checkpoint_asked(model, environment, [player == seat for player in (0, 1)])
    setup = {
        "seed": seed,
        "decks": decks,
        "seats": [
            {"control": "caller"} if player == seat else {"control": "pilot"} for player in (0, 1)
        ],
        "keep_record": True,
        "limit": 150,
    }
    with environment.start(setup) as episode:
        while True:
            event = episode.advance()
            if event["decision"] is not None:
                episode.choose(asked(event["decision"])["model"])
            if event["over"] is not None:
                record = event["over"]["record"]
                break
    return {
        "opponent": "model",
        "player_name": "Tester",
        "player_deck_name": None,
        # An episode makes a seat's pilot with its seed and the seat's
        # number; the page makes its own with this seed and one more.
        "policy_seed": (seed + seat - 1) % 2**64,
        "duel": {
            "seed": record["seed"],
            "human": 1 - seat,
            # A record lists a player's cards as the page's save does: the
            # engine is given the last of them first.
            "decks": record["decks"],
            "responses": record["responses"],
            "started_at": 0,
        },
    }, record


def test_a_saved_duel_is_read_as_the_engine_gives_it_or_as_the_tab_keeps_it(tmp_path):
    saved = {"opponent": "model", "policy_seed": 5, "duel": {"seed": [1, 2, 3, 4], "human": 0}}
    plain = tmp_path / "saved.json"
    plain.write_text(json.dumps(saved), encoding="utf8")
    kept = tmp_path / "tab.json"
    kept.write_text(json.dumps({"opponent": {"id": "model"}, "saved": json.dumps(saved)}))
    # A finished duel exported from the page's history: a log around the same text.
    log = tmp_path / "claudi-oh-duel.json"
    log.write_text(
        json.dumps({"format": 1, "record": {"result": "player"}, "replay": json.dumps(saved)})
    )

    assert tool.read(plain) == saved == tool.read(kept) == tool.read(log)

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"seed": [1, 2, 3, 4]}), encoding="utf8")
    with pytest.raises(SystemExit, match="not a duel the web client saved"):
        tool.read(other)


def test_the_cards_are_listed_for_a_replay_as_the_engine_was_given_them():
    saved = {
        "policy_seed": 5,
        "duel": {
            "seed": [1, 2, 3, 4],
            "human": 1,
            "decks": [{"main": [1, 2, 3], "extra": [8, 9]}, {"main": [4, 5], "extra": []}],
            "responses": [[0, "ffffffff"], [1, "04000100"]],
        },
    }
    assert tool.kept(saved) == {
        "seed": [1, 2, 3, 4],
        "flags": 0xD0700,
        "start_lp": 8000,
        "start_hand": 5,
        "draw_count": 1,
        "decks": [{"main": [3, 2, 1], "extra": [9, 8]}, {"main": [5, 4], "extra": []}],
        "responses": ["ffffffff", "04000100"],
    }


def test_what_was_found_is_said():
    found = {"network": 0, "decks": ["blackwing", None], "kept": 40, "answers": 21, "decisions": 9}
    same = tool.said("duel.json", {**found, "differing": []})
    assert same[0].startswith("duel.json: a deck that is no list of the pool against the network")
    assert "which went first; 40 answers kept, 21 of them its seat's, 9 of those" in same[0]
    assert same[1] == "  every answer of its seat is this seat's, byte for byte"

    zones = tool.said("duel.json", {**found, "differing": [(3, "000801", "000800")]})
    assert zones[1] == "  answers of its seat that are others here: 1 of 21"
    assert zones[2] == "    answer 3: here 000801, the page 000800"
    assert "not made with the page's seed" in zones[-1]
    # Another answer than a zone is no matter of the pilot's seed.
    mixed = [(3, "000801", "000800"), (7, "ffffffff", "00000000")]
    other = tool.said("duel.json", {**found, "differing": mixed})
    assert "the page's seed" not in other[-1]

    left = tool.said("duel.json", {"left": "it is not played again here"})
    assert left == ["duel.json: it is not played again here"]


@pytest.mark.integration
@needs_the_environment
@pytest.mark.parametrize("seat", [0, 1])
def test_the_trainers_seat_gives_the_answers_a_page_kept(environment, seat):
    model = network(3)
    saved, record = saved_by_a_page(environment, model, seat=seat)
    # The page hosts the rules the trainer deals a duel with.
    assert (record["flags"], record["start_lp"], record["start_hand"], record["draw_count"]) == (
        tool.MASTER_RULE_1,
        tool.LIFE_POINTS,
        tool.HAND,
        tool.DRAWN,
    )

    found = tool.again(environment, model, saved)

    assert "left" not in found and found["differing"] == []
    assert (found["network"], found["decks"]) == (seat, DECKS if seat == 0 else DECKS[::-1])
    assert found["kept"] == len(saved["duel"]["responses"])
    assert found["answers"] > found["decisions"] > 10

    # Another network does not give those answers: its seat answers
    # otherwise, or the other player's answers no longer fit.
    another = tool.again(environment, network(4), saved)
    assert "left" in another or another["differing"]

    # The seat that plays another deck than the checkpoint's is not it.
    assert "not monarch" in tool.again(environment, model, saved, learner="monarch")["left"]


@pytest.mark.integration
@needs_the_environment
def test_a_duel_of_another_engine_or_other_cards_is_not_played_again(environment):
    saved, _ = saved_by_a_page(environment, network(3), seat=0)
    # Another rule set deals the same cards and asks other questions.
    found = tool.again(environment, network(3), saved, flags=0)
    assert "left" in found or found["differing"]

    cards = json.loads(json.dumps(saved))
    cards["duel"]["decks"][1]["main"] = cards["duel"]["decks"][1]["main"][::-1]
    found = tool.again(environment, network(3), cards)
    assert "left" in found or found["differing"]
