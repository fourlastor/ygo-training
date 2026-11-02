"""`tools/deck_replays.py`: what is read from a deck's games played again."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deck_replays", ROOT / "tools/deck_replays.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

CARD = 111


def move(transaction: int, value: float, offered: bool, kind: str = "Pass", code: int = 0) -> dict:
    return {
        "by": "network",
        "kind": kind,
        "code": code,
        "value": value,
        "transaction": transaction,
        "offered": [CARD] if offered else [],
    }


def game(moves: list[dict], winner: int | None, seat: int = 0, deal: int = 0) -> dict:
    return {
        "deal": deal,
        "seat": seat,
        "other": "rival",
        "names": ("network", "other") if seat == 0 else ("other", "network"),
        "moves": moves,
        "over": {"winner": 2 if winner is None else winner, "reason": 2, "turns": 10},
    }


def test_the_credit_of_a_game_is_worked_from_its_end():
    found = [{"value": 0.5}, {"value": 0.25}]
    assert tool.credits(found, 1.0, 1.0) == pytest.approx([0.5, 0.75])
    assert tool.credits(found, 1.0, 0.5) == pytest.approx([-0.25 + 0.5 * 0.75, 0.75])


def test_a_transactions_first_answer_gives_its_value_and_the_others_seat_is_no_step():
    moves = [move(1, 0.5, False), move(1, 0.9, False), {**move(2, 0.1, False), "by": "other"}]
    found = tool.steps(game([*moves, move(3, 0.25, False)], 0))
    assert [step["value"] for step in found] == [0.5, 0.25]
    assert [len(step["moves"]) for step in found] == [2, 1]


def test_what_a_card_did_counts_uses_and_games_and_tells_a_draw_from_a_loss():
    games = [
        game(
            [
                move(1, 0, True, "Activate", CARD),
                move(2, 0, True, "Activate", CARD),
                move(3, 0, False, "Activate", 222),
            ],
            0,
        ),
        game([move(1, 0, True)], 1),
        game([move(1, 0, False)], None),
    ]
    told = tool.cards(games, {CARD})
    assert told["wins"] == pytest.approx(1 / 3)
    assert told["uses"] == pytest.approx(2 / 3)
    assert told["in_games"] == pytest.approx(1 / 3)
    assert told["passed_over"] == 1
    assert told["ends"] == {"won by deck-out": 1, "lost by deck-out": 1, "drawn by deck-out": 1}


def test_a_seat_of_the_network_wins_by_who_sits_there():
    assert tool.won(game([], 1, seat=1))
    assert not tool.won(game([], 0, seat=1))
    assert not tool.won(game([], None))
    assert tool.outcome(game([], 0, seat=1)) == -1.0
    assert tool.outcome(game([], None)) == 0.0


def test_two_files_are_paired_deal_by_deal():
    one = [game([], 0, 0, 0), game([], 1, 1, 0), game([], 0, 0, 1), game([], 0, 1, 1)]
    two = [game([], 1, 0, 0), game([], 1, 1, 0), game([], 0, 0, 1), game([], 1, 1, 1)]
    gap, error, deals = tool.paired(one, two)
    # The first deal +1 from one seat and 0 from the other, the second 0 and -1.
    assert deals == 2
    assert gap == pytest.approx(0.0)
    assert error == pytest.approx(0.5)


def test_the_edge_is_of_the_steps_a_card_was_played_over_those_it_was_passed_by():
    plays = [
        game([move(1, 0.0, True, "Activate", CARD), move(2, 0.0, False)], 0, deal=deal)
        for deal in range(8)
    ]
    passes = [game([move(1, 0.0, True), move(2, 0.0, False)], 1, deal=deal) for deal in range(8)]
    told, times, before = tool.edges(plays + passes, {CARD})
    assert times == 8
    assert before == 2.0
    edge, error = told[1.0]
    assert edge == pytest.approx(2.0, abs=0.01)
    assert math.isfinite(error)


def behind(turn: int, could: bool, did: bool, would: bool, network: bool = True) -> list:
    return [network, turn, could, did, would, 6, 0]


def test_a_turn_held_back_is_one_its_pilot_would_have_summoned_in():
    rows = [
        # Held back: it could, its pilot would, and it did not, at either answer.
        behind(10, True, False, True),
        behind(10, True, False, False),
        # It did, at the turn's second answer.
        behind(12, True, False, True),
        behind(12, True, True, True),
        # Nothing to summon: not a turn that counts.
        behind(14, False, False, False),
        # The other side's.
        behind(16, True, False, True, network=False),
    ]
    told = tool.held_back([{**game([], 0), "rows": rows}])
    assert told["behind"] == 1.0
    assert told["turns"] == 3.0
    assert told["could"] == 2.0
    assert told["did"] == pytest.approx(0.5)
    assert told["would"] == 1.0
    assert told["held back"] == pytest.approx(0.5)
    assert tool.held_back([{**game([], 0), "rows": rows}], network=False)["held back"] == 1.0


def test_a_game_with_no_wall_counts_for_nothing_held_back():
    told = tool.held_back([{**game([], 0), "rows": []}])
    assert told["behind"] == 0.0
    assert told["held back"] == 0.0


def test_the_gap_is_told_from_each_seat():
    one = [game([], 0, 0, 0), game([], 1, 1, 0), game([], 0, 0, 1), game([], 0, 1, 1)]
    two = [game([], 1, 0, 0), game([], 1, 1, 0), game([], 0, 0, 1), game([], 1, 1, 1)]
    told = tool.from_each_seat(one, two)
    assert told[0][0] == pytest.approx(0.5)
    assert told[1][0] == pytest.approx(-0.5)
