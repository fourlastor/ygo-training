"""The named plays are counted as the record tells them."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from ygo_training.environment import library_path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("named_plays", ROOT / "tools/named_plays.py")
assert SPEC is not None and SPEC.loader is not None
plays = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plays)

ALLURE, DARK, SPELL, OTHER_SPELL, LIGHT = 1, 2, 3, 4, 5
NAMES = {ALLURE: plays.ALLURE, DARK: "Blackwing", SPELL: "X", OTHER_SPELL: "Y", LIGHT: "Veiler"}
KINDS = {ALLURE: "spell", DARK: "dark", SPELL: "spell", OTHER_SPELL: "spell", LIGHT: "monster"}
ACTIVATE = f"Idle: Activate {plays.ALLURE}"
HAND, FIELD = plays.HAND, 4


def name(code: int) -> str:
    return NAMES.get(code, "")


def kind(code: int) -> str:
    return KINDS[code]


def decision(game: str, said: list[str], chosen: int, **more) -> dict:
    """A decision of the checkpoint's seat, as the games keep it."""
    row = {
        "game": game,
        "opponent": "monarch",
        "seat": 0,
        "turn": 1,
        "turn_player": 0,
        "phase": "Main1",
        "said": said,
        "chosen": chosen,
        "pilot": None,
        "cards": [],
        "events": [],
        "chain": [],
        "from": [HAND] * len(said),
    }
    row.update(more)
    return row


def hand(*codes: int) -> list[list[int]]:
    return [[0, HAND, code, 0, 0] for code in codes]


def over(game: str, winner: int, turns: int = 9, **more) -> dict:
    return {
        "game": game,
        "opponent": "monarch",
        "seat": 0,
        "over": {"winner": winner, "turns": turns, "limit": False},
        **more,
    }


def test_allure_is_told_by_the_turn_and_by_what_else_is_in_hand():
    end = "Idle: EndTurn"
    set_x, set_y = "Idle: SetSpellTrap X", "Idle: SetSpellTrap Y"
    own = f"Idle: SetSpellTrap {plays.ALLURE}"
    rows = [
        # With a DARK monster to banish: activated.
        decision("a", [ACTIVATE, end], 0, turn=3, cards=hand(ALLURE, DARK)),
        # Two Spells beside it: one is set, then the card is activated while
        # the other could still be set, and the hand goes to the Graveyard.
        decision(
            "b", [ACTIVATE, set_x, set_y, end], 1, turn=5, cards=hand(ALLURE, SPELL, OTHER_SPELL)
        ),
        decision("b", [ACTIVATE, set_y, end], 0, turn=5, cards=hand(ALLURE, OTHER_SPELL)),
        decision("b", [end], 0, turn=5),
        # Nothing else in hand: kept one turn, activated the next, and a DARK
        # monster is drawn.
        decision("c", [ACTIVATE, end], 1, turn=2, cards=hand(ALLURE)),
        decision("c", [ACTIVATE, end], 0, turn=4, cards=hand(ALLURE)),
        decision("c", [end], 0, turn=4, cards=hand(DARK)),
        # Its Spell set first, and nothing left to set but the card itself.
        decision("d", [ACTIVATE, set_x, end], 1, turn=1, cards=hand(ALLURE, SPELL)),
        decision("d", [ACTIVATE, own, end], 0, turn=1, cards=hand(ALLURE)),
        decision("d", [end], 0, turn=1),
        # Set on an earlier turn and activated from the field: not a turn it
        # was held in hand.
        decision("e", [ACTIVATE, end], 0, turn=7, cards=hand(SPELL), **{"from": [FIELD, HAND]}),
        # A monster that is not DARK beside it is neither kind of hand.
        decision("f", [ACTIVATE, end], 1, turn=2, cards=hand(ALLURE, LIGHT)),
    ]
    games, _ = plays.by_game(rows)
    assert plays.allure(games, name, kind) == {
        "turns": {plays.DARK: 1, plays.SPELLS: 2, plays.ALONE: 2, plays.OTHER: 1},
        "activated": {plays.DARK: 1, plays.SPELLS: 2, plays.ALONE: 1},
        "with_cards_still_to_set": 1,
        # In b, c and d, by the hand at the activation itself.
        "activations_without_a_dark_monster": 3,
        "hand_lost": 2,
        "hand_lost_with_other_cards": 1,
    }


def window(game: str, card: str, takes: bool, pilot_takes: bool, **more) -> dict:
    said = [f"Chain: Activate {card}", "Chain: Pass"]
    return decision(game, said, 0 if takes else 1, pilot=0 if pilot_takes else 1, **more)


def summoned(code: int, attack: int) -> dict:
    """A monster of theirs that has just arrived."""
    return {
        "cards": [[1, plays.MONSTERS, code, plays.FACE_UP_ATTACK, attack]],
        "events": [{"controller": 1, "code": code}],
    }


def test_traps_are_told_by_the_windows_that_offer_them():
    rows = [
        window("a", "Trap Hole", True, True, **summoned(70, 1400)),
        window("a", "Trap Hole", False, True, **summoned(71, 1800)),
        window("b", "Trap Hole", False, True, **summoned(70, 1400)),
        # A monster that was there before is not the one summoned.
        window(
            "c",
            "Trap Hole",
            True,
            False,
            cards=[
                [1, plays.MONSTERS, 72, plays.FACE_UP_ATTACK, 1200],
                [1, plays.MONSTERS, 71, plays.FACE_UP_ATTACK, 1800],
            ],
            events=[{"controller": 1, "code": 71}],
        ),
        window("c", "Dimensional Prison", True, False),
        decision("c", ["Idle: EndTurn"], 0),
    ]
    games, _ = plays.by_game(rows)
    assert plays.traps(games) == {
        "Trap Hole": {"windows": 4, "taken": 2, "pilot": 3},
        "Bottomless Trap Hole": {"windows": 0, "taken": 0, "pilot": 0},
        "Dimensional Prison": {"windows": 1, "taken": 1, "pilot": 0},
        "Trap Hole, 1,000 to 1,499 ATK": {"windows": 2, "taken": 1, "pilot": 2},
        "Trap Hole, first window of a game": {"windows": 3, "taken": 2, "pilot": 2},
    }


def test_threatening_roar_is_told_by_where_it_is_activated_and_what_it_answers():
    attacker = [[1, plays.MONSTERS, 70, plays.FACE_UP_ATTACK, 1800]]
    defender = [[1, plays.MONSTERS, 70, 4, 1800]]
    theirs = {"turn_player": 1}

    def at(game: str, takes: bool, pilot_takes: bool = False, **more) -> dict:
        return window(game, plays.ROAR, takes, pilot_takes, **{**theirs, **more})

    rows = [
        # At their attack, where the pilot plays it.
        at("a", True, True, phase="BattleStart", cards=attacker),
        at("a", False, True, phase="BattleStart", cards=attacker),
        # A Battle Phase entered with no monster, and one with none attacking.
        at("b", False, phase="BattleStart"),
        at("b", False, phase="BattleStep", cards=defender),
        # In their Main Phase 1: in answer to a card on the chain, to a
        # monster arriving, and to nothing while they have no attacker.
        at(
            "c",
            True,
            phase="Main1",
            chain=[{"controller": 0, "code": 1}, {"controller": 1, "code": 9}],
        ),
        at("c", True, phase="Main1", **summoned(70, 1800)),
        at("d", True, phase="Main1"),
        at("d", True, phase="Main1", cards=attacker),
        at("d", False, phase="Draw"),
        # In its own turn.
        window("e", plays.ROAR, False, False, phase="Main1"),
    ]
    games, _ = plays.by_game(rows)
    assert plays.roar(games) == {
        "activations": 5,
        "windows": {
            "battle, attackers": 2,
            "battle, no monster": 1,
            "battle, none attacking": 1,
            "Main1": 4,
            "Draw": 1,
            "own turn": 1,
        },
        "taken": {"battle, attackers": 1, "Main1": 4},
        "pilot": {
            "battle, attackers": 2,
            "battle, no monster": 0,
            "battle, none attacking": 0,
            "Main1": 0,
            "Draw": 0,
            "own turn": 0,
        },
        "answering": {
            "battle, attackers: nothing": 1,
            "Main1: chain": 1,
            "Main1: arrival": 1,
            "Main1: nothing": 2,
        },
        # Only the one with no attacker of theirs on the field.
        "spent_early": 1,
    }


def test_a_checkpoints_games_are_counted_and_set_beside_anothers():
    copy = "Idle: Summon Blackwing"
    rows = [
        # The other copy of a card is the pilot's answer too.
        decision("a", [copy, copy, "Idle: EndTurn"], 1, pilot=0),
        decision("a", [copy, "Idle: EndTurn"], 1, pilot=0),
        decision("b", ["Idle: EndTurn"], 0, opponent="machina"),
        over("a", 0),
        over("b", 1, opponent="machina"),
    ]
    idle = [over("i", 0, turns=7), over("j", 0, turns=11), over("k", 1, turns=30)]
    found = plays.counted(rows, idle, name, kind)
    assert (found["games"], found["won"]) == (2, 1)
    assert found["by_opponent"] == {"machina": [0, 1], "monarch": [1, 1]}
    assert (found["decisions"], found["pilot_answers"]) == (3, 1)
    assert found["idle"] == {"games": 3, "won": 2, "turns": 9.0}
    told = dict(plays.lines(found))
    assert told["games won"] == "1 of 2 (50%)"
    assert told["won against machina"] == "0 of 1 (0%)"
    assert told["... in turns, on the average"] == "9.0"
    assert told["Trap Hole at a window that offers it"] == "-"
    # Two checkpoints beside each other: a row for each thing told of one.
    table = plays.table([("one", found), ("other", found)]).splitlines()
    assert len(table) == len(told) + 1
    assert table[0].split() == ["one", "other"]
    assert table[1].startswith("games won") and table[1].count("1 of 2 (50%)") == 2


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_a_checkpoints_decisions_are_kept_and_read_again(tmp_path, monkeypatch):
    """A few games of a network as it is first drawn: every decision of its
    seat is kept, and the same checkpoint and games are read again, not
    played again."""
    import torch

    from ygo_training.checkpoints import save
    from ygo_training.model import PolicyValueNet

    torch.set_num_threads(2)

    def checkpoint(seed: int) -> Path:
        torch.manual_seed(seed)
        # A checkpoint is loaded into a network of the trainer's own size.
        model = PolicyValueNet()
        path = tmp_path / f"drawn-{seed}" / "model-00000.pt"
        save(path, model, torch.optim.Adam(model.parameters()), 1, {})
        return path

    args = argparse.Namespace(
        root=ROOT,
        device="cpu",
        opponents="monarch,blackwing",
        games=2,
        idle_opponents="blackwing",
        idle_games=2,
        seed=50_000,
    )
    first = checkpoint(3)
    decisions, idle = plays.decisions_of(first, tmp_path / "kept", "drawn", args)
    games, ended = plays.by_game(decisions)
    assert sorted(ended) == ["blackwing-0", "blackwing-1", "monarch-0", "monarch-1"]
    assert {row["seat"] for row in ended.values()} == {0, 1}
    taken = [row for rows in games.values() for row in rows]
    assert len(taken) > 40 and all(len(row["said"]) == len(row["from"]) > 1 for row in taken)
    assert all(0 <= row["chosen"] < len(row["said"]) for row in taken)
    assert any(row["pilot"] is not None and row["chosen"] != row["pilot"] for row in taken)
    assert sorted(plays.by_game(idle)[1]) == ["blackwing-0", "blackwing-1"]
    found = plays.counted(decisions, idle, *plays.card_data(ROOT))
    assert found["games"] == 4 and found["idle"]["games"] == 2
    assert plays.table([("drawn", found)]).startswith(" ")

    kept = tmp_path / "kept" / "drawn.jsonl"
    assert json.loads(kept.read_text().splitlines()[0])["described"]["games"] == 2
    monkeypatch.setattr(plays, "played", lambda *_, **__: pytest.fail("played again"))
    assert plays.decisions_of(first, tmp_path / "kept", "drawn", args) == (decisions, idle)
    # Another checkpoint under the name, or other games, are not taken for it.
    with pytest.raises(SystemExit, match="another checkpoint or of other games"):
        plays.decisions_of(checkpoint(4), tmp_path / "kept", "drawn", args)
    args.games = 4
    with pytest.raises(SystemExit, match="another checkpoint or of other games"):
        plays.decisions_of(first, tmp_path / "kept", "drawn", args)
