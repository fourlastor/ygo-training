"""`tools/attack_targets.py`: the attacks of a learner's seat, sorted by what
it could see of the target, how each came out, and the bar on attacks into a
monster it could see would win the battle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ygo_training.environment import library_path
from ygo_training.schema import ACTION_SCALAR_COUNT, CONTEXT_COUNT, NUMERIC_COUNT, SCALAR_COUNT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("attack_targets", ROOT / "tools/attack_targets.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# Its dataclass looks its module up.
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)

BATTLE, SELECT, OTHER, YES_NO = 2, 7, 3, 4
ATTACK, TOGGLE, MAIN2, YES, NO = 7, 21, 9, 13, 14
ATTACK_TARGET = 22
UP_ATTACK, UP_DEFENSE, DOWN_DEFENSE = 0x1, 0x4, 0x8
ATTACKER, BEATEN, BIGGER, WALL = 1001, 2002, 3003, 4004
# Attackers whose texts make something of a battle, and two tricks.
BOOSTED, RECRUITER, GLADIATOR, BLACKWING, MACHINE = 5005, 6006, 7007, 8008, 8009
KALUT, LIMITER = 9009, 9010
HAND = 2

BATTLE_STEP, DAMAGE, MAIN_2 = 5, 6, 9


class Questions(tool.Questions):
    """The numbers of the library's vocabulary, without opening the engine."""

    def __init__(self) -> None:
        self.attack, self.yes, self.battle, self.select = ATTACK, YES, BATTLE, SELECT
        self.yes_no, self.attack_target, self.damage_step = YES_NO, ATTACK_TARGET, {DAMAGE, 7}
        self.hand = HAND


BATTLES = tool.Battles(
    printed={
        BOOSTED: tool.Printed(boost=300),
        RECRUITER: tool.Printed(gains=True),
        GLADIATOR: tool.Printed(tags_out=True),
        BLACKWING: tool.Printed(setcodes=(tool.BLACKWING,)),
        MACHINE: tool.Printed(race=tool.MACHINE),
    },
    tricks={
        KALUT: tool.TRICKS["Blackwing - Kalut the Moon Shadow"],
        LIMITER: tool.TRICKS["Limiter Removal"],
    },
    hand=HAND,
)


def scaled(value: int) -> float:
    return value / tool.COMBAT_SCALE


def view(
    *,
    question: int,
    answers: list[tuple[int, int, int, int, int, int]],
    board: list[tuple[int, int, int, int, int]],
    turn: int = 3,
    life_points: int = 8000,
    hint: int = 0,
    phase: int = BATTLE_STEP,
    acting: tuple[int, int] | None = None,
    hand: tuple[int, ...] = (),
    spells: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    """One row: answers as (kind, code, location, controller, position, ATK),
    the field as (code, controller, position, ATK, DEF), then the seat's own
    hand and Spells and Traps."""
    count, cards = max(len(answers), 1), max(len(board) + len(hand) + len(spells), 1)
    arrays: dict[str, np.ndarray] = {
        "decision_kind": np.array([question]),
        "decision_hint": np.array([hint]),
        "phase": np.array([phase]),
        "action_kind": np.zeros((1, count), dtype=np.int64),
        "action_card": np.zeros((1, count), dtype=np.int64),
        "action_location": np.zeros((1, count), dtype=np.int64),
        "action_controller": np.zeros((1, count), dtype=np.int64),
        "action_position": np.zeros((1, count), dtype=np.int64),
        "action_numeric": np.zeros((1, count, NUMERIC_COUNT), dtype=np.float32),
        "action_scalars": np.zeros((1, count, ACTION_SCALAR_COUNT), dtype=np.float32),
        "state_lengths": np.array([len(board) + len(hand) + len(spells)]),
        "state_card": np.zeros((1, cards), dtype=np.int64),
        "state_location": np.zeros((1, cards), dtype=np.int64),
        "state_controller": np.zeros((1, cards), dtype=np.int64),
        "state_position": np.zeros((1, cards), dtype=np.int64),
        "state_numeric": np.zeros((1, cards, NUMERIC_COUNT), dtype=np.float32),
        "context_mask": np.zeros((1, CONTEXT_COUNT), dtype=bool),
        "context_card": np.zeros((1, CONTEXT_COUNT), dtype=np.int64),
        "context_controller": np.zeros((1, CONTEXT_COUNT), dtype=np.int64),
        "context_position": np.zeros((1, CONTEXT_COUNT), dtype=np.int64),
        "context_numeric": np.zeros((1, CONTEXT_COUNT, NUMERIC_COUNT), dtype=np.float32),
        "scalars": np.zeros((1, SCALAR_COUNT), dtype=np.float32),
    }
    for answer, (kind, code, location, controller, position, atk) in enumerate(answers):
        arrays["action_kind"][0, answer] = kind
        arrays["action_card"][0, answer] = code
        arrays["action_location"][0, answer] = location
        arrays["action_controller"][0, answer] = controller
        arrays["action_position"][0, answer] = position
        arrays["action_numeric"][0, answer, 0] = scaled(atk)
    for place, (code, controller, position, atk, defense) in enumerate(board):
        arrays["state_card"][0, place] = code
        arrays["state_location"][0, place] = tool.MONSTER_ZONE
        arrays["state_controller"][0, place] = controller
        arrays["state_position"][0, place] = position
        arrays["state_numeric"][0, place, :2] = (scaled(atk), scaled(defense))
    held = [(code, HAND) for code in hand] + [(code, tool.SPELL_TRAP_ZONE) for code in spells]
    for place, (code, location) in enumerate(held, start=len(board)):
        arrays["state_card"][0, place] = code
        arrays["state_location"][0, place] = location
    if acting is not None:
        arrays["context_mask"][0, tool.ACTING] = True
        arrays["context_card"][0, tool.ACTING] = acting[0]
        arrays["context_numeric"][0, tool.ACTING, 0] = scaled(acting[1])
    arrays["scalars"][0, tool.TURN] = turn / tool.TURN_SCALE
    arrays["scalars"][0, tool.LIFE_POINTS] = life_points / tool.LIFE_SCALE
    return arrays


def attack_with(
    atk: int,
    board: list[tuple[int, int, int, int, int]],
    *,
    direct: bool = False,
    attacker: int = ATTACKER,
    **told: Any,
) -> dict[str, np.ndarray]:
    arrays = view(
        question=BATTLE,
        answers=[(ATTACK, attacker, tool.MONSTER_ZONE, 0, UP_ATTACK, atk), (MAIN2, 0, 0, 0, 0, 0)],
        board=[(attacker, 0, UP_ATTACK, atk, 0), *board],
        **told,
    )
    # The library's value 1 of 16.
    arrays["action_scalars"][0, 0, tool.MAY_ATTACK_DIRECTLY] = 1 / 16 if direct else 0.0
    return arrays


def asked_direct(board: list[tuple[int, int, int, int, int]]) -> dict[str, np.ndarray]:
    """Whether the attacker attacks directly, asked as it attacks."""
    return view(
        question=YES_NO,
        answers=[(YES, 0, 0, 0, 0, 0), (NO, 0, 0, 0, 0, 0)],
        board=board,
        acting=(ATTACKER, 1800),
    )


def after(board: list[tuple[int, int, int, int, int]], **told: int) -> dict[str, np.ndarray]:
    """The seat's next decision once the battle is over."""
    return view(question=OTHER, answers=[(MAIN2, 0, 0, 0, 0, 0)], board=board, **told)


class Seat:
    """Feeds one duel's rows to `Attacks`, one round each."""

    def __init__(self) -> None:
        self.attacks = tool.Attacks(SimpleNamespace(_about={7: {"index": 0}}), Questions(), BATTLES)

    def answer(self, arrays: dict[str, np.ndarray], answer: int = 0) -> None:
        self.attacks(
            SimpleNamespace(arrays=arrays, duel=np.array([7])), np.array([0]), np.array([answer])
        )

    def told(self) -> dict[str, int]:
        self.attacks.flush()
        return dict(self.attacks.games[0])


def test_a_target_is_what_its_visible_numbers_make_it():
    assert tool.target_kind(1800, 0, DOWN_DEFENSE, 0, 0) == tool.FACE_DOWN
    assert tool.target_kind(1800, BIGGER, UP_ATTACK, 1700, 2500) == tool.WEAKER
    assert tool.target_kind(1800, BIGGER, UP_ATTACK, 1800, 0) == tool.EVEN
    assert tool.target_kind(1800, BIGGER, UP_ATTACK, 1900, 0) == tool.STRONGER
    # In Defense Position its DEF counts, not its ATK.
    assert tool.target_kind(1800, WALL, UP_DEFENSE, 2500, 1500) == tool.WEAKER
    assert tool.target_kind(1800, WALL, UP_DEFENSE, 0, 2000) == tool.STRONGER


def test_the_cards_texts_make_a_lost_battle_planned_or_armed_and_add_a_boost():
    stronger = (BIGGER, UP_ATTACK, 1900, 0)
    # An attacker's own boost when it attacks counts, as Jain's.
    assert BATTLES.kind(BOOSTED, 1800, stronger, []) == tool.WEAKER
    assert BATTLES.kind(BOOSTED, 1800, (BIGGER, UP_ATTACK, 2100, 0), []) == tool.EVEN
    assert BATTLES.kind(BOOSTED, 1800, (BIGGER, UP_ATTACK, 2200, 0), []) == tool.STRONGER
    assert BATTLES.kind(RECRUITER, 1400, stronger, []) == tool.PLANNED
    # A Gladiator Beast tags out after a battle it survives.
    assert BATTLES.kind(GLADIATOR, 1600, (WALL, UP_DEFENSE, 0, 2000), []) == tool.PLANNED
    assert BATTLES.kind(GLADIATOR, 1600, stronger, []) == tool.STRONGER
    # A trick held where it can be used, on an attacker it raises.
    assert BATTLES.kind(BLACKWING, 1600, stronger, [(KALUT, True)]) == tool.ARMED
    assert BATTLES.kind(BLACKWING, 1600, stronger, [(KALUT, False)]) == tool.STRONGER
    assert BATTLES.kind(ATTACKER, 1600, stronger, [(KALUT, True)]) == tool.STRONGER
    assert BATTLES.kind(MACHINE, 1600, stronger, [(LIMITER, False)]) == tool.ARMED
    # A face-down target stays one.
    assert BATTLES.kind(RECRUITER, 1400, (0, DOWN_DEFENSE, 0, 0), []) == tool.FACE_DOWN


def test_the_seats_tricks_are_read_from_its_hand_and_spells():
    arrays = attack_with(1600, [], hand=(KALUT, BEATEN), spells=(LIMITER,))
    assert BATTLES.held(arrays, 0) == [(KALUT, True), (LIMITER, False)]


@pytest.mark.skipif(not tool._database().exists(), reason="the card database is not checked out")
def test_the_rules_read_the_texts_of_the_fourteen_decks_cards():
    battles = tool.Battles.read(tool._database(), HAND)
    names = {name: code for code, name in tool.Names(tool._database()).names.items()}

    def card(name: str) -> Any:
        return battles.printed[names[name]]

    assert card("Jain, Lightsworn Paladin").boost == 300
    assert card("Crystal Beast Topaz Tiger").boost == 400
    assert card("Blackwing Armed Wing").boost == 500
    for name in ("UFO Turtle", "Neo-Spacian Grand Mole", "Crystal Beast Sapphire Pegasus"):
        assert card(name).gains, name
    assert card("Elemental HERO Absolute Zero").gains
    assert card("Gladiator Beast Laquari").tags_out
    # Its counters go at the end of the Battle Phase: no gain.
    assert not card("Mythical Beast Cerberus").tags_out
    assert not card("Thunder King Rai-Oh").gains
    assert {names["Honest"], names["Blackwing - Kalut the Moon Shadow"]} <= set(battles.tricks)
    assert tool.BLACKWING in card("Blackwing - Gale the Whirlwind").setcodes


def test_the_board_is_read_to_its_length_and_by_player():
    arrays = attack_with(1800, [(BIGGER, 1, UP_ATTACK, 1900, 1000), (0, 1, DOWN_DEFENSE, 0, 0)])
    assert tool.monsters(arrays, 0, tool.OPPONENT) == [
        (BIGGER, UP_ATTACK, 1900, 1000),
        (0, DOWN_DEFENSE, 0, 0),
    ]
    assert tool.monsters(arrays, 0, tool.OWN) == [(ATTACKER, UP_ATTACK, 1800, 0)]
    arrays["state_lengths"][0] = 1
    assert tool.monsters(arrays, 0, tool.OPPONENT) == []


def test_a_direct_attack_and_how_it_came_out():
    seat = Seat()
    seat.answer(attack_with(1800, []))
    seat.answer(after([(ATTACKER, 0, UP_ATTACK, 1800, 0)]))
    assert seat.told() == {tool.DIRECT: 1, f"{tool.DIRECT}:{tool.FINE}": 1}


def test_a_single_target_is_read_from_the_board():
    seat = Seat()
    seat.answer(attack_with(1800, [(BIGGER, 1, UP_ATTACK, 1900, 1000)]))
    # The attacker is gone at the next decision of the turn.
    seat.answer(after([(BIGGER, 1, UP_ATTACK, 1900, 1000)]))
    assert seat.told() == {tool.STRONGER: 1, f"{tool.STRONGER}:{tool.LOST}": 1}


def test_a_target_among_several_is_the_one_chosen():
    seat = Seat()
    board = [(BEATEN, 1, UP_ATTACK, 1000, 1000), (WALL, 1, UP_DEFENSE, 0, 2000)]
    seat.answer(attack_with(1800, board))
    selection = view(
        question=SELECT,
        hint=ATTACK_TARGET,
        answers=[
            (TOGGLE, BEATEN, tool.MONSTER_ZONE, 1, UP_ATTACK, 1000),
            (TOGGLE, WALL, tool.MONSTER_ZONE, 1, UP_DEFENSE, 0),
        ],
        board=[(ATTACKER, 0, UP_ATTACK, 1800, 0), *board],
        acting=(ATTACKER, 1800),
    )
    selection["action_numeric"][0, 1, 1] = scaled(2000)
    seat.answer(selection, answer=1)
    # A window of the damage step tells nothing yet.
    seat.answer(after([(ATTACKER, 0, UP_ATTACK, 1800, 0), *board], phase=DAMAGE))
    # It stayed, and its player took the difference.
    seat.answer(after([(ATTACKER, 0, UP_ATTACK, 1800, 0), *board], life_points=7800))
    assert seat.told() == {tool.STRONGER: 1, f"{tool.STRONGER}:{tool.HURT}": 1}


def test_an_attacker_that_may_attack_directly_is_told_by_its_answer():
    board = [(BIGGER, 1, UP_ATTACK, 1900, 1000)]
    seat = Seat()
    seat.answer(attack_with(1800, board, direct=True))
    seat.answer(asked_direct([(ATTACKER, 0, UP_ATTACK, 1800, 0), *board]), answer=0)
    seat.answer(after([(ATTACKER, 0, UP_ATTACK, 1800, 0), *board]))
    assert seat.told() == {tool.DIRECT: 1, f"{tool.DIRECT}:{tool.FINE}": 1}
    # Answered no, it attacks the monster.
    seat = Seat()
    seat.answer(attack_with(1800, board, direct=True))
    seat.answer(asked_direct([(ATTACKER, 0, UP_ATTACK, 1800, 0), *board]), answer=1)
    seat.answer(after(board))
    assert seat.told() == {tool.STRONGER: 1, f"{tool.STRONGER}:{tool.LOST}": 1}


def test_an_attack_with_no_decision_of_its_turn_after_it_is_unseen():
    seat = Seat()
    seat.answer(attack_with(1800, [(BEATEN, 1, UP_ATTACK, 1000, 1000)]))
    seat.answer(after([], turn=4))
    seat.answer(attack_with(1800, []))
    assert seat.told() == {
        tool.WEAKER: 1,
        f"{tool.WEAKER}:{tool.UNSEEN}": 1,
        tool.DIRECT: 1,
        f"{tool.DIRECT}:{tool.UNSEEN}": 1,
    }


def bar(arrays: dict[str, np.ndarray]) -> list[bool]:
    barred = tool.Bar(Questions(), BATTLES)(SimpleNamespace(arrays=arrays), np.array([0]))
    return barred[0].tolist()


def test_an_attack_is_barred_only_when_every_target_would_win():
    assert bar(attack_with(1800, [(BIGGER, 1, UP_ATTACK, 1900, 0)])) == [True, False]
    assert bar(
        attack_with(1800, [(BIGGER, 1, UP_ATTACK, 1900, 0), (WALL, 1, UP_DEFENSE, 0, 2000)])
    ) == [
        True,
        False,
    ]
    # A face-down monster or a weaker one leaves it whole, as does an empty field.
    assert bar(
        attack_with(1800, [(BIGGER, 1, UP_ATTACK, 1900, 0), (0, 1, DOWN_DEFENSE, 0, 0)])
    ) == [
        False,
        False,
    ]
    assert bar(attack_with(1800, [(BEATEN, 1, UP_ATTACK, 1000, 0)])) == [False, False]
    assert bar(attack_with(1800, [])) == [False, False]
    # Nor one whose battle its text plans, or that a trick it holds raises.
    stronger = [(BIGGER, 1, UP_ATTACK, 1900, 0)]
    assert bar(attack_with(1400, stronger, attacker=RECRUITER)) == [False, False]
    assert bar(attack_with(1600, stronger, attacker=BLACKWING, hand=(KALUT,))) == [False, False]
    assert bar(attack_with(1600, stronger, attacker=BLACKWING)) == [True, False]
    # Nor is an attacker barred that may attack directly.
    assert bar(attack_with(1800, [(BIGGER, 1, UP_ATTACK, 1900, 0)], direct=True)) == [False, False]
    # The ATK the engine gives the attacker now decides, as an equip's.
    assert bar(attack_with(2800, [(BIGGER, 1, UP_ATTACK, 1900, 0)])) == [False, False]


def test_where_the_target_is_asked_the_stronger_ones_are_barred():
    selection = view(
        question=SELECT,
        hint=ATTACK_TARGET,
        answers=[
            (TOGGLE, BEATEN, tool.MONSTER_ZONE, 1, UP_ATTACK, 1000),
            (TOGGLE, BIGGER, tool.MONSTER_ZONE, 1, UP_ATTACK, 1900),
        ],
        board=[],
        acting=(ATTACKER, 1800),
    )
    assert bar(selection) == [False, True]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_every_attack_of_a_few_games_is_told_once():
    import torch

    from ygo_training.model import PolicyValueNet

    torch.manual_seed(0)
    model = PolicyValueNet().eval()
    run = tool.play(
        model,
        learners=["lightsworn"],
        group="blackwing,heroes",
        games=2,
        seed=7,
        bar=True,
        threads=1,
        slots=4,
    )
    assert len(run["results"]) == 4
    assert all("barred" in result for result in run["results"])
    told = sum(count for _, _, _, count in run["cards"])
    kinds = sum(r["attacks"].get(kind, 0) for r in run["results"] for kind in tool.KINDS)
    outcomes = sum(
        r["attacks"].get(f"{kind}:{outcome}", 0)
        for r in run["results"]
        for kind in tool.KINDS
        for outcome in tool.OUTCOMES
    )
    assert told == kinds == outcomes
