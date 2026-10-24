"""Decisions made by hand, in the form the duel environment reports them.

The tests of the policy's input and of the optimizer need decisions with a
known content and no engine behind them.  These builders give every field of
a view a plain default; the library makes the model's input of them, as it
does of every decision.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from ygo_training.environment import Environment

ROOT = Path(__file__).resolve().parents[1]

# Cards of the pool, by what they are.
MONSTER = 22835145  # Blackwing - Blizzard the Far North
OTHER_MONSTER = 58820853  # Blackwing - Shura the Blue Flame
SPELL = 5318639  # Mystical Space Typhoon
TRAP = 53567095  # Icarus Attack

HAND, MONSTER_ZONE, SPELL_TRAP_ZONE, GRAVEYARD = 2, 3, 4, 5
FACE_UP_ATTACK, FACE_DOWN_DEFENSE = 1, 8
IDLE, BATTLE, CHAIN, SELECT_CARDS = 1, 2, 3, 7
NORMAL_SUMMON, ACTIVATE, ATTACK, END_TURN, PASS, TOGGLE = 1, 6, 7, 10, 12, 21


def card(code: int = MONSTER, **fields: Any) -> dict[str, Any]:
    """A card of a view; a monster of the seat's own, face-up on the field."""
    return {
        "controller": 0,
        "location": MONSTER_ZONE,
        "code": code,
        "position": FACE_UP_ATTACK,
        "attack": 1300,
        "defense": 0,
        "level": 2,
        "can_attack": False,
        "battles": 0,
        "counters": 0,
        "coin": 0,
        "selected": False,
        "attacking": False,
        "attacked": False,
        "event": False,
        "chained": False,
        "targeted": False,
        **fields,
    }


def hidden(**fields: Any) -> dict[str, Any]:
    """A card of the other player the seat cannot see: its place alone."""
    return card(
        0,
        controller=1,
        position=FACE_DOWN_DEFENSE,
        attack=0,
        defense=0,
        level=0,
        **fields,
    )


def named(index: int = -1, code: int = MONSTER, **fields: Any) -> dict[str, Any]:
    """A card a choice names: its place among the view's cards, or -1."""
    return {
        "card": index,
        "code": code,
        "controller": 0,
        "location": MONSTER_ZONE,
        "position": FACE_UP_ATTACK,
        "value": 0,
        "required": False,
        **fields,
    }


def choice(kind: int = PASS, card: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "card": card, "members": [], "description": 0, **fields}


def view(**fields: Any) -> dict[str, Any]:
    """A decision: by default two answers to an open chain window, no cards."""
    return {
        "player": 0,
        "answer": 0,
        "step": 0,
        "transaction": 1,
        "turn": 1,
        "turn_player": 0,
        "phase": 3,
        "life_points": [8000, 8000],
        "piles": [[35, 35], [5, 5], [0, 0], [0, 0], [0, 0]],
        "summon_used": False,
        "cards": [],
        "chain": [],
        "events": [],
        "acting": None,
        "coin_toss": None,
        "question": {
            "kind": CHAIN,
            "forced": False,
            "triggers": False,
            "hint": 0,
            "minimum": 0,
            "maximum": 0,
            "subject": 0,
        },
        "choices": [choice(ACTIVATE, named(code=TRAP, location=SPELL_TRAP_ZONE)), choice(PASS)],
        "label": None,
        "pilot": None,
        "chosen": None,
        "history": [],
        **fields,
    }


@cache
def library() -> Environment:
    """The library that makes the model's input, opened once for the tests."""
    return Environment(ROOT)


def packed(**fields: Any) -> dict[str, Any]:
    """The arrays of :func:`view` with ``fields`` changed, as the library
    makes them of the decision alone."""
    return library().pack(view(**fields))


def batched(*views: dict[str, Any]) -> dict[str, Any]:
    """Decisions as the batch the model is given, built by the library."""
    import torch

    return {name: torch.from_numpy(array) for name, array in library().batch(list(views)).items()}


class Shown:
    """Decisions held as the pool holds those of its duels, each under a
    row: any of them are given back as the batch the model is given, as an
    optimizer reads its steps (``PpoOptimizer.kept``)."""

    def __init__(self) -> None:
        self.views: list[dict[str, Any]] = []

    def row(self, shown: dict[str, Any]) -> int:
        self.views.append(shown)
        return len(self.views) - 1

    def __call__(self, rows: list[int]) -> dict[str, Any]:
        return batched(*(self.views[row] for row in rows))


# Where the steps of the tests are kept, for every optimizer they are given to.
SHOWN = Shown()


def row(**fields: Any) -> int:
    """The row :func:`view` with ``fields`` changed is kept under."""
    return SHOWN.row(view(**fields))


def step(**fields: Any) -> dict[str, Any]:
    """A rollout step on a two-answer decision, as the pool's games report one."""
    return {
        "row": row(),
        "chosen": 0,
        "log_prob": -0.6931471805599453,
        "value": 0.0,
        "player": 0,
        **fields,
    }
