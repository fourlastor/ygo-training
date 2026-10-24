"""The decisions the shared input crate is held to.

    uv run python tools/model_input_vectors.py \\
        native/ygo-model-input/input/vectors/decisions.json

The model's input has one definition: the crates of `native/ygo-model-input`,
which build its arrays for the trainer's library and for whatever else seats
a checkpoint.  This writes what that repository checks itself against:
decisions of real duels and a few made by hand, each as the view the library
reports, with what the library makes of it.  Of every array its name, its
type, its shape and a digest of its bytes are kept, one decision at a time
and all of them padded into one batch; and of every card they name, its row
of the card database.  Any change of what the arrays are then fails there,
and here (`tests/test_model_input_vectors.py`), until the file is written
again.

Run it again, and commit the file there, whenever the input is changed on
purpose.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.views import (
    ACTIVATE,
    HAND,
    MONSTER,
    PASS,
    SPELL,
    TOGGLE,
    TRAP,
    card,
    choice,
    hidden,
    named,
    view,
)
from ygo_training.checkpoints import FORMAT_VERSION
from ygo_training.environment import POLICIES_DIRECTORY, Environment
from ygo_training.schema import MAX_ACTION_MEMBERS, MAX_CARDS, MAX_HISTORY

DATABASE = "vendor/BabelCdb/cards.cdb"
COLUMNS = "id, type, atk, def, level, race, attribute, setcode"
# Decisions kept of one kind: the same question about the same sort of thing.
ALIKE = 1
# Duels played for the decisions, each once by its pilots and once otherwise.
GAMES = 4


def digest(values: np.ndarray) -> str:
    """FNV-1a, 64 bits, of an array's bytes as NumPy keeps them: small
    enough to write again wherever the arrays are built."""
    found = 0xCBF29CE484222325
    for byte in values.tobytes():
        found = ((found ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{found:016x}"


def told(arrays: dict[str, np.ndarray]) -> list[list[Any]]:
    return [
        [name, str(values.dtype), list(values.shape), digest(values)]
        for name, values in arrays.items()
    ]


def played(environment: Environment, games: int) -> list[dict[str, Any]]:
    """Decisions of duels over the pool: pilots' games with both seats seen,
    then games a caller plays otherwise than the pilot, which have a
    command's card with the selections it leads to."""
    names = sorted(environment.decks)
    found = []
    for game in range(games):
        seated = [names[game * 7 % len(names)], names[(game * 5 + 3) % len(names)]]
        for control in ("pilot", "caller"):
            seat = {"control": control, **({"observe": True} if control == "pilot" else {})}
            setup = {"seed": 9_300 + game, "decks": seated, "seats": [seat, seat]}
            with environment.start(setup) as episode:
                while True:
                    event = episode.advance()
                    shown = [*event["samples"]]
                    if control == "caller" and event["decision"] is not None:
                        shown.append(event["decision"])
                        answers = len(event["decision"]["choices"])
                        episode.choose((len(found) * 7 + 3) % answers)
                    found.extend(shown)
                    if event["over"] is not None:
                        break
    return found


def kept(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A few decisions of every kind seen."""
    seen: dict[tuple, int] = {}
    chosen = []
    for shown in decisions:
        members = max((len(offered["members"]) for offered in shown["choices"]), default=0)
        kind = (
            shown["question"]["kind"],
            shown["question"]["hint"],
            shown["acting"] is not None,
            min(len(shown["chain"]), 2),
            bool(shown["events"]),
            min(len(shown["choices"]), 6) // 3,
            min(members, 4) // 2,
            min(len(shown["history"]), MAX_HISTORY) // MAX_HISTORY,
            shown["coin_toss"] is not None,
            shown["step"] > 0,
        )
        seen[kind] = seen.get(kind, 0) + 1
        if seen[kind] <= ALIKE:
            chosen.append(shown)
    return chosen


def by_hand() -> list[dict[str, Any]]:
    """What real duels seldom show: the limits, hidden cards, cards named by
    code alone, a coin toss, values with a second half, descriptions of an
    effect."""
    busy = [
        card(code=MONSTER + index % 3, counters=index % 14, battles=index % 5)
        for index in range(MAX_CARDS + 9)
    ]
    busy[3] = hidden()
    busy[5] = card(SPELL, location=HAND, position=0, attack=-2, defense=12_500, level=13)
    busy[6] = card(attacking=True)
    busy[8] = card(attacked=True, chained=True, targeted=True, selected=True, event=True, coin=2)
    busy[9] = card(TRAP, coin=1)
    many = [
        named(index % 11, MONSTER + index % 3, value=index * 70_000 - 5)
        for index in range(MAX_ACTION_MEMBERS + 6)
    ]
    crowded = view(
        cards=busy,
        turn=41,
        turn_player=1,
        step=12,
        life_points=[-300, 16_000],
        piles=[[3, 44], [9, 0], [17, 2], [1, 0], [15, 6]],
        summon_used=True,
        coin_toss=[2, 5],
        chain=[
            {"code": TRAP, "controller": 1, "source": 9, "targets": [6, 8]},
            {"code": SPELL, "controller": 0, "source": -1, "targets": []},
            {"code": MONSTER, "controller": 0, "source": MAX_CARDS + 2, "targets": []},
        ],
        events=[{"card": 400, "code": SPELL, "controller": 1, "location": 5}],
        acting=named(5, SPELL, location=HAND, position=0, value=3, required=True),
        question={
            "kind": 7,
            "forced": True,
            "triggers": True,
            "hint": 23,
            "minimum": 2,
            "maximum": 40,
            "subject": TRAP,
        },
        choices=[
            choice(TOGGLE, named(8), members=many, description=(SPELL << 20) + 19),
            choice(ACTIVATE, named(MAX_CARDS + 4, SPELL, value=-1, required=True), description=7),
            choice(ACTIVATE, named(2, 4_007), members=many[:3]),
            choice(PASS),
        ],
        history=[
            {
                "kind": ACTIVATE,
                "code": MONSTER + index,
                "location": index % 9,
                "description": (index << 20) + index,
            }
            for index in range(MAX_HISTORY + 5)
        ],
    )
    unseen = view(
        cards=[hidden(), hidden(location=HAND)],
        choices=[choice(PASS), choice(ACTIVATE, named(0, 0))],
    )
    return [crowded, view(turn_player=-1, coin_toss=[0, 0]), unseen]


def codes(value: Any) -> set[int]:
    """Every card a view names, by any of the fields that hold a code."""
    if isinstance(value, dict):
        own = {int(value[key]) for key in ("code", "subject") if key in value}
        return own.union(*(codes(inner) for inner in value.values()))
    if isinstance(value, list):
        return set().union(*(codes(inner) for inner in value))
    return set()


def made(environment: Environment, decisions: list[dict[str, Any]]) -> tuple[list, dict]:
    """What the library makes of each decision alone, and of all of them in
    one batch."""
    return [environment.pack(shown) for shown in decisions], environment.batch(decisions)


def vectors(games: int) -> dict[str, Any]:
    with Environment(ROOT) as environment:
        decisions = [*kept(played(environment, games)), *by_hand()]
        packed, batch = made(environment, decisions)
    wanted = sorted(codes(decisions))
    database = sqlite3.connect(f"file:{ROOT / POLICIES_DIRECTORY / DATABASE}?mode=ro", uri=True)
    rows = [
        list(row)
        for code in wanted
        for row in database.execute(f"SELECT {COLUMNS} FROM datas WHERE id = ?", (code,))
    ]
    database.close()
    return {
        "about": (
            "Decisions as views, with what the crates of this repository make of each: "
            "every array's name, type, shape and the FNV-1a 64 digest of its bytes. "
            "Written by the trainer's tools/model_input_vectors.py; not to be edited."
        ),
        "checkpoint_format": FORMAT_VERSION,
        "columns": COLUMNS,
        "cards": rows,
        "decisions": [
            {"view": shown, "arrays": told(arrays_)}
            for shown, arrays_ in zip(decisions, packed, strict=True)
        ],
        "batch": told(batch),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--games", type=int, default=GAMES, help="duels played for the decisions")
    args = parser.parse_args()
    found = vectors(args.games)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(found, separators=(",", ":")) + "\n", encoding="utf8")
    kinds = sorted({decision["view"]["question"]["kind"] for decision in found["decisions"]})
    print(
        f"{len(found['decisions'])} decisions (question kinds {kinds}), "
        f"{len(found['cards'])} cards, "
        f"{args.output.stat().st_size / 1e6:.2f} MB: {args.output}"
    )


if __name__ == "__main__":
    main()
