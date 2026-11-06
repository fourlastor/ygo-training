"""`tools/failed_duels.py`: failed duels grouped by the script line that failed."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def loaded(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = loaded("failed_duels")


def row(failure: str, decks: list[str], mains: list[list[int]]) -> str:
    return json.dumps(
        {
            "failure": failure,
            "setup": {"decks": decks},
            "record": {"decks": [{"main": main, "extra": []} for main in mains]},
        }
    )


def test_a_scripts_failure_is_told_by_its_card_and_line_and_the_deck_holding_it(tmp_path):
    path = tmp_path / "failed-duels.jsonl"
    path.write_text(
        "\n".join(
            [
                row(
                    '[string "c11.lua"]:19: Attempting to access deleted object.',
                    ["game-0001", "agents"],
                    [[11], [22]],
                ),
                row(
                    '[string "c11.lua"]:19: Attempting to access deleted object.',
                    ["zombie", "game-0001"],
                    [[33], [11]],
                ),
                row(
                    '[string "c11.lua"]:40: attempt to index a nil value',
                    ["game-0001", "game-0002"],
                    [[11], [11]],
                ),
                # The engine's log comes before the message.
                row("41230939\n\n\nThe arena is full", ["agents", "zombie"], [[22], [33]]),
                row("\n\nThe arena is full", ["agents", "zombie"], [[22], [33]]),
                row(
                    '[string "proc_union.lua"]:143: attempt to index a nil value',
                    ["agents", "zombie"],
                    [[22], [33]],
                ),
            ]
        )
        + "\n"
    )
    counts, held = tool.grouped(path)
    deleted = 11, "19: Attempting to access deleted object."
    indexed = 11, "40: attempt to index a nil value"
    arena = None, "The arena is full"
    union = None, "proc_union.lua:143: attempt to index a nil value"
    assert counts == {deleted: 2, indexed: 1, arena: 2, union: 1}
    # Both seats of the third duel hold the card: one duel, two holders.
    assert held[deleted] == {"game-0001": 2}
    assert held[indexed] == {"game-0001": 1, "game-0002": 1}
    assert held[arena] == {}
    table = tool.table({"run": (counts, held)})
    assert table[1].endswith(" 2  game-0001")
    assert table[-1].split() == ["all", "6"]


def test_names_come_from_the_card_database(tmp_path):
    database = tmp_path / "cards.cdb"
    with sqlite3.connect(database) as connection:
        connection.execute("create table texts (id integer, name text)")
        connection.execute("insert into texts values (11, 'Beast Rising')")
    assert tool.names([11, 12, None], database) == {11: "Beast Rising"}
