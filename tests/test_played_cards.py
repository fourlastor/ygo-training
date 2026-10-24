"""`tools/played_cards.py`: how much of the slots of tournament decklists a
set of decks holds, on a count written as `analysis/meta-decks-2011/cards.csv`
is."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("played_cards", ROOT / "tools/played_cards.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

FIELDS = ["passcode", "name", "share_of_main_extra_slots", "share_of_decks", "in_wc2011_pool"]
FIELDS += ["wc2011_count"]


def count(path: Path, rows: list[tuple]) -> Path:
    with path.open("w", encoding="utf8", newline="") as found:
        writer = csv.writer(found)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path


def test_the_slots_a_set_of_decks_holds(tmp_path):
    path = count(
        tmp_path / "cards.csv",
        [
            (5318639, "Mystical Space Typhoon", 0.4, 0.9, "true", 2),
            (37520316, "Mind Control", 0.3, 0.3, "true", 3),
            (44508094, "Stardust Dragon", 0.2, 0.8, "true", 3),
            # Forbidden in the pool, and outside it: not read.
            (55144522, "Pot of Greed", 0.05, 0.1, "true", 0),
            (1, "Not in the pool", 0.05, 0.1, "false", ""),
        ],
    )
    name = tool.Names()
    cards = tool.played(path, name)
    assert [row["card"] for row in cards] == [
        "Mystical Space Typhoon",
        "Mind Control",
        "Stardust Dragon",
    ]
    result = tool.report(
        cards,
        {
            "Blackwing": {"Mystical Space Typhoon"},
            "more": {"Mystical Space Typhoon", "Stardust Dragon"},
        },
    )
    assert result["held"]["Blackwing"]["slots"] == pytest.approx(0.4 / 0.9)
    assert result["held"]["more"]["cards"] == 2
    assert result["concentration"]["0.5"] == 2
    assert result["missing"]["most_played"][0]["card"] == "Mind Control"
    assert result["missing"]["slots"] == pytest.approx(0.3 / 0.9)
    assert "Mind Control" in tool.show(result)


def test_the_pins_decks_are_read_by_their_names():
    decks = tool.pinned()
    assert set(tool.THIRTEEN) <= set(decks) and all(decks.values())
    name = tool.Names()
    assert "Blackwing - Bora the Spear" in {name(code) for code in decks["blackwing"]}


def test_a_lists_copies_are_counted_at_the_pools_limits(tmp_path):
    lists = [
        {
            "included_in_statistics": True,
            "cards": [
                # Book of Moon is limited to one in the pool.
                {"passcode": 14087893, "copies": 3, "section": "main"},
                {"passcode": 5318639, "copies": 1, "section": "main"},
                {"passcode": 44508094, "copies": 2, "section": "side"},
            ],
        },
        {"included_in_statistics": False, "cards": [{"passcode": 5318639, "copies": 3}]},
    ]
    path = tmp_path / "decks.json"
    path.write_text(json.dumps(lists))
    cards = tool.capped(path, tool.Names())
    assert {row["card"]: row["slots"] for row in cards} == {
        "Book of Moon": pytest.approx(0.5),
        "Mystical Space Typhoon": pytest.approx(0.5),
    }
