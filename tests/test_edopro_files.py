"""An EDOPro's files read as EDOPro reads them, beside the trainer's."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("edopro_files", ROOT / "tools/edopro_files.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

UPDATES = "https://example.org/updates"
# A card as a database tells it: its code, the card it is an artwork of, its
# set code, and what it is called.
SWORDSMAN = (10, 0, 5, "Swordsman")
ARTWORK = (11, 10, 5, "Swordsman")
TRAP = (20, 0, 0, "Trap")


def database(path: Path, *cards: tuple[int, int, int, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = sqlite3.connect(path)
    kept.execute(
        "create table datas(id integer primary key, ot integer, alias integer, setcode integer,"
        " type integer, atk integer, def integer, level integer, race integer,"
        " attribute integer, category integer)"
    )
    kept.execute("create table texts(id integer primary key, name text, desc text)")
    for code, alias, setcode, name in cards:
        kept.execute(
            "insert into datas values(?,3,?,?,33,1000,1000,4,1,16,0)", (code, alias, setcode)
        )
        kept.execute("insert into texts values(?,?,'')", (code, name))
    kept.commit()
    kept.close()
    return path


def written(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")
    return path


@pytest.fixture
def trainer(tmp_path):
    """The trainer's side: its card rows, its scripts and a deck list."""
    vendor = tmp_path / "vendor"
    database(vendor / "BabelCdb/cards.cdb", SWORDSMAN, ARTWORK, TRAP)
    written(vendor / "CardScripts/official/c10.lua", "the swordsman, as it is now")
    written(vendor / "CardScripts/official/c20.lua", "the trap")
    written(vendor / "CardScripts/utility.lua", "the helper, as it is now")
    written(vendor / "CardScripts/constant.lua", "the numbers")
    written(tmp_path / "decks/Swords.ydk", "#main\n11\n20\n20\n#extra\n!side\n")
    return vendor, tmp_path / "decks"


@pytest.fixture
def edopro(tmp_path):
    """An EDOPro as it came: the artwork is not in its database, the
    swordsman is of another set there and its script is an older one.  It is
    set up with a repository of updates it has not fetched."""
    root = tmp_path / "edopro"
    repositories = [
        {
            "url": UPDATES,
            "repo_path": "./repositories/updates",
            "has_core": True,
            "core_path": "bin",
            "data_path": "",
            "script_path": "script",
        },
        {
            "url": "https://example.org/lists",
            "repo_path": "./repositories/lists",
            "lflist_path": ".",
        },
    ]
    written(root / "config/configs.json", json.dumps({"repos": repositories}))
    written(root / "config/user_configs.json", json.dumps({"repos": []}))
    database(root / "expansions/cards.cdb", (10, 0, 4, "Swordsman"), TRAP)
    database(root / "expansions/other.cdb", (900, 0, 0, "Another game's card"))
    written(root / "script/official/c10.lua", "the swordsman, as it was")
    written(root / "script/official/c20.lua", "the trap")
    written(root / "script/utility.lua", "the helper, as it was")
    written(root / "script/constant.lua", "the numbers")
    written(root / "libocgcore.so", "")
    return root


def fetched(root: Path) -> Path:
    """The repository of updates, fetched: the swordsman's row and script
    and the helper as they are now, the artwork, and an engine."""
    updates = root / "repositories/updates"
    (updates / ".git").mkdir(parents=True)
    database(updates / "cards.delta.cdb", SWORDSMAN, ARTWORK)
    written(updates / "script/official/c10.lua", "the swordsman, as it is now")
    written(updates / "script/utility.lua", "the helper, as it is now")
    written(updates / "script/VERSION", "0123456789abcdef\n")
    written(updates / "bin/libocgcore.so", "")
    return root


def test_an_installation_is_read_as_edopro_reads_it(edopro):
    found = tool.installation(edopro)

    # Not fetched: it plays what it came with, and the repository is said.
    assert found.scripts == [edopro / "script"]
    assert found.databases == [edopro / "expansions/cards.cdb", edopro / "expansions/other.cdb"]
    assert found.engine == edopro / "libocgcore.so"
    # A repository of lists brings nothing a duel is played with.
    assert found.unfetched == [UPDATES]

    found = tool.installation(fetched(edopro))
    updates = edopro / "repositories/updates"
    assert found.scripts == [updates / "script", edopro / "script"]
    assert found.databases[-1] == updates / "cards.delta.cdb" and len(found.databases) == 3
    assert found.engine == updates / "bin/libocgcore.so"
    assert found.unfetched == []
    # A script is the repository's where it has one, a card is as it tells it.
    held = tool.named(found.scripts)
    assert held["c10.lua"] == updates / "script/official/c10.lua"
    assert held["c20.lua"] == edopro / "script/official/c20.lua"
    assert held["utility.lua"] == updates / "script/utility.lua"
    assert tool.helpers(found.scripts) == {"utility.lua", "constant.lua"}
    told = tool.rows(found.databases)
    assert told[10][:3] == (10, 0, 5) and told[11][:3] == (11, 10, 5) and 900 in told

    with pytest.raises(SystemExit, match="no EDOPro folder"):
        tool.installation(edopro / "script")


def test_the_cards_of_the_pool_are_set_beside_the_trainers(edopro, trainer):
    vendor, decks = trainer
    assert tool.pool(decks) == {11: ["Swords"], 20: ["Swords"]}

    lines, differences = tool.files(tool.installation(edopro), vendor, decks)
    text = "\n".join(lines)
    # The artwork is missing, and the script it plays by is an older one.
    assert differences == 2
    assert "2 cards in 1 deck lists" in text
    assert "rows     1 as the trainer's, 0 told otherwise, 1 missing" in text
    assert "missing: 11 (Swordsman; Swords)" in text
    assert "scripts  1 as the trainer's, 1 another, 0 missing" in text
    assert (
        "another: 10 (Swordsman; the card another is an artwork of): script/official/c10.lua"
        in text
    )
    # A helper that differs is said and is no difference by itself.
    assert "helpers  1 of 2 as the trainer's; another: utility.lua" in text

    lines, differences = tool.files(tool.installation(fetched(edopro)), vendor, decks)
    text = "\n".join(lines)
    assert differences == 0
    assert "rows     2 as the trainer's, 0 told otherwise, 0 missing" in text
    assert "scripts  2 as the trainer's, 0 another, 0 missing" in text
    assert "helpers  2 of 2 as the trainer's" in text

    # A card of the pool told otherwise is said by what differs.
    written(decks / "Old.ydk", "10\n")
    as_it_came = tool.installation(edopro)._replace(databases=[edopro / "expansions/cards.cdb"])
    lines, differences = tool.files(as_it_came, vendor, decks)
    assert "    told otherwise: 10 (Swordsman; Old): setcode 5 here and 4 there" in lines
    assert differences == 2


def test_what_edopro_reads_is_laid_out_for_the_bench(edopro, trainer, tmp_path):
    vendor, _ = trainer
    cards, scripts = tool.assembled(tool.installation(fetched(edopro)), vendor, tmp_path / "laid")

    # One database of every card, each as the last that holds it tells it.
    assert {code: row[:3] for code, row in tool.rows([cards]).items()} == {
        10: (10, 0, 5),
        11: (11, 10, 5),
        20: (20, 0, 0),
        900: (900, 0, 0),
    }
    names = sqlite3.connect(cards)
    assert dict(names.execute("select id, name from texts"))[11] == "Swordsman"
    names.close()
    # The trainer's layout, with the scripts EDOPro would read.
    assert sorted(path.name for path in scripts.glob("*.lua")) == ["constant.lua", "utility.lua"]
    assert (scripts / "utility.lua").read_text() == "the helper, as it is now"
    assert (scripts / "official/c10.lua").read_text() == "the swordsman, as it is now"
    assert (scripts / "official/c20.lua").read_text() == "the trap"
    # EDOPro's files themselves are left as they were.
    assert tool.rows([edopro / "expansions/cards.cdb"])[10][:3] == (10, 0, 4)


def test_the_games_are_set_beside_each_other():
    def game(digest: str, winner: int = 0) -> dict:
        return {"digest": digest, "winner": winner, "reason": 1, "turns": 9, "decisions": 80}

    ours = {
        ("blackwing", "monarch", 1, 0, "game"): game("a"),
        ("blackwing", "monarch", 2, 1, "game"): game("b"),
        ("burn", "blackwing", 3, 0, "game"): game("c"),
    }
    lines, apart = tool.beside(ours, dict(ours), "blackwing")
    assert apart == 0
    assert lines == ["  3 of 3 the same question for question, 0 another, 0 failed"]

    theirs = {
        **ours,
        ("blackwing", "monarch", 2, 1, "game"): game("other"),
        ("burn", "blackwing", 3, 0, "game"): {"error": "Deck card 11 is absent from the database"},
    }
    lines, apart = tool.beside(ours, theirs, "blackwing")
    assert apart == 2
    assert lines == [
        "  1 of 3 the same question for question, 1 another, 1 failed",
        "    another against: monarch",
        "    failed against: burn",
        "    the first failure: Deck card 11 is absent from the database",
    ]

    # A game the trainer's own files fail, or runs of other games, tell nothing.
    with pytest.raises(SystemExit, match="failed"):
        tool.beside(theirs, ours, "blackwing")
    with pytest.raises(SystemExit, match="not of the same games"):
        tool.beside(ours, {key: ours[key] for key in list(ours)[:2]}, "blackwing")
