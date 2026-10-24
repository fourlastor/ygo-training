"""How much of what people played around WC2011 the decks of training hold:
the cards of tournament decklists of 2010 and 2011, counted by their share of
main- and extra-deck slots, against the cards of sets of the pool's decks.

    uv run python tools/played_cards.py analysis/meta-decks-2011/cards.csv \\
        --branch origin/ygo-agent-benchmark-work

`cards.csv` is the count of the collected decklists (`analysis/meta-decks-2011`,
gathered for the experiment on Darker Than Black). A card is told by its name,
an alternate artwork by its original's; only the cards the WC2011 pool allows
(a count above 0) are read. With `--decks` (the lists, `decks.json`) the slots
are counted again at the pool's limits: a list's copies of a card no more
than the WC2011 list allows it. The sets are the thirteen decks of training, the
pool's decks at the pin and, with `--branch`, the decks of that revision of
the policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.environment import pool

POLICIES = ROOT / "native" / "ygo-policies"
DATABASE = POLICIES / "vendor" / "BabelCdb" / "cards.cdb"
THIRTEEN = [
    "blackwing",
    "lightsworn",
    "gishki",
    "burn",
    "crystal",
    "dragunity",
    "gladiator",
    "heroes",
    "infernity",
    "monarch",
    "morphtronic",
    "rock-block",
    "spellcaster",
]
MARKS = (0.5, 0.8, 0.9, 0.95, 0.99)


class Names:
    """A card's name by its code: an alternate artwork is its original card,
    and a card whose alias is of another name (Harpie Lady 1) is its own."""

    def __init__(self, database: Path = DATABASE) -> None:
        connection = sqlite3.connect(database)
        self.names = dict(connection.execute("select id, name from texts"))
        self.alias = dict(connection.execute("select id, alias from datas"))

    def __call__(self, code: int) -> str:
        original = self.alias.get(code) or code
        if code in self.names and self.names.get(original) != self.names[code]:
            return self.names[code]
        return self.names.get(original, str(code))


def codes(text: str) -> list[int]:
    """The main- and extra-deck codes of a `.ydk` list."""
    found, part = [], False
    for line in text.splitlines():
        line = line.strip()
        if line in ("#main", "#extra"):
            part = True
        elif line.startswith("!"):
            part = False
        elif part and line.isdigit():
            found.append(int(line))
    return found


def pinned() -> dict[str, list[int]]:
    """The pool's decks at the pin, by policy id."""
    return {
        deck: codes((POLICIES / "decks" / f"{name}.ydk").read_text(encoding="utf8"))
        for deck, name in pool(ROOT).items()
    }


def of_revision(revision: str) -> list[list[int]]:
    """The decks of a revision of the policies, read from git."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(POLICIES), *arguments], capture_output=True, text=True, check=True
        ).stdout

    files = [name for name in git("ls-tree", "--name-only", revision, "decks/").split("\n")]
    return [codes(git("show", f"{revision}:{name}")) for name in files if name.endswith(".ydk")]


def played(path: Path, name: Names) -> list[dict[str, Any]]:
    """The cards of the decklists the WC2011 pool allows, most played first."""
    rows = []
    for row in csv.DictReader(path.open(encoding="utf8")):
        if row["in_wc2011_pool"] != "true" or row["wc2011_count"] in ("", "0"):
            continue
        rows.append(
            {
                "card": name(int(row["passcode"])),
                "slots": float(row["share_of_main_extra_slots"]),
                "decks": float(row["share_of_decks"]),
            }
        )
    return sorted(rows, key=lambda row: -row["slots"])


def limits(name: Names) -> dict[str, int]:
    """How many copies of each card the WC2011 list allows, by its name."""
    found: dict[str, int] = {}
    for line in (POLICIES / "data" / "wc2011.lflist.conf").read_text(encoding="utf8").split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            card = name(int(parts[0]))
            found[card] = max(int(parts[1]), found.get(card, 0))
    return found


def capped(path: Path, name: Names) -> list[dict[str, Any]]:
    """The cards of the counted lists the pool allows, by their main- and
    extra-deck copies in each list no more than the pool's limit, most played
    first; their slots as a share of all such copies."""
    allowed = limits(name)
    copies: dict[str, float] = {}
    for deck in json.loads(path.read_text(encoding="utf8")):
        if not deck["included_in_statistics"]:
            continue
        held: dict[str, int] = {}
        for card in deck["cards"]:
            if card.get("section") in ("main", "extra") and card.get("passcode"):
                told = name(int(card["passcode"]))
                held[told] = held.get(told, 0) + int(card["copies"])
        for card, count in held.items():
            if allowed.get(card, 0):
                copies[card] = copies.get(card, 0) + min(count, allowed[card])
    whole = sum(copies.values())
    rows = [{"card": card, "slots": count / whole, "decks": 0.0} for card, count in copies.items()]
    return sorted(rows, key=lambda row: -row["slots"])


def report(cards: list[dict[str, Any]], sets: dict[str, set[str]], top: int = 30) -> dict:
    whole = sum(row["slots"] for row in cards)
    result: dict[str, Any] = {"cards": len(cards), "held": {}, "concentration": {}}
    for label, held in sets.items():
        got = [row for row in cards if row["card"] in held]
        result["held"][label] = {
            "cards": len(got),
            "slots": sum(row["slots"] for row in got) / whole,
            "of_top_100": sum(row["card"] in held for row in cards[:100]),
        }
    total, marks = 0.0, list(MARKS)
    for count, row in enumerate(cards, 1):
        total += row["slots"]
        while marks and total >= marks[0] * whole:
            result["concentration"][str(marks.pop(0))] = count
    widest = list(sets.values())[-1]
    missing = [row for row in cards if row["card"] not in widest]
    result["missing"] = {
        "of": list(sets)[-1],
        "cards": len(missing),
        "slots": sum(row["slots"] for row in missing) / whole,
        "most_played": [
            {"card": row["card"], "slots": row["slots"] / whole, "decks": row["decks"]}
            for row in missing[:top]
        ],
    }
    return result


def show(result: dict[str, Any]) -> str:
    lines = [f"{result['cards']} cards played that the WC2011 pool allows, held by:"]
    for label, held in result["held"].items():
        lines.append(
            f"  {label:28}{held['cards']:>5} cards, {100 * held['slots']:5.1f}% of their slots,"
            f" {held['of_top_100']} of the 100 most played"
        )
    lines.append("The most played cards that fill so much of the slots:")
    for mark, count in result["concentration"].items():
        lines.append(f"  {100 * float(mark):3.0f}%: {count}")
    missing = result["missing"]
    lines.append(
        f"Held by none of {missing['of']}: {missing['cards']} cards,"
        f" {100 * missing['slots']:.1f}% of the slots; the most played (slots, decks):"
    )
    for row in missing["most_played"]:
        lines.append(f"  {row['card'][:44]:44}{100 * row['slots']:6.2f}%{100 * row['decks']:7.1f}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cards", type=Path)
    parser.add_argument("--branch", help="a revision of the policies whose decks are read too")
    parser.add_argument("--decks", type=Path, help="the lists counted, to count them again capped")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    name = Names()
    decks = pinned()
    sets = {
        f"the {len(THIRTEEN)} of training": {name(c) for d in THIRTEEN for c in decks[d]},
        f"the {len(decks)} of the pin": {name(c) for d in decks.values() for c in d},
    }
    if args.branch:
        found = of_revision(args.branch)
        sets[f"the {len(found)} of {args.branch}"] = {name(c) for d in found for c in d}
    result = report(played(args.cards, name), sets)
    if args.decks:
        within = report(capped(args.decks, name), sets)
        result["at_the_pools_limits"] = within["held"]
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    print(show(result))
    if args.decks:
        print("At the pool's limits, a list's copies of a card no more than it allows:")
        for label, held in result["at_the_pools_limits"].items():
            print(f"  {label:28}{100 * held['slots']:5.1f}% of the slots")


if __name__ == "__main__":
    main()
