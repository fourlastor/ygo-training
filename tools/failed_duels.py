"""Failed duels grouped by the card script and the line that failed.

    uv run python tools/failed_duels.py \
        quarter=analysis/expanded-pool/replay-quarter/failed-duels.jsonl \
        distilled=analysis/expanded-pool/replay-distilled/failed-duels.jsonl

Each row of a run's `failed-duels.jsonl` is a duel the engine ended with an error. A
script's error names its file and line (`[string "c39163598.lua"]:19: ...`); any other
failure is grouped by its message. One column a run, the most failures first, with the
Decks that hold the card: a game's Decks are new, any other one of the seventy-one.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "native/ygo-policies/vendor/BabelCdb/cards.cdb"
SCRIPT = re.compile(r'\[string "([^"]+)\.lua"\]:(\d+):\s*(.*)')
CARD = re.compile(r"c(\d+)")
HOLDERS = 3


def failure(text: str) -> tuple[int | None, str]:
    """The card whose script failed, None for any other failure, and where.

    A failure carries the engine's log before its message: the message is its last line.
    """
    match = SCRIPT.search(text)
    if match is None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return None, (lines[-1] if lines else "")[:70]
    script, line, message = match.groups()
    where = f"{line}: {message.strip()[:50]}"
    card = CARD.fullmatch(script)
    if card is None:
        return None, f"{script}.lua:{where}"
    return int(card.group(1)), where


def holders(row: dict, code: int | None) -> list[str]:
    """The Decks seated in a failed duel that hold the card."""
    if code is None:
        return []
    decks = row["setup"]["decks"]
    lists = row["record"]["decks"]
    return [
        deck
        for deck, cards in zip(decks, lists, strict=True)
        if code in cards.get("main", []) or code in cards.get("extra", [])
    ]


def grouped(path: Path) -> tuple[Counter, dict[tuple[int | None, str], Counter]]:
    """Each failure's count of duels, and of the Decks holding its card in them."""
    counts: Counter = Counter()
    held: dict = defaultdict(Counter)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = failure(row["failure"])
        counts[key] += 1
        held[key].update(holders(row, key[0]))
    return counts, held


def names(codes, database: Path = DATABASE) -> dict[int, str]:
    wanted = sorted(code for code in codes if code is not None)
    if not wanted or not database.exists():
        return {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        marks = ",".join("?" * len(wanted))
        rows = connection.execute(f"select id, name from texts where id in ({marks})", wanted)
        return dict(rows.fetchall())


def table(runs: dict[str, tuple[Counter, dict]]) -> list[str]:
    keys = {key for counts, _ in runs.values() for key in counts}
    known = names(code for code, _ in keys)
    totals = {key: [counts[key] for counts, _ in runs.values()] for key in keys}
    lines = [f"{'card':34} {'line':48} " + " ".join(f"{name:>10}" for name in runs) + "  held by"]
    for key in sorted(keys, key=lambda k: (-sum(totals[k]), str(k))):
        code, where = key
        card = f"{known.get(code, code)} ({code})" if code is not None else "-"
        held: Counter = Counter()
        for _, decks in runs.values():
            held.update(decks.get(key, Counter()))
        holding = ", ".join(deck for deck, _ in held.most_common(HOLDERS))
        counts = " ".join(f"{count:10}" for count in totals[key])
        lines.append(f"{card[:34]:34} {where[:48]:48} {counts}  {holding}")
    every = " ".join(f"{sum(counts.values()):10}" for counts, _ in runs.values())
    lines.append(f"{'all':34} {'':48} {every}")
    return lines


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = {}
    for argument in sys.argv[1:]:
        name, _, path = argument.rpartition("=")
        runs[name or Path(path).parent.name] = grouped(Path(path))
    print("\n".join(table(runs)))


if __name__ == "__main__":
    main()
