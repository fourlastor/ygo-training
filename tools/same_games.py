"""Whether two measurements of a suite are the same games: every case's
outcomes, game for game. A measurement made at once with others, through an
MPS server, held to one made alone (experiments/commands/fourteen-decks.sh).

    uv run python tools/same_games.py BESIDE.json ALONE.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def outcomes(path: Path) -> list[dict[str, list[str]]]:
    """Each checkpoint's outcomes of each case, in the file's order."""
    records = json.loads(path.read_text(encoding="utf8"))["records"]
    return [
        {name: case["outcomes"] for name, case in record["evaluation_suite"].items()}
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first, second = outcomes(args.first), outcomes(args.second)
    games = sum(len(games) for record in first for games in record.values())
    if first != second:
        raise SystemExit(f"{args.first} and {args.second} are other games")
    print(f"{args.first} and {args.second} are the same {games} games")


if __name__ == "__main__":
    main()
