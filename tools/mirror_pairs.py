"""Networks' mirrors paired deal by deal against the first network's: each of the
seventy-one mirrors head to head against its champion on the deals of the check suites
(`analysis/seventy-one-champions/checks/suites`), as `experiments/commands/pool-batch.sh`
measures them (experiments/2026-11-02-expanded-pool.md, the batch again).

    uv run python tools/mirror_pairs.py \\
        champions=analysis/seventy-one-champions/checks/20160 \\
        quarter=analysis/expanded-pool/replay-quarter/9000/mirrors

Each directory holds `DECK.plays.json`, the network the learner against the champion,
and `DECK.held.json`, the champion the learner against the network. A network's share of
a deal is its result, a draw half of one; its difference to the first network's is taken
deal by deal over the decks both measured, with the standard error of those differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learner_report import mean_error, points
from two_decks_report import mean

SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}
# How many of the decks that moved most are named each way.
NAMED = 8


def outcomes(path: Path) -> list[str]:
    return json.loads(path.read_text())["records"][0]["evaluation"]["outcomes"]


def shares(directory: Path) -> dict[str, list[float]]:
    """Each mirror's deals, the network's share of each: its own games, then the
    champion's at the other side of their result."""
    found = {}
    for plays in sorted(directory.glob("*.plays.json")):
        deck = plays.name.removesuffix(".plays.json")
        held = outcomes(directory / f"{deck}.held.json")
        found[deck] = [SCORE[o] for o in outcomes(plays)] + [1.0 - SCORE[o] for o in held]
    return found


def against(first: dict[str, list[float]], second: dict[str, list[float]]) -> dict[str, Any]:
    """The second network's mirrors less the first's, deal by deal."""
    decks = sorted(set(first) & set(second))
    differences = []
    for deck in decks:
        if len(first[deck]) != len(second[deck]):
            raise ValueError(f"{deck}: {len(first[deck])} deals against {len(second[deck])}")
        differences += [b - a for a, b in zip(first[deck], second[deck])]
    difference, error = mean_error(differences)
    by_deck = sorted((mean(second[deck]) - mean(first[deck]), deck) for deck in decks)
    return {
        "decks": len(decks),
        "difference": difference,
        "error": error,
        "lower": sum(moved < 0 for moved, _ in by_deck),
        "by deck": by_deck,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "networks",
        nargs="+",
        metavar="NAME=DIRECTORY",
        help="the first is the one the others are paired against",
    )
    args = parser.parse_args()
    named = []
    for argument in args.networks:
        name, _, directory = argument.partition("=")
        named.append((name, shares(Path(directory))))
    for name, found in named:
        games = sum(len(deals) for deals in found.values())
        share = mean([mean(deals) for deals in found.values()])
        print(f"{name}: {len(found)} decks, {100 * share:.1f}%, {games:,} games")
    (first_name, first), others = named[0], named[1:]
    for name, found in others:
        paired = against(first, found)
        print(
            f"{name} against {first_name}: {points(paired['difference'], paired['error'])} "
            f"points over {paired['decks']} decks, lower on {paired['lower']}"
        )
        moved = paired["by deck"]
        print("  most lost:", ", ".join(f"{deck} {points(x)}" for x, deck in moved[:NAMED]))
        print("  most won:", ", ".join(f"{deck} {points(x)}" for x, deck in moved[-NAMED:]))


if __name__ == "__main__":
    main()
