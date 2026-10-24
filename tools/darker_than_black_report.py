"""The reading of Darker Than Black, a variant of the Blackwing list
(experiments/2026-10-19-darker-than-black.md).

    uv run python tools/darker_than_black_report.py analysis/darker-than-black \\
        --output analysis/darker-than-black/report.json

The directory is what `experiments/commands/darker-than-black.sh` leaves:
Blackwing's pilot and each network (half, of the experiment on thirteen
decks, and all, of the experiment on thirty-one decks), a network by its
checkpoints after 1,680, 1,800 and 1,920 updates of PPO, on the wide
Blackwing screen with Blackwing's own list (`PLAYER.blackwing.json`) and
with Darker Than Black (`PLAYER.darker-than-black.json`), the same suite on
the same seed.

A game is read by the share of a network's three checkpoints that win it, a
won game a win (a drawn one is not). A network's edge with a list is that
share less the pilot's win, game by game: the network and the pilot play
the list on the same deals against the same opponents, so that what the
list is worth against them is the same for both.

The reading, fixed on 2026-10-19 before anything was built, for each
network: its edge with Darker Than Black less its edge with Blackwing's
list, over the thirteen opponents: kept when the gap less two standard
errors is above -2 points, lost when the gap and two standard errors are
below -2 points, otherwise not decided.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_descriptions_report import kept_rule
from more_decks_report import outcomes
from ppo_head_report import difference
from search_leads_report import percent, points
from thirteen_decks_report import END, dealt, kept

NETWORKS = ("half", "all")
PILOT = "pilot"
LISTS = ("blackwing", "darker-than-black")
MIRROR = "blackwing-vs-blackwing"
NAMES = {
    "half": "half: thirteen decks",
    "all": "all: the thirty-one",
    PILOT: "Blackwing's pilot",
    "blackwing": "Blackwing's list",
    "darker-than-black": "Darker Than Black",
}
READINGS = {"equivalent": "kept", "worse": "lost", "not decided": "not decided"}


def played(path: Path, deals: Any) -> tuple[Any, list[list[float]]]:
    """Whether each game of a file was won, a list for each of its players
    (the pilot, or each checkpoint in the order of its updates), and what
    tells its games, which must be `deals` where they are given."""
    found = kept(path)
    if found is None:
        raise SystemExit(f"Missing {path}")
    told = dealt(found)
    if deals is not None and told != deals:
        raise SystemExit(f"{path}: other games than the others")
    records = sorted(found["records"], key=lambda record: int(record.get("checkpoint_step") or 0))
    for record in records:
        failed = [name for name, case in record["evaluation_suite"].items() if "error" in case]
        if failed:
            raise SystemExit(f"{path}: {', '.join(failed)} failed")
    return told, [outcomes(record["evaluation_suite"]) for record in records]


def cases(deals: Any) -> dict[str, range]:
    """Where each case's games are among a file's."""
    found, start = {}, 0
    for name, games in deals[1]:
        found[name] = range(start, start + games)
        start += games
    return found


def edge(network: list[float], pilot: list[float], games: range | None = None) -> dict[str, float]:
    """A network's share less the pilot's win, game by game, over `games`."""
    if games is not None:
        network, pilot = [network[i] for i in games], [pilot[i] for i in games]
    gap, error = difference(network, pilot)
    return {"gap": gap, "error": error, "games": len(network)}


def report(directory: Path) -> dict[str, Any]:
    games: dict[str, dict[str, list[float]]] = {}
    deals: Any = None
    for listed in LISTS:
        games[listed] = {}
        deals, pilots = played(directory / f"{PILOT}.{listed}.json", deals)
        if len(pilots) != 1:
            raise SystemExit(f"{PILOT}.{listed}: {len(pilots)} players, not the pilot alone")
        games[listed][PILOT] = pilots[0]
        for network in NETWORKS:
            deals, checkpoints = played(directory / f"{network}.{listed}.json", deals)
            if len(checkpoints) != END:
                raise SystemExit(f"{network}.{listed}: {len(checkpoints)} checkpoints, not {END}")
            games[listed][network] = [sum(game) / END for game in zip(*checkpoints, strict=True)]
    where = cases(deals)
    result: dict[str, Any] = {"rates": {}, "edges": {}, "readings": {}, "mirror": {}}
    for listed in LISTS:
        result["rates"][listed] = {
            player: sum(won) / len(won) for player, won in games[listed].items()
        }
    for network in NETWORKS:
        edges = {listed: edge(games[listed][network], games[listed][PILOT]) for listed in LISTS}
        result["edges"][network] = edges
        gap = edges["darker-than-black"]["gap"] - edges["blackwing"]["gap"]
        error = math.hypot(edges["darker-than-black"]["error"], edges["blackwing"]["error"])
        result["readings"][network] = {
            "gap": gap,
            "error": error,
            "games": edges["darker-than-black"]["games"],
            "verdict": READINGS[kept_rule(gap, error)],
        }
        result["mirror"][network] = {
            listed: edge(games[listed][network], games[listed][PILOT], where[MIRROR])
            for listed in LISTS
        }
    result["by_opponent"] = {
        name.split("-vs-")[1]: {
            network: {
                listed: edge(games[listed][network], games[listed][PILOT], span)["gap"]
                for listed in LISTS
            }
            for network in NETWORKS
        }
        for name, span in where.items()
    }
    result["mirror"][PILOT] = {
        listed: sum(games[listed][PILOT][i] for i in where[MIRROR]) / len(where[MIRROR])
        for listed in LISTS
    }
    return result


def show(result: dict[str, Any]) -> str:
    lines = ["The wide Blackwing screen, against the thirteen decks' pilots: won with"]
    lines.append(f"  {'':24}{NAMES['blackwing']:>20}{NAMES['darker-than-black']:>20}")
    for player in (*NETWORKS, PILOT):
        rates = [percent(result["rates"][listed][player]) for listed in LISTS]
        lines.append(f"  {NAMES[player]:24}{rates[0]:>20}{rates[1]:>20}")
    lines.append("Each network's edge over Blackwing's pilot on the same games:")
    for network in NETWORKS:
        told = [
            points(
                result["edges"][network][listed]["gap"], result["edges"][network][listed]["error"]
            )
            for listed in LISTS
        ]
        lines.append(f"  {NAMES[network]:24}{told[0]:>20}{told[1]:>20}")
    lines.append("Against Blackwing's pilot with Blackwing's list alone (the mirror):")
    rates = [percent(result["mirror"][PILOT][listed]) for listed in LISTS]
    lines.append(f"  {NAMES[PILOT] + ', won':24}{rates[0]:>20}{rates[1]:>20}")
    for network in NETWORKS:
        told = [
            points(
                result["mirror"][network][listed]["gap"], result["mirror"][network][listed]["error"]
            )
            for listed in LISTS
        ]
        lines.append(f"  {NAMES[network] + ', its edge':24}{told[0]:>20}{told[1]:>20}")
    lines.append(
        "Opponent by opponent, each network's edge with each list"
        " (BW Blackwing's, DTB Darker Than Black):"
    )
    heads = [f"{network} {short}" for network in NETWORKS for short in ("BW", "DTB")]
    lines.append(f"  {'':16}" + "".join(f"{head:>10}" for head in heads))
    for opponent, edges in result["by_opponent"].items():
        told = [points(edges[network][listed]) for network in NETWORKS for listed in LISTS]
        lines.append(f"  {opponent:16}" + "".join(f"{value:>10}" for value in told))
    lines.append("The reading:")
    for network, reading in result["readings"].items():
        lines.append(
            f"  {NAMES[network]}: the edge with Darker Than Black less that with Blackwing's "
            f"list: {points(reading['gap'], reading['error'])} over {reading['games']:,} games: "
            f"{reading['verdict']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
