"""Blackwing against another deck, head to head (experiments/2026-10-14-more-decks.md,
"Blackwing against another deck, head to head").

    uv run python tools/matchups_report.py analysis/matchups

The directory holds, for each network held in the other seat (`OPPONENT`)
and each deck it plays there (`X`), what `evaluate-suite --output` leaves of
Blackwing against X (`OPPONENT/X.json`), or against X's pilot
(`pilots/X.json`): one record a network playing Blackwing, all on the same
deals, each network named by the run its checkpoint is of. Blackwing's share
is its games won, a drawn one not won; two networks are set beside each other
deal by deal, over each deck and over every deck together. The readings,
fixed on 2026-10-20 before any was measured: against each network held, over
the decks together, `decks-05` against `decks-01`: the matchup's own games
count when `decks-05` is over at 2 points and two standard errors more, the
deck's other games count for more when `decks-01` is, otherwise not told
apart. The pilots are set beside them, without a reading.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_head_report import difference
from thirteen_decks_report import MARK, verdict

NETWORK = re.compile(r"decks-\d+")
MATCHUP = "decks-05"
ALONE = "decks-01"
PILOTS = "pilots"


def played(path: Path) -> dict[str, list[float]]:
    """Whether Blackwing won each game, by the network that played it."""
    found = {}
    for record in json.loads(path.read_text(encoding="utf8"))["records"]:
        (case,) = record["evaluation_suite"].values()
        names = NETWORK.findall(record["checkpoint"])
        if not names:
            raise SystemExit(f"{path}: no network named in {record['checkpoint']}")
        found[names[-1]] = [1.0 if outcome == "win" else 0.0 for outcome in case["outcomes"]]
    return found


def report(directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for held in sorted(path for path in directory.iterdir() if path.is_dir()):
        decks = {path.stem: played(path) for path in sorted(held.glob("*.json"))}
        if not decks:
            continue
        players = sorted(set.intersection(*(set(games) for games in decks.values())))
        together = {
            name: [won for games in decks.values() for won in games[name]] for name in players
        }
        told: dict[str, Any] = {
            "shares": {
                deck: {name: sum(games[name]) / len(games[name]) for name in players}
                for deck, games in decks.items()
            },
            "together": {name: sum(won) / len(won) for name, won in together.items()},
            "games": len(together[players[0]]),
            "beside": {},
        }
        for name in players:
            if name == ALONE:
                continue
            gap, error = difference(together[name], together[ALONE])
            by_deck = {
                deck: dict(
                    zip(("gap", "error"), difference(games[name], games[ALONE]), strict=True)
                )
                for deck, games in decks.items()
            }
            told["beside"][name] = {"gap": gap, "error": error, "by_deck": by_deck}
        if MATCHUP in told["beside"] and held.name != PILOTS:
            found = told["beside"][MATCHUP]
            told["reading"] = verdict(
                found["gap"],
                found["error"],
                MARK,
                "the matchup's own games count",
                "the deck's other games count for more",
            )
        result[held.name] = told
    return result


def show(result: dict[str, Any]) -> str:
    lines = []
    for held, told in result.items():
        players = list(told["together"])
        against = "the decks' pilots" if held == PILOTS else f"{held} playing the other deck"
        lines.append(f"Blackwing against {against}, Blackwing's share:")
        lines.append(f"  {'':12}" + "".join(f"{name:>10}" for name in players))
        for deck, shares in told["shares"].items():
            lines.append(f"  {deck:12}" + "".join(f"{shares[name]:>10.1%}" for name in players))
        lines.append(
            f"  {'together':12}" + "".join(f"{told['together'][name]:>10.1%}" for name in players)
        )
        for name, found in told["beside"].items():
            decks = ", ".join(
                f"{deck} {by['gap'] * 100:+.1f} ± {by['error'] * 100:.1f}"
                for deck, by in found["by_deck"].items()
            )
            lines.append(
                f"  {name} less {ALONE}: {found['gap'] * 100:+.1f} ± {found['error'] * 100:.1f}"
                f" over {told['games']:,} games ({decks})"
            )
        if "reading" in told:
            lines.append(f"  The reading: {told['reading']}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
