"""The improved pilots on `all71`'s yardstick
(experiments/2026-10-23-improved-pilots.md), told without a mark.

    uv run python tools/improved_pilots.py PINNED BRANCH --network analysis/seventy-one-decks

PINNED holds the pilots as the trainer pins them (`new.json`, `old.json`)
and BRANCH as the branch has them, each the forty decks' and the eighteen
decks' suite
played as seventy-one decks played them (`evaluate-suite --pilot`, seed 2012,
24 workers), and where both hold them the thirteen decks' screens
(`screen-DECK.json`). A deck the branch left as it was wins exactly the
games it won; one that does not is told. For each deck whose games changed:
each pilot's share, the gain with its standard error (the same games,
paired), `all71`'s share of the same games by its three checkpoints at R3
(`--network`: `new/all71-10560.json` and `old/all71-10560.json` there, and
`screens/all71-10560.DECK.json`), and the share of the gap between the pinned
pilot and `all71` the branch's pilot closes. Where BRANCH says which decks'
pilots the branch changed (`changed.txt`, the diff of the policies), the
games against those of the thirteen are no longer those `all71` played: a
deck is measured on its games against the others, and against each changed
opponent the games of every deck whose own pilot is as it was are told
together, without a mark: how much more or less its improved pilot gives the
others.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# The suites of seventy-one decks, as its measurements name them: the forty
# decks new to the pool, and the eighteen other decks of thirty-one, each
# against the thirteen decks' pilots; and each of the thirteen's own screen.
THIRTEEN = (
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
)
SUITES = ("new", "old", *(f"screen-{deck}" for deck in THIRTEEN))
NETWORK = "all71-10560"
# A deck's pilot, where the diff of the policies changes it.
CHANGED = re.compile(r"src/decks/([a-z_]+)\.rs")


def deck_of(case: str) -> str:
    return case.split("-vs-")[0]


def won(path: Path) -> dict[str, list[list[float]]]:
    """For each case of a file, each record's games: 1 a win, 0 otherwise."""
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    records = json.loads(path.read_text(encoding="utf8"))["records"]
    found: dict[str, list[list[float]]] = {}
    for record in records:
        for name, case in record["evaluation_suite"].items():
            if "outcomes" not in case:
                raise SystemExit(f"{path}: {name} has no outcomes")
            found.setdefault(name, []).append([float(told == "win") for told in case["outcomes"]])
    return found


def shares(found: dict[str, list[list[float]]]) -> dict[str, list[float]]:
    """Each game's share of the records that win it."""
    return {
        name: [sum(game) / len(game) for game in zip(*records, strict=True)]
        for name, records in found.items()
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def error(values: list[float]) -> float:
    centre = mean(values)
    return math.sqrt(
        sum((value - centre) ** 2 for value in values) / (len(values) - 1)
    ) / math.sqrt(len(values))


def by_deck(games: dict[str, list[float]], decks: set[str] | None = None) -> dict[str, list[float]]:
    """Each deck's games, its cases in the order of their names."""
    found: dict[str, list[float]] = {}
    for name in sorted(games):
        deck = deck_of(name)
        if decks is None or deck in decks:
            found.setdefault(deck, []).extend(games[name])
    return found


def network_file(network: Path, kind: str) -> Path:
    """Where `all71` played a suite: a screen's under `screens/`."""
    if kind.startswith("screen-"):
        return network / "screens" / f"{NETWORK}.{kind.removeprefix('screen-')}.json"
    return network / kind / f"{NETWORK}.json"


def improved(branch_at: Path) -> set[str] | None:
    """The decks whose pilots the branch changed, by the diff it keeps."""
    path = branch_at / "changed.txt"
    if not path.is_file():
        return None
    return {name.replace("_", "-") for name in CHANGED.findall(path.read_text(encoding="utf8"))}


def opponent_of(case: str) -> str:
    return case.split("-vs-", 1)[1]


def gained(new: list[float], old: list[float]) -> dict[str, Any]:
    gain = [one - other for one, other in zip(new, old, strict=True)]
    return {
        "games": len(new),
        "pinned": mean(old),
        "branch": mean(new),
        "gain": mean(gain),
        "gain_error": error(gain),
    }


def report(pinned_at: Path, branch_at: Path, network: Path) -> dict[str, Any]:
    pinned: dict[str, list[float]] = {}
    branch: dict[str, list[float]] = {}
    theirs: dict[str, list[float]] = {}
    kinds = [
        kind
        for kind in SUITES
        if kind in ("new", "old")
        or ((pinned_at / f"{kind}.json").is_file() and (branch_at / f"{kind}.json").is_file())
    ]
    for kind in kinds:
        pinned.update(shares(won(pinned_at / f"{kind}.json")))
        branch.update(shares(won(branch_at / f"{kind}.json")))
        theirs.update(shares(won(network_file(network, kind))))
    if set(pinned) != set(branch):
        raise SystemExit("The pinned pilots and the branch's played other cases")
    own = improved(branch_at)
    # The thirteen opponents whose pilots the branch changed: games against
    # them are no longer those all71 played, and are told apart.
    moved = set(THIRTEEN) & own if own is not None else set()

    def part(games: dict[str, list[float]], against_moved: bool) -> dict[str, list[float]]:
        return {
            name: values
            for name, values in games.items()
            if (opponent_of(name) in moved) == against_moved
        }

    ours, before = by_deck(part(branch, False)), by_deck(part(pinned, False))
    changed = sorted(deck for deck in ours if ours[deck] != before[deck])
    network_games = by_deck(part(theirs, False), set(changed))
    result: dict[str, Any] = {
        "decks": len(ours),
        "unchanged": len(ours) - len(changed),
        "changed": {},
        "screens": [kind.removeprefix("screen-") for kind in kinds if kind.startswith("screen-")],
        "moved": sorted(moved),
    }
    for deck in changed:
        new, old, net = ours[deck], before[deck], network_games[deck]
        gap = [one - other for one, other in zip(net, old, strict=True)]
        told = gained(new, old)
        told.update({"network": mean(net), "gap": mean(gap), "gap_error": error(gap)})
        if own is not None and deck not in own:
            # Its pilot as it was, and its games changed against opponents
            # as they were: something else the branch changed.
            told["unexplained"] = True
        gain = [one - other for one, other in zip(new, old, strict=True)]
        if told["gap"] > 0:
            ratio = told["gain"] / told["gap"]
            rest = [one - ratio * other for one, other in zip(gain, gap, strict=True)]
            told["closed"] = ratio
            told["closed_error"] = error(rest) / told["gap"]
        result["changed"][deck] = told
    if moved:
        # Against each changed opponent, the games of every deck whose own
        # pilot is as it was, together: how much more, or less, its improved
        # pilot gives the others.
        against = {
            name: values
            for name, values in part(branch, True).items()
            if deck_of(name) not in (own or set())
        }
        result["against the moved"] = {
            opponent: gained(
                [
                    game
                    for name in sorted(against)
                    if opponent_of(name) == opponent
                    for game in against[name]
                ],
                [
                    game
                    for name in sorted(against)
                    if opponent_of(name) == opponent
                    for game in pinned[name]
                ],
            )
            for opponent in sorted(moved)
        }
    return result


def shown(result: dict[str, Any]) -> str:
    moved = result.get("moved") or []
    lines = [
        (
            f"The branch's pilots on all71's yardstick: {len(result['changed'])} of "
            f"{result['decks']} decks play other games, {result['unchanged']} the same games"
            + (
                f", against the opponents whose pilots are as they were ({len(moved)} changed: "
                f"{', '.join(moved)})."
                if moved
                else "."
            )
        )
    ]
    for deck, told in result["changed"].items():
        line = (
            f"  {deck}: {told['branch']:.1%} against {told['pinned']:.1%} as pinned, "
            f"{100 * told['gain']:+.1f} ± {100 * told['gain_error']:.1f} points over "
            f"{told['games']} games; all71 {told['network']:.1%} "
            f"({100 * told['gap']:+.1f} ± {100 * told['gap_error']:.1f} over the pinned pilot)"
        )
        if "closed" in told:
            line += f"; the gap closed {told['closed']:.0%} ± {told['closed_error']:.0%}"
        if told.get("unexplained"):
            line += "; its own pilot as it was"
        lines.append(line)
    for opponent, told in (result.get("against the moved") or {}).items():
        lines.append(
            f"  Against {opponent}'s pilot as the branch has it, the decks of pilots as they were: "
            f"{told['branch']:.1%} against {told['pinned']:.1%}, "
            f"{100 * told['gain']:+.1f} ± {100 * told['gain_error']:.1f} points over "
            f"{told['games']} games"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pinned", type=Path, help="the pilots as the trainer pins them")
    parser.add_argument("branch", type=Path, help="the branch's pilots")
    parser.add_argument("--network", type=Path, default=ROOT / "analysis/seventy-one-decks")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.pinned, args.branch, args.network)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(shown(result))


if __name__ == "__main__":
    main()
