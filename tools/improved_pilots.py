"""The improved pilots on `all71`'s yardstick
(experiments/2026-10-23-improved-pilots.md), told without a mark.

    uv run python tools/improved_pilots.py PINNED BRANCH --network analysis/seventy-one-decks

PINNED holds the pilots as the trainer pins them (`new.json`, `old.json`)
and BRANCH as the branch has them, each the forty decks' and the eighteen
decks' suite
played as seventy-one decks played them (`evaluate-suite --pilot`, seed 2012,
24 workers). A deck the branch left as it was wins exactly the games it won;
one that does not is told. For each deck whose games changed: each pilot's
share, the gain with its standard error (the same games, paired), `all71`'s
share of the same games by its three checkpoints at R3 (`--network`:
`new/all71-10560.json` and `old/all71-10560.json` there), and the share of the
gap between the pinned pilot and `all71` the branch's pilot closes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# The suites of seventy-one decks, as its measurements name them: the forty
# decks new to the pool, and the eighteen other decks of thirty-one, each
# against the thirteen decks' pilots.
SUITES = ("new", "old")
NETWORK = "all71-10560"


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


def report(pinned_at: Path, branch_at: Path, network: Path) -> dict[str, Any]:
    pinned: dict[str, list[float]] = {}
    branch: dict[str, list[float]] = {}
    theirs: dict[str, list[float]] = {}
    for kind in SUITES:
        pinned.update(shares(won(pinned_at / f"{kind}.json")))
        branch.update(shares(won(branch_at / f"{kind}.json")))
        theirs.update(shares(won(network / kind / f"{NETWORK}.json")))
    if set(pinned) != set(branch):
        raise SystemExit("The pinned pilots and the branch's played other cases")
    ours, before = by_deck(branch), by_deck(pinned)
    changed = sorted(deck for deck in ours if ours[deck] != before[deck])
    network_games = by_deck(theirs, set(changed))
    result: dict[str, Any] = {
        "decks": len(ours),
        "unchanged": len(ours) - len(changed),
        "changed": {},
    }
    for deck in changed:
        new, old, net = ours[deck], before[deck], network_games[deck]
        gain = [one - other for one, other in zip(new, old, strict=True)]
        gap = [one - other for one, other in zip(net, old, strict=True)]
        told: dict[str, Any] = {
            "games": len(new),
            "pinned": mean(old),
            "branch": mean(new),
            "network": mean(net),
            "gain": mean(gain),
            "gain_error": error(gain),
            "gap": mean(gap),
            "gap_error": error(gap),
        }
        if told["gap"] > 0:
            ratio = told["gain"] / told["gap"]
            rest = [one - ratio * other for one, other in zip(gain, gap, strict=True)]
            told["closed"] = ratio
            told["closed_error"] = error(rest) / told["gap"]
        result["changed"][deck] = told
    return result


def shown(result: dict[str, Any]) -> str:
    lines = [
        (
            f"The branch's pilots on all71's yardstick: {len(result['changed'])} of "
            f"{result['decks']} decks play other games, {result['unchanged']} the same games."
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
        lines.append(line)
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
