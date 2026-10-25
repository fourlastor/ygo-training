"""The weights of a learner's duels against itself, by how far behind each
pairing of its decks is (experiments/2026-10-24-four-decks.md, its weighted
arm): from a measurement of the learner against `all71` on the other side of
every ordered pairing, the file `train --self-play-weights` reads.

    uv run python tools/pairing_weights.py measured.json \\
        --decks blackwing,cyber,lightsworn,infernity --output weights.json \\
        --shares shares.json [--before earlier-shares.json]

A pairing's share is the learner's against `all71`: a mirror's from its own
case, every deal from both seats; two decks' the mean of the two cases, each
deck once in the learner's hands. With `--before`, the mean of this
measurement's and the one before. Each pairing then weighs (1 - share)^POWER
times its ordered pairs (one for a mirror, two for two decks), its share of
the weights at least FLOOR of its share where every ordered pair counts the
same; two decks' weight is split evenly between their two seatings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

POWER = 2.0
FLOOR = 0.25


def ordered_shares(measured: dict[str, Any], suite: dict[str, Any]) -> dict[tuple[str, str], float]:
    """The learner's share of each ordered pairing of a measurement, by its
    suite's cases."""
    (record,) = measured["records"]
    played = record["evaluation_suite"]
    found = {}
    for case in suite["cases"]:
        outcomes = played[case["name"]]["outcomes"]
        found[(case["learner"], case["opponent"])] = sum(o == "win" for o in outcomes) / len(
            outcomes
        )
    return found


def pairing_shares(
    ordered: dict[tuple[str, str], float], decks: list[str]
) -> dict[tuple[str, str], float]:
    """Each pairing's share, a mirror's its own, two decks' the mean of their
    two seatings: by the decks in their order."""
    found = {}
    for place, one in enumerate(decks):
        for other in decks[place:]:
            seatings = [(one, other)] if one == other else [(one, other), (other, one)]
            missing = [seating for seating in seatings if seating not in ordered]
            if missing:
                raise SystemExit(f"The measurement plays no {' or '.join(map(str, missing))}")
            found[(one, other)] = sum(ordered[seating] for seating in seatings) / len(seatings)
    return found


def weights_of(
    shares: dict[tuple[str, str], float], decks: list[str], power: float, floor: float
) -> dict[str, float]:
    """The ordered pairs' weights, summing to one."""
    seatings = {pairing: 1 if pairing[0] == pairing[1] else 2 for pairing in shares}
    raw = {pairing: seatings[pairing] * (1 - share) ** power for pairing, share in shares.items()}
    least = {pairing: floor * seatings[pairing] / len(decks) ** 2 for pairing in shares}
    fixed: dict[tuple[str, str], float] = {}
    while True:
        free = {pairing: value for pairing, value in raw.items() if pairing not in fixed}
        left = 1 - sum(fixed.values())
        total = sum(free.values())
        shared = {
            pairing: left * value / total if total > 0 else left / len(free)
            for pairing, value in free.items()
        }
        under = [pairing for pairing, value in shared.items() if value < least[pairing]]
        if not under:
            break
        for pairing in under:
            fixed[pairing] = least[pairing]
    fractions = {**fixed, **shared}
    weights = {}
    for (one, other), fraction in fractions.items():
        if one == other:
            weights[f"{one}>{other}"] = fraction
        else:
            weights[f"{one}>{other}"] = weights[f"{other}>{one}"] = fraction / 2
    return dict(sorted(weights.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("measured", type=Path, help="evaluate-suite's output on the pairings")
    parser.add_argument("--decks", required=True, help="the learner's decks, by commas")
    parser.add_argument("--output", type=Path, required=True, help="the weights, for train")
    parser.add_argument("--shares", type=Path, required=True, help="the shares, for the record")
    parser.add_argument("--before", type=Path, help="the shares of the measurement before")
    parser.add_argument("--suite", type=Path, help="the suite measured (the measurement's own)")
    parser.add_argument("--power", type=float, default=POWER)
    parser.add_argument("--floor", type=float, default=FLOOR)
    args = parser.parse_args()
    decks = [deck.strip() for deck in args.decks.split(",") if deck.strip()]
    measured = json.loads(args.measured.read_text(encoding="utf8"))
    suite = json.loads(Path(args.suite or measured["suite"]).read_text(encoding="utf8"))
    ordered = ordered_shares(measured, suite)
    now = pairing_shares(ordered, decks)
    shares = dict(now)
    if args.before is not None:
        before = json.loads(args.before.read_text(encoding="utf8"))["measured"]
        shares = {
            pairing: (share + before[f"{pairing[0]}|{pairing[1]}"]) / 2
            for pairing, share in now.items()
        }
    weights = weights_of(shares, decks, args.power, args.floor)
    args.output.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf8")
    args.shares.write_text(
        json.dumps(
            {
                "measured": {f"{one}|{other}": share for (one, other), share in now.items()},
                "shares": {f"{one}|{other}": share for (one, other), share in shares.items()},
                "ordered": {f"{one}>{other}": share for (one, other), share in ordered.items()},
                "weights": weights,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    for (one, other), share in shares.items():
        told = weights[f"{one}>{other}"] * (1 if one == other else 2)
        print(f"  {one} against {other}: {share:.1%}, {told:.1%} of the duels against itself")


if __name__ == "__main__":
    main()
