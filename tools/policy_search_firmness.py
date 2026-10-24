"""How firmly a search decided where it left its pilot.

Reads `policy-bench search --log true` rows, where every decision the search
examined is recorded with how each alternative did in every world it was
played in.  The search tries the alternatives in stages (8, 32 and 96 worlds)
and takes one only when it is ahead of the pilot's answer by a paired z over
the same worlds.  This says how much of such a decision is the position and
how much is the worlds that happened to be drawn:

- how far the test went at each decision, and at how many an alternative ended
  ahead and was not taken;
- the lead of the alternatives that went on after the middle stage, in the
  worlds that chose them and in the later worlds, which had no part in it;
- how often each deviation is found again when its worlds are drawn again
  from those it was played in.  The deviations are the ones that passed, so
  their leads are flattering and this is an upper bound.

Usage, from the repository root::

    uv run python tools/policy_search_firmness.py analysis/search/blackwing-seed730000-32-logged.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# As in the bench's `engine/search.rs`.
EARLY_ACCEPT = 2.33
EARLY_REJECT = 0.67
KEPT = 2
PAYOFF = {"+": 1.0, "-": -1.0, "0": 0.0}


def payoffs(played: str) -> list[float | None]:
    """A seat's payoff in each world an answer was played in."""
    return [PAYOFF.get(mark) for mark in played]


def paired(answer: list[float | None], pilot: list[float | None]) -> tuple[float, float]:
    """Mean and z of an answer's payoff over the pilot's, world by world."""
    differences = [a - b for a, b in zip(answer, pilot) if a is not None and b is not None]
    count = len(differences)
    if count < 2:
        return 0.0, 0.0
    mean = sum(differences) / count
    variance = sum((d - mean) ** 2 for d in differences) / (count - 1)
    if variance > 0.0:
        return mean, mean / math.sqrt(variance / count)
    return mean, math.inf if mean > 0.0 else -math.inf if mean < 0.0 else 0.0


def staged(results: list[list[float | None]], stages: tuple[int, ...], z: float) -> int | None:
    """The answer the search takes, given every answer's payoffs (the pilot's
    first): its place, or None when a stage needs worlds that were not played."""
    pilot = results[0]
    if all(payoff is None or payoff >= 1.0 for payoff in pilot[: stages[0]]):
        return 0
    alive = list(range(1, len(results)))
    for stage, total in enumerate(stages):
        if len(pilot) < total or any(len(results[index]) < total for index in alive):
            return None
        ranked = []
        for index in alive:
            mean, score = paired(results[index][:total], pilot[:total])
            if mean > 0.0:
                ranked.append((index, mean, score))
        ranked.sort(key=lambda entry: -entry[1])
        if not ranked:
            return 0
        best, last = ranked[0], stage + 1 == len(stages)
        if best[2] >= z if last else stage > 0 and best[2] >= EARLY_ACCEPT:
            return best[0]
        if last or (stage > 0 and best[2] < EARLY_REJECT):
            return 0
        alive = [entry[0] for entry in ranked[:KEPT]]
    return 0


def found_again(
    pilot: list[float | None],
    answer: list[float | None],
    stages: tuple[int, ...],
    z: float,
    draws: int,
    random: np.random.Generator,
) -> float:
    """How often the staged test takes `answer` again when the worlds are
    drawn, with replacement, from those both answers were played in."""
    pairs = np.array([(p, a) for p, a in zip(pilot, answer) if p is not None and a is not None])
    sample = pairs[random.integers(0, len(pairs), size=(draws, stages[-1]))]
    base, difference = sample[:, :, 0], sample[:, :, 1] - sample[:, :, 0]

    def score(total: int) -> tuple[np.ndarray, np.ndarray]:
        mean = difference[:, :total].mean(axis=1)
        error = difference[:, :total].std(axis=1, ddof=1) / math.sqrt(total)
        with np.errstate(divide="ignore", invalid="ignore"):
            return mean, np.where(error > 0.0, mean / error, np.sign(mean) * np.inf)

    going = ~(base[:, : stages[0]] >= 1.0).all(axis=1)
    taken = np.zeros(draws, dtype=bool)
    for stage, total in enumerate(stages):
        mean, value = score(total)
        going &= mean > 0.0
        last = stage + 1 == len(stages)
        accepted = going & (value >= (z if last else EARLY_ACCEPT)) & (last or stage > 0)
        taken |= accepted
        going &= ~accepted
        if stage > 0:
            going &= value >= EARLY_REJECT
    return float(taken.mean())


def quantiles(values: list[float], points: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> str:
    return "  ".join(f"{np.quantile(values, point):.2f}" for point in points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rows", type=Path)
    parser.add_argument("--stages", type=int, nargs=3, default=(8, 32, 96))
    parser.add_argument("--z", type=float, default=1.645)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stages = tuple(args.stages)
    middle, final = stages[1], stages[2]
    random = np.random.default_rng(args.seed)

    games = examined = skipped = mismatched = 0
    reached: Counter[int] = Counter()
    ahead: Counter[int] = Counter()
    taken_at: Counter[int] = Counter()
    deviations: list[dict[str, Any]] = []
    # The alternative leading after the middle stage, where the test went on:
    # its lead up to there, its lead after, and whether it was taken.
    finalists: list[tuple[float, float, bool]] = []
    # Every alternative played to the end, on the later worlds alone.
    later_scores: list[float] = []
    with args.rows.open(encoding="utf8") as rows:
        for text in rows:
            if not text.strip():
                continue
            row = json.loads(text)
            if "error" in row["search"]:
                continue
            games += 1
            left = {item["decision"]: item for item in row["search"]["deviations"]}
            for entry in row["search"]["examined"]:
                examined += 1
                results = [payoffs(played) for played in entry["played"]]
                chosen = int(entry["chosen"])
                mismatched += staged(results, stages, args.z) != chosen
                if entry["skipped"]:
                    skipped += 1
                    continue
                worlds = len(results[0])
                reached[worlds] += 1
                if chosen:
                    taken_at[worlds] += 1
                    item = left[entry["decision"]]
                    deviations.append(
                        {
                            "kind": entry["kind"],
                            "worlds": worlds,
                            "z": float(item["z"]),
                            "gain": float(item["gain"]),
                            "again": found_again(
                                results[0], results[chosen], stages, args.z, args.draws, random
                            ),
                        }
                    )
                elif any(
                    paired(answer, results[0][: len(answer)])[0] > 0.0
                    for answer in results[1:]
                    if len(answer) == worlds
                ):
                    ahead[worlds] += 1
                if worlds < final:
                    continue
                played_out = [
                    index for index in range(1, len(results)) if len(results[index]) >= final
                ]
                for index in played_out:
                    later_scores.append(
                        paired(results[index][middle:final], results[0][middle:final])[1]
                    )
                leader = max(
                    played_out,
                    key=lambda index: paired(results[index][:middle], results[0][:middle])[0],
                )
                finalists.append(
                    (
                        paired(results[leader][:middle], results[0][:middle])[0],
                        paired(results[leader][middle:final], results[0][middle:final])[0],
                        chosen == leader,
                    )
                )

    print(f"{games} games, {examined} decisions examined, {len(deviations)} deviations")
    if mismatched:
        print(f"WARNING: the staged test read here gives another answer at {mismatched} decisions")
    print(f"  the pilot's answer wins every first-stage world, nothing else tried: {skipped}")
    for worlds in sorted(reached):
        print(
            f"  decided after {worlds:>3} worlds: {reached[worlds]:>6}, an alternative taken at "
            f"{taken_at[worlds]}, one ahead and not taken at {ahead[worlds]}"
        )

    # A payoff is 1, 0 or -1: half a difference of payoffs is a difference of win rates.
    first = [entry[0] for entry in finalists]
    later = [entry[1] for entry in finalists]
    taken = [entry for entry in finalists if entry[2]]
    kept = [entry for entry in finalists if not entry[2]]
    print()
    print(
        f"The alternative leading after {middle} worlds, where the test went on ({len(finalists)} decisions),"
    )
    print(
        f"in points of win rate over the pilot's answer: in those {middle} worlds, and in the next {final - middle}:"
    )
    for name, group in (("all", finalists), ("taken in the end", taken), ("not taken", kept)):
        if group:
            print(
                f"  {name:<17}{len(group):>6}   {50 * np.mean([e[0] for e in group]):+5.1f}"
                f"   {50 * np.mean([e[1] for e in group]):+5.1f}"
            )
    correlation = float(np.corrcoef(first, later)[0, 1]) if len(finalists) > 2 else math.nan
    print(f"  correlation of the two leads: {correlation:+.2f}")
    above = sum(score >= args.z for score in later_scores)
    below = sum(score <= -args.z for score in later_scores)
    print(
        f"  of {len(later_scores)} alternatives played to the end, on the later worlds alone: "
        f"{above} at z >= {args.z}, {below} at z <= -{args.z}"
    )

    again = [item["again"] for item in deviations]
    print()
    print(f"The {len(deviations)} deviations (quantiles at 10, 25, 50, 75 and 90%):")
    print(f"  z                         {quantiles([item['z'] for item in deviations])}")
    print(f"  gain, points of win rate  {quantiles([50 * item['gain'] for item in deviations])}")
    print(f"  found again, worlds drawn again from those played: {np.mean(again):.1%} on average")
    print(f"    by deviation            {quantiles(again)}")
    print(
        f"    under one time in two   {np.mean([value < 0.5 for value in again]):.1%} of the deviations"
    )
    for kind in sorted({item["kind"] for item in deviations}):
        group = [item["again"] for item in deviations if item["kind"] == kind]
        print(f"    {kind:<9}{len(group):>5}   {np.mean(group):.1%}")
    # The ranges `search_imitation_report.py --records` sets a model's answers apart by.
    by_z: dict[str, dict[str, float]] = {}
    low = 0.0
    for limit, name in (
        (2.0, "z under 2"),
        (2.5, "2 to 2.5"),
        (3.0, "2.5 to 3"),
        (math.inf, "3 and more"),
    ):
        group = [item for item in deviations if low <= item["z"] < limit]
        low = limit
        if group:
            by_z[name] = {
                "deviations": len(group),
                "found_again": float(np.mean([item["again"] for item in group])),
                "median_gain": float(np.median([item["gain"] for item in group])) / 2,
            }
            print(
                f"    {name:<11}{len(group):>5}   {by_z[name]['found_again']:.1%}"
                f"   median gain {100 * by_z[name]['median_gain']:.1f} points"
            )

    if args.output:
        report = {
            "games": games,
            "examined": examined,
            "skipped": skipped,
            "mismatched": mismatched,
            "decided_after": {str(worlds): reached[worlds] for worlds in sorted(reached)},
            "taken_after": {str(worlds): taken_at[worlds] for worlds in sorted(taken_at)},
            "ahead_not_taken_after": {str(worlds): ahead[worlds] for worlds in sorted(ahead)},
            "finalists": len(finalists),
            "finalists_lead_first": float(np.mean(first)) / 2 if finalists else None,
            "finalists_lead_later": float(np.mean(later)) / 2 if finalists else None,
            "finalists_lead_correlation": correlation,
            "later_alone_above": above,
            "later_alone_below": below,
            "deviations": len(deviations),
            "found_again": float(np.mean(again)) if again else None,
            "found_again_under_half": float(np.mean([value < 0.5 for value in again]))
            if again
            else None,
            "by_z": by_z,
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")


if __name__ == "__main__":
    main()
