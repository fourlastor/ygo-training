"""The readings of the follow-up on PPO in one pass: a second run of it, a run
at another learning rate, and the first run made long
(experiments/2026-10-09-ppo-with-a-head-that-predicts.md).

    uv run python tools/ppo_long_report.py analysis/ppo-long --long 1920 \\
        --output analysis/ppo-long/report.json

The directory is what `experiments/commands/ppo-long.sh` leaves.  Of the
experiment's first runs: `one-pass.metrics.jsonl`, `heldout/one-pass.json`
and `heldout/start.json`, the checkpoints of the run they all start from.
The two short runs, as long as `one-pass` and from the same checkpoint:
`one-pass-seed`, whose games are drawn with another seed, and
`one-pass-rate`, which learns at another rate.  And `one-pass-long`, which
is `one-pass` on from its last checkpoint: its figures, and one
`heldout/one-pass-long.NNNN.json` for each of its checkpoints measured.

A short run is read as the experiment read its arms: deal by deal, by the
share of its checkpoints after the first at which it wins a deal, against
the share of its start's checkpoints.  The long run is read by its last
three checkpoints, and only when it has had the updates of `--long`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learner_report import column, mean_error, measured, metrics, reading, share, suite
from ppo_head_report import KINDS, difference, ends, left, over_checkpoints, plain
from search_leads_report import percent, points, wins

FIRST = "one-pass"
SEED = "one-pass-seed"
RATE = "one-pass-rate"
LONG = "one-pass-long"
START = "start"
SHORT = (FIRST, SEED, RATE)
# The updates a value head's fit is taken over in a short run.
LAST = 40
# The checkpoints the long run is read by: its last ones.
END = 3
# The marks of the readings.
BETTER = 0.02
ERRORS = 2.0


def run_said(directory: Path, name: str) -> dict[str, Any]:
    """What a run was started with."""
    path = directory / f"{name}.run.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else {}


def long_measured(directory: Path, name: str = LONG) -> dict[int, dict[str, Any]]:
    """A long run's checkpoints on the held-out games, one file each."""
    found: dict[int, dict[str, Any]] = {}
    for path in sorted((directory / "heldout").glob(f"{name}.*.json")):
        for entry in json.loads(path.read_text(encoding="utf8"))["checkpoints"]:
            found[int(entry["checkpoint_step"])] = entry
    return found


def over_pilot(entry: dict[str, Any]) -> tuple[float, float] | None:
    """A checkpoint's deals against the pilot's on the same deals, from the
    count of who won each."""
    counts = entry["deals"].get("paired", {}).get("pilot")
    if not counts or "model_only" not in counts:
        return None
    total = sum(counts.values())
    mine, theirs = counts["model_only"], counts["pilot_only"]
    centre = (mine - theirs) / total
    variance = (mine + theirs - (mine - theirs) ** 2 / total) / total**2
    return centre, math.sqrt(max(variance, 0.0))


def pilot_deals(entries: list[dict[str, Any]]) -> list[float] | None:
    """The pilot's result on every deal, where a measurement holds it."""
    for entry in entries:
        outcomes = entry["deals"].get("references", {}).get("pilot", {}).get("outcomes")
        if outcomes:
            return [float(outcome == "win") for outcome in outcomes]
    return None


def between(rows: dict[int, dict[str, Any]], key: str, after: int, until: int) -> float | None:
    """A figure over the updates after `after` and up to `until`."""
    values = [
        float(rows[update][key])
        for update in range(after + 1, until + 1)
        if update in rows and rows[update].get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def report(
    directory: Path,
    long: int | None = None,
    first: str = FIRST,
    beside: Path | None = None,
) -> dict[str, Any]:
    """`first` names the run the long one goes on from, where it is not
    `one-pass`, and `beside` the folder of the first long run, which this
    one is then set beside checkpoint by checkpoint."""
    continued = f"{first}-long"
    rows = {name: metrics(directory, name) for name in (*SHORT, continued)}
    held = {name: measured(directory, "heldout", name) for name in SHORT}
    start = measured(directory, "heldout", START)
    began = max(start) if start else 0
    result: dict[str, Any] = {"runs": {}, "differences": {}, "readings": {}}
    scores: dict[str, list[float]] = {}

    def over(first: str, second: str, label: str | None = None) -> tuple[float, float] | None:
        if first not in scores or second not in scores:
            return None
        found = difference(scores[first], scores[second])
        result["differences"][label or f"{first} over {second}"] = found
        return found

    def said(found: tuple[float, float] | None) -> str:
        return "not measured" if found is None else points(*found)

    # The runs of the same length.  `one-pass-rate` starts from the same
    # model with the same seed, so its first update has the games of
    # `one-pass`; `one-pass-seed` has others.
    firsts = {
        name: found[0].get("rollout_decision_fingerprint") for name, found in rows.items() if found
    }
    if FIRST in firsts and RATE in firsts:
        same = firsts[FIRST] == firsts[RATE]
        result["rate_same_first_games"] = same
        print(f"The first update of {RATE} has the games of {FIRST}: " + ("yes" if same else "NO"))
    if FIRST in firsts and SEED in firsts:
        other = firsts[FIRST] != firsts[SEED]
        result["seed_other_first_games"] = other
        print(f"The first update of {SEED} has other games: " + ("yes" if other else "NO"))
    for name in SHORT:
        told = run_said(directory, name)
        if told:
            result["runs"].setdefault(name, {}).update(
                training_seed=told.get("training_seed"), learning_rate=told.get("learning_rate")
            )
            print(
                f"  {name}: its games drawn with seed {told.get('training_seed')}, "
                f"a learning rate of {told.get('learning_rate')}"
            )

    print()
    print("The runs of one length: what each head explains of the results of its own games")
    print("before it trains on them, over its last updates; the entropy of its answers and")
    print("the games it wins while it samples them, both at its first and its last updates;")
    print("how far an update moves it; and the suite at its last two evaluations.")
    print(
        f"{'run':<20}{'updates':>9}{'fit':>18}{'correlation':>13}{'entropy':>14}"
        f"{'wins':>16}{'step KL':>9}{'suite':>8}"
    )
    for name in SHORT:
        count = len(rows[name])
        if not count:
            continue
        updates = range(max(1, count - LAST + 1), count + 1)
        fit, error = mean_error(column(rows[name], "duel_outcome_explained_variance", updates))
        correlation, _ = mean_error(column(rows[name], "duel_outcome_correlation", updates))
        moved, _ = mean_error(column(rows[name], "duel_approximate_kl", updates))
        entropy = ends(rows[name], "duel_entropy")
        sampled = ends(rows[name], "learner_win_rate")
        played = [suite(row) for row in rows[name] if "evaluation_suite" in row][-2:]
        result["runs"].setdefault(name, {}).update(
            updates=count,
            fit=fit,
            fit_error=error,
            correlation=correlation,
            entropy=entropy,
            sampled_wins=sampled,
            step_kl=moved,
            suite=sum(played) / len(played) if played else None,
        )
        print(
            f"{name:<20}"
            f"{f'{updates.start}-{count}':>9}"
            f"{share(fit, error):>18}"
            f"{share(correlation):>13}"
            f"{f'{plain(entropy[0])} to {plain(entropy[1])}':>14}"
            f"{f'{percent(sampled[0])} to {percent(sampled[1])}':>16}"
            f"{plain(moved, 3):>9}"
            f"{percent(result['runs'][name]['suite']):>8}"
        )

    measured_runs = [name for name in SHORT if held[name]]
    steps = (
        sorted(set.intersection(*(set(held[name]) for name in measured_runs)))
        if measured_runs
        else []
    )
    later = steps[1:] if len(steps) > 1 else steps
    result["checkpoints"] = {
        "start": sorted(start),
        "runs": [step - began for step in steps],
        "read": [step - began for step in later],
    }
    print()
    if not start or not steps:
        print("The copies: the start or the runs are not measured.")
    else:
        scores[START] = over_checkpoints(start, sorted(start))
        deals = len(scores[START])
        print("The copies: held-out deals won.  The start by its own updates, a run by the")
        print("updates of PPO it had; `read` is what a copy is read by: all of the start's")
        print(f"checkpoints, a run's after {', '.join(str(step - began) for step in later)}.")
        print(f"{START:<20}" + "".join(f"{step:>8}" for step in sorted(start)) + f"{'read':>8}")
        each = [sum(wins(start[step]["deals"])) / deals for step in sorted(start)]
        result["runs"][START] = {
            "deals": dict(zip(sorted(start), each)),
            "deals_read": sum(scores[START]) / deals,
        }
        print(
            f"{'':<20}"
            + "".join(f"{percent(value):>8}" for value in each)
            + f"{percent(result['runs'][START]['deals_read']):>8}"
        )
        print(f"{'run':<20}" + "".join(f"{step - began:>8}" for step in steps) + f"{'read':>8}")
        for name in measured_runs:
            scores[name] = over_checkpoints(held[name], later)
            each = [sum(wins(held[name][step]["deals"])) / deals for step in steps]
            result["runs"].setdefault(name, {}).update(
                deals=dict(zip([step - began for step in steps], each)),
                deals_read=sum(scores[name]) / deals,
            )
            print(
                f"{name:<20}"
                + "".join(f"{percent(value):>8}" for value in each)
                + f"{percent(result['runs'][name]['deals_read']):>8}"
            )
        counts = start[max(start)]["deals"].get("paired", {}).get("pilot")
        if counts:
            print(
                "  on those deals the pilot wins "
                f"{percent((counts['both'] + counts['pilot_only']) / deals)}"
            )

        print()
        print("Deal by deal, over the checkpoints a copy is read by:")
        for name in SHORT:
            print(f"  {name} over its start: {said(over(name, START))}")
        print(f"  {SEED} over {FIRST}: {said(over(SEED, FIRST))}")
        print(f"  {RATE} over {FIRST}: {said(over(RATE, FIRST))}")
        print(f"  {RATE} over {SEED}: {said(over(RATE, SEED))}")

        print()
        print("Where the search kept the pilot's answer, how often the first choice is another:")
        print("at all those decisions and by their kind, at the start's last checkpoint and at")
        print("each run's.")
        print(f"{'':<20}{'all':>8}" + "".join(f"{kind:>8}" for kind in KINDS))
        for name, entry in [(START, start[max(start)])] + [
            (name, held[name][steps[-1]]) for name in measured_runs
        ]:
            rates = [left(entry)] + [left(entry, kind) for kind in KINDS]
            result["runs"][name]["leaves_the_pilot"] = dict(zip(("all", *KINDS), rates))
            print(f"{name:<20}" + "".join(f"{percent(rate):>8}" for rate in rates))

    # The long run: `one-pass` and what was made of it after, as one run.
    measured_long = long_measured(directory, continued)
    joined = {
        int(row["version"]) - began + 1: row for name in (first, continued) for row in rows[name]
    }
    points_measured = {
        step - began: entry for step, entry in {**held[first], **measured_long}.items()
    }
    longest = max(points_measured, default=0)
    result["long"] = {"updates": max(joined, default=0), "measured": sorted(points_measured)}
    # A run that goes on keeps the rate its optimizer was saved with and
    # the seed its games are drawn with: where the two descriptions differ
    # in either, the folder sets the long run after another run's updates.
    apart = [
        f"{what} {before[key]} and {after[key]}"
        for before, after in [(run_said(directory, first), run_said(directory, continued))]
        for key, what in (("learning_rate", "a learning rate of"), ("training_seed", "seed"))
        if key in before and key in after and before[key] != after[key]
    ]
    result["long"]["one_run"] = not apart
    print()
    if not measured_long:
        print(f"The long run: no checkpoint of {continued} is measured.")
    else:
        duels = int(next(iter(joined.values())).get("duels", 0)) if joined else 0
        print(
            f"The long run: {first} and {continued}, which goes on from it, as one run.  After so"
        )
        print("many updates and games: the held-out deals it wins and how many more than the")
        print("pilot on the same deals, the suite, and where its first choice is another answer")
        print("than the pilot's that the search kept, at all such decisions and in chains.")
        print("Over the updates since the row before: the games it wins while it samples, the")
        print("entropy of its answers, and what its head explains of those games' results.")
        print(
            f"{'updates':>8}{'games':>11}{'deals':>8}{'over the pilot':>16}{'suite':>8}"
            f"{'another':>9}{'chains':>8}{'sampled':>9}{'entropy':>9}{'fit':>8}"
        )
        table = []
        before = 0
        for update in sorted(points_measured):
            entry = points_measured[update]
            won = sum(wins(entry["deals"])) / len(entry["deals"]["outcomes"])
            row = joined.get(update)
            played = suite(row) if row and "evaluation_suite" in row else None
            line = {
                "updates": update,
                "games": update * duels,
                "deals": won,
                "over_the_pilot": over_pilot(entry),
                "suite": played,
                "leaves_the_pilot": left(entry),
                "leaves_the_pilot_in_chains": left(entry, "Chain"),
                "sampled_wins": between(joined, "learner_win_rate", before, update),
                "entropy": between(joined, "duel_entropy", before, update),
                "fit": between(joined, "duel_outcome_explained_variance", before, update),
            }
            table.append(line)
            print(
                f"{update:>8}{line['games']:>11,}{percent(won):>8}"
                f"{said(line['over_the_pilot']):>16}{percent(played):>8}"
                f"{percent(line['leaves_the_pilot']):>9}"
                f"{percent(line['leaves_the_pilot_in_chains']):>8}"
                f"{percent(line['sampled_wins']):>9}{plain(line['entropy']):>9}"
                f"{share(line['fit']):>8}"
            )
            before = update
        result["long"]["checkpoints"] = table

    # The first long run, where this is another: the two after the same updates.
    other: dict[int, dict[str, Any]] = {}
    if beside is not None:
        other = {
            step - began: entry
            for step, entry in {
                **measured(beside, "heldout", FIRST),
                **long_measured(beside, LONG),
            }.items()
        }
        shared = sorted(set(other) & set(points_measured))
        result["beside"] = {"folder": str(beside), "updates": shared, "over": {}}
        if shared:
            print()
            print(f"Beside the first long run ({beside}): the held-out deals each wins after the")
            print("same updates, and this run over it deal by deal.")
            print(f"{'updates':>8}{'this run':>10}{'the first':>11}{'over it':>14}")
            for update in shared:
                mine = [float(won) for won in wins(points_measured[update]["deals"])]
                theirs = [float(won) for won in wins(other[update]["deals"])]
                gap = difference(mine, theirs) if len(mine) == len(theirs) else None
                result["beside"]["over"][update] = gap
                print(
                    f"{update:>8}{percent(sum(mine) / len(mine)):>10}"
                    f"{percent(sum(theirs) / len(theirs)):>11}{said(gap):>14}"
                )

    if measured_long and apart:
        print()
        print(
            f"NOT ONE RUN: {continued} does not go on from {first} ({first} then it: "
            f"{'; '.join(apart)})."
        )
        print(
            f"The rows up to {max((step - began for step in held[first]), default=0)} updates "
            "above are another run's, and so is what the long run would be set against:"
        )
        print("its readings are not taken.")
    # The long run's readings are of the run as long as it was to be.
    complete = long is not None and longest == long and len(measured_long) >= END and not apart
    if measured_long and not complete and not apart:
        print(
            f"  the long run is measured as far as {longest} updates"
            + ("" if long is None else f" of {long}")
            + ": its readings are not taken"
        )
    if complete and start:
        last = sorted(points_measured)[-END:]
        entries = [points_measured[update] for update in last]
        scores["end"] = [
            sum(deal) / len(deal) for deal in zip(*(wins(entry["deals"]) for entry in entries))
        ]
        short = sorted(held[first])
        if short:
            scores["after"] = over_checkpoints(held[first], short[-1:])
            scores["early"] = over_checkpoints(held[first], short[1:] if len(short) > 1 else short)
        pilot = pilot_deals(entries)
        if pilot is not None:
            scores["pilot"] = pilot
        result["long"]["read"] = last
        result["long"]["deals_read"] = sum(scores["end"]) / len(scores["end"])
        reference = entries[-1]["deals"].get("references", {})
        print()
        print(
            "Deal by deal, the long run by its checkpoints after "
            f"{', '.join(str(update) for update in last)} updates "
            f"({percent(result['long']['deals_read'])} of the deals):"
        )
        ran = short[-1] - began if short else 0
        print(f"  over its checkpoint after {ran}: {said(over('end', 'after', 'end over after'))}")
        print(
            "  over its checkpoints after "
            f"{', '.join(str(step - began) for step in short[1:])}: "
            f"{said(over('end', 'early', 'end over early'))}"
        )
        print(f"  over its start: {said(over('end', START, 'end over start'))}")
        print(f"  over the pilot: {said(over('end', 'pilot', 'end over pilot'))}")
        if "search" in reference:
            print(
                "  the pilot with its search wins "
                f"{percent(reference['search'].get('win_rate'))} of those deals"
            )
        if other and all(update in other for update in last):
            theirs = [wins(other[update]["deals"]) for update in last]
            scores["other"] = [sum(deal) / len(deal) for deal in zip(*theirs, strict=True)]
            if len(scores["other"]) == len(scores["end"]):
                print(
                    "  over the first long run by its checkpoints after the same updates "
                    f"({percent(sum(scores['other']) / len(scores['other']))}): "
                    f"{said(over('end', 'other', 'end over the first long run'))}"
                )

    def no_worse(found: tuple[float, float] | None) -> bool | None:
        return None if found is None else found[0] >= -ERRORS * found[1]

    def better(found: tuple[float, float] | None) -> bool | None:
        return None if found is None else found[0] >= BETTER and found[0] > ERRORS * found[1]

    differences = result["differences"]
    second = differences.get(f"{SEED} over {START}")
    kept = differences.get("end over after")
    passed = differences.get("end over pilot")
    readings = {
        "second_run_gains": better(second),
        "long_holds": no_worse(kept),
        "long_adds": better(kept),
        "past_the_pilot": better(passed),
    }
    result["readings"] = readings
    mark = f"the mark {BETTER * 100:.0f} points and {ERRORS:.0f} standard errors"
    print()
    print("The readings:")
    print(
        reading(
            f"the gain is not one run's luck ({SEED} over its start {said(second)}, {mark})",
            readings["second_run_gains"],
        )
    )
    print(
        reading(
            "run long, one pass does not undo what it added (the end over the checkpoint "
            f"it goes on from {said(kept)})",
            readings["long_holds"],
        )
    )
    print(reading(f"PPO goes on adding ({said(kept)}, {mark})", readings["long_adds"]))
    print(
        reading(
            f"it is past its pilot (the end over the pilot {said(passed)}, {mark})",
            readings["past_the_pilot"],
        )
    )
    if beside is not None:
        ahead = differences.get("end over the first long run")
        readings["over_the_first_long_run"] = better(ahead)
        print(
            reading(
                f"run long, it ends over the first long run ({said(ahead)}, {mark})",
                readings["over_the_first_long_run"],
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--long", type=int, help="the updates the long run was to have, for its readings"
    )
    parser.add_argument(
        "--first",
        default=FIRST,
        help="the run the long one goes on from, where it is not one-pass",
    )
    parser.add_argument(
        "--beside",
        type=Path,
        help="the folder of the first long run, to set this one beside it",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.directory, args.long, args.first, args.beside)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
