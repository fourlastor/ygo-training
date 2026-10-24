"""The readings of the pass beside the next update's games
(experiments/2026-10-23-overlap.md).

    uv run python tools/overlap_report.py analysis/overlap/run \\
        --base analysis/duels-in-flight/run --output analysis/overlap/run/report.json

The directory is what `PARTS=measure` of `experiments/commands/overlap.sh`
leaves: the arm (`overlap`) by its checkpoints after the last three `--step`
of its `--point` updates of PPO, on the screen of each of the thirteen decks
(`screens/overlap.DECK.json`), on the eighteen other decks of thirty-one
(`old/overlap.json`) and on the forty decks new to the pool
(`new/overlap.json`); and the runs' own figures, `overlap.metrics.jsonl`,
`fast.metrics.jsonl` and the trial's `trial-fast.metrics.jsonl` (updates of
`fast`'s settings at the arm's code) and `trial-overlap.metrics.jsonl`.
`--base` is where duels in flight measured `fast` on the same games
(`screens/fast.DECK.json`, `old/fast.json`, `new/fast.json`).

A game is read by the share of a run's three checkpoints that win it, a won
game a win (a drawn one is not).

The readings, fixed on 2026-10-23 before anything was built:

1. It learns as well: the arm against `fast` at R1, both by their
   checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by game
   over the three suites together: equivalent when the gap less two
   standard errors is above -2 points, worse when the gap plus two standard
   errors is below -2 points, otherwise not decided.
2. It is faster: the arm's first 20 updates against the trial's 20 of
   `fast`'s settings, at the same code and from the same start, by duels an
   hour: faster when it plays at least 1.25 times as many.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_descriptions_report import kept_rule
from duels_in_flight_report import duels_an_hour, parts, rows_of
from privileged_critic_run_report import KINDS
from privileged_critic_run_report import report as measured
from search_leads_report import percent, points
from thirteen_decks_report import DECKS, mean

ARM = "overlap"
BASE = "fast"
TRIAL = "trial-fast"
# Reading 2: so many times the duels an hour of `fast`'s settings.
FASTER = 1.25
# What an update of the arm tells of its games beside the pass, beside what
# every update tells.
BESIDE = {
    "overlap_wait_seconds": "waited for the games",
    "overlap_beside_seconds": "the games beside the pass",
    "overlap_games_lock_seconds": "the games waited for the pass's calls",
    "overlap_pass_lock_seconds": "the pass waited for the games' calls",
    "overlap_copy_seconds": "the weights copied",
}


def faster(ratio: float) -> str:
    """Reading 2: faster, or not."""
    return "faster" if ratio >= FASTER else "not faster"


def speed(directory: Path, first: int) -> dict[str, Any]:
    """Reading 2: the arm's first updates against the trial's of `fast`'s
    settings, by duels an hour."""
    ours = rows_of(directory / f"{ARM}.metrics.jsonl")[:first]
    theirs = rows_of(directory / f"{TRIAL}.metrics.jsonl")[:first]
    if len(ours) < first or len(theirs) < first:
        raise SystemExit(f"Reading 2 sets {first} updates of each beside the other")
    ratio = duels_an_hour(ours) / duels_an_hour(theirs)
    return {
        "updates": first,
        ARM: duels_an_hour(ours),
        TRIAL: duels_an_hour(theirs),
        "ratio": ratio,
        "verdict": faster(ratio),
    }


def whole(rows: list[dict[str, Any]]) -> dict[str, float]:
    """A run's figures over all its updates (`duels_in_flight_report.parts`),
    how often PPO's clip bound, and, where the games went beside the pass,
    what each waited for."""
    found = parts(rows)
    found["clip_fraction"] = mean([float(row.get("duel_clip_fraction", 0.0)) for row in rows])
    for name in BESIDE:
        told = [float(row[name]) for row in rows if name in row]
        if told:
            found[name] = mean(told)
    return found


def report(
    directory: Path,
    base: Path,
    thirteen: list[str],
    copied: int = 17_040,
    point: int = 1_920,
    step: int = 120,
    first: int = 20,
) -> dict[str, Any]:
    result = measured(
        directory, base, thirteen, copied, point, step, arm=ARM, base_name=BASE, base_run=BASE
    )
    reading = result["reading"]
    reading["verdict"] = kept_rule(reading["gap"], reading["error"])
    result["speed"] = speed(directory, first)
    result["whole"] = {
        run: whole(rows_of(directory / f"{run}.metrics.jsonl"))
        for run in (ARM, BASE)
        if (directory / f"{run}.metrics.jsonl").is_file()
    }
    trial = directory / f"trial-{ARM}.metrics.jsonl"
    if trial.is_file():
        result["trial"] = duels_an_hour(rows_of(trial)[:first])
    return result


def show(result: dict[str, Any]) -> str:
    reading, fast = result["reading"], result["speed"]
    lines = [
        (
            f"The run {ARM} against {BASE} at {result['point']:,} updates of PPO, each by its "
            "last three checkpoints, game by game:"
        ),
        (
            f"1. It learns as well: {reading['verdict']}. Over the three suites "
            f"{percent(reading[ARM])} against {percent(reading[BASE])}: "
            f"{points(reading['gap'], reading['error'])} over {reading['games']:,} games."
        ),
        (
            f"2. It is faster: {fast['verdict']}. Its first {fast['updates']} updates "
            f"{fast[ARM]:,.0f} duels an hour, {BASE}'s settings at this code "
            f"{fast[TRIAL]:,.0f}: {fast['ratio']:.2f} times."
        ),
        "Beside them:",
    ]
    for kind, what in KINDS.items():
        told = result["by_kind"][kind]
        lines.append(
            f"  {what}: {percent(told[ARM])} against {percent(told[BASE])}, "
            f"{points(told['gap'], told['error'])} ({told['games']:,} games)"
        )
    lines.append(f"  each screen, {ARM} against {BASE}:")
    for deck, told in result["by_screen"].items():
        ours, theirs = told[ARM], told[BASE]
        lines.append(
            f"    {deck}: {percent(ours)} against {percent(theirs)} ({points(ours - theirs)})"
        )
    if "trial" in result:
        lines.append(
            f"  the trial's {fast['updates']} updates of {ARM}: {result['trial']:,.0f} duels an hour"
        )
    for run, told in result["whole"].items():
        line = (
            f"  {run} over its {told['updates']:,.0f} updates: {told['duels_an_hour']:,.0f} duels "
            f"an hour, {told['update_seconds']:.2f} s an update (games {told['games_seconds']:.2f}, "
            f"the model's calls in them {told['calls_seconds']:.2f}, the pass "
            f"{told['pass_seconds']:.2f}); {told['cut_off']:.0f} duels cut off; PPO's clip "
            f"bound at {told['clip_fraction']:.3f} of its decisions"
        )
        if "age" in told:
            line += (
                f"; a duel trained on begun {told['age']:.2f} updates before on average, "
                f"{told['oldest']:.0f} at most"
            )
        lines.append(line)
        beside = [f"{what} {told[name]:.2f} s" for name, what in BESIDE.items() if name in told]
        if beside:
            lines.append(f"    an update, on average: {'; '.join(beside)}")
    blocks = result["along"]
    theirs = {block["first"]: block for block in blocks.get(BASE, [])}
    if blocks.get(ARM):
        lines.append(f"  against the pilots in training, by blocks ({ARM}, {BASE}):")
        for block in blocks[ARM]:
            other = theirs.get(block["first"], {})
            if "pilots" in block and "pilots" in other:
                lines.append(
                    f"    updates {block['first']:5,}-{block['last']:5,}: "
                    f"{percent(block['pilots'])}, {percent(other['pilots'])}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base", type=Path, required=True, help="where fast was measured")
    parser.add_argument("--thirteen", nargs="+", default=list(DECKS))
    parser.add_argument("--copied", type=int, default=17_040, help="updates of all71's copy")
    parser.add_argument("--point", type=int, default=1_920, help="updates of PPO read at")
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--first", type=int, default=20, help="updates of reading 2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.directory, args.base, args.thirteen, args.copied, args.point, args.step, args.first
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(show(result))


if __name__ == "__main__":
    main()
