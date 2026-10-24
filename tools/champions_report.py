"""The readings of champions of single decks by the recipe
(experiments/2026-10-24-four-decks.md).

    uv run python tools/champions_report.py analysis/four-decks \\
        lightsworn infernity cyber-2013=cyber-old

The directory is what `experiments/commands/champions.sh` leaves, in the
files of `tools/two_decks_report.py`: pairs `STEM.plays.json` and
`STEM.held.json` on the 2,048 deals of a mirror's suite, the first network
named in the learner's seat and then the second, STEM its updates of PPO.
Each run named, `NAME`:

- `NAME/`: its last three checkpoints against `all71` in its deck's mirror;
- `NAME-copy.metrics.jsonl` and `NAME.metrics.jsonl`: its copy and its PPO.

A run of another seed, `NAME=FIRST`, is set beside the deck's first run:
`FIRST/` against `all71` as well, on the same deals, and
`NAME-against-FIRST/` the two head to head, checkpoint for checkpoint.

The reading, as the Cyber champion's (fixed on 2026-10-23, taken for these
on 2026-10-24 before anything was run): a run's last three checkpoints
against `all71` head to head in its deck's mirror, every deal from both
seats, a deal's games together: a champion when it wins at least 52% by
two standard errors (the share less two standard errors at 52% or more),
weaker at 48% or less by two standard errors, otherwise not told apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_decks_report import deals, hours, paired, told

CHAMPION = 0.52
WEAKER = 0.48
ERRORS = 2.0


def verdict(share: float, error_: float) -> str:
    """The reading, for a run."""
    if share - ERRORS * error_ >= CHAMPION:
        return "a champion"
    if share + ERRORS * error_ <= WEAKER:
        return "weaker"
    return "not told apart"


def summed(*parts: float | None) -> float | None:
    """Hours added up, where all of them are known."""
    known = [part for part in parts if part is not None]
    return None if len(known) < len(parts) else sum(known)


def report(directory: Path, runs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for given in runs:
        name, _, first = given.partition("=")
        found = told(deals(directory / name))
        found["verdict"] = verdict(found["share"], found["error"])
        found["hours"] = summed(hours(directory, f"{name}-copy"), hours(directory, name))
        if first:
            ours, theirs = deals(directory / name), deals(directory / first)
            found["first"] = {
                "name": first,
                "against all71": told(theirs),
                "over it, against all71": paired(ours, theirs),
                "head to head": told(deals(directory / f"{name}-against-{first}")),
            }
        result[name] = found
    return result


def shown(result: dict[str, Any]) -> str:
    lines = ["Against all71 in its deck's mirror, a champion at 52% by two standard errors:"]
    for name, found in result.items():
        spent = "" if found["hours"] is None else f"; {found['hours']:.2f} hours"
        lines.append(
            f"  {name}: {found['verdict']}. {found['share']:.1%} ± {found['error']:.1%} "
            f"({found['deals']:,} deals, each from both seats){spent}"
        )
        first = found.get("first")
        if first is not None:
            over = first["over it, against all71"]
            lines.append(
                f"    beside {first['name']}, the deck's first run: it wins "
                f"{first['against all71']['share']:.1%} against all71 on the same deals, "
                f"{100 * over['gap']:+.1f} ± {100 * over['error']:.1f} points to it, the deals "
                f"paired; head to head {first['head to head']['share']:.1%} ± "
                f"{first['head to head']['error']:.1%}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("runs", nargs="+", help="NAME, or NAME=FIRST for a run of another seed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.runs)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(shown(result))


if __name__ == "__main__":
    main()
