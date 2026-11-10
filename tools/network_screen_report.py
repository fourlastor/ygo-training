"""The readings of the screen of the network under PPO
(experiments/2026-11-02-expanded-pool.md, its section beside the batches).

    uv run python tools/network_screen_report.py analysis/network-screen \\
        cyber cyber-tokens cyber-no-history

The directory is what `experiments/commands/champions.sh` leaves for a
control and its runs of a shape (`DECK@SHAPE`), in the files of
`tools/two_decks_report.py`: `VARIANT-against-CONTROL/` the two head to
head in the deck's mirror, checkpoint for checkpoint, the variant in the
learner's seat and then the control; `NAME/` each run against `all71`;
`NAME-copy.metrics.jsonl` and `NAME.metrics.jsonl` its copy and its PPO.
A variant is named after its control: `CONTROL-tokens`, the board as its
cards, `CONTROL-no-history`, without its own earlier answers, or
`CONTROL-switched`, the control given the board as its cards part of the way
(`tools/switch_board.py`), which has no copy of its own.

The readings, fixed on 2026-11-09 before anything was built: a variant's
last three checkpoints against the control's, every deal from both seats, a
deal's games together, its share with two standard errors of the deals.

- The board as its cards: **stronger** at 54% or more by two standard
  errors, **weaker** at 46% or less by two, otherwise **not a large lever
  for one deck**.
- Without its history: the history is **needed** when the variant wins at
  most 46% by two standard errors, otherwise it is **not shown to be
  needed**, which frees nothing.

The second session's, fixed on 2026-11-10 before anything was built: the
board as its cards as above, for Blackwing and for Cyber after twice the
training; and the switch, which **pays** at 54% or more by two standard
errors, **harms** at 46% or less by two, and is otherwise **not shown**.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_decks_report import deals, hours, mean, told

STRONGER = 0.54
WEAKER = 0.46
ERRORS = 2.0
SHAPES = ("tokens", "no-history", "switched")


def reading(shape: str, share: float, error_: float) -> str:
    """A variant's reading against its control."""
    if shape == "tokens":
        if share - ERRORS * error_ >= STRONGER:
            return "stronger"
        if share + ERRORS * error_ <= WEAKER:
            return "weaker"
        return "not a large lever for one deck"
    if shape == "no-history":
        if share + ERRORS * error_ <= WEAKER:
            return "the history needed"
        return "the history not shown to be needed"
    if shape == "switched":
        if share - ERRORS * error_ >= STRONGER:
            return "the switch pays"
        if share + ERRORS * error_ <= WEAKER:
            return "the switch harms"
        return "the switch not shown"
    raise SystemExit(f"A variant is of the shape {' or '.join(SHAPES)}, not {shape}")


def seconds(directory: Path, run: str) -> float | None:
    """The run's mean seconds an update."""
    path = directory / f"{run}.metrics.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    return mean([float(row["wall_seconds"]) for row in rows]) if rows else None


def spent(directory: Path, run: str) -> dict[str, float | None]:
    copy, ppo = hours(directory, f"{run}-copy"), hours(directory, run)
    return {
        "hours": None if copy is None or ppo is None else copy + ppo,
        "copy seconds an update": seconds(directory, f"{run}-copy"),
        "seconds an update": seconds(directory, run),
    }


def report(directory: Path, control: str, variants: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "control": {"name": control, "against all71": told(deals(directory / control))}
    }
    result["control"].update(spent(directory, control))
    result["variants"] = {}
    for variant in variants:
        shape = variant.removeprefix(f"{control}-")
        if variant == shape or shape not in SHAPES:
            raise SystemExit(
                f"A variant is named {control}-{' or -'.join(SHAPES)} after its control, not {variant}"
            )
        found = told(deals(directory / f"{variant}-against-{control}"))
        found["shape"] = shape
        found["reading"] = reading(shape, found["share"], found["error"])
        found["against all71"] = told(deals(directory / variant))
        found.update(spent(directory, variant))
        result["variants"][variant] = found
    return result


def timed(found: dict[str, Any]) -> str:
    parts = []
    if found["seconds an update"] is not None:
        parts.append(f"{found['seconds an update']:.1f} s an update of PPO")
    if found["hours"] is not None:
        parts.append(f"{found['hours']:.2f} hours")
    return f"; {', '.join(parts)}" if parts else ""


def shown(result: dict[str, Any]) -> str:
    control = result["control"]
    lines = [
        (
            "Each variant head to head against the control in the mirror, its last three "
            "checkpoints, every deal from both seats:"
        )
    ]
    for name, found in result["variants"].items():
        lines.append(
            f"  {name}: {found['reading']}. {found['share']:.1%} ± {found['error']:.1%} "
            f"against {control['name']} ({found['deals']:,} deals)"
        )
    lines.append("Beside them, each against all71 in the mirror:")
    for name, found in [(control["name"], control), *result["variants"].items()]:
        against = found["against all71"]
        lines.append(f"  {name}: {against['share']:.1%} ± {against['error']:.1%}{timed(found)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("control", help="the control's run, as champions.sh names it")
    parser.add_argument("variants", nargs="+", help="CONTROL-tokens, CONTROL-no-history")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.control, args.variants)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(shown(result))


if __name__ == "__main__":
    main()
