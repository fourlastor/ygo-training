"""The readings of a Cyber champion (experiments/2026-10-23-cyber-champion.md).

    uv run python tools/cyber_champion_report.py analysis/cyber-champion \\
        --references analysis/seventy-one-decks --copied 240

The directory is what `PARTS=measure` of `experiments/commands/cyber-champion.sh`
leaves:

- `yardstick/ARM.json` for each arm (`cyber-new`, `cyber-old`): its
  checkpoints after each `--step` of its updates of PPO on Cyber's 208 games of
  the forty decks' suite (`eval_suites/cyber-yardstick.json`, the deals
  `all71` and the pilots were measured on there);
- `mirror/STEM.plays.json` and `mirror/STEM.held.json` for each of
  `cyber-new`'s last three checkpoints: the Cyber mirror's 2,048 deals
  (`eval_suites/cyber-against-checkpoint.json`) with the checkpoint in the
  learner's seat and `all71` in the other, then the other way round;
- `ARM-copy.metrics.jsonl` and `ARM.metrics.jsonl`, each arm's two runs.

`--references` is where seventy-one decks measured `all71` and the pilots on
the same games (`new/all71-10560.json`, `new/pilot.json`); `--new-pilot`, the
improved pilot's own games (measurement A's `new.json`).

The readings, fixed on 2026-10-23 before anything was built:

1. A champion: `cyber-new`'s last three checkpoints against `all71` head to
   head in the Cyber mirror, every deal from both seats, a deal's games
   together: a champion when it wins at least 52% by two standard errors,
   weaker at 48% or less by two standard errors, otherwise not told apart.
2. The pilot matters: `cyber-new` against `cyber-old` on the Cyber yardstick
   (the 208 games against the thirteen decks' pilots), after 120 updates of
   PPO and after 720: it matters at a point when `cyber-new` is at least 2
   points over by two standard errors, the same games paired.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("cyber-new", "cyber-old")
DECK = "cyber"
MIRROR = "cyber-vs-checkpoint"
CHAMPION = 0.52
WEAKER = 0.48
ERRORS = 2.0
# Reading 2: so many points over, at these updates of PPO.
MATTERS = 0.02
READ_AT = (120, 720)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def error(values: list[float]) -> float:
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1)) / (
        math.sqrt(len(values))
    )


def records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf8"))["records"]


def games_of(record: dict[str, Any]) -> list[float]:
    """Cyber's games of a record, won or not, its cases in the order of their
    names."""
    cases = record["evaluation_suite"]
    names = sorted(name for name in cases if name.startswith(f"{DECK}-vs-"))
    if not names:
        raise SystemExit("No case of Cyber in a record")
    return [float(told == "win") for name in names for told in cases[name]["outcomes"]]


def by_point(path: Path, copied: int) -> dict[int, list[float]]:
    """A run's games on the yardstick, by its updates of PPO."""
    return {int(record["checkpoint_step"]) - copied: games_of(record) for record in records(path)}


def shared(path: Path) -> list[float]:
    """Each game's share of a file's records that win it."""
    games = [games_of(record) for record in records(path)]
    return [sum(game) / len(game) for game in zip(*games, strict=True)]


def champion(share: float, error_: float) -> str:
    """Reading 1."""
    if share - ERRORS * error_ >= CHAMPION:
        return "a champion"
    if share + ERRORS * error_ <= WEAKER:
        return "weaker"
    return "not told apart"


def mirror(directory: Path) -> dict[str, Any]:
    """Reading 1: cyber-new's checkpoints against all71 in the mirror, a deal's
    games of every checkpoint and both seats together."""
    found: list[list[float]] = []
    for plays in sorted((directory / "mirror").glob("*.plays.json")):
        held = plays.with_name(plays.name.replace(".plays.", ".held."))
        if not held.is_file():
            raise SystemExit(f"Missing {held}")
        (ours,) = records(plays)
        (theirs,) = records(held)
        won = [float(told == "win") for told in ours["evaluation_suite"][MIRROR]["outcomes"]]
        # all71 plays these: the checkpoint wins what it loses.
        lost = [float(told == "loss") for told in theirs["evaluation_suite"][MIRROR]["outcomes"]]
        if len(won) != len(lost):
            raise SystemExit(f"{plays.name} and {held.name} play other deals")
        found.append([(one + other) / 2 for one, other in zip(won, lost, strict=True)])
    if not found:
        raise SystemExit("No checkpoint of the mirror is measured")
    deals = [mean(list(told)) for told in zip(*found, strict=True)]
    share, error_ = mean(deals), error(deals)
    return {
        "checkpoints": len(found),
        "deals": len(deals),
        "share": share,
        "error": error_,
        "verdict": champion(share, error_),
    }


def matters(gain: float, error_: float) -> str:
    """Reading 2 at a point."""
    return "matters" if gain - ERRORS * error_ >= MATTERS else "not shown"


def hours(directory: Path, arm: str) -> dict[str, float]:
    found = {}
    for part, name in (("copy", f"{arm}-copy"), ("ppo", arm)):
        path = directory / f"{name}.metrics.jsonl"
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
            found[part] = sum(float(row["wall_seconds"]) for row in rows if row) / 3600
    return found


def report(
    directory: Path,
    references: Path,
    copied: int = 240,
    new_pilot: Path | None = None,
    read_at: tuple[int, ...] = READ_AT,
) -> dict[str, Any]:
    curves = {arm: by_point(directory / "yardstick" / f"{arm}.json", copied) for arm in ARMS}
    result: dict[str, Any] = {"mirror": mirror(directory), "matters": {}, "curves": {}}
    for point in read_at:
        if any(point not in curves[arm] for arm in ARMS):
            raise SystemExit(f"The arms are not both measured after {point} updates of PPO")
        gain = [
            one - other for one, other in zip(*(curves[arm][point] for arm in ARMS), strict=True)
        ]
        told = {"gain": mean(gain), "error": error(gain), "games": len(gain)}
        told["verdict"] = matters(told["gain"], told["error"])
        result["matters"][point] = told
    for arm in ARMS:
        result["curves"][arm] = {point: mean(games) for point, games in sorted(curves[arm].items())}
    result["references"] = {
        "all71": mean(shared(references / "new" / "all71-10560.json")),
        "pilot as pinned": mean(shared(references / "new" / "pilot.json")),
    }
    if new_pilot is not None:
        result["references"]["improved pilot"] = mean(shared(new_pilot))
    result["hours"] = {arm: hours(directory, arm) for arm in ARMS}
    return result


def shown(result: dict[str, Any]) -> str:
    told = result["mirror"]
    lines = [
        (
            f"1. A champion: {told['verdict']}. cyber-new's last {told['checkpoints']} checkpoints "
            f"win {told['share']:.1%} ± {told['error']:.1%} of the Cyber mirror against all71 "
            f"({told['deals']:,} deals, each from both seats)."
        ),
        "2. The pilot matters, cyber-new against cyber-old on Cyber's yardstick:",
    ]
    for point, told in result["matters"].items():
        lines.append(
            f"   after {point} updates of PPO: {told['verdict']}, {100 * told['gain']:+.1f} ± "
            f"{100 * told['error']:.1f} points over {told['games']} games"
        )
    lines.append("Beside them, the yardstick by updates of PPO:")
    for arm, curve in result["curves"].items():
        points = ", ".join(f"{point}: {share:.1%}" for point, share in curve.items())
        lines.append(f"  {arm}: {points}")
    references = ", ".join(f"{name} {share:.1%}" for name, share in result["references"].items())
    lines.append(f"  on the same games: {references}")
    for arm, told in result["hours"].items():
        parts = ", ".join(f"{part} {value:.2f} h" for part, value in told.items())
        lines.append(f"  {arm} took {parts}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--copied", type=int, default=240, help="updates of each arm's copy")
    parser.add_argument("--new-pilot", type=Path, help="the improved pilot's games")
    parser.add_argument(
        "--read-at", type=int, nargs="+", default=list(READ_AT), help="updates of reading 2"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.directory, args.references, args.copied, args.new_pilot, tuple(args.read_at)
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(shown(result))


if __name__ == "__main__":
    main()
