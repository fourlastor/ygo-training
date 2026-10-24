"""Measurements of one suite beside each other, game by game.

    uv run python tools/suite_beside.py analysis/unseen/paired/pilot.json \\
        analysis/unseen/paired/start-and-960.json analysis/unseen/ppo-1920.json \\
        --names pilot start 960 1920

Each file is what `evaluate-suite --output` leaves: one record a checkpoint,
or the pilot's, with how each game of every case ended (`outcomes`).  Played
with one seed, the records are of the same deals: the players are set
beside each other case by case, and each is told over the first and over the
one before it by the games one wins and the other does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_head_report import difference
from search_leads_report import percent, points


def records(paths: list[Path]) -> list[tuple[str, Any, dict[str, list[float]]]]:
    """Every record of the files, in order: its name, what tells its games
    (the suite and the seed), and whether it won each game of each case."""
    found = []
    for path in paths:
        kept = json.loads(path.read_text(encoding="utf8"))
        for record in kept["records"]:
            cases = record["evaluation_suite"]
            missing = sorted(case for case in cases if "outcomes" not in cases[case])
            if missing:
                raise SystemExit(
                    f"{path}: {record.get('checkpoint_label')} does not tell how each game of "
                    f"{missing[0]} ended: measure it again with this checkout's evaluate-suite"
                )
            games = {
                case: [float(outcome == "win") for outcome in cases[case]["outcomes"]]
                for case in cases
            }
            dealt = (kept.get("seed"), tuple((case, len(games[case])) for case in sorted(games)))
            found.append((str(record.get("checkpoint_label")), dealt, games))
    return found


def report(paths: list[Path], names: list[str] | None = None) -> dict[str, Any]:
    found = records(paths)
    if names is not None:
        if len(names) != len(found):
            raise SystemExit(f"{len(names)} names for {len(found)} records")
        found = [(name, dealt, games) for name, (_, dealt, games) in zip(names, found, strict=True)]
    if len({dealt for _, dealt, _ in found}) > 1:
        raise SystemExit("the records are not of the same suite and seed: not the same games")
    cases = sorted(found[0][2])
    width = max(len(case) for case in [*cases, "all"]) + 2
    result: dict[str, Any] = {"cases": {}, "all": {}, "differences": {}}
    print(f"{'':<{width}}" + "".join(f"{name:>10}" for name, _, _ in found))
    for case in cases:
        rates = [sum(games[case]) / len(games[case]) for _, _, games in found]
        result["cases"][case] = dict(zip([name for name, _, _ in found], rates, strict=True))
        print(f"{case:<{width}}" + "".join(f"{percent(rate):>10}" for rate in rates))
    pooled = {name: [won for case in cases for won in games[case]] for name, _, games in found}
    result["all"] = {name: sum(won) / len(won) for name, won in pooled.items()}
    print(f"{'all':<{width}}" + "".join(f"{percent(result['all'][name]):>10}" for name in pooled))
    print()
    print(f"Game by game, of {len(next(iter(pooled.values()))):,}:")
    order = list(pooled)
    for place, name in enumerate(order[1:], start=1):
        for other in dict.fromkeys((order[0], order[place - 1])):
            gap = difference(pooled[name], pooled[other])
            result["differences"][f"{name} over {other}"] = gap
            print(f"  {name} over {other}: {points(*gap)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--names", nargs="+", help="a name for each record, in their order")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.files, args.names)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
