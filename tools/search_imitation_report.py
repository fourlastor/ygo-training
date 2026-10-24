"""The search-imitation experiment in one table.

Usage, from the repository root::

    uv run python tools/search_imitation_report.py analysis/search-imitation/arms \\
        --output analysis/search-imitation/arms/report.json

The directory holds the `evaluate-search` results the arms script writes:
`seed.json` for the model both arms start from, and `search-N.json` and
`pilot-N.json` for the two arms on `N` games, each with one entry for every
measured update.  All are measured on the same held-out run, so the two arms
are compared deal by deal and deviation by deviation.  A `seed-continued.json`
there, the seed's own run measured further on, is given beside the seed.

With `--records`, the held-out run's own rows, the deviations are also set
apart by how far ahead of the pilot's answer the search found its own (the
paired z it took it at).  Then come the deviations the starting model gave
next to no weight: what a run makes its first choice there it has from the
games it imitated, and no longer from having copied its pilot badly.

Where the measurements read the models' answers by the search's worlds (the
held-out run was logged), each run's other answers than the pilot's are given
with the lead the search measured for them, and the two arms' leads are set
against each other game by game, at every measured update.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from ygo_training.records import recorded_games

# Where a deviation's z falls: the search takes an alternative from 1.645 on.
FIRMNESS = ((2.0, "z under 2"), (2.5, "2 to 2.5"), (3.0, "2.5 to 3"), (math.inf, "3 and more"))
# The weight on the search's answer under which the starting model did not
# have it, and the z that parts the deviations in two near halves.
UNSEEN = 0.01
FIRM = 2.5


def percent(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{100.0 * value:.{digits}f}%"


def paired(first: list[bool], second: list[bool]) -> tuple[float, float]:
    """The difference of two rates on the same items, and its standard error."""
    total = len(first)
    if not total or total != len(second):
        return math.nan, math.nan
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    difference = (only_first - only_second) / total
    variance = (only_first + only_second - (only_first - only_second) ** 2 / total) / total**2
    return difference, math.sqrt(max(variance, 0.0))


def measure(entry: dict[str, Any]) -> dict[str, Any]:
    """What the report reads from one checkpoint's evaluation."""
    key, deals = entry["answer_key"], entry["deals"]
    return {
        "checkpoint": entry["checkpoint_label"],
        "step": entry["checkpoint_step"],
        "win_rate": deals["win_rate"],
        "wins": [outcome == "win" for outcome in deals["outcomes"]],
        "search_rate": key["search_rate"],
        "pilot_rate": key["pilot_rate"],
        "kept_left_rate": key["kept_left_rate"],
        "examined_kept_left_rate": key.get("examined_kept_left_rate"),
        # Whether the model gave the search's answer, deviation by deviation.
        "followed": {
            (item["game"], item["answer"]): item["reading"] == "search"
            for item in key["deviation_list"]
        },
        # The weight it gave the search's answer, deviation by deviation.
        "weight": {
            (item["game"], item["answer"]): item["search_probability"]
            for item in key["deviation_list"]
        },
        # Its answers by what the search measured for them, when it logged it.
        "tried": key.get("tried"),
    }


def firmness(records: Path) -> dict[tuple[int, int], float]:
    """The z of every deviation of a search run, by the game's place among the
    run's games as the measurements number them, and the answer's place in it."""
    found: dict[tuple[str, int, int], dict[int, float]] = {}
    with records.open(encoding="utf8") as rows:
        for text in rows:
            if text.strip():
                row = json.loads(text)
                found[(str(row["b"]), int(row["seed"]), int(row["seat"]))] = {
                    int(item["decision"]): float(item["z"]) for item in row["search"]["deviations"]
                }
    return {
        (index, answer): z
        for index, game in enumerate(recorded_games(records, "search"))
        for answer, z in found[(game.opponent, game.seed, game.learner_player)].items()
    }


def bucket(z: float) -> str:
    return next(name for limit, name in FIRMNESS if z < limit)


def points(value: float | None, error: float | None = None) -> str:
    """A difference of win rates, in points."""
    if value is None or math.isnan(value):
        return "-"
    return f"{100 * value:+.1f}" + ("" if error is None else f" ± {100 * error:.1f}")


def mean(values: list[float]) -> tuple[float, float]:
    """A mean and its standard error."""
    count = len(values)
    if count < 2:
        return (values[0] if values else math.nan), math.nan
    centre = sum(values) / count
    return centre, math.sqrt(sum((value - centre) ** 2 for value in values) / (count - 1) / count)


def share(hits: int, total: int) -> str:
    """A rate in percent, with its standard error."""
    if not total:
        return "-"
    rate = hits / total
    return f"{100 * rate:.1f} ± {100 * math.sqrt(rate * (1 - rate) / total):.1f}"


def tried_line(name: str, tried: dict[str, Any]) -> str:
    others = tried["decisions"] - tried["pilot"]
    return (
        f"{name:<14}{others / tried['games']:>8.2f}{tried['outright']:>10}{tried['tried']:>8}"
        f"{tried['ahead']:>7}{tried['level']:>7}{tried['behind']:>8}"
        f"{points(tried['lead'], tried['lead_error']):>14}"
        f"{points(tried['lead_per_game'], tried['lead_per_game_error']):>14}"
    )


def line(name: str, update: str, item: dict[str, Any]) -> str:
    return (
        f"{name:<14}{update:>7}{percent(item['win_rate']):>8}{percent(item['search_rate']):>10}"
        f"{percent(item['pilot_rate']):>9}{percent(item['kept_left_rate']):>9}"
        f"{percent(item['examined_kept_left_rate']):>10}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--records", type=Path, help="the held-out run's rows")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seed = json.loads((args.directory / "seed.json").read_text(encoding="utf8"))
    start = measure(seed["checkpoints"][0])
    references = seed["checkpoints"][0]["deals"]["references"]
    seeds = [("seed", start)]
    continued = args.directory / "seed-continued.json"
    if continued.exists():
        entry = json.loads(continued.read_text(encoding="utf8"))["checkpoints"][0]
        seeds.append(("seed-continued", measure(entry)))
    arms: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in sorted(args.directory.glob("*-*.json")):
        match = re.fullmatch(r"(search|pilot)-(\d+)", path.stem)
        if match:
            result = json.loads(path.read_text(encoding="utf8"))
            arms[(match[1], int(match[2]))] = [measure(entry) for entry in result["checkpoints"]]

    print(
        f"{seed['games']} held-out deals: pilot {percent(references['pilot']['win_rate'])}, "
        f"search {percent(references['search']['win_rate'])}"
    )
    print("At the deviations: the model's first choice is the search's answer, or still the")
    print("pilot's. Kept: another answer than the pilot's where the search kept it, among")
    print("all such decisions and among those the search examined.")
    print(
        f"{'run':<14}{'update':>7}{'wins':>8}{'search':>10}{'pilot':>9}{'kept':>9}{'examined':>10}"
    )
    kept_out = ("wins", "followed", "weight", "tried")
    report: dict[str, Any] = {}
    for name, item in seeds:
        print(line(name, str(int(item["step"]) - int(start["step"])), item))
        report[name.replace("-", "_")] = {k: v for k, v in item.items() if k not in kept_out}
    report["references"] = {name: reference["win_rate"] for name, reference in references.items()}
    report["arms"] = {}
    report["search_against_pilot"] = {}
    for (arm, size), items in sorted(arms.items(), key=lambda entry: (entry[0][1], entry[0][0])):
        for item in items:
            update = int(item["step"]) - int(start["step"])
            print(line(f"{arm}-{size}", str(update), item))
            report["arms"].setdefault(f"{arm}-{size}", {})[str(update)] = {
                k: v for k, v in item.items() if k not in kept_out
            }

    print()
    print("The Search arm against the Pilot arm, same games and updates:")
    print(f"{'games':>7}{'update':>7}{'wins':>18}{'search at deviations':>24}")
    for size in sorted({size for _, size in arms}):
        if ("search", size) not in arms or ("pilot", size) not in arms:
            continue
        for searching, piloting in zip(arms[("search", size)], arms[("pilot", size)]):
            update = int(searching["step"]) - int(start["step"])
            wins = paired(searching["wins"], piloting["wins"])
            shared = sorted(searching["followed"].keys() & piloting["followed"].keys())
            followed = paired(
                [searching["followed"][key] for key in shared],
                [piloting["followed"][key] for key in shared],
            )
            print(
                f"{size:>7}{update:>7}"
                f"{f'{100 * wins[0]:+.1f} ± {100 * wins[1]:.1f}':>18}"
                f"{f'{100 * followed[0]:+.1f} ± {100 * followed[1]:.1f}':>24}"
            )
            report["search_against_pilot"].setdefault(str(size), {})[str(update)] = {
                "wins": wins[0],
                "wins_standard_error": wins[1],
                "search_rate": followed[0],
                "search_rate_standard_error": followed[1],
            }
    if start["tried"]:
        last = [(name, item) for name, item in seeds if item["tried"]] + [
            (f"{arm}-{size}", items[-1])
            for (arm, size), items in sorted(
                arms.items(), key=lambda entry: (entry[0][1], entry[0][0])
            )
            if items[-1]["tried"]
        ]
        seen = start["tried"]
        print()
        print(
            f"By the search's worlds, at the {seen['decisions']} decisions it examined (after the"
        )
        print("last measured update): the other answers than the pilot's a game, those where")
        print("the pilot's wins outright, and those the search tried with how they did there,")
        print("in points of win rate; the last column adds a game's leads up.")
        print(
            f"{'run':<14}{'a game':>8}{'outright':>10}{'tried':>8}{'ahead':>7}{'level':>7}"
            f"{'behind':>8}{'lead':>14}{'sum a game':>14}"
        )
        report["tried"] = {}
        for name, item in last:
            print(tried_line(name, item["tried"]))
            report["tried"][name] = {
                k: v for k, v in item["tried"].items() if k not in ("lead_by_game", "by_question")
            }
        print(tried_line("the search", seen["search"] | {"games": seen["games"]}))
        other = seen["any_other"]
        print(
            f"any other answer the search tried ({other['tried']}): "
            f"{points(other['lead'], other['lead_error'])}"
        )
        print()
        print("A game's leads added up, for each arm, and the Search arm's against the Pilot")
        print("arm's game by game; then what no lead is measured for, each arm's other answers")
        print("a game where the pilot's wins outright:")
        print(
            f"{'games':>7}{'update':>7}{'search':>14}{'pilot':>14}{'difference':>14}"
            f"{'outright: search':>19}{'pilot':>7}"
        )
        report["tried_search_against_pilot"] = {}
        for size in sorted({size for _, size in arms}):
            if ("search", size) not in arms or ("pilot", size) not in arms:
                continue
            for searching, piloting in zip(arms[("search", size)], arms[("pilot", size)]):
                pair = (searching["tried"], piloting["tried"])
                if not all(pair):
                    continue
                update = int(searching["step"]) - int(start["step"])
                difference = mean(
                    [a - b for a, b in zip(pair[0]["lead_by_game"], pair[1]["lead_by_game"])]
                )
                print(
                    f"{size:>7}{update:>7}"
                    + "".join(
                        f"{points(arm['lead_per_game'], arm['lead_per_game_error']):>14}"
                        for arm in pair
                    )
                    + f"{points(*difference):>14}"
                    + f"{pair[0]['outright'] / pair[0]['games']:>19.2f}"
                    + f"{pair[1]['outright'] / pair[1]['games']:>7.2f}"
                )
                report["tried_search_against_pilot"].setdefault(str(size), {})[str(update)] = {
                    "search": pair[0]["lead_per_game"],
                    "pilot": pair[1]["lead_per_game"],
                    "lead_per_game": difference[0],
                    "lead_per_game_standard_error": difference[1],
                    "search_outright_per_game": pair[0]["outright"] / pair[0]["games"],
                    "pilot_outright_per_game": pair[1]["outright"] / pair[1]["games"],
                }
    if args.records:
        scores = firmness(args.records)
        if scores.keys() != start["followed"].keys():
            raise SystemExit(f"{args.records} is not the run these measurements were made on")
        names = [name for _, name in FIRMNESS]
        runs = seeds + [
            (f"{arm}-{size}", items[-1])
            for (arm, size), items in sorted(
                arms.items(), key=lambda entry: (entry[0][1], entry[0][0])
            )
        ]
        print()
        print("The search's answer as first choice, by how firmly the search took it (after the")
        print("last measured update):")
        print(f"{'run':<14}" + "".join(f"{name:>14}" for name in names))
        counts = {name: sum(bucket(z) == name for z in scores.values()) for name in names}
        print(f"{'deviations':<14}" + "".join(f"{counts[name]:>14}" for name in names))
        report["by_firmness"] = {"deviations": counts}
        for name, item in runs:
            followed = {
                group: sum(item["followed"][key] for key, z in scores.items() if bucket(z) == group)
                for group in names
            }
            print(
                f"{name:<14}"
                + "".join(
                    f"{percent(followed[group] / counts[group] if counts[group] else None):>14}"
                    for group in names
                )
            )
            report["by_firmness"][name] = followed
        fresh = [key for key, weight in start["weight"].items() if weight < UNSEEN]
        groups = {
            "all": fresh,
            f"z under {FIRM}": [key for key in fresh if scores[key] < FIRM],
            f"{FIRM} and more": [key for key in fresh if scores[key] >= FIRM],
        }
        print()
        print(
            f"Where the seed gave the search's answer under {UNSEEN:.0%} of its weight "
            f"({len(fresh)} of the"
        )
        print(f"{len(scores)} deviations): the search's answer as first choice, in percent, among")
        print("all of those and by the z the search took them at.")
        print(f"{'run':<14}{'update':>7}" + "".join(f"{name:>16}" for name in groups))
        print(f"{'deviations':<21}" + "".join(f"{len(keys):>16}" for keys in groups.values()))
        report["unseen"] = {"deviations": {name: len(keys) for name, keys in groups.items()}}
        rows = seeds[1:] + [
            (f"{arm}-{size}", item)
            for (arm, size), items in sorted(
                arms.items(), key=lambda entry: (entry[0][1], entry[0][0])
            )
            for item in items
        ]
        for name, item in rows:
            update = int(item["step"]) - int(start["step"])
            hits = {
                group: sum(item["followed"][key] for key in keys) for group, keys in groups.items()
            }
            print(
                f"{name:<14}{update:>7}"
                + "".join(f"{share(hits[group], len(keys)):>16}" for group, keys in groups.items())
            )
            report["unseen"].setdefault(name, {})[str(update)] = hits
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")


if __name__ == "__main__":
    main()
