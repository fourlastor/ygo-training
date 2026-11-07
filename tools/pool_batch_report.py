"""The readings of a batch of the expanded pool (experiments/2026-11-02-expanded-pool.md).

    uv run python tools/pool_batch_report.py analysis/expanded-pool/batch-one \\
        --points 4020,8040

The directory is what `experiments/commands/pool-batch.sh` leaves: at 0 and at each of
the points `pairings.json`, each new deck against the frozen network at each old deck;
at the last point, and at the points before where they were measured, `old.json`, every
ordered pairing of the old decks head to head against it; at 0 and at the last point
`held-out.json` and `pilots.json`; at the last point `mirrors/`, each of the seventy-one
mirrors head to head against its champion.

The readings, fixed on 2026-11-03 before anything was built:

1. The new decks learn: the mean over the new decks of their share against the frozen
   network at the last point, less at the start, the deals paired: learns when at
   least 5 points by two standard errors, does not when at most 2 points by two
   standard errors, otherwise not decided.
2. The old decks hold: the network head to head against the frozen one on the old
   decks' pairings at the last point: hold when at least 49% by two standard errors,
   slip when at most 49% by two standard errors, otherwise not decided.
3. The mirrors come back: the seventy-one mirrors head to head against their
   champions at the last point: kept when their mean is at least 48% by two standard
   errors and none is at most 45% by two standard errors, lost when the mean is at
   most 48% by two standard errors or one is at most 45% by two standard errors,
   otherwise not decided.

From batch three (fixed on 2026-11-07), with `--reference`, reading 3 is the
seventy-one mirrors deal by deal against the reference network's on its deals, a draw
half a win (`tools/mirror_pairs.py`): kept when the difference is at least -2 points by
two standard errors, lost when at most -2 points by two, otherwise not decided. No one
mirror has a floor of its own: `seventy-one-champions`' own check fails the one above.

Told beside them, without a mark: the gain and the old decks at the points before the
last; the old decks' share by the two decks of a pairing, both of the seventy-one, both
game Decks or one of each, where a batch before made game Decks old; each new deck's
gain; the mirrors' mean against their champions with a reference, and deal by deal
against another network's with `--beside`; the next decks, trained on nothing; the new
decks against the pilots; the hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mirror_pairs
from first_batch_report import cases, games, holds, learns
from fourteen_decks_report import ERRORS, kept_mirrors, number
from specialist_distillation_report import shares, spent
from two_decks_report import mean, paired, told

# Reading 3 against a reference: the mirrors kept when at least this far under its by two
# standard errors, lost when at most.
UNDER = -0.02
# The decks named that moved most against the reference.
NAMED = 5


def kept_against(difference: float, error_: float) -> str:
    """Reading 3 against a reference, of the mirrors' difference to its, deal by deal."""
    if difference - ERRORS * error_ >= UNDER:
        return "kept"
    if difference + ERRORS * error_ <= UNDER:
        return "lost"
    return "not decided"


def by_pairing(path: Path) -> dict[str, list[float]]:
    """The old decks' games in a file by the two decks of their pairing: both of the
    seventy-one, both game Decks, or one of each, won or not. Each of these is played
    from both sides, so what a deck is worth against the others cancels, as it does
    not in a deck's own games against all of them: the seventy-one win most of theirs
    against the game Decks whichever network plays them."""
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    found: dict[str, list[float]] = {"seventy-one": [], "game decks": [], "one of each": []}
    for name, case in record["evaluation_suite"].items():
        for opponent, told_ in case["by_opponent"].items():
            game = (name.startswith("game-"), opponent.startswith("game-"))
            kind = "game decks" if all(game) else "one of each" if any(game) else "seventy-one"
            found[kind] += [1.0] * told_["wins"] + [0.0] * (told_["games"] - told_["wins"])
    return found


def report(
    directory: Path,
    points: list[int],
    run: str,
    reference: Path | None = None,
    beside: Path | None = None,
) -> dict[str, Any]:
    start = cases(directory / "0" / "pairings.json")
    tables = {point: cases(directory / str(point) / "pairings.json") for point in points}
    last = points[-1]
    gains = {point: paired(games(tables[point]), games(start)) for point in points}
    old = {
        point: told(games(cases(directory / str(point) / "old.json")))
        for point in points
        if point == last or (directory / str(point) / "old.json").is_file()
    }
    found: dict[str, Any] = {
        "start": mean(games(start)),
        "gains": gains,
        "each": {name: mean(tables[last][name]) - mean(start[name]) for name in sorted(start)},
        "learns": learns(gains[last]["gap"], gains[last]["error"]),
        "old": old,
        "holds": holds(old[last]["share"], old[last]["error"]),
    }
    # The old decks' share at the last point by the two decks of a pairing, where a batch
    # before made game Decks old.
    apart = by_pairing(directory / str(last) / "old.json")
    if apart["seventy-one"] and apart["game decks"]:
        found["old apart"] = {kind: told(games_) for kind, games_ in apart.items()}
    mirrors = sorted((directory / str(last) / "mirrors").glob("*.held.json"))
    if mirrors:
        each = {
            path.name.split(".")[0]: told(shares(path.with_name(path.name.split(".")[0])))
            for path in mirrors
        }
        found["mirrors"] = {**kept_mirrors(each), "each": each}
        if reference is not None or beside is not None:
            ours = mirror_pairs.shares(directory / str(last) / "mirrors")
        if reference is not None:
            moved = mirror_pairs.against(mirror_pairs.shares(reference), ours)
            found["mirrors"] = {
                "mean": found["mirrors"]["mean"],
                "error": found["mirrors"]["error"],
                "against": {**moved, "reference": str(reference)},
                "verdict": kept_against(moved["difference"], moved["error"]),
                "each": each,
            }
        if beside is not None:
            moved = mirror_pairs.against(mirror_pairs.shares(beside), ours)
            found["mirrors"]["beside"] = {**moved, "network": str(beside)}
    for part in ("held-out", "pilots"):
        at_last = directory / str(last) / f"{part}.json"
        if at_last.is_file():
            before = games(cases(directory / "0" / f"{part}.json"))
            found[part] = {"start": mean(before), **paired(games(cases(at_last)), before)}
    found["hours"] = spent(directory, run)
    return found


def shown(found: dict[str, Any], points: list[int]) -> str:
    last = points[-1]
    gain = found["gains"][last]
    old = found["old"][last]
    lines = [
        (
            f"1. A new deck {found['learns']}: at {last:,} updates "
            f"{100 * gain['gap']:+.1f} ± {100 * gain['error']:.1f} points against the frozen "
            f"network, from {100 * found['start']:.1f}%"
        ),
        (
            f"2. The old decks {found['holds']}: {100 * old['share']:.1f}% ± "
            f"{100 * old['error']:.1f}% head to head against the frozen network"
        ),
    ]
    if "mirrors" in found and "against" in found["mirrors"]:
        mirrors = found["mirrors"]
        moved = mirrors["against"]
        lines.append(
            f"3. The mirrors {mirrors['verdict']}: {100 * moved['difference']:+.1f} ± "
            f"{100 * moved['error']:.1f} points deal by deal against {moved['reference']}, "
            f"lower on {moved['lower']} of {moved['decks']}"
        )
    elif "mirrors" in found:
        mirrors = found["mirrors"]
        lines.append(
            f"3. The mirrors {mirrors['verdict']}: {100 * mirrors['mean']:.1f}% ± "
            f"{100 * mirrors['error']:.1f}% against their champions"
            + (
                f", under 45%: {', '.join(mirrors['under the floor'])}"
                if mirrors["under the floor"]
                else ""
            )
        )
    lines.append("Beside them:")
    lines.append(
        "  The gain along the run: "
        + ", ".join(
            f"{point:,}: {100 * found['gains'][point]['gap']:+.1f} ± "
            f"{100 * found['gains'][point]['error']:.1f}"
            for point in points
        )
    )
    if len(found["old"]) > 1:
        lines.append(
            "  The old decks along it: "
            + ", ".join(
                f"{point:,}: {100 * told_['share']:.1f}%" for point, told_ in found["old"].items()
            )
        )
    if "old apart" in found:
        apart = found["old apart"]
        lines.append(
            f"  The old decks at {last:,} by the decks of a pairing: "
            + ", ".join(
                f"{label} {100 * apart[kind]['share']:.1f}% ± {100 * apart[kind]['error']:.1f}%"
                for kind, label in (
                    ("seventy-one", "both of the seventy-one"),
                    ("game decks", "both game Decks"),
                    ("one of each", "one of each"),
                )
            )
        )
    ranked = sorted(found["each"].items(), key=lambda item: item[1])
    lines.append(
        "  The new decks' least gains: "
        + ", ".join(f"{name.split('-vs-')[0]} {100 * gap:+.1f}" for name, gap in ranked[:5])
        + "; most: "
        + ", ".join(f"{name.split('-vs-')[0]} {100 * gap:+.1f}" for name, gap in ranked[-3:])
    )
    if "mirrors" in found:
        mirrors = found["mirrors"]
        if "against" in mirrors:
            lines.append(
                f"  The mirrors against their champions: {100 * mirrors['mean']:.1f}% ± "
                f"{100 * mirrors['error']:.1f}%"
            )
            moved = mirrors["against"]["by deck"]
            lines.append(
                "  The mirrors most lost against the reference: "
                + ", ".join(f"{deck} {100 * gap:+.1f}" for gap, deck in moved[:NAMED])
                + "; most won: "
                + ", ".join(f"{deck} {100 * gap:+.1f}" for gap, deck in moved[-NAMED:])
            )
        if "beside" in mirrors:
            beside = mirrors["beside"]
            lines.append(
                f"  The mirrors deal by deal against {beside['network']}: "
                f"{100 * beside['difference']:+.1f} ± {100 * beside['error']:.1f} points, "
                f"lower on {beside['lower']} of {beside['decks']}"
            )
        each = mirrors["each"]
        lowest = sorted(each, key=lambda deck: each[deck]["share"])[:5]
        lines.append(
            "  The lowest mirrors: "
            + ", ".join(f"{deck} {100 * each[deck]['share']:.1f}%" for deck in lowest)
        )
    if "held-out" in found:
        told_ = found["held-out"]
        lines.append(
            f"  The next decks, trained on nothing: {100 * told_['gap']:+.1f} ± "
            f"{100 * told_['error']:.1f} points, from {100 * told_['start']:.1f}%"
        )
    if "pilots" in found:
        told_ = found["pilots"]
        lines.append(
            f"  The new decks against the pilots: {100 * (told_['start'] + told_['gap']):.1f}%, "
            f"from {100 * told_['start']:.1f}%, {100 * told_['gap']:+.1f} ± {100 * told_['error']:.1f}"
        )
    lines.append(f"  The run took {number(found['hours'])}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--points", required=True)
    parser.add_argument("--run", default="batch-one", help="the run whose metrics give the hours")
    parser.add_argument(
        "--reference",
        type=Path,
        help="the mirrors of the network reading 3 is read against, deal by deal",
    )
    parser.add_argument(
        "--beside", type=Path, help="the mirrors of a network told beside, deal by deal"
    )
    args = parser.parse_args()
    points = [int(point) for point in args.points.split(",")]
    found = report(args.directory, points, args.run, args.reference, args.beside)
    (args.directory / "report.json").write_text(
        json.dumps(found, indent=1, default=str), encoding="utf8"
    )
    print(shown(found, points))


if __name__ == "__main__":
    main()
