"""The readings of the network of the seventy-one decks
(experiments/2026-10-28-seventy-one-champions.md).

    uv run python tools/seventy_one_champions_report.py analysis/seventy-one-champions \\
        --decks blackwing,cyber,... --champions analysis/champions \\
        --teachers analysis/fourteen-decks/teachers

The directory is what `PARTS=copy-measure` of
`experiments/commands/seventy-one-champions.sh` leaves. A pair
`STEM.plays.json` and `STEM.held.json` is a suite played with the first
network named in the learner's seat and then the second.

- `heads-taught/DECK`: the copy's last checkpoint head to head against each
  teacher in its mirror;
- `taught-DECK/POINT`: the copy after POINT updates against `all71` in each
  mirror;
- `pilots/taught.json`, `pilots/all71.json`: the copy's last checkpoint and
  `all71`, each deck against each of the thirteen decks' pilots;
- `taught.metrics.jsonl`.

And where PPO is measured (`PARTS=ppo-measure`):

- `heads-ppo/DECK`: PPO's last checkpoint head to head against each teacher
  in its mirror;
- `pairings/ppo-PART.json`, `pairings/taught-PART.json`: PPO's last
  checkpoint and the copy's against `all71` on every ordered pairing, the
  parts of one suite;
- `pilots/ppo.json`: PPO's last checkpoint against the pilots;
- `checks/`: PPO's checks as it trained (tools/seventy_one_checks.py);
- `ppo.metrics.jsonl`.

Each teacher's own figure against `all71` in its mirror is its run's, from
its making (`CHAMPIONS/DECK/report.json`, tools/champions_report.py); for
the decks of the champions' release made before that report, its figure as
the fourteen decks measured it (`TEACHERS/DECK`, the same suites and seed).

The reading of the copy, fixed on 2026-10-28 before anything was built:

1. The copy keeps the teachers: head to head against each teacher in its
   mirror, by the last checkpoint, every deal from both seats: kept when the
   mean over the seventy-one mirrors is at least 48% by two standard errors
   and no mirror is at most 45% by two standard errors, lost when the mean
   is at most 48% by two standard errors or a mirror at most 45%, and
   otherwise not decided.

Told beside it, without a mark: each mirror against `all71` less its
teacher's figure; the copy along its updates; against the pilots beside
`all71`; the hours.

The readings of PPO, by its last checkpoint, fixed on 2026-10-28 once the
fourteen decks' mirror arms were read, before anything of it was built:

1. The mirrors kept, by the copy's rule.
2. Against `all71`, deck by deck: each deck's share, played by the network
   against `all71` playing each of the seventy-one decks: it beats all71
   when every deck is over 50% by two standard errors, falls short when one
   is at most 50% by two standard errors, and is otherwise not decided.
3. Against the copy: the mean over the ordered pairings against `all71`,
   the deals paired: PPO pays when it is at least 1 point over the copy by
   two standard errors, does not pay when it is at most 1 point over it by
   two standard errors, and is otherwise not decided.

Told beside them, without a mark: the mirrors against `all71` beside the
copy's; against the pilots beside the copy and `all71`; each check; the
hours.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourteen_decks_report import (
    ERRORS,
    against_pilots,
    beats,
    joined,
    kept_mirrors,
    number,
    outcomes_of,
)
from seventy_one_checks import read as checks_read
from specialist_distillation_report import shares, spent, stems
from two_decks_report import mean, paired, told

# PPO's reading 3: over the copy by so much.
PAYS = 0.01


def teacher_figure(deck: str, champions: Path, teachers: Path) -> dict[str, Any] | None:
    """A teacher's own share against `all71` in its mirror and its error:
    its run's, from its making, or as the fourteen decks measured it."""
    report_ = champions / deck / "report.json"
    if report_.is_file():
        found = json.loads(report_.read_text(encoding="utf8"))[deck]
        return {"share": found["share"], "error": found["error"], "from": "its making"}
    if (teachers / f"{deck}.held.json").is_file():
        return {**told(shares(teachers / deck)), "from": "the fourteen decks"}
    return None


def curve(directory: Path, decks: list[str]) -> list[dict[str, Any]]:
    """The copy along its updates against `all71` in each mirror: each
    deck's share, their mean and the weakest."""
    found = []
    for point in stems(directory / f"taught-{decks[0]}"):
        each = {deck: mean(shares(directory / f"taught-{deck}" / str(point))) for deck in decks}
        weakest = min(each, key=lambda deck: each[deck])
        found.append(
            {
                "updates": point,
                "mean": mean(list(each.values())),
                "weakest": weakest,
                "shares": each,
            }
        )
    return found


def against_teachers(
    directory: Path, decks: list[str], champions: Path, teachers: Path, last: int
) -> dict[str, Any]:
    """Each mirror against `all71` by the copy's last checkpoint, less its
    teacher's figure, the two measured apart."""
    each = {}
    for deck in decks:
        ours = told(shares(directory / f"taught-{deck}" / str(last)))
        theirs = teacher_figure(deck, champions, teachers)
        each[deck] = {**ours, "teacher": theirs}
        if theirs is not None:
            each[deck]["gap"] = ours["share"] - theirs["share"]
            each[deck]["gap error"] = math.hypot(ours["error"], theirs["error"])
    known = [one for one in each.values() if "gap" in one]
    return {
        "each": each,
        "gap": mean([one["gap"] for one in known]) if known else None,
        "error": joined([one["gap error"] for one in known]) if known else None,
        "unknown": [deck for deck, one in each.items() if "gap" not in one],
    }


def report(directory: Path, decks: list[str], champions: Path, teachers: Path) -> dict[str, Any]:
    heads = {deck: told(shares(directory / "heads-taught" / deck)) for deck in decks}
    along = curve(directory, decks)
    result: dict[str, Any] = {
        "teachers": {**kept_mirrors(heads), "each": heads},
        "curve": along,
        "against all71": against_teachers(
            directory, decks, champions, teachers, along[-1]["updates"]
        ),
    }
    pilots = directory / "pilots"
    if (pilots / "taught.json").is_file() and (pilots / "all71.json").is_file():
        ours = against_pilots(pilots / "taught.json", decks)
        theirs = against_pilots(pilots / "all71.json", decks)
        result["pilots"] = {
            "mean": ours["mean"],
            "decks": ours["decks"],
            "all71": theirs["mean"],
            "all71 decks": theirs["decks"],
            **paired(ours["games"], theirs["games"]),
        }
    result["hours"] = spent(directory, "taught")
    if (directory / "heads-ppo").is_dir():
        result["ppo"] = ppo_report(directory, decks)
    return result


def against_all71(folder: Path, network: str) -> dict[str, list[str]]:
    """How each game of every ordered pairing against `all71` ended, the
    suite's parts together."""
    played: dict[str, list[str]] = {}
    for path in sorted(folder.glob(f"{network}-*.json")):
        played.update(outcomes_of(path))
    if not played:
        raise SystemExit(f"Nothing of {network} is measured in {folder}")
    return played


def won(played: dict[str, list[str]], cases: list[str] | None = None) -> list[float]:
    """Each game won or not, in the order of the cases' names."""
    names = played if cases is None else cases
    return [float(game == "win") for name in sorted(names) for game in played[name]]


def pays(gap: float, error_: float) -> str:
    """PPO's reading 3."""
    if gap - ERRORS * error_ >= PAYS:
        return "pays"
    if gap + ERRORS * error_ <= PAYS:
        return "does not pay"
    return "not decided"


def ppo_report(directory: Path, decks: list[str]) -> dict[str, Any]:
    """PPO's readings by its last checkpoint, and what is told beside them."""
    heads = {deck: told(shares(directory / "heads-ppo" / deck)) for deck in decks}
    ours = against_all71(directory / "pairings", "ppo")
    theirs = against_all71(directory / "pairings", "taught")
    if sorted(ours) != sorted(theirs):
        raise SystemExit("PPO and the copy were played on other pairings")
    by_deck = {
        deck: told(won(ours, [name for name in ours if name.split("-vs-")[0] == deck]))
        for deck in decks
    }
    weakest = min(by_deck, key=lambda deck: by_deck[deck]["share"])
    gain = paired(won(ours), won(theirs))
    mirrors = [f"{deck}-vs-{deck}-checkpoint" for deck in decks]
    found: dict[str, Any] = {
        "mirrors": {**kept_mirrors(heads), "each": heads},
        "against all71": {
            "verdict": beats(by_deck),
            "mean": mean(won(ours)),
            "weakest": weakest,
            "each": by_deck,
        },
        "against the copy": {**gain, "verdict": pays(gain["gap"], gain["error"])},
        "mirrors against all71": {
            "ppo": mean(won(ours, mirrors)),
            "taught": mean(won(theirs, mirrors)),
        },
        "hours": spent(directory, "ppo"),
    }
    pilots = directory / "pilots"
    if all((pilots / f"{who}.json").is_file() for who in ("ppo", "taught", "all71")):
        ppo, copy, all71 = (
            against_pilots(pilots / f"{who}.json", decks) for who in ("ppo", "taught", "all71")
        )
        found["pilots"] = {
            "mean": ppo["mean"],
            "taught": copy["mean"],
            "all71": all71["mean"],
            "against the copy": paired(ppo["games"], copy["games"]),
            "against all71": paired(ppo["games"], all71["games"]),
        }
    if (directory / "checks" / "copy").is_dir():
        found["checks"] = checks_read(directory / "checks", decks)
    return found


def ppo_shown(found: dict[str, Any]) -> list[str]:
    mirrors, against, gain = found["mirrors"], found["against all71"], found["against the copy"]
    each = mirrors["each"]
    lowest = min(each, key=lambda deck: each[deck]["share"])
    floor = mirrors["under the floor"]
    weakest = against["each"][against["weakest"]]
    lines = [
        "PPO, by its last checkpoint:",
        (
            f"  1. The mirrors: {mirrors['verdict']}. Head to head against each teacher in its "
            f"mirror, the mean {mirrors['mean']:.1%} ± {mirrors['error']:.1%}, the lowest "
            f"{lowest} {each[lowest]['share']:.1%} ± {each[lowest]['error']:.1%}"
            + (f"; at most 45% by two standard errors: {', '.join(floor)}" if floor else "")
        ),
        (
            f"  2. Against all71: {against['verdict']}. Deck by deck, the mean over the pairings "
            f"{against['mean']:.1%}, the weakest {against['weakest']} {weakest['share']:.1%} ± "
            f"{weakest['error']:.1%}"
        ),
        (
            f"  3. Against the copy: {gain['verdict']}, {100 * gain['gap']:+.1f} ± "
            f"{100 * gain['error']:.1f} points over the pairings against all71, the deals paired"
        ),
        "Beside them:",
        "  Head to head against each teacher: "
        + ", ".join(f"{deck} {one['share']:.1%}" for deck, one in each.items()),
        "  Against all71, each deck over its pairings: "
        + ", ".join(f"{deck} {one['share']:.1%}" for deck, one in against["each"].items()),
        (
            f"  The mirrors against all71 {found['mirrors against all71']['ppo']:.1%}, the copy "
            f"{found['mirrors against all71']['taught']:.1%}"
        ),
    ]
    if "pilots" in found:
        pilots = found["pilots"]
        over_copy, over_all71 = pilots["against the copy"], pilots["against all71"]
        lines.append(
            f"  Against the thirteen decks' pilots {pilots['mean']:.1%}: the copy "
            f"{pilots['taught']:.1%} ({100 * over_copy['gap']:+.1f} ± "
            f"{100 * over_copy['error']:.1f} points), all71 {pilots['all71']:.1%} "
            f"({100 * over_all71['gap']:+.1f} ± {100 * over_all71['error']:.1f}), the games paired"
        )
    if "checks" in found:
        for point, check in found["checks"]["checks"].items():
            lines.append(
                f"  Its check after {point:,} updates against the copy's: the mirrors "
                f"{100 * check['mirrors']['gap']:+.1f} ± {100 * check['mirrors']['error']:.1f} "
                f"points, the pairings {100 * check['pairings']['gap']:+.1f} ± "
                f"{100 * check['pairings']['error']:.1f}"
            )
    lines.append(f"  PPO took {number(found['hours'])}.")
    return lines


def shown(result: dict[str, Any], decks: list[str]) -> str:
    teachers = result["teachers"]
    each = teachers["each"]
    lowest = min(each, key=lambda deck: each[deck]["share"])
    last = result["curve"][-1]["updates"]
    floor = teachers["under the floor"]
    lines = [
        f"The copy of the seventy-one decks, by its checkpoint after {last:,} updates:",
        (
            f"  1. The teachers: {teachers['verdict']}. Head to head against each teacher in its "
            f"mirror, the mean {teachers['mean']:.1%} ± {teachers['error']:.1%}, the lowest "
            f"{lowest} {each[lowest]['share']:.1%} ± {each[lowest]['error']:.1%}"
            + (f"; at most 45% by two standard errors: {', '.join(floor)}" if floor else "")
        ),
        "Beside it:",
        "  Head to head against each teacher: "
        + ", ".join(f"{deck} {each[deck]['share']:.1%}" for deck in decks),
        "  Against all71 in each mirror along the copy, the mean over the decks (the weakest): "
        + ", ".join(
            f"{point['updates']:,}: {point['mean']:.1%} ({point['weakest']} "
            f"{point['shares'][point['weakest']]:.0%})"
            for point in result["curve"]
        ),
    ]
    against = result["against all71"]
    if against["gap"] is not None:
        lines.append(
            f"  Against all71 in each mirror after {last:,} updates, less its teacher's figure: "
            f"the mean {100 * against['gap']:+.1f} ± {100 * against['error']:.1f} points"
        )
        lines.append(
            "    "
            + ", ".join(
                f"{deck} {100 * one['gap']:+.1f} ({one['share']:.1%} to {one['teacher']['share']:.1%})"
                for deck, one in against["each"].items()
                if "gap" in one
            )
        )
    if against["unknown"]:
        lines.append(f"    No teacher's figure: {', '.join(against['unknown'])}")
    if "pilots" in result:
        pilots = result["pilots"]
        weakest = sorted(pilots["decks"], key=lambda deck: pilots["decks"][deck])[:5]
        lines.append(
            f"  Against the thirteen decks' pilots {pilots['mean']:.1%}, all71 {pilots['all71']:.1%} "
            f"on the same games: {100 * pilots['gap']:+.1f} ± {100 * pilots['error']:.1f} points; "
            "the weakest decks "
            + ", ".join(
                f"{deck} {pilots['decks'][deck]:.1%} ({pilots['all71 decks'][deck]:.1%})"
                for deck in weakest
            )
        )
    lines.append(f"  The copy took {number(result['hours'])}.")
    if "ppo" in result:
        lines += ppo_shown(result["ppo"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decks", required=True)
    parser.add_argument("--champions", type=Path, default=Path("analysis/champions"))
    parser.add_argument("--teachers", type=Path, default=Path("analysis/fourteen-decks/teachers"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decks = [deck.strip() for deck in args.decks.split(",") if deck.strip()]
    result = report(args.directory, decks, args.champions, args.teachers)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf8")
    print(shown(result, decks))


if __name__ == "__main__":
    main()
