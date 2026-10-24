"""The readings of the network of two decks with its duels against itself in
the mirrors only (experiments/2026-10-24-two-decks.md, its follow-up on the
mirrors only).

    uv run python tools/two_decks_mirrors_report.py analysis/two-decks/mirrors \\
        --both analysis/two-decks \\
        --cyber-alone analysis/cyber-champion/cyber-old/mirror

The directory is what `PARTS=measure` of
`experiments/commands/two-decks-mirrors.sh` leaves, in the files of
`tools/two_decks_report.py`: pairs `STEM.plays.json` and `STEM.held.json` on
the 2,048 deals of a suite, the first network named in the learner's seat
and then the second, STEM the first's updates of PPO.

- `all71-blackwing/`, `all71-cyber/` and `cross/`: the network of the
  mirrors (`mirrors`) against `all71` in each deck's mirror and in Blackwing
  against Cyber, `mirrors` after 960, 1,200 and 1,440 updates of PPO;
- `blackwing/` and `cyber/`: `mirrors` against each deck's champion in its
  mirror, Blackwing alone and `cyber-old` after 480, 600 and 720;
- `both-blackwing/` and `both-cyber/`: `mirrors` against the network of both
  (`both`) in each mirror, `both` after 1,920, 2,400 and 2,880;
- `mirrors.metrics.jsonl`: the run.

At the same points each network has had as many games of each mirror.
`--both` is where `tools/two_decks_report.py` read `both` and Blackwing
alone (`all71-blackwing/`, `all71-cyber/`, `cross/` and `all71-alone/`);
`--cyber-alone`, where `cyber-old` was played against `all71` in the Cyber
mirror.

The readings, fixed on 2026-10-24 before anything was built:

1. What the cross games bring to a mirror: for each deck, `both` against
   `mirrors` by their shares against `all71` in that deck's mirror, at as
   many games of it, the deals paired: the cross games bring strength there
   when `both` is at least 2 points over by two standard errors, cost
   strength when it is at least 2 points under by two standard errors, and
   are otherwise not told apart.
2. The same games in one network: `mirrors` against each deck's champion
   head to head in its mirror, at as many games of it, every deal from both
   seats: it matches when its share less two standard errors is at least
   48%, is weaker when its share plus two standard errors is at most 48%,
   and is otherwise not decided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_decks_report import deals, hours, paired, told, verdict

# Reading 1: so many points between the two networks against all71.
BRINGS = 0.02
ERRORS = 2.0
# Each deck: its name, its folder's part and its champion.
DECKS = (("Blackwing", "blackwing", "Blackwing alone"), ("Cyber", "cyber", "cyber-old"))


def brings(gap: float, error_: float) -> str:
    """Reading 1, for a deck."""
    if gap - ERRORS * error_ >= BRINGS:
        return "the cross games bring strength"
    if gap + ERRORS * error_ <= -BRINGS:
        return "the cross games cost strength"
    return "not told apart"


def report(directory: Path, both: Path, cyber_alone: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"cross games": {}, "champions": {}, "beside": {}}
    alone = {"blackwing": both / "all71-alone", "cyber": cyber_alone}
    for deck, part, champion in DECKS:
        ours = deals(directory / f"all71-{part}")
        theirs = deals(both / f"all71-{part}")
        found = paired(theirs, ours)
        found["verdict"] = brings(found["gap"], found["error"])
        found["both"], found["mirrors"] = told(theirs)["share"], told(ours)["share"]
        result["cross games"][deck] = found
        head = told(deals(directory / part))
        head["champion"] = champion
        head["verdict"] = verdict(head["share"], head["error"])
        result["champions"][deck] = head
        beside = {"against both": told(deals(directory / f"both-{part}"))}
        if alone[part] is not None:
            beside["against all71, over the champion"] = paired(ours, deals(alone[part]))
        result["beside"][deck] = beside
    ours, theirs = deals(directory / "cross"), deals(both / "cross")
    result["beside"]["Blackwing against Cyber, against all71"] = {
        "mirrors": told(ours),
        "both": told(theirs),
        "paired": paired(ours, theirs),
    }
    spent = hours(directory, "mirrors")
    result["hours"] = {} if spent is None else {"mirrors": spent}
    return result


def shown(result: dict[str, Any]) -> str:
    lines = ["1. What the cross games bring to a mirror, both against mirrors against all71:"]
    for deck, found in result["cross games"].items():
        lines.append(
            f"   {deck}: {found['verdict']}. both {found['both']:.1%}, mirrors "
            f"{found['mirrors']:.1%}: {100 * found['gap']:+.1f} ± {100 * found['error']:.1f} "
            "points, the deals paired"
        )
    lines.append("2. The same games in one network, mirrors against each champion head to head:")
    for deck, found in result["champions"].items():
        lines.append(
            f"   {deck}: {found['verdict']}. mirrors wins {found['share']:.1%} ± "
            f"{found['error']:.1%} of the {deck} mirror against {found['champion']} "
            f"({found['deals']:,} deals, each from both seats)"
        )
    lines.append("Beside them:")
    for deck, _, _ in DECKS:
        found = result["beside"][deck]
        line = f"  {deck}: mirrors against both {found['against both']['share']:.1%}"
        over = found.get("against all71, over the champion")
        if over is not None:
            line += (
                f"; against all71, {100 * over['gap']:+.1f} ± {100 * over['error']:.1f} "
                "points over the champion, the deals paired"
            )
        lines.append(line)
    cross = result["beside"]["Blackwing against Cyber, against all71"]
    lines.append(
        f"  Blackwing against Cyber, against all71: mirrors {cross['mirrors']['share']:.1%}, "
        f"both {cross['both']['share']:.1%}: {100 * cross['paired']['gap']:+.1f} ± "
        f"{100 * cross['paired']['error']:.1f} points"
    )
    if result["hours"]:
        lines.append(f"  hours: mirrors {result['hours']['mirrors']:.2f} h")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--both", type=Path, required=True, help="where both was measured")
    parser.add_argument("--cyber-alone", type=Path, help="cyber-old against all71 in its mirror")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.both, args.cyber_alone)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(shown(result))


if __name__ == "__main__":
    main()
