"""The readings of the copies of a specialist: new networks that imitate a
checkpoint as the pilot's copy imitated its pilot.

    uv run python tools/specialist_copy_report.py analysis/specialist-copy --updates 200 \\
        --output analysis/specialist-copy/report.json

The directory is what `experiments/commands/specialist-copy.sh` leaves.  The
copies, each with its figures (`NAME.metrics.jsonl`, `NAME.run.json`) and its
checkpoints on the held-out logged games (`heldout/NAME.json`): `choice`,
which is taught the teacher's first choice at each decision, and
`distribution`, which is taught what the teacher gives every answer.  Beside
them `pilot-copy`, the network that imitated the pilot from the same start,
on the same deals: its figures, its checkpoints after the same updates
(`heldout/pilot-copy.json`) and its later ones (`heldout/pilot-copy-later.json`).
And the teacher on the same held-out games (`heldout/teacher.json`), with,
where they are there, the checkpoints of the runs it came from
(`heldout/line.*.json`) and those runs' figures (`teacher.metrics.jsonl`).

A copy is read deal by deal, by the share of its last checkpoints at which it
wins a deal, and only when it has had the updates of `--updates`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learner_report import measured, metrics, reading, share, suite
from ppo_head_report import difference, over_checkpoints, plain
from search_leads_report import percent, points, wins

COPIES = ("choice", "distribution")
# The copy the readings are of: taught as the pilot's copy was.
MAIN = "choice"
OTHER = "distribution"
CONTROL = "pilot-copy"
LATER = "pilot-copy-later"
TEACHER = "teacher"
LINE = "line"
# The checkpoints a copy is read by: its last ones.
READ = 2
# The updates before a point that a copy's figures while it learns are of.
BEFORE = 10
# The marks of the readings.
BETTER = 0.02
ERRORS = 2.0
# The share of what the teacher has over its pilot that a copy is to hold.
HELD = 0.5


def said(directory: Path, name: str) -> dict[str, Any]:
    """What a run was started with."""
    path = directory / f"{name}.run.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else {}


def by_update(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """A run's rows by the updates it had had, the row's own among them."""
    return {int(row["version"]) + 1: row for row in rows}


def before(rows: dict[int, dict[str, Any]], key: str, point: int) -> float | None:
    """A figure over the updates that end at `point`."""
    values = [
        float(rows[update][key])
        for update in range(max(1, point - BEFORE + 1), point + 1)
        if update in rows and rows[update].get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def games_of(path: Path) -> str | None:
    """The logged games a measurement is of, by their fingerprint."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8")).get("fingerprint")


def line(directory: Path, games: str | None) -> dict[int, dict[str, Any]]:
    """The checkpoints of the runs the teacher came from, by their updates:
    those measured on the logged games `games`."""
    found: dict[int, dict[str, Any]] = {}
    for path in sorted((directory / "heldout").glob(f"{LINE}.*.json")):
        if games_of(path) != games:
            continue
        for entry in json.loads(path.read_text(encoding="utf8"))["checkpoints"]:
            found[int(entry["checkpoint_step"])] = entry
    return found


def won(entry: dict[str, Any]) -> float:
    deals = wins(entry["deals"])
    return sum(deals) / len(deals)


def pilot_deals(entries: list[dict[str, Any]]) -> list[float] | None:
    """The pilot's result on every deal, where a measurement holds it."""
    for entry in entries:
        outcomes = entry["deals"].get("references", {}).get("pilot", {}).get("outcomes")
        if outcomes:
            return [float(outcome == "win") for outcome in outcomes]
    return None


def report(directory: Path, updates: int | None = None) -> dict[str, Any]:
    rows = {name: by_update(metrics(directory, name)) for name in (*COPIES, CONTROL)}
    told = {name: said(directory, name) for name in (*COPIES, CONTROL)}
    held = {name: measured(directory, "heldout", name) for name in (*COPIES, CONTROL, LATER)}
    held[TEACHER] = measured(directory, "heldout", TEACHER)
    # A copy is set beside what was measured on the logged games it was
    # measured on: the same deals.  Anything measured on others is left out.
    logged = {name: games_of(directory / "heldout" / f"{name}.json") for name in held}
    measured_on = next((logged[name] for name in COPIES if held[name]), None)
    elsewhere = sorted(name for name in held if held[name] and logged[name] != measured_on)
    for name in elsewhere:
        held[name] = {}
    teacher_held = held.pop(TEACHER)
    result: dict[str, Any] = {
        "copies": {},
        "differences": {},
        "readings": {},
        "measured_on_other_games": elsewhere,
    }
    scores: dict[str, list[float]] = {}

    def over(first: str, second: str) -> tuple[float, float] | None:
        if first not in scores or second not in scores:
            return None
        found = difference(scores[first], scores[second])
        result["differences"][f"{first} over {second}"] = found
        return found

    def told_of(found: tuple[float, float] | None) -> str:
        return "not measured" if found is None else points(*found)

    # The teacher, and whether the copies and what they are set beside are
    # what the readings take them to be.
    teachers = {name: told[name].get("imitation_teacher") for name in COPIES if told[name]}
    described = next((teacher for teacher in teachers.values() if teacher), None)
    result["teacher"] = described
    if described:
        print(f"The teacher: {described['path']}, after {described['step']} updates.")
        same = len({teacher["fingerprint"] for teacher in teachers.values() if teacher}) == 1
        result["one_teacher"] = same and all(teachers.values())
        if len(teachers) > 1:
            print(
                "  the copies have the one teacher: " + ("yes" if result["one_teacher"] else "NO")
            )
    for name in COPIES:
        if told[name] and told[CONTROL]:
            alike = told[name].get("initial_model_fingerprint") == told[CONTROL].get(
                "initial_model_fingerprint"
            )
            result["copies"].setdefault(name, {})["starts_as_the_pilots_copy"] = alike
            print(
                f"  {name} starts from the network the pilot's copy started from: "
                + ("yes" if alike else "NO")
            )
    shared = sorted(set(rows[MAIN]) & set(rows[OTHER]))
    if shared:
        alike = all(
            rows[MAIN][update].get("rollout_decision_fingerprint")
            == rows[OTHER][update].get("rollout_decision_fingerprint")
            for update in shared
        )
        result["same_games"] = alike
        print(
            f"  the copies are taught on the same games, over their {len(shared)} updates "
            "in common: " + ("yes" if alike else "NO")
        )
    for name in COPIES:
        played = [row for row in rows[name].values() if row.get("learner_win_rate") is not None]
        if not played:
            continue
        games = sum(int(row.get("duels", 0)) for row in played)
        rate = sum(float(row["learner_win_rate"]) * int(row.get("duels", 0)) for row in played) / (
            games or 1
        )
        left = [
            float(row["imitation_departures"]) / float(row["samples"])
            for row in played
            if row.get("imitation_departures") is not None and row.get("samples")
        ]
        result["teacher_games"] = {
            "games": games,
            "win_rate": rate,
            "leaves_its_pilot": sum(left) / len(left) if left else None,
        }
        print(
            f"  in the {games:,} games it is imitated in, by its first choice, the teacher wins "
            f"{percent(rate)} ± {100 * math.sqrt(rate * (1 - rate) / max(games, 1)):.1f}; "
            f"it leaves its pilot's answer at {percent(result['teacher_games']['leaves_its_pilot'])} "
            "of its decisions"
        )
        break

    # The points a copy is measured at, and those it is read by.
    measured_copies = [name for name in COPIES if held[name]]
    steps = (
        sorted(set.intersection(*(set(held[name]) for name in measured_copies)))
        if measured_copies
        else []
    )
    shown = steps or sorted(
        update for update, row in rows[MAIN].items() if "evaluation_suite" in row
    )
    if not shown and rows[MAIN]:
        shown = [max(rows[MAIN])]
    read = steps[-READ:]
    complete = bool(steps) and updates is not None and steps[-1] == updates and len(read) == READ
    result["checkpoints"] = {"measured": steps, "read": read if complete else []}

    print()
    print(f"While they learn, over the {BEFORE} updates that end at each point and on games new")
    print("to the copy: how often its first choice is its teacher's (the pilot, for the pilot's")
    print("copy); where the teacher leaves its pilot, how often it is the teacher's answer and")
    print("how often still the pilot's; how far its answers are from the teacher's, in nats a")
    print("decision; and what its value head explains of the games' results.")
    print(
        f"{'copy':<14}{'updates':>8}{'agrees':>9}{'the teacher':>13}{'the pilot':>11}"
        f"{'apart':>8}{'head':>9}"
    )
    for name in (*COPIES, CONTROL):
        learned = []
        for point in shown:
            if point not in rows[name]:
                continue
            figures = {
                "updates": point,
                "agrees": before(rows[name], "imitation_model_teacher_agreement", point),
                "departures_followed": before(
                    rows[name], "imitation_model_departure_agreement", point
                ),
                "departures_refused": before(
                    rows[name], "imitation_model_departure_pilot_agreement", point
                ),
                "divergence": before(rows[name], "imitation_teacher_divergence", point),
                "head": before(rows[name], "imitation_value_explained_variance", point),
            }
            learned.append(figures)
            print(
                f"{name:<14}{point:>8}{percent(figures['agrees']):>9}"
                f"{percent(figures['departures_followed']):>13}"
                f"{percent(figures['departures_refused']):>11}"
                f"{plain(figures['divergence'], 3):>8}{share(figures['head']):>9}"
            )
        if learned:
            result["copies"].setdefault(name, {})["learning"] = learned

    print()
    if not steps:
        print("The copies are not measured on the held-out games.")
    else:
        if elsewhere:
            print(
                f"Measured on other logged games than {measured_copies[0]}, and left out: "
                + ", ".join(elsewhere)
            )
        deals = len(wins(held[measured_copies[0]][steps[0]]["deals"]))
        print(f"Held-out deals won, of {deals:,}, by the updates a copy had; `read` is what a")
        print(
            "copy is read by: its checkpoints after "
            + " and ".join(str(step) for step in read)
            + " updates."
        )
        print(f"{'copy':<14}" + "".join(f"{step:>8}" for step in steps) + f"{'read':>8}")
        for name in (*measured_copies, CONTROL):
            if not held[name]:
                continue
            each = {step: won(held[name][step]) for step in steps if step in held[name]}
            if all(step in held[name] for step in read):
                scores[name] = over_checkpoints(held[name], read)
            result["copies"].setdefault(name, {}).update(
                deals=each,
                deals_read=sum(scores[name]) / deals if name in scores else None,
            )
            print(
                f"{name:<14}"
                + "".join(f"{percent(each.get(step)):>8}" for step in steps)
                + f"{percent(result['copies'][name]['deals_read']):>8}"
            )
        if held[LATER]:
            later = sorted(held[LATER])
            scores[LATER] = over_checkpoints(held[LATER], later)
            result["copies"][LATER] = {
                "deals": {step: won(held[LATER][step]) for step in later},
                "deals_read": sum(scores[LATER]) / deals,
            }
            print(
                f"  the pilot's copy after {', '.join(str(step) for step in later)} updates: "
                + ", ".join(percent(won(held[LATER][step])) for step in later)
                + f" ({percent(result['copies'][LATER]['deals_read'])} over them)"
            )
        entries = [held[name][step] for name in measured_copies for step in steps]
        pilot = pilot_deals(entries)
        references = entries[-1]["deals"].get("references", {})
        beside = []
        if pilot is not None:
            scores["pilot"] = pilot
            result["pilot"] = sum(pilot) / deals
            beside.append(f"the pilot wins {percent(result['pilot'])}")
        if "search" in references:
            beside.append(
                f"the pilot with its search {percent(references['search'].get('win_rate'))}"
            )
        # The teacher as it was measured, when it is the one imitated.
        imitated = described["step"] if described else None
        if teacher_held and imitated in teacher_held:
            scores[TEACHER] = [float(deal) for deal in wins(teacher_held[imitated]["deals"])]
            result["teacher_deals"] = sum(scores[TEACHER]) / deals
            beside.append(f"the teacher {percent(result['teacher_deals'])}")
        elif teacher_held:
            print(
                f"  the checkpoint measured as the teacher has had {sorted(teacher_held)} updates, "
                f"the teacher {imitated}: it is not set beside the copies"
            )
        if beside:
            print("  on those deals " + ", ".join(beside))
        if pilot is not None and TEACHER in scores:
            # What a copy is to hold: its pilot's deals and so much of what
            # the teacher has over them.
            scores["mark"] = [
                (1 - HELD) * low + HELD * high
                for low, high in zip(pilot, scores[TEACHER], strict=True)
            ]

        if not complete:
            print(
                f"  the copies are measured as far as {steps[-1]} updates"
                + ("" if updates is None else f" of {updates}")
                + ": they are not read"
            )
        else:
            print()
            print("Deal by deal, a copy by the checkpoints it is read by:")
            for name in measured_copies:
                print(f"  {name} over the pilot: {told_of(over(name, 'pilot'))}")
                print(
                    f"  {name} over the pilot's copy after as many updates: "
                    f"{told_of(over(name, CONTROL))}"
                )
                print(f"  {name} over the pilot's copy at its end: {told_of(over(name, LATER))}")
                print(f"  {name} over its teacher: {told_of(over(name, TEACHER))}")
                print(
                    f"  {name} over its pilot and {HELD:.0%} of what the teacher has over it: "
                    f"{told_of(over(name, 'mark'))}"
                )
            if OTHER in scores and MAIN in scores:
                print(f"  {OTHER} over {MAIN}: {told_of(over(OTHER, MAIN))}")
            print(f"  the teacher over the pilot: {told_of(over(TEACHER, 'pilot'))}")

        # The runs the teacher came from: when they had what a copy has.
        before_it = line(directory, measured_on)
        behind = {**held[CONTROL], **held[LATER], **before_it}
        duels = int(next(iter(rows[MAIN].values())).get("duels", 0)) if rows[MAIN] else 0
        if complete and before_it:
            print()
            print("The line the teacher came from (its pilot's copy, then PPO from it), after so")
            print("many updates and games, on the same deals:")
            print(
                "  " + ", ".join(f"{step}: {percent(won(behind[step]))}" for step in sorted(behind))
            )
            result["line"] = {step: won(behind[step]) for step in sorted(behind)}
            for name in measured_copies:
                level = result["copies"][name]["deals_read"]
                reached = next(
                    (step for step in sorted(behind) if won(behind[step]) >= level), None
                )
                result["copies"][name]["line_reached_after"] = reached
                print(
                    f"  {name} wins {percent(level)} after {read[-1]} updates"
                    + (f" ({read[-1] * duels:,} games)" if duels else "")
                    + ": the line first won as many after "
                    + (
                        "none of its checkpoints"
                        if reached is None
                        else f"{reached} updates"
                        + (f" ({reached * duels:,} games)" if duels else "")
                    )
                )

    # The suite, as the runs played it while they learned.
    played = {
        name: {
            update: suite(row) for update, row in rows[name].items() if "evaluation_suite" in row
        }
        for name in (*COPIES, CONTROL)
    }
    evaluated = sorted(set().union(*(set(played[name]) for name in COPIES)))
    if evaluated:
        print()
        print("The suite, by the updates a copy had:")
        print(f"{'copy':<14}" + "".join(f"{update:>8}" for update in evaluated))
        for name in (*COPIES, CONTROL):
            if not played[name]:
                continue
            result["copies"].setdefault(name, {})["suite"] = played[name]
            print(
                f"{name:<14}"
                + "".join(f"{percent(played[name].get(update)):>8}" for update in evaluated)
            )
        taught = by_update(metrics(directory, TEACHER))
        imitated = described["step"] if described else None
        if imitated in taught and "evaluation_suite" in taught[imitated]:
            result["teacher_suite"] = suite(taught[imitated])
            print(f"  the teacher wins {percent(result['teacher_suite'])} of it")

    def better(found: tuple[float, float] | None) -> bool | None:
        return None if found is None else found[0] >= BETTER and found[0] > ERRORS * found[1]

    differences = result["differences"]
    past = differences.get(f"{MAIN} over pilot")
    holds = differences.get(f"{MAIN} over mark")
    more = differences.get(f"{OTHER} over {MAIN}")
    readings = {
        "past_the_pilot": better(past),
        "holds_its_share": None if holds is None else holds[0] >= 0.0,
        "distribution_teaches_more": better(more),
    }
    result["readings"] = readings
    mark = f"the mark {BETTER * 100:.0f} points and {ERRORS:.0f} standard errors"
    print()
    print("The readings:")
    print(
        reading(
            f"a copy of the specialist is past the pilot ({MAIN} over the pilot {told_of(past)}, "
            f"{mark})",
            readings["past_the_pilot"],
        )
    )
    print(
        reading(
            f"it holds {HELD:.0%} of what its teacher has over the pilot ({MAIN} over that "
            f"much {told_of(holds)}, the mark 0)",
            readings["holds_its_share"],
        )
    )
    print(
        reading(
            f"the teacher's distribution teaches more than its choice ({OTHER} over {MAIN} "
            f"{told_of(more)}, {mark})",
            readings["distribution_teaches_more"],
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--updates", type=int, help="the updates a copy was to have, for its readings"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.directory, args.updates)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
