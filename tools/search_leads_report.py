"""The search-leads experiment: the weight of the lead loss, the choice on
the validation games and the reading of the held-out ones.

Usage, from the repository root::

    uv run python tools/search_leads_report.py seed analysis/search-leads/pilot-suite.json \\
        analysis/search-leads/seed-suite.json --checkpoint runs/seed/checkpoints/model-00089.pt
    uv run python tools/search_leads_report.py weight runs/search-leads/calibration/metrics.jsonl
    uv run python tools/search_leads_report.py read analysis/search-leads/arms \\
        --choice analysis/search-leads/arms/choice.json
    uv run python tools/search_leads_report.py read analysis/search-leads/arms \\
        --output analysis/search-leads/arms/report.json
    uv run python tools/search_leads_report.py more analysis/search-leads/arms

`seed` holds the model the arms start from against its pilot on the
evaluation suite, from the two `evaluate-suite` results, and fails where the
model wins more than five points less than the pilot or is not the
checkpoint named.

`weight` reads the updates of a calibration run, which trained the lead head
with the weight 1.  Each reports how hard the imitation loss and the lead
loss pull on the parameters they share, before it moves anything.  The
weight is the first over the second, over the run's updates but the first,
where the head is still untrained: with it the two pull alike.

`read` takes the directory of the `evaluate-search` results the arms' script
writes.
`validation/ARM.json` has every arm at every measured update on the
validation run, and `validation/seed.json` the model the arms start from.
An arm with a lead head is read there by its rule: its policy's first
choice, unless the head predicts for another answer a lead over it of at
least a margin.  The update and the margin at which the rule gains most are
its choice, and `--choice` writes the choices down; an arm whose rule gains
nothing above zero anywhere has none.

`heldout/ARM.json` is then each such arm at its choice, with the run's deals
played by the rule, and the other arms at the updates so chosen.  The
readings are applied to them as the experiment's record fixes them: the
mark is a quarter of what the search gains over its pilot on those deals.

`more` takes the same directory once `more/ARM.json` holds every arm at
every measured update on the held-out run, measured after the readings and
changing none of them.  It sets every arm against the `pilot` arm at the
same update, by the deals won and by what its other answers are worth, and
gives the rule's changes apart by where they start: from the pilot's
answer, back to it, or from one other answer to another.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# The arms whose head is trained, and so read by its rule; the others.
LED = ("leads", "leads-read")
OTHERS = ("pilot", "search")
# The share of the search's gain the readings ask for, and the standard
# errors above zero: of the rule's gain, and of the deals won.
SHARE = 0.25
GAIN_ERRORS = 3.0
WIN_ERRORS = 2.0
# How far under its pilot on the suite the arms' starting model may be.
SEED_SLACK = 0.05


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def points(value: float | None, error: float | None = None) -> str:
    """A difference of win rates, in points."""
    if value is None or math.isnan(value):
        return "-"
    return f"{100 * value:+.1f}" + ("" if error is None else f" ± {100 * error:.1f}")


def paired(first: list[bool], second: list[bool]) -> tuple[float, float]:
    """The difference of two rates on the same deals, and its standard error."""
    total = len(first)
    if not total or total != len(second):
        raise ValueError(f"{total} deals against {len(second)}: not the same run")
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    difference = (only_first - only_second) / total
    variance = (only_first + only_second - (only_first - only_second) ** 2 / total) / total**2
    return difference, math.sqrt(max(variance, 0.0))


def wins(deals: dict[str, Any]) -> list[bool]:
    return [outcome == "win" for outcome in deals["outcomes"]]


def load(folder: Path) -> dict[str, list[dict[str, Any]]]:
    """Every result of a folder: the checkpoints measured, by the file's name.

    A report written beside the results holds no checkpoints and is left out.
    """
    found = {}
    for path in sorted(folder.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf8"))
        if "checkpoints" in result:
            found[path.stem] = result["checkpoints"]
    return found


def led(entry: dict[str, Any], margin: float) -> dict[str, Any]:
    """An entry's reading by its rule at a margin."""
    tried = entry["answer_key"]["tried"]
    for reading in tried.get("led") or []:
        if math.isclose(reading["margin"], margin):
            return reading
    raise ValueError(f"{entry['checkpoint_label']} is not read at a margin of {margin}")


def choose(entries: list[dict[str, Any]], seed_step: int) -> dict[str, Any] | None:
    """The update and the margin at which an arm's rule gains most a game.

    Of equal gains the earlier update, then the larger margin.  None where
    it gains nothing above zero at any of them.
    """
    best: tuple[float, int, float] | None = None
    chosen: dict[str, Any] | None = None
    for entry in entries:
        update = int(entry["checkpoint_step"]) - seed_step
        for reading in entry["answer_key"]["tried"]["led"]:
            key = (float(reading["gain_per_game"]), -update, float(reading["margin"]))
            if best is None or key > best:
                best = key
                chosen = {
                    "update": update,
                    "version": entry["version"],
                    "checkpoint": entry["checkpoint"],
                    "margin": reading["margin"],
                    "gain_per_game": reading["gain_per_game"],
                    "gain_per_game_error": reading["gain_per_game_error"],
                    "changed": reading["changed"],
                }
    if best is None or best[0] <= 0.0:
        return None
    return chosen


def lead_weight(records: list[dict[str, Any]]) -> tuple[float, list[tuple[int, float, float]]]:
    """The weight at which the lead loss pulls on the shared parameters as
    hard as the imitation loss, from a calibration run's updates but the
    first, and the two gradients of every update it holds."""
    measured = [
        (
            int(record["version"]),
            float(record["imitation_shared_gradient_norm"]),
            float(record["imitation_lead_shared_gradient_norm"]),
        )
        for record in records
        if "imitation_lead_shared_gradient_norm" in record
    ]
    used = measured[1:]
    if len(used) < 2:
        raise ValueError(f"{len(measured)} updates with a lead loss: too few to set its weight")
    lead = sum(item[2] for item in used)
    if lead <= 0.0:
        raise ValueError("the lead loss has no gradient on the shared parameters")
    return sum(item[1] for item in used) / lead, measured


def suite_rate(record: dict[str, Any]) -> tuple[float, int]:
    """What one player won of an evaluation suite, its cases pooled."""
    cases = record["evaluation_suite"]
    failed = [name for name, case in cases.items() if "wins" not in case]
    if failed:
        raise ValueError(f"cases of the suite gave no result: {', '.join(sorted(failed))}")
    games = sum(int(case["games"]) for case in cases.values())
    return sum(int(case["wins"]) for case in cases.values()) / games, games


def seed_gap(pilot: dict[str, Any], model: dict[str, Any]) -> tuple[float, float, int]:
    """The pilot's and a model's rates on the same suite, and its games."""
    reference, games = suite_rate(pilot)
    rate, played = suite_rate(model)
    if played != games or pilot["evaluation_suite"].keys() != model["evaluation_suite"].keys():
        raise ValueError("the pilot and the model were not measured on the same suite")
    return reference, rate, games


def references(entry: dict[str, Any]) -> tuple[float, float]:
    """What the pilot and the search won of a run's deals."""
    found = entry["deals"]["references"]
    return found["pilot"]["win_rate"], found["search"]["win_rate"]


def others(tried: dict[str, Any]) -> float:
    """A model's answers other than the pilot's, a game."""
    return (tried["decisions"] - tried["pilot"]) / tried["games"]


def validation(
    arms: dict[str, list[dict[str, Any]]], seed_step: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The validation run: every arm at every update, the arms with a head by
    their rule at every margin, and their choices."""
    start = arms["seed"][0]
    pilot, search = references(start)
    games = len(start["deals"]["outcomes"])
    print(f"{games} validation deals: pilot {percent(pilot)}, search {percent(search)}")
    print("Each arm's own first choices: deals won, its other answers than the pilot's")
    print("a game, and their leads added up over a game by the search's measurements.")
    print(f"{'arm':<12}{'update':>7}{'wins':>8}{'others':>8}{'sum a game':>14}")
    table: dict[str, Any] = {}
    for name in ("seed", *OTHERS, *LED):
        for entry in arms.get(name, []):
            tried = entry["answer_key"]["tried"]
            update = int(entry["checkpoint_step"]) - seed_step
            print(
                f"{name:<12}{update:>7}{percent(entry['deals']['win_rate']):>8}"
                f"{others(tried):>8.2f}"
                f"{points(tried['lead_per_game'], tried['lead_per_game_error']):>14}"
            )
            table.setdefault(name, {})[str(update)] = {
                "win_rate": entry["deals"]["win_rate"],
                "others_per_game": others(tried),
                "lead_per_game": tried["lead_per_game"],
                "lead_per_game_error": tried["lead_per_game_error"],
            }
    choices: dict[str, Any] = {}
    for name in LED:
        entries = arms.get(name)
        if not entries:
            continue
        margins = [reading["margin"] for reading in entries[0]["answer_key"]["tried"]["led"]]
        print()
        print(f"{name} by its rule: what it gains a game at each margin, and in brackets the")
        print("first choices it changes a game.")
        print(
            f"{'update':>7}" + "".join(f"{f'{100 * margin:.0f} points':>22}" for margin in margins)
        )
        for entry in entries:
            update = int(entry["checkpoint_step"]) - seed_step
            cells = []
            for margin in margins:
                reading = led(entry, margin)
                gain = points(reading["gain_per_game"], reading["gain_per_game_error"])
                cells.append(f"{gain} ({reading['changed'] / games:.2f})")
                table[name][str(update)].setdefault("led", {})[str(margin)] = {
                    "gain_per_game": reading["gain_per_game"],
                    "gain_per_game_error": reading["gain_per_game_error"],
                    "changed": reading["changed"],
                }
            print(f"{update:>7}" + "".join(f"{cell:>22}" for cell in cells))
        print("what the head knows: its predictions against the measured leads, and the")
        print("measured lead of the alternatives it ranks highest.")
        shares = [item["share"] for item in entries[0]["answer_key"]["tried"]["head"]["highest"]]
        print(
            f"{'update':>7}{'correlation':>13}"
            + "".join(f"{f'top {100 * share:g}%':>12}" for share in shares)
        )
        for entry in entries:
            head = entry["answer_key"]["tried"]["head"]
            update = int(entry["checkpoint_step"]) - seed_step
            print(
                f"{update:>7}{head['correlation']:>13.3f}"
                + "".join(f"{points(item['lead']):>12}" for item in head["highest"])
            )
            table[name][str(update)]["head"] = {
                "correlation": head["correlation"],
                "highest": {str(item["share"]): item["lead"] for item in head["highest"]},
            }
        choice = choose(entries, seed_step)
        choices[name] = choice
        if choice is None:
            print(f"{name}: its rule gains nothing above zero at any update and margin.")
            print("Not learnable this way; the held-out games are not read for it.")
        else:
            gain = points(choice["gain_per_game"], choice["gain_per_game_error"])
            print(
                f"{name}: chosen update {choice['update']}, margin "
                f"{100 * choice['margin']:.0f} points ({gain} a game here)"
            )
    return table, choices


def heldout(
    arms: dict[str, list[dict[str, Any]]], choices: dict[str, Any], seed_step: int
) -> dict[str, Any]:
    """The held-out run, at the choices, and the readings."""
    start = arms["seed"][0]
    pilot, search = references(start)
    games = len(start["deals"]["outcomes"])
    mark = SHARE * (search - pilot)
    print()
    print(
        f"{games} held-out deals: pilot {percent(pilot)}, search {percent(search)}. The search "
        f"gains {points(search - pilot)}"
    )
    print(f"points and the mark of the readings is a quarter of it: {points(mark)}.")
    if mark <= 0.0:
        print("A search that gains nothing over its pilot sets no mark: nothing is shown here.")
    print("Each arm's own first choices as above, and the arms with a head by their rule:")
    print("deals won, first choices changed a game, and what the rule gains a game.")
    print(
        f"{'arm':<12}{'update':>7}{'wins':>8}{'others':>8}{'sum a game':>14}"
        f"{'margin':>8}{'by rule':>9}{'changed':>9}{'gain a game':>14}"
    )
    report: dict[str, Any] = {
        "games": games,
        "pilot": pilot,
        "search": search,
        "mark": mark,
        "arms": {},
        "readings": {},
    }
    by_update: dict[str, dict[int, dict[str, Any]]] = {}
    for name in ("seed", *OTHERS, *LED):
        for entry in arms.get(name, []):
            tried = entry["answer_key"]["tried"]
            update = int(entry["checkpoint_step"]) - seed_step
            by_update.setdefault(name, {})[update] = entry
            row = (
                f"{name:<12}{update:>7}{percent(entry['deals']['win_rate']):>8}"
                f"{others(tried):>8.2f}"
                f"{points(tried['lead_per_game'], tried['lead_per_game_error']):>14}"
            )
            item: dict[str, Any] = {
                "win_rate": entry["deals"]["win_rate"],
                "others_per_game": others(tried),
                "lead_per_game": tried["lead_per_game"],
                "lead_per_game_error": tried["lead_per_game_error"],
            }
            if name in LED:
                choice = choices.get(name)
                played = entry.get("led_deals")
                if choice is None or played is None:
                    raise ValueError(f"heldout/{name}.json is read without a choice or a rule")
                if entry["version"] != choice["version"] or not math.isclose(
                    played["margin"], choice["margin"]
                ):
                    raise ValueError(
                        f"heldout/{name}.json is version {entry['version']} at a margin of "
                        f"{played['margin']}, and the choice is version {choice['version']} "
                        f"at {choice['margin']}"
                    )
                reading = led(entry, choice["margin"])
                row += (
                    f"{100 * choice['margin']:>8.0f}{percent(played['win_rate']):>9}"
                    f"{reading['changed'] / games:>9.2f}"
                    f"{points(reading['gain_per_game'], reading['gain_per_game_error']):>14}"
                )
                item.update(
                    {
                        "margin": choice["margin"],
                        "led_win_rate": played["win_rate"],
                        "changed": reading["changed"],
                        "gain_per_game": reading["gain_per_game"],
                        "gain_per_game_error": reading["gain_per_game_error"],
                        "head": entry["answer_key"]["tried"]["head"],
                    }
                )
            print(row)
            report["arms"].setdefault(name, {})[str(update)] = item
    for name in LED:
        choice = choices.get(name)
        if choice is None or name not in by_update:
            continue
        update = choice["update"]
        entry = by_update[name][update]
        control = by_update.get("pilot", {}).get(update)
        if control is None:
            raise ValueError(f"heldout/pilot.json does not hold update {update}")
        reading = led(entry, choice["margin"])
        gain, gain_error = reading["gain_per_game"], reading["gain_per_game_error"]
        ruled = wins(entry["led_deals"])
        against_pilot = paired(ruled, wins(control["deals"]))
        against_itself = paired(ruled, wins(entry["deals"]))
        learnable = 0.0 < mark <= gain and gain > GAIN_ERRORS * gain_error
        better = 0.0 < mark <= against_pilot[0] and against_pilot[0] > WIN_ERRORS * against_pilot[1]
        head = entry["answer_key"]["tried"]["head"]
        print()
        print(f"{name}, update {update}, margin {100 * choice['margin']:.0f} points:")
        print(
            f"  the rule gains {points(gain, gain_error)} a game; the mark is {points(mark)} "
            f"and three"
        )
        print(
            f"  standard errors are {points(GAIN_ERRORS * gain_error)}: "
            + ("the leads are learnable" if learnable else "the leads are not shown learnable")
        )
        print(
            f"  by its rule against the pilot arm at the same update: "
            f"{points(*against_pilot)} of deals won;"
        )
        print(
            f"  two standard errors are {points(WIN_ERRORS * against_pilot[1])}: "
            + ("it plays better" if better else "it is not shown to play better")
        )
        print(f"  by its rule against its own first choices: {points(*against_itself)}")
        print(
            f"  the head over {head['alternatives']} alternatives: correlation "
            f"{head['correlation']:.3f}; those it ranks highest lead by "
            + ", ".join(
                f"{points(item['lead'], item['lead_error'])} (top {100 * item['share']:g}%)"
                for item in head["highest"]
            )
        )
        found: dict[str, Any] = {
            "update": update,
            "margin": choice["margin"],
            "gain_per_game": gain,
            "gain_per_game_error": gain_error,
            "learnable": learnable,
            "against_pilot": against_pilot,
            "plays_better": better,
            "against_own_first_choices": against_itself,
        }
        searched = by_update.get("search", {}).get(update)
        if searched is not None:
            found["search_against_pilot"] = paired(wins(searched["deals"]), wins(control["deals"]))
            print(
                f"  the search arm against the pilot arm at that update: "
                f"{points(*found['search_against_pilot'])}"
            )
        report["readings"][name] = found
    return report


def mean_error(values: list[float]) -> tuple[float, float]:
    """A mean and its standard error."""
    count = len(values)
    centre = sum(values) / count
    if count < 2:
        return centre, math.nan
    return centre, math.sqrt(sum((value - centre) ** 2 for value in values) / (count - 1) / count)


def against(
    mine: dict[int, list[float]], theirs: dict[int, list[float]]
) -> dict[str, tuple[float, float]]:
    """Two arms on the same deals at the same updates: the first's values less
    the second's at each update, and over all of them, deal by deal."""
    found: dict[str, tuple[float, float]] = {}
    shared = sorted(mine.keys() & theirs.keys())
    for update in shared:
        if len(mine[update]) != len(theirs[update]):
            raise ValueError(f"update {update} is not measured on the same deals")
        found[str(update)] = mean_error([a - b for a, b in zip(mine[update], theirs[update])])
    if len(shared) > 1:
        # A deal's difference averaged over the updates: the deals are the
        # sample, the checkpoints are what they are.
        deals = range(len(mine[shared[0]]))
        found["all"] = mean_error(
            [
                sum(mine[update][deal] - theirs[update][deal] for update in shared) / len(shared)
                for deal in deals
            ]
        )
    return found


def compared(
    title: list[str],
    arms: dict[str, dict[int, list[float]]],
    show: Any,
) -> dict[str, Any]:
    """Every arm at every update, and beside it what it has over the `pilot`
    arm at the same update."""
    for line in title:
        print(line)
    names = [name for name in (*OTHERS, *LED) if name in arms]
    control = arms.get("pilot", {})
    updates = sorted({update for name in names for update in arms[name]})
    print(f"{'update':>7}" + "".join(f"{name:>28}" for name in names))
    report: dict[str, Any] = {}
    rows = [(str(update), update) for update in updates]
    if len(updates) > 1:
        rows.append(("all", None))
    differences = {name: against(arms[name], control) for name in names if name != "pilot"}
    for label, update in rows:
        cells = []
        for name in names:
            if update is None:
                values = [value for chosen in sorted(arms[name]) for value in arms[name][chosen]]
            else:
                values = arms[name].get(update, [])
            if not values:
                cells.append("-")
                continue
            centre = sum(values) / len(values)
            cell = show(centre)
            item: dict[str, Any] = {"value": centre}
            difference = differences.get(name, {}).get(label)
            if difference is not None:
                cell += f" ({points(*difference)})"
                item["against_pilot"] = difference
            cells.append(cell)
            report.setdefault(name, {})[label] = item
        print(f"{label:>7}" + "".join(f"{cell:>28}" for cell in cells))
    return report


def more(args: argparse.Namespace) -> None:
    first = load(args.directory / "heldout")
    measured = load(args.directory / "more")
    if "seed" not in first or not measured:
        raise SystemExit(f"{args.directory} holds no heldout/seed.json or nothing under more/")
    start = first["seed"][0]
    seed_step = int(start["checkpoint_step"])
    choices = json.loads((args.directory / "choice.json").read_text(encoding="utf8"))
    pilot, search = references(start)
    games = len(start["deals"]["outcomes"])
    entries = {
        name: {int(entry["checkpoint_step"]) - seed_step: entry for entry in found}
        for name, found in measured.items()
    }
    # What was read first is measured again here: it must come out the same.
    for name, found in first.items():
        for entry in found:
            again = entries.get(name, {}).get(int(entry["checkpoint_step"]) - seed_step)
            if again is not None and again["deals"]["outcomes"] != entry["deals"]["outcomes"]:
                raise SystemExit(
                    f"more/{name}.json does not play the games of heldout/{name}.json "
                    f"at version {entry['version']}"
                )
    print(
        f"{games} held-out deals, after the readings: pilot {percent(pilot)}, "
        f"search {percent(search)}."
    )
    report: dict[str, Any] = {"games": games, "pilot": pilot, "search": search}
    report["deals_won"] = compared(
        [
            "Deals won by each arm's own first choices, and in brackets what it has over the",
            "pilot arm at the same update, deal by deal.",
        ],
        {
            name: {
                update: [float(won) for won in wins(entry["deals"])]
                for update, entry in found.items()
            }
            for name, found in entries.items()
        },
        percent,
    )
    print()
    report["leads"] = compared(
        [
            "What each arm's other answers than the pilot's are worth a game by the search's",
            "measurements, and in brackets against the pilot arm's, game by game.",
        ],
        {
            name: {
                update: entry["answer_key"]["tried"]["lead_by_game"]
                for update, entry in found.items()
            }
            for name, found in entries.items()
        },
        points,
    )
    report["rule"] = {}
    for name in LED:
        found = entries.get(name)
        choice = choices.get(name)
        if not found:
            continue
        margins = [
            reading["margin"]
            for reading in next(iter(found.values()))["answer_key"]["tried"]["led"]
        ]
        print()
        chosen = (
            "it has no choice"
            if choice is None
            else f"its choice was update {choice['update']} at {100 * choice['margin']:.0f} points"
        )
        print(f"{name} by its rule on the held-out games, at every update and margin ({chosen}):")
        print("what it gains a game, and in brackets the first choices it changes a game.")
        print(
            f"{'update':>7}" + "".join(f"{f'{100 * margin:.0f} points':>22}" for margin in margins)
        )
        gains: dict[str, Any] = {}
        for update in sorted(found):
            cells = []
            for margin in margins:
                reading = led(found[update], margin)
                gain = points(reading["gain_per_game"], reading["gain_per_game_error"])
                cells.append(f"{gain} ({reading['changed'] / games:.2f})")
                gains.setdefault(str(update), {})[str(margin)] = {
                    "gain_per_game": reading["gain_per_game"],
                    "gain_per_game_error": reading["gain_per_game_error"],
                    "changed": reading["changed"],
                    "split": reading.get("split"),
                }
            print(f"{update:>7}" + "".join(f"{cell:>22}" for cell in cells))
        report["rule"][name] = gains
        if choice is None or choice["update"] not in found:
            continue
        print(
            f"{name} at update {choice['update']}, its changes apart by where they start: "
            "how many, how many"
        )
        print("of them are ahead of, level with and behind the first choice they replace, and what")
        print("they gain a game.")
        starts = (
            ("from_pilot", "from the pilot's answer"),
            ("to_pilot", "back to the pilot's answer"),
            ("between_others", "between other answers"),
        )
        print(f"{'margin':>7}" + "".join(f"{title:>40}" for _, title in starts))
        for margin in margins:
            reading = led(found[choice["update"]], margin)
            if "split" not in reading:
                raise SystemExit(f"more/{name}.json was measured without the changes apart")
            cells = []
            for key, _ in starts:
                part = reading["split"][key]
                cells.append(
                    f"{part['changed']} ({part['ahead']}/{part['level']}/{part['behind']}) "
                    f"{points(part['gain_per_game'], part['gain_per_game_error'])}"
                )
            print(f"{100 * margin:>7.0f}" + "".join(f"{cell:>40}" for cell in cells))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")


def seed(args: argparse.Namespace) -> None:
    pilot = json.loads(args.pilot.read_text(encoding="utf8"))["records"][0]
    model = json.loads(args.model.read_text(encoding="utf8"))["records"][0]
    measured = Path(model["checkpoint"]).resolve()
    if measured != args.checkpoint.resolve():
        raise SystemExit(f"{args.model} measures {measured}, not {args.checkpoint}")
    reference, rate, games = seed_gap(pilot, model)
    under = reference - rate
    print(
        f"On the suite ({games} games): the pilot wins {percent(reference)} and "
        f"{model['checkpoint_label']} {percent(rate)},"
    )
    print(
        f"{100 * under:.1f} points under. The arms start from it up to "
        f"{100 * SEED_SLACK:.0f} points under."
    )
    if under > SEED_SLACK:
        raise SystemExit("It is first to be continued against these opponents.")


def weight(args: argparse.Namespace) -> None:
    with args.metrics.open(encoding="utf8") as lines:
        records = [json.loads(line) for line in lines if line.strip()]
    found, measured = lead_weight(records)
    print("The gradient of each loss on the shared parameters, before each update:")
    print(f"{'version':>8}{'imitation':>12}{'lead':>12}{'ratio':>9}")
    for version, imitation, lead in measured:
        print(f"{version:>8}{imitation:>12.4f}{lead:>12.4f}{imitation / lead:>9.2f}")
    print(f"weight: {found:.3g}")


def read(args: argparse.Namespace) -> None:
    measured = load(args.directory / "validation")
    if "seed" not in measured:
        raise SystemExit(f"{args.directory}/validation/seed.json is missing")
    seed_step = int(measured["seed"][0]["checkpoint_step"])
    table, choices = validation(measured, seed_step)
    if args.choice:
        args.choice.write_text(json.dumps(choices, indent=2, sort_keys=True), encoding="utf8")
    report: dict[str, Any] = {"validation": table, "choices": choices}
    kept = args.directory / "choice.json"
    read = load(args.directory / "heldout") if (args.directory / "heldout").is_dir() else {}
    if read:
        if "seed" not in read:
            raise SystemExit(f"{args.directory}/heldout/seed.json is missing")
        if not kept.exists():
            raise SystemExit(f"{kept} is missing: the held-out run is read at a choice")
        # The choice the held-out run was read at, which a later measurement
        # of the validation run must not move.
        choices = json.loads(kept.read_text(encoding="utf8"))
        report["choices"] = choices
        report["heldout"] = heldout(read, choices, seed_step)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("seed", help="the starting model against its pilot")
    command.add_argument("pilot", type=Path, help="`evaluate-suite --pilot` on the suite")
    command.add_argument("model", type=Path, help="`evaluate-suite` of the starting model")
    command.add_argument("--checkpoint", type=Path, required=True, help="the model meant")
    command.set_defaults(func=seed)
    command = commands.add_parser("weight", help="the weight of the lead loss")
    command.add_argument("metrics", type=Path, help="the calibration run's metrics.jsonl")
    command.set_defaults(func=weight)
    command = commands.add_parser("read", help="the choice and the readings")
    command.add_argument("directory", type=Path)
    command.add_argument("--choice", type=Path, help="write the choices made on the validation run")
    command.add_argument("--output", type=Path)
    command.set_defaults(func=read)
    command = commands.add_parser("more", help="the held-out run at every update")
    command.add_argument("directory", type=Path)
    command.add_argument("--output", type=Path)
    command.set_defaults(func=more)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
