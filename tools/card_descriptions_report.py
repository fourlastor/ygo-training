"""The readings of card descriptions and hidden identities
(experiments/2026-10-17-card-descriptions.md).

    uv run python tools/card_descriptions_report.py analysis/card-descriptions \\
        --output analysis/card-descriptions/report.json

The directory is what `PARTS=measure` of
`experiments/commands/card-descriptions.sh` leaves: each arm (both,
descriptions, hidden) and the control (half of the experiment on thirteen
decks) by its checkpoints after the last three `--step` of `--updates`
updates of PPO, on the screen of each deck of training
(`screens/RUN.DECK.json`) and on the decks outside training
(`unseen/RUN.json`); each one's copy at its last checkpoint and the decks'
own pilots on those decks too (`unseen/RUN-copy.json`, `unseen/pilot.json`);
and each arm's own figures (`RUN.metrics.jsonl`, the control's over as many
updates as the arms').

A game is read by the share of a run's three checkpoints that win it, a won
game a win (a drawn one is not).

The readings, fixed on 2026-10-17 before anything was built, each by an
arm's checkpoints after 1,680, 1,800 and 1,920 updates of PPO against the
control's same three, game by game:

1. The decks outside training: both (descriptions, and identities hidden)
   against the control: better at 2 points and two standard errors more,
   worse at as many fewer, otherwise not told apart.
2. The decks of training kept: both against the control over the thirteen
   screens: equivalent when the gap less two standard errors is above -2
   points, worse when the gap and two standard errors are below -2 points,
   otherwise not decided.
3. Which part carries it: descriptions against the control, and hidden
   against the control, on the decks outside training, by the rule of 1.

Told beside them: the copies on the decks outside training; each such deck
by what it shares with the decks of training (its cards that are theirs,
those of the same effect, how near on average its cards come to one of
theirs, by the descriptions of `data/card-descriptions.npz` as they were
first measured) and the share of its pilot's wins each network keeps; both
against the control on the nine decks least covered and the nine most;
each network against those decks' own pilots; and the runs' own figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_head_report import difference
from search_leads_report import percent, points
from thirteen_decks_report import DECKS, MARK, Games, figures, mean, points_of, verdict

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("both", "descriptions", "hidden")
CONTROL = "control"
PILOT = "pilot"
COPY = "copy"
TABLE = ROOT / "data" / "card-descriptions.npz"
# Two cards are of the same effect from this closeness of their
# descriptions: where cards of the same effect sat when the descriptions
# were first measured (Solemn Judgment and Solemn Warning, 0.90).
SAME_EFFECT = 0.8
# The decks outside training are split by how near their cards come.
HALVES = 9
NAMES = {
    "both": "both: descriptions, identities hidden",
    "descriptions": "descriptions",
    "hidden": "identities hidden",
    CONTROL: "the control, half",
    PILOT: "the decks' own pilots",
}


def kept_rule(gap: float, error: float) -> str:
    """Reading 2: equivalent, worse, or neither."""
    if gap - 2.0 * error > -MARK:
        return "equivalent"
    if gap + 2.0 * error < -MARK:
        return "worse"
    return "not decided"


def described(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The table's codes, the row of each, and its descriptions as the check
    first measured them, of length one (the rows less the empty first)."""
    with np.load(path) as found:
        vectors = found["descriptions"][1:].astype(np.float64) * found["spread"] + found["mean"]
        codes, rows = found["codes"], found["rows"]
    return codes, rows, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def coverage(
    decks: dict[str, list[int]],
    training: list[str],
    codes: np.ndarray,
    rows: np.ndarray,
    unit: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Each deck outside training by what it shares with the decks of
    training, its cards counted with their copies (an alternate artwork as
    its original, as `decks` gives them), as they come in its games: its
    cards that are theirs; those, and those of its others whose description
    is at SAME_EFFECT or nearer to one of theirs; and how near on average
    its cards come to one of theirs, 1 for one of theirs, a card without a
    text (a Normal Monster) not counted there."""
    trained = set().union(*(set(decks[name]) for name in training))

    def row_of(code: int) -> int:
        place = int(np.searchsorted(codes, code))
        return int(rows[place]) if place < len(codes) and codes[place] == code else 0

    theirs = unit[sorted({row_of(code) - 1 for code in trained} - {-1})]
    found = {}
    for name, cards in decks.items():
        if name in training:
            continue
        same = effect = 0
        near = []
        for code in cards:
            if code in trained:
                same += 1
                effect += 1
                near.append(1.0)
                continue
            row = row_of(code)
            if not row:
                continue
            closest = float((theirs @ unit[row - 1]).max())
            near.append(closest)
            effect += closest >= SAME_EFFECT
        found[name] = {
            "cards": len(cards),
            "same_cards": same / len(cards),
            "same_effect": effect / len(cards),
            "nearest": mean(near),
        }
    return found


def deck_cards(root: Path) -> dict[str, list[int]]:
    """Every deck's cards with their copies, an alternate artwork as its
    original."""
    import card_embedding_check as texts

    from ygo_training.environment import Environment

    cards = texts.database_cards()

    def original(code: int) -> int:
        _, alias, _, name, _ = cards[code]
        return alias if alias in cards and cards[alias][3] == name else code

    with Environment(root) as environment:
        return {
            name: [original(code) for code in [*deck["main"], *deck["extra"]]]
            for name, deck in environment.decks.items()
        }


def played_by(games: Games) -> list[str]:
    """The deck each game of the decks outside training is played with, in
    the order of the games."""
    assert games.deals is not None
    return [case.split("-vs-")[0] for case, count in games.deals[1] for _ in range(count)]


def report(
    directory: Path,
    decks: list[str],
    cards: dict[str, list[int]] | None = None,
    table: Path = TABLE,
    copy: int = 240,
    updates: int = 1920,
    step: int = 120,
) -> dict[str, Any]:
    copied = len(decks) * copy
    end = points_of(updates, step)
    runs = (*ARMS, CONTROL)
    result: dict[str, Any] = {"decks": decks, "end": end, "readings": {}}

    def steps(prefix: str) -> dict[str, int]:
        return {f"{prefix}@{point}": copied + point for point in end}

    # The screens of the decks of training.
    screens: dict[str, Games] = {}
    for deck in decks:
        found = Games(f"the screen of {deck}")
        for run in runs:
            found.add(directory / "screens" / f"{run}.{deck}.json", steps(run))
        screens[deck] = found

    def together(run: str) -> list[float] | None:
        games: list[float] = []
        for found in screens.values():
            told = found.three(run, end)
            if told is None:
                return None
            games.extend(told)
        return games

    on_screens = {run: together(run) for run in runs}
    result["screens"] = {run: mean(games) for run, games in on_screens.items() if games}
    result["by_screen"] = {
        deck: {run: mean(told) for run in runs if (told := found.three(run, end))}
        for deck, found in screens.items()
    }

    # The decks outside training.
    unseen = Games("the decks outside training")
    for run in runs:
        unseen.add(directory / "unseen" / f"{run}.json", steps(run))
        unseen.add(directory / "unseen" / f"{run}-copy.json", {f"{run}-copy": copied})
    unseen.add(directory / "unseen" / f"{PILOT}.json", None)
    outside = {run: unseen.three(run, end) for run in runs}
    outside |= {f"{run}-copy": unseen.games.get(f"{run}-copy") for run in runs}
    outside[PILOT] = unseen.games.get(PILOT)
    result["unseen"] = {name: mean(games) for name, games in outside.items() if games}

    readings = result["readings"]
    control = outside[CONTROL]
    for arm, key in (("both", "outside"), ("descriptions", "descriptions"), ("hidden", "hidden")):
        if outside[arm] is not None and control is not None:
            gap, error = difference(outside[arm], control)
            readings[key] = {
                "gap": gap,
                "error": error,
                "games": len(control),
                "verdict": verdict(gap, error, MARK, "better", "worse"),
            }
    if on_screens["both"] is not None and on_screens[CONTROL] is not None:
        gap, error = difference(on_screens["both"], on_screens[CONTROL])
        readings["kept"] = {
            "gap": gap,
            "error": error,
            "games": len(on_screens[CONTROL]),
            "verdict": kept_rule(gap, error),
        }

    # Beside the readings: each deck outside training, what it shares with
    # the decks of training and what each network keeps of its pilot's wins.
    if unseen.deals is None:
        return result
    playing = played_by(unseen)
    names = sorted(set(playing))
    by_deck: dict[str, dict[str, Any]] = {}
    for name in names:
        places = [index for index, deck in enumerate(playing) if deck == name]
        row: dict[str, Any] = {}
        for run, games in outside.items():
            if games is not None:
                row[run] = mean([games[index] for index in places])
        pilot = row.get(PILOT)
        if pilot:
            row["kept"] = {
                run: row[run] / pilot for run in (*runs, *map("{}-copy".format, runs)) if run in row
            }
        by_deck[name] = row
    if cards is not None:
        codes, rows, unit = described(table)
        shared = coverage(
            {name: cards[name] for name in [*decks, *names]}, decks, codes, rows, unit
        )
        for name in names:
            by_deck[name]["shares"] = shared[name]
        ordered = sorted(names, key=lambda name: shared[name]["nearest"])
        for half, chosen in (("least", ordered[:HALVES]), ("most", ordered[-HALVES:])):
            places = [index for index, deck in enumerate(playing) if deck in chosen]
            if outside["both"] is not None and control is not None:
                gap, error = difference(
                    [outside["both"][index] for index in places],
                    [control[index] for index in places],
                )
                result.setdefault("covered", {})[half] = {
                    "decks": chosen,
                    "gap": gap,
                    "error": error,
                    "games": len(places),
                }
    result["by_deck"] = by_deck
    if outside[PILOT] is not None:
        for run in runs:
            if outside[run] is not None:
                gap, error = difference(outside[run], outside[PILOT])
                result.setdefault("from_pilots", {})[run] = {"gap": gap, "error": error}
    for run in runs:
        found = figures(directory / f"{run}.metrics.jsonl", step)
        if found is not None:
            result.setdefault("figures", {})[run] = found
    return result


def show(result: dict[str, Any]) -> str:
    end = ", ".join(map(str, result["end"]))
    lines = [
        "The decks outside training, each against the pilots of the decks of training: each",
        f"network by its checkpoints after {end} updates of PPO, its copy at its last",
        "checkpoint, and the decks' own pilots.",
    ]
    unseen = result["unseen"]
    for run in (*ARMS, CONTROL):
        if run in unseen:
            copy = unseen.get(f"{run}-copy")
            lines.append(
                f"  {NAMES[run]:<40}{percent(unseen[run]):>8}"
                + ("" if copy is None else f"   its copy {percent(copy)}")
            )
    if PILOT in unseen:
        lines.append(f"  {NAMES[PILOT]:<40}{percent(unseen[PILOT]):>8}")
    if "from_pilots" in result:
        lines.append("Against those decks' own pilots, the same games:")
        for run, found in result["from_pilots"].items():
            lines.append(f"  {NAMES[run]:<40}{points(found['gap'], found['error']):>14}")
    if result.get("by_deck"):
        lines += [
            "Deck by deck: its cards that the decks of training hold, those or of the same",
            "effect, how near its cards come on average; then each network's wins and the",
            "share of the pilot's wins it keeps (both, descriptions, hidden, the control).",
            f"  {'':<16}{'same':>6}{'effect':>8}{'near':>6}{'pilot':>8}"
            + "".join(f"{name:>14}" for name in ("both", "descriptions", "hidden", "control")),
        ]
        for name, row in result["by_deck"].items():
            shares = row.get("shares", {})
            kept = row.get("kept", {})
            lines.append(
                f"  {name:<16}"
                + (
                    f"{percent(shares['same_cards']):>6}{percent(shares['same_effect']):>8}"
                    f"{shares['nearest']:>6.2f}"
                    if shares
                    else f"{'':>20}"
                )
                + f"{percent(row.get(PILOT)):>8}"
                + "".join(
                    f"{percent(row.get(run)):>7}{percent(kept.get(run)):>7}"
                    for run in (*ARMS, CONTROL)
                )
            )
    if "covered" in result:
        lines.append(
            "Both against the control on the decks outside training, by how near they come:"
        )
        for half, found in result["covered"].items():
            lines.append(
                f"  the {len(found['decks'])} {half} covered: {points(found['gap'], found['error'])} "
                f"over {found['games']:,} games ({', '.join(found['decks'])})"
            )
    if result.get("screens"):
        lines.append(f"The {len(result['decks'])} screens of the decks of training together:")
        for run, rate in result["screens"].items():
            lines.append(f"  {NAMES[run]:<40}{percent(rate):>8}")
    if "figures" in result:
        lines.append("The runs' own figures over their last updates: an update, duels an hour.")
        for run, found in result["figures"].items():
            lines.append(
                f"  {NAMES[run]:<40}{found['update_seconds']:>6.1f} s"
                f"{found['duels_an_hour']:>10,.0f}"
            )
    readings = result["readings"]
    lines.append("The readings:")
    labels = (
        ("outside", "1. both against the control on the decks outside training"),
        ("kept", "2. both against the control on the screens of the decks of training"),
        ("descriptions", "3. descriptions against the control on the decks outside training"),
        ("hidden", "   hidden against the control on the decks outside training"),
    )
    for key, label in labels:
        if key in readings:
            found = readings[key]
            lines.append(
                f"  {label}: {points(found['gap'], found['error'])} over {found['games']:,} games: "
                f"{found['verdict']}"
            )
        else:
            lines.append(f"  {label}: not measured")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decks", nargs="+", default=list(DECKS))
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument(
        "--no-cards",
        action="store_true",
        help="leave out what the decks share, which opens the duel environment for their lists",
    )
    parser.add_argument("--copy", type=int, default=240, help="a copy's updates for each deck")
    parser.add_argument("--updates", type=int, default=1920, help="an arm's updates of PPO")
    parser.add_argument("--step", type=int, default=120, help="the updates between checkpoints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cards = None if args.no_cards else deck_cards(ROOT)
    result = report(
        args.directory, args.decks, cards, args.table, args.copy, args.updates, args.step
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
