"""A deck's games played again with every answer kept, and what became of a card
in them: why PPO takes a deck down where a card pays late
(experiments/2026-10-28-seventy-one-champions.md).

    uv run python tools/deck_replays.py play NETWORK --deck countdown \\
        --against blackwing,gishki --pilots --deals 32 --output countdown.jsonl
    uv run python tools/deck_replays.py play NETWORK --deck deckout --other CHAMPION \\
        --deals 512 --output mirror.jsonl
    uv run python tools/deck_replays.py card countdown.jsonl --cards 95308449
    uv run python tools/deck_replays.py paired forced.jsonl countdown.jsonl
    uv run python tools/deck_replays.py credit countdown.jsonl --cards 95308449

`play` seats NETWORK at the deck against each deck of `--against`, its mirror
without them, on every deal from both seats, each network taking its first
choice. The other seat is `--other`, or NETWORK again where none is named, or
with `--pilots` the pilot of the deck there; NETWORK `pilot` answers as the
pilot of its own deck would. `--force CODE` makes NETWORK play a card wherever
it may, `--as-pilot CODE,...` wherever its pilot would, and `--refuse CODE,...`
answer no wherever a card asks: the same deals without them say what a habit
costs.

`card` tells the wins, how the games ended, and what the network did with the
cards; `paired` the wins of one file less another's, deal by deal, a deal being
both of its seats; `credit` the credit the cards get as the trainer works it, a
step being one of the network's transactions: the mean at the steps they were
played less the mean at the steps they could have been and were not, with the
credit fading by each of `FADES`, and its standard error by resampling games.
Self-play's games are the ones to read for that.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The deals are none of a suite's.
FIRST_SEED = 900000
# What the credit is taken forward by a transaction: the trainer's, then less fading.
FADES = (0.95, 0.98, 0.99, 1.0)
RESAMPLES = 400
REASONS = {1: "life points", 2: "deck-out", 17: "a card's own win"}


def played(
    network: str,
    other: str | None,
    pilots: bool,
    deck: str,
    against: list[str],
    deals: int,
    force: set[int],
    as_pilot: set[int],
    refuse: set[int] = frozenset(),
) -> list[dict[str, Any]]:
    """Every deal of ``deck`` against each of ``against``, from both seats."""
    from ygo_training import checkpoints
    from ygo_training.environment import Environment
    from ygo_training.inference import scored

    environment = Environment(str(ROOT))
    models = {}
    if network != "pilot":
        models["network"] = checkpoints.load_inference_model(Path(network))[0].eval()
    if not pilots:
        models["other"] = (
            models["network"]
            if other is None
            else checkpoints.load_inference_model(Path(other))[0].eval()
        )
    names = {number: name for name, number in environment.vocabulary["choice"].items()}

    def activates(choice: dict[str, Any]) -> int:
        """The card a choice activates, or 0."""
        if names[int(choice["kind"])] != "Activate":
            return 0
        return int((choice.get("card") or {}).get("code") or 0)

    def refuses(choice: dict[str, Any]) -> int:
        """The card a choice says no to, or 0."""
        if names[int(choice["kind"])] != "No":
            return 0
        return int((choice.get("card") or {}).get("code") or 0)

    caller = {"control": "caller"}
    games = []
    for rival in against:
        for deal in range(deals):
            for seat in (0, 1):
                seats = [caller, caller]
                if pilots:
                    seats[1 - seat] = {"control": "pilot"}
                setup = {
                    "seed": FIRST_SEED + deal,
                    "decks": [deck, rival] if seat == 0 else [rival, deck],
                    "seats": seats,
                }
                games.append(
                    {
                        "deal": deal,
                        "seat": seat,
                        "other": rival,
                        "names": ("network", "other") if seat == 0 else ("other", "network"),
                        "episode": environment.start(setup),
                        "moves": [],
                        "over": None,
                        "pending": None,
                    }
                )
    live = list(games)
    while live:
        waiting: dict[str, list[dict[str, Any]]] = {"network": [], "other": []}
        for game in live:
            while game["pending"] is None and game["over"] is None:
                event = game["episode"].advance()
                game["pending"] = event["decision"]
                game["over"] = event["over"]
            if game["pending"] is not None:
                waiting[game["names"][int(game["pending"]["player"])]].append(game)
        for name, rows in waiting.items():
            if not rows:
                continue
            views = [game["pending"] for game in rows]
            if name in models:
                scores, values = scored(models[name], environment, views)
            for row, (game, view) in enumerate(zip(rows, views, strict=True)):
                choices = view["choices"]
                if name in models:
                    answer = int(scores[row][: len(choices)].argmax())
                else:
                    answer = int(view["label"])
                if name == "network":
                    guide = view["pilot"]
                    if guide is not None and activates(choices[int(guide)]) in as_pilot:
                        answer = int(guide)
                    for place, choice in enumerate(choices):
                        if activates(choice) in force or refuses(choice) in refuse:
                            answer = place
                choice = choices[answer]
                game["moves"].append(
                    {
                        "by": name,
                        "turn": int(view["turn"]),
                        "kind": names[int(choice["kind"])],
                        "code": int((choice.get("card") or {}).get("code") or 0),
                        "life": view["life_points"],
                        "value": float(values[row]) if name in models else None,
                        "transaction": int(view["transaction"]),
                        "offered": sorted({activates(one) for one in choices} - {0}),
                    }
                )
                game["episode"].choose(answer)
                game["pending"] = None
        live = [game for game in live if game["over"] is None]
    for game in games:
        game["episode"].close()
    keys = ("deal", "seat", "other", "names", "moves", "over")
    return [{key: game[key] for key in keys} for game in games]


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def won(game: dict[str, Any]) -> bool:
    winner = game["over"]["winner"]
    return winner in (0, 1) and game["names"][winner] == "network"


def mean(rows: list[float]) -> float:
    return sum(rows) / max(len(rows), 1)


def taken(move: dict[str, Any], codes: set[int]) -> bool:
    return move["kind"] == "Activate" and move["code"] in codes


def cards(games: list[dict[str, Any]], codes: set[int]) -> dict[str, Any]:
    """The wins, how the games ended and what the network did with ``codes``."""
    ends: Counter[str] = Counter()
    uses = in_games = passed_over = 0
    for game in games:
        mine = [move for move in game["moves"] if move["by"] == "network"]
        count = sum(taken(move, codes) for move in mine)
        uses += count
        in_games += count > 0
        passed_over += not count and any(codes & set(move["offered"]) for move in mine)
        result = "won" if won(game) else "drawn" if game["over"]["winner"] not in (0, 1) else "lost"
        reason = game["over"]["reason"]
        ends[f"{result} by {REASONS.get(reason, reason)}"] += 1
    return {
        "games": len(games),
        "wins": mean([float(won(game)) for game in games]),
        "turns": mean([float(game["over"]["turns"]) for game in games]),
        "ends": dict(ends.most_common()),
        "uses": uses / max(len(games), 1),
        "in_games": in_games / max(len(games), 1),
        "passed_over": passed_over,
    }


def paired(one: list[dict[str, Any]], two: list[dict[str, Any]]) -> tuple[float, float, int]:
    """The wins of ``one`` less those of ``two`` on the same deals, its standard
    error, and how many deals."""

    def wins(games: list[dict[str, Any]]) -> dict[tuple[str, int, int], float]:
        return {(game["other"], game["deal"], game["seat"]): float(won(game)) for game in games}

    first, second = wins(one), wins(two)
    deals = sorted({key[:2] for key in first})
    gaps = [
        sum(first[(*deal, seat)] - second[(*deal, seat)] for seat in (0, 1)) / 2 for deal in deals
    ]
    centre = mean(gaps)
    spread = sum((gap - centre) ** 2 for gap in gaps) / max(len(gaps) - 1, 1)
    return centre, math.sqrt(spread / max(len(gaps), 1)), len(gaps)


def steps(game: dict[str, Any]) -> list[dict[str, Any]]:
    """The network's transactions: the value at each one's first answer, and
    its answers."""
    found: list[dict[str, Any]] = []
    for move in game["moves"]:
        if move["by"] != "network":
            continue
        if found and found[-1]["transaction"] == move["transaction"]:
            found[-1]["moves"].append(move)
        else:
            found.append(
                {"transaction": move["transaction"], "value": move["value"], "moves": [move]}
            )
    return found


def credits(found: list[dict[str, Any]], result: float, fade: float) -> list[float]:
    """Generalized advantage estimation over the steps of a game."""
    credit = [0.0] * len(found)
    running = 0.0
    following = result
    for place in range(len(found) - 1, -1, -1):
        running = following - found[place]["value"] + fade * running
        credit[place] = running
        following = found[place]["value"]
    return credit


def outcome(game: dict[str, Any]) -> float:
    if game["over"]["winner"] not in (0, 1):
        return 0.0
    return 1.0 if won(game) else -1.0


def edges(
    games: list[dict[str, Any]], codes: set[int]
) -> tuple[dict[float, tuple[float, float]], int, float]:
    """For each of `FADES` the edge of playing ``codes`` over passing them by,
    with its error; how many times they were played; and how many of the
    network's transactions before a game's end, on average."""
    rows = []
    left = []
    for game in games:
        found = steps(game)
        tables = {fade: credits(found, outcome(game), fade) for fade in FADES}
        row = {fade: [[0.0, 0], [0.0, 0]] for fade in FADES}
        for place, step in enumerate(found):
            if not any(codes & set(move["offered"]) for move in step["moves"]):
                continue
            was_taken = any(taken(move, codes) for move in step["moves"])
            if was_taken:
                left.append(float(len(found) - place))
            for fade in FADES:
                row[fade][0 if was_taken else 1][0] += tables[fade][place]
                row[fade][0 if was_taken else 1][1] += 1
        rows.append(row)
    drawn = random.Random(7)
    told = {}
    for fade in FADES:
        found_edges = []
        for _ in range(RESAMPLES):
            sample = [rows[drawn.randrange(len(rows))] for _ in rows]
            played_there = sum(row[fade][0][0] for row in sample)
            passed_there = sum(row[fade][1][0] for row in sample)
            found_edges.append(
                played_there / max(sum(row[fade][0][1] for row in sample), 1)
                - passed_there / max(sum(row[fade][1][1] for row in sample), 1)
            )
        centre = mean(found_edges)
        spread = sum((edge - centre) ** 2 for edge in found_edges) / (RESAMPLES - 1)
        told[fade] = (centre, math.sqrt(spread))
    return told, len(left), mean(left)


def codes_of(text: str) -> set[int]:
    return {int(code) for code in text.split(",") if code}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parts = parser.add_subparsers(dest="part", required=True)
    play = parts.add_parser("play")
    play.add_argument("network")
    play.add_argument("--deck", required=True)
    play.add_argument("--against", default="")
    play.add_argument("--other")
    play.add_argument("--pilots", action="store_true")
    play.add_argument("--deals", type=int, required=True)
    play.add_argument("--force", default="")
    play.add_argument("--as-pilot", default="")
    play.add_argument("--refuse", default="")
    play.add_argument("--output", type=Path, required=True)
    card = parts.add_parser("card")
    card.add_argument("files", type=Path, nargs="+")
    card.add_argument("--cards", required=True)
    pair = parts.add_parser("paired")
    pair.add_argument("one", type=Path)
    pair.add_argument("two", type=Path)
    credit = parts.add_parser("credit")
    credit.add_argument("files", type=Path, nargs="+")
    credit.add_argument("--cards", required=True)
    args = parser.parse_args()
    if args.part == "play":
        games = played(
            args.network,
            args.other,
            args.pilots,
            args.deck,
            args.against.split(",") if args.against else [args.deck],
            args.deals,
            codes_of(args.force),
            codes_of(args.as_pilot),
            codes_of(args.refuse),
        )
        args.output.write_text("".join(json.dumps(game) + "\n" for game in games), encoding="utf8")
    elif args.part == "card":
        for path in args.files:
            told = cards(read(path), codes_of(args.cards))
            print(
                f"{path.name}: {told['games']} games, {100 * told['wins']:.1f}% won in "
                f"{told['turns']:.1f} turns; played in {100 * told['in_games']:.1f}% of them, "
                f"{told['uses']:.2f} a game, offered and never played in {told['passed_over']}; "
                f"{told['ends']}"
            )
    elif args.part == "paired":
        gap, error, deals = paired(read(args.one), read(args.two))
        print(
            f"{args.one.name} less {args.two.name}: "
            f"{100 * gap:+.1f} ± {100 * error:.1f} points on {deals} deals"
        )
    else:
        for path in args.files:
            told, times, before = edges(read(path), codes_of(args.cards))
            print(f"{path.name}: played {times} times, {before:.0f} transactions before the end")
            for fade, (edge, error) in told.items():
                print(f"  the play's edge at {fade}: {edge:+.3f} ± {error:.3f}")


if __name__ == "__main__":
    main()
