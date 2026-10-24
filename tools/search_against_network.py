"""The staged search against the network
(experiments/2026-10-23-search-against-network.md).

    .venv/bin/python tools/search_against_network.py play CHECKPOINT --run-json RUN.json \\
        --arm generic --output OUT.json
    .venv/bin/python tools/search_against_network.py report OUT.json

`play` builds the trainer as the run's `run.json` says (as
`tools/update_signal.py` does), the network as the checkpoint has it, and
takes DEALS of the run's own duels against itself AWAY updates on: two of
its decks, in the seats they were dealt. Each deal is played twice, the
network in one seat and the search in the other, then the other way round:
the same seed, the same decks in the same seats. The network gives its first
choice, as a measured suite has it. The search is the policies' staged
search at the settings upstream measured (a search's seat of the pool,
`crates/ygo-env`), its pilot and its model of the other player by ARM:

- `generic`: the generic pilot, the other player taken to be the generic
  pilot, as upstream measured it;
- `specialist`: the deck's own pilot, the other player taken to be the other
  deck's.

A duel the limit stops is a draw.

`report` takes the reading as it was fixed on 2026-10-23, before anything
was built: the network is stronger when it wins at least 52% of the arm's
games, by two standard errors (the deals drawn again, each deal's two games
together); weaker when it wins 48% or less, by two standard errors;
otherwise not told apart.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from update_signal import trainer_for, update_games

# The reading's marks.
STRONGER = 0.52
WEAKER = 0.48
ERRORS = 2.0
# The improved pilot's: the network wins so many points less against the
# search with it, by two standard errors (`compare`).
DROP = 0.02
ARMS = ("generic", "specialist")


def reading(rate: float, error: float) -> str:
    if rate - ERRORS * error >= STRONGER:
        return "the network is stronger"
    if rate + ERRORS * error <= WEAKER:
        return "the network is weaker"
    return "not told apart"


def scored(games: list[dict[str, Any]]) -> tuple[float, float, int]:
    """The network's share of the games it won, its error with each deal's
    games together, and the deals."""
    by_deal: dict[int, list[float]] = {}
    for game in games:
        by_deal.setdefault(int(game["deal"]), []).append(float(game["result"] == "win"))
    shares = np.array([np.mean(won) for won in by_deal.values()])
    if len(shares) < 2:
        return float(shares.mean()) if len(shares) else 0.0, math.inf, len(shares)
    return float(shares.mean()), float(shares.std(ddof=1) / math.sqrt(len(shares))), len(shares)


def search_seat(arm: str, own: str, other: str) -> dict[str, Any]:
    """The seat of the search of ``arm``, playing the deck ``own`` against
    the deck ``other``."""
    if arm == "generic":
        return {"control": "search", "alone": True, "policy": "generic"}
    if arm == "specialist":
        return {"control": "search", "alone": True, "policy": own, "model": other}
    raise SystemExit(f"--arm is one of {', '.join(ARMS)}, not {arm}")


def deals_of(trainer: Any, count: int, away: int, deck: str | None = None) -> list[dict[str, Any]]:
    """``count`` of the run's duels against itself from ``away`` updates on:
    their seed and the decks in their seats; with ``deck``, those it is
    in, in either seat."""
    found: list[dict[str, Any]] = []
    later = away
    while len(found) < count:
        _, games = update_games(trainer, later)
        for game in games:
            if game.get("other") != "self":
                continue
            own, other, seat = (
                str(game["learner"]),
                str(game["opponent"]),
                int(game["learner_player"]),
            )
            decks = [own, other] if seat == 0 else [other, own]
            if deck is None or deck in decks:
                found.append({"seed": int(game["seed"]), "decks": decks})
        later += 1
    return found[:count]


def play(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from ygo_training.pool import DuelPool, _first_choices

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    model = trainer.model
    model.eval()
    deals = deals_of(trainer, args.deals, args.away, args.deck)
    games: list[dict[str, Any]] = []
    setups = []
    for number, deal in enumerate(deals):
        for network in (0, 1):
            seats: list[dict[str, Any]] = [{}, {}]
            seats[network] = {"control": "caller", "alone": True}
            seats[1 - network] = search_seat(
                args.arm, deal["decks"][1 - network], deal["decks"][network]
            )
            setups.append(
                {
                    "seed": deal["seed"],
                    "decks": deal["decks"],
                    "seats": seats,
                    "limit": trainer.config.limit,
                }
            )
            games.append(
                {"deal": number, "seed": deal["seed"], "decks": deal["decks"], "network": network}
            )
    started = time.perf_counter()
    with DuelPool(ROOT, threads=args.threads, slots=args.slots) as pool:
        first = pool.submit(setups, keep=False)
        answers: Any = ()
        ended = 0
        while True:
            round_ = pool.step(answers)
            for over in round_.finished:
                game = games[int(over["duel"]) - first]
                winner = over.get("winner")
                game["result"] = (
                    "draw"
                    if over.get("limit") or winner not in (0, 1)
                    else ("win" if winner == game["network"] else "loss")
                )
                for name in ("winner", "limit", "turns", "decisions", "search"):
                    game[name] = over.get(name)
                ended += 1
                if ended % args.every == 0:
                    won = sum(told.get("result") == "win" for told in games)
                    print(
                        f"{ended} of {len(games)} games, the network won {won}, "
                        f"{time.perf_counter() - started:.0f} s",
                        flush=True,
                    )
            if not len(round_):
                if round_.in_flight:
                    raise SystemExit(f"{round_.in_flight} duels neither ended nor asked")
                break
            rows = np.arange(len(round_))
            answers = _first_choices(model, round_.arrays, rows, "cpu").astype(np.uint32)
    trainer.close()
    return {
        "checkpoint": str(args.checkpoint),
        "measured_from": trainer.measured_from,
        "arm": args.arm,
        "away": args.away,
        "deals": len(deals),
        "deck": args.deck,
        "seconds": time.perf_counter() - started,
        "threads": args.threads,
        "games": games,
    }


def report(found: dict[str, Any]) -> dict[str, Any]:
    games = [game for game in found["games"] if "result" in game]
    rate, error, deals = scored(games)
    result: dict[str, Any] = {
        "arm": found["arm"],
        "games": len(games),
        "deals": deals,
        "rate": rate,
        "error": error,
        "reading": reading(rate, error),
        "wins": sum(game["result"] == "win" for game in games),
        "losses": sum(game["result"] == "loss" for game in games),
        "draws": sum(game["result"] == "draw" for game in games),
        "seconds": found["seconds"],
        "threads": found["threads"],
    }
    for kind, kept in (
        ("mirrors", [game for game in games if game["decks"][0] == game["decks"][1]]),
        ("other pairings", [game for game in games if game["decks"][0] != game["decks"][1]]),
    ):
        if kept:
            share, share_error, count = scored(kept)
            result[kind] = {"games": len(kept), "deals": count, "rate": share, "error": share_error}
    searches = [told for game in games for told in (game.get("search") or [])]
    if searches:
        total = {
            name: float(sum(float(told.get(name, 0) or 0) for told in searches))
            for name in ("asked", "searched", "changed", "seconds")
        }
        result["search"] = {
            "asked_a_game": total["asked"] / len(searches),
            "searched_a_game": total["searched"] / len(searches),
            "changed_a_game": total["changed"] / len(searches),
            "seconds_a_searched_decision": total["seconds"] / max(total["searched"], 1.0),
            "seconds_a_game": total["seconds"] / len(searches),
        }
    return result


def compared(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The network against two searches on the same deals, deal by deal:
    the search with the pilot as pinned (``before``) and with the improved
    one (``after``). The search with it is stronger when the network wins
    at least ``DROP`` less against it by two standard errors, each deal's
    two games together."""

    def shares(found: dict[str, Any]) -> dict[int, float]:
        by_deal: dict[int, list[float]] = {}
        for game in found["games"]:
            if "result" not in game:
                raise SystemExit(f"A game of deal {game['deal']} did not end")
            by_deal.setdefault(int(game["deal"]), []).append(float(game["result"] == "win"))
        return {deal: sum(won) / len(won) for deal, won in by_deal.items()}

    dealt = [
        [(game["seed"], game["decks"]) for game in found["games"]] for found in (before, after)
    ]
    if dealt[0] != dealt[1]:
        raise SystemExit("The two searches were played on other deals")
    ours, theirs = shares(before), shares(after)
    drop = [ours[deal] - theirs[deal] for deal in sorted(ours)]
    mean = sum(drop) / len(drop)
    error = math.sqrt(sum((one - mean) ** 2 for one in drop) / (len(drop) - 1) / len(drop))
    return {
        "deals": len(drop),
        "before": report(before),
        "after": report(after),
        "drop": mean,
        "error": error,
        "reading": (
            "the improved pilot makes the search stronger"
            if mean - ERRORS * error >= DROP
            else "not shown"
        ),
    }


def shown_compared(result: dict[str, Any]) -> str:
    lines = [
        (
            f"Reading: {result['reading']}. Against the search with the improved pilot the "
            f"network wins {100 * result['drop']:.1f} ± {100 * result['error']:.1f} points less "
            f"than against the search with the pilot as pinned, {result['deals']:,} deals paired."
        )
    ]
    for name in ("before", "after"):
        lines.append(f"{name}:")
        lines.extend(f"  {line}" for line in shown(result[name]).splitlines())
    return "\n".join(lines)


def shown(result: dict[str, Any]) -> str:
    lines = [
        (
            f"The {result['arm']} search against the network, {result['games']:,} games of "
            f"{result['deals']:,} deals (each both ways): the network won {result['wins']:,}, "
            f"lost {result['losses']:,}, drew {result['draws']:,}."
        ),
        (
            f"Reading: {result['reading']}. The network wins {result['rate']:.1%} ± "
            f"{result['error']:.1%} of the games."
        ),
    ]
    for kind in ("mirrors", "other pairings"):
        if kind in result:
            told = result[kind]
            lines.append(
                f"  {kind}: {told['rate']:.1%} ± {told['error']:.1%} ({told['games']:,} games)"
            )
    if "search" in result:
        told = result["search"]
        lines.append(
            f"  the search, a game: asked {told['asked_a_game']:.0f} times, searched "
            f"{told['searched_a_game']:.0f}, changed its pilot's answer {told['changed_a_game']:.1f} "
            f"times; {told['seconds_a_searched_decision']:.2f} s a searched decision, "
            f"{told['seconds_a_game']:.0f} s a game"
        )
    lines.append(
        f"  {result['seconds'] / 3600:.2f} h on {result['threads']} threads, "
        f"{result['seconds'] * result['threads'] / max(result['games'], 1):.0f} s of a thread a game"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    playing = commands.add_parser("play")
    playing.add_argument("checkpoint", type=Path)
    playing.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
    playing.add_argument("--arm", choices=ARMS, required=True)
    playing.add_argument("--output", type=Path, required=True)
    playing.add_argument("--deals", type=int, default=1040)
    playing.add_argument("--away", type=int, default=30_000, help="updates away for the deals")
    playing.add_argument("--deck", help="only the deals this deck is in, in either seat")
    playing.add_argument("--every", type=int, default=50, help="games between reports")
    playing.add_argument("--learners", default=None, help="comma-separated; the run's by default")
    playing.add_argument("--opponents", default=None, help="comma-separated; the run's by default")
    playing.add_argument("--threads", type=int, default=22, help="the pool's threads")
    playing.add_argument("--slots", type=int, default=24)
    playing.add_argument("--torch-threads", type=int, default=2)
    playing.add_argument(
        "--scratch-run",
        type=Path,
        default=ROOT / "runs" / "search-against-network",
        help="the run the trainer is told of; nothing is written there",
    )
    reported = commands.add_parser("report")
    reported.add_argument("found", type=Path)
    comparing = commands.add_parser("compare")
    comparing.add_argument("before", type=Path, help="the search with the pilot as pinned")
    comparing.add_argument("after", type=Path, help="the search with the improved pilot")
    args = parser.parse_args()
    if args.command == "report":
        print(shown(report(json.loads(args.found.read_text()))))
        return
    if args.command == "compare":
        found = [json.loads(path.read_text()) for path in (args.before, args.after)]
        print(shown_compared(compared(*found)))
        return
    found = play(args)
    args.output.write_text(json.dumps(found))
    print(shown(report(found)))


if __name__ == "__main__":
    main()
