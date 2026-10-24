"""Pairing at the decision: the second measurement of
experiments/2026-10-23-credit-offline.md.

    .venv/bin/python tools/decision_pairing.py play CHECKPOINT --run-json RUN.json --output OUT.json
    .venv/bin/python tools/decision_pairing.py report OUT.json

`play` builds the trainer as the run's `run.json` says (as
`tools/update_signal.py` does), the policy as the checkpoint has it and
never changed, and plays DUELS of the run's games AWAY updates on, the
network drawing its answers as the run's own updates draw them
(`inference._sample_with_seeds`, a draw seeded by the duel and the
decision's place in it). In each duel one decision is drawn: the first
decision of a transaction of a seat the network plays. Its answer taken is
*a*, and *b* another, drawn from the policy's chances without *a*. The duel
is then played again from its start PAIRS times answering *a* there and
PAIRS times *b*: up to that decision with the draws it was played with, so
that it comes to the same position (the seat's view there is checked
against the first play's; a play that comes elsewhere is counted and left
out), and on from it with the draws of its pair: the k-th play of *a* and
the k-th of *b* draw alike, the engine's random stream goes on from the
same point in both, and the pilots' with it, so the two share their future
as far as their answers let them. Of each play it keeps the result for the
seat that decided (+1, -1, or 0 for a draw or a duel the limit stops), the
value head's estimate at that seat's next decision (its result where the
duel ends first), and how many of the network's decisions came after.

`report` takes the readings as they were fixed on 2026-10-23, before
anything was built:

1. A shared future cancels the luck: pairing pays when the paired
   difference of the two answers' results has at most half the variance of
   the same difference with independent futures, by two standard errors;
   it does not pay at a quarter or less cancelled, by two standard errors;
   otherwise it is not told apart.
2. A paired play judges better than the head: one paired play's difference
   strays less from what the two answers are worth (the mean of the pairs,
   its own noise allowed for) than the head's difference does (its estimate
   at the seat's next decision after a, less the same after b), by two
   standard errors.

The errors by drawing the decisions again (`update_signal.drawn_again`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from update_signal import drawn_again, trainer_for, update_games

# The readings' marks.
PAYS = 0.5
DOES_NOT = 0.25
ERRORS = 2.0
# The bands of the chance of the answer taken, told beside the readings.
BANDS = ((0.0, 0.9), (0.9, 0.99), (0.99, 1.01))


# The measures, on numbers alone: ``a`` and ``b`` hold each decision's
# results in a row, the k-th of each a pair; ``head`` the head's difference
# of each pair.


def within(values: np.ndarray) -> np.ndarray:
    """Each row's variance, about its own mean."""
    return values.var(axis=1, ddof=1)


def cancelled(a: np.ndarray, b: np.ndarray) -> float:
    """Of the variance of the answers' difference with independent futures,
    the share a shared future cancels."""
    apart = float((within(a) + within(b)).mean())
    if apart <= 0.0:
        return 0.0
    return 1.0 - float(within(a - b).mean()) / apart


def play_noise(a: np.ndarray, b: np.ndarray) -> float:
    """How far one pair's difference strays from what the answers are worth:
    its variance about the decision's mean."""
    return float(within(a - b).mean())


def head_error(a: np.ndarray, b: np.ndarray, head: np.ndarray) -> float:
    """How far the head's difference strays from what the answers are worth:
    its squared distance from the mean of the other pairs, less what that
    mean strays by itself (the other pairs' noise)."""
    difference = a - b
    pairs = difference.shape[1]
    others = (difference.sum(axis=1, keepdims=True) - difference) / (pairs - 1)
    return float(((head - others) ** 2).mean() - within(difference).mean() / (pairs - 1))


def judged(a: np.ndarray, b: np.ndarray, head: np.ndarray) -> float:
    """What the head strays by beyond one paired play: over zero where the
    play judges the two answers better."""
    return head_error(a, b, head) - play_noise(a, b)


def worth_spread(a: np.ndarray, b: np.ndarray) -> float:
    """How much what the two answers are worth differs from one decision to
    another: the variance of the pairs' means less their own noise."""
    difference = a - b
    return float(
        difference.mean(axis=1).var(ddof=1) - within(difference).mean() / difference.shape[1]
    )


def head_correlation(a: np.ndarray, b: np.ndarray, head: np.ndarray) -> float:
    """How closely the head's difference follows what the two answers are
    worth: its covariance with the other pairs' differences, which share no
    play with it, over both spreads."""
    difference = a - b
    pairs = difference.shape[1]
    others = (difference.sum(axis=1, keepdims=True) - difference) / (pairs - 1)
    covariance = float(((head - head.mean()) * (others - others.mean())).mean())
    spread = worth_spread(a, b) * float(head.var())
    return covariance / float(np.sqrt(spread)) if spread > 0.0 else 0.0


# The readings.


def pairing_reading(share: float, error: float) -> str:
    if share - ERRORS * error >= PAYS:
        return "pairing pays"
    if share + ERRORS * error <= DOES_NOT:
        return "pairing does not pay"
    return "not told apart"


def judging_reading(gap: float, error: float) -> str:
    if gap - ERRORS * error > 0.0:
        return "a paired play judges better"
    return "not shown"


# The plays.


def digest(view: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(view, sort_keys=True).encode()).hexdigest()


def result_for(over: dict[str, Any], seat: int) -> float:
    """A duel's result for ``seat``: +1, -1, or 0 for a draw or a duel the
    limit stops."""
    winner = over.get("winner")
    if over.get("limit") or winner not in (0, 1):
        return 0.0
    return 1.0 if winner == seat else -1.0


def played(
    trainer: Any, plays: list[dict[str, Any]], threads: int, slots: int
) -> list[dict[str, Any]]:
    """``plays`` played to their end, each a game (`game`, `rollout_seed`)
    and, where it is played on from a decision, that decision's place among
    the network's (`branch`), the answer given there (`forced`), the view it
    must come to (`expected`), the seat (`seat`) and the seed of the draws
    after it (`pair_seed`). What each found: for a first play, its decisions
    that begin a transaction of the network's seats; for a play from a
    decision, whether it came there, the head's estimate at the seat's next
    decision, and the network's decisions after it; for both, the duel's end."""
    import torch

    from ygo_training.inference import _sample_with_seeds
    from ygo_training.pool import DuelPool, _setups
    from ygo_training.schedules import duel_sample_seed

    config = trainer.config
    model = trainer.model
    model.eval()
    setups = [
        _setups(config.learner_deck, [play["game"]], {"control": "caller"}, config.limit)[0]
        for play in plays
    ]
    found: list[dict[str, Any]] = [
        {"decisions": 0, "starts": [], "keys": set(), "came": None, "after": None, "later": 0}
        for _ in plays
    ]
    with DuelPool(ROOT, threads=threads, slots=slots) as pool:
        first = pool.submit(setups, keep=False, show=True)
        answers: Any = ()
        while True:
            round_ = pool.step(answers)
            for over in round_.finished:
                found[int(over["duel"]) - first]["over"] = {
                    name: over.get(name) for name in ("winner", "limit", "decisions", "turns")
                }
            if not len(round_):
                if round_.in_flight:
                    raise SystemExit(f"{round_.in_flight} duels neither ended nor asked")
                break
            batch = {name: torch.from_numpy(array) for name, array in round_.arrays.items()}
            with torch.no_grad():
                logits, values = model(batch)
            places = (round_.duel - first).tolist()
            seeds = []
            for place in places:
                play, told = plays[place], found[place]
                step = told["decisions"]
                drawn_with = play["rollout_seed"]
                if "branch" in play and step > play["branch"]:
                    drawn_with = play["pair_seed"]
                seeds.append(duel_sample_seed(drawn_with, int(play["game"]["episode_index"]), step))
            temperature = config.rollout_temperature
            chosen = _sample_with_seeds(logits, seeds, temperature).tolist()
            chances = torch.softmax(logits / temperature, dim=-1).numpy()
            for row, place in enumerate(places):
                play, told = plays[place], found[place]
                step = told["decisions"]
                player = int(round_.player[row])
                key = (player, int(round_.transaction[row]))
                begins = key not in told["keys"]
                told["keys"].add(key)
                if "branch" not in play:
                    if begins:
                        count = int((chances[row] > 0).sum())
                        told["starts"].append(
                            {
                                "step": step,
                                "player": player,
                                "chosen": int(chosen[row]),
                                "chances": chances[row][:count].astype(float).tolist(),
                                "value": float(values[row]),
                                "view": digest(round_.views[row]),
                            }
                        )
                elif step == play["branch"]:
                    told["came"] = digest(round_.views[row]) == play["expected"]
                    chosen[row] = play["forced"]
                elif step > play["branch"]:
                    told["later"] += 1
                    if told["after"] is None and player == play["seat"]:
                        told["after"] = float(values[row])
                told["decisions"] = step + 1
            answers = np.array(chosen, dtype=np.uint32)
    for told in found:
        del told["keys"]
    return found


def play(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    schedule = trainer.coordinator.schedule
    games: list[dict[str, Any]] = []
    later = args.away
    while len(games) < args.duels:
        seed, dealt = update_games(trainer, later)
        games.extend({"game": game, "rollout_seed": seed} for game in dealt)
        later += 1
    games = games[: args.duels]
    started = time.perf_counter()
    first = played(trainer, games, args.threads, args.slots)
    print(f"{len(games)} duels played, {time.perf_counter() - started:.0f} s", flush=True)
    generator = np.random.default_rng(args.seed)
    pair_seeds = [
        schedule.rollout_seed(trainer.measured_from + args.away_pairs + pair)
        for pair in range(args.pairs)
    ]
    decisions: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    for place, (dealt, told) in enumerate(zip(games, first, strict=True)):
        if not told["starts"]:
            continue
        start = told["starts"][int(generator.integers(len(told["starts"])))]
        chances = np.array(start["chances"])
        if len(chances) < 2:
            continue
        a = start["chosen"]
        others = chances.copy()
        others[a] = 0.0
        b = int(generator.choice(len(chances), p=others / others.sum()))
        game = dealt["game"]
        seat = start["player"]
        decisions.append(
            {
                "duel": place,
                "against": "itself" if game.get("other") == "self" else "the pilots",
                "seat": seat,
                "step": start["step"],
                "answers": len(chances),
                "a": a,
                "b": b,
                "chance_a": float(chances[a]),
                "chance_b": float(chances[b]),
                "value": start["value"],
                "result": result_for(told["over"], seat),
                "decisions": told["decisions"],
            }
        )
        for pair in range(args.pairs):
            for answer in (a, b):
                branches.append(
                    {
                        **dealt,
                        "branch": start["step"],
                        "forced": answer,
                        "expected": start["view"],
                        "seat": seat,
                        "pair_seed": pair_seeds[pair],
                    }
                )
    print(f"{len(decisions)} decisions, {len(branches)} plays from them", flush=True)
    found_again = played(trainer, branches, args.threads, args.slots)
    trainer.close()
    print(f"played on: {time.perf_counter() - started:.0f} s", flush=True)
    for number, decision in enumerate(decisions):
        plays = found_again[number * 2 * args.pairs : (number + 1) * 2 * args.pairs]
        decision["came"] = [bool(told["came"]) for told in plays]
        decision["results"] = [result_for(told["over"], decision["seat"]) for told in plays]
        decision["after"] = [
            result_for(told["over"], decision["seat"]) if told["after"] is None else told["after"]
            for told in plays
        ]
        decision["later"] = [told["later"] for told in plays]
    return {
        "checkpoint": str(args.checkpoint),
        "measured_from": trainer.measured_from,
        "away": args.away,
        "pairs": args.pairs,
        "duels": len(games),
        "seconds": time.perf_counter() - started,
        # Each decision's plays in pairs: a, b, a, b, ...
        "decisions": decisions,
    }


# The report.


def arrays(decisions: list[dict[str, Any]], pairs: int) -> dict[str, np.ndarray]:
    """Of the decisions whose plays all came to them: the results of a's and
    b's plays, the head's difference of each pair, and the rest."""
    kept = [decision for decision in decisions if all(decision["came"])]
    results = np.array([decision["results"] for decision in kept], dtype=float).reshape(
        -1, pairs, 2
    )
    after = np.array([decision["after"] for decision in kept], dtype=float).reshape(-1, pairs, 2)
    later = np.array([decision["later"] for decision in kept], dtype=float)
    return {
        "a": results[:, :, 0],
        "b": results[:, :, 1],
        "head": after[:, :, 0] - after[:, :, 1],
        "chance": np.array([decision["chance_a"] for decision in kept]),
        "itself": np.array([decision["against"] == "itself" for decision in kept]),
        "later": later.mean(axis=1),
        "share_later": later.mean(axis=1) / np.array([decision["decisions"] for decision in kept]),
    }


def measured(found: dict[str, np.ndarray], chosen: np.ndarray, draws: int) -> dict[str, Any]:
    a, b, head = found["a"][chosen], found["b"][chosen], found["head"][chosen]
    count = len(chosen)

    def picked(measure: Any) -> tuple[float, float]:
        return drawn_again(
            lambda drawn: measure(a[drawn], b[drawn], head[drawn]), count, draws=draws
        )

    share, share_error = picked(lambda x, y, _: cancelled(x, y))
    noise, noise_error = picked(lambda x, y, _: play_noise(x, y))
    head_wrong, head_wrong_error = picked(head_error)
    gap, gap_error = picked(judged)
    follows, follows_error = picked(head_correlation)
    return {
        "decisions": count,
        "cancelled": share,
        "cancelled_error": share_error,
        "apart": float((within(a) + within(b)).mean()),
        "play_noise": noise,
        "play_noise_error": noise_error,
        "head_error": head_wrong,
        "head_error_error": head_wrong_error,
        "gap": gap,
        "gap_error": gap_error,
        "worth": float(np.abs((a - b).mean(axis=1)).mean()),
        "worth_spread": worth_spread(a, b),
        "head_correlation": follows,
        "head_correlation_error": follows_error,
        "later": float(found["later"][chosen].mean()),
        "share_later": float(found["share_later"][chosen].mean()),
    }


def report(found: dict[str, Any], draws: int = 1000) -> dict[str, Any]:
    pairs = int(found["pairs"])
    decisions = found["decisions"]
    told = arrays(decisions, pairs)
    every = np.arange(len(told["a"]))
    result: dict[str, Any] = {
        "decisions": len(decisions),
        "left_out": len(decisions) - len(every),
        "all": measured(told, every, draws),
    }
    result["reading_1"] = pairing_reading(
        result["all"]["cancelled"], result["all"]["cancelled_error"]
    )
    result["reading_2"] = judging_reading(result["all"]["gap"], result["all"]["gap_error"])
    for low, high in BANDS:
        inside = np.flatnonzero((told["chance"] >= low) & (told["chance"] < high))
        if len(inside) > 2:
            result[f"chance {low:g}-{min(high, 1):g}"] = measured(told, inside, draws)
    for kind, inside in (
        ("against itself", told["itself"]),
        ("against the pilots", ~told["itself"]),
    ):
        listed = np.flatnonzero(inside)
        if len(listed) > 2:
            result[kind] = measured(told, listed, draws)
    return result


def shown(result: dict[str, Any]) -> str:
    every = result["all"]
    lines = [
        (
            f"{every['decisions']:,} decisions played on ({result['left_out']} left out, a play "
            "not coming to them)"
        ),
        (
            f"1. A shared future cancels the luck: {result['reading_1']}. Cancelled "
            f"{every['cancelled']:.3f} ± {every['cancelled_error']:.3f} of the variance of the "
            f"answers' difference ({every['apart']:.3f} with independent futures)."
        ),
        (
            f"2. A paired play judges better than the head: {result['reading_2']}. One pair strays "
            f"by {every['play_noise']:.3f} ± {every['play_noise_error']:.3f}, the head by "
            f"{every['head_error']:.3f} ± {every['head_error_error']:.3f}: the head beyond the "
            f"play {every['gap']:+.3f} ± {every['gap_error']:.3f}."
        ),
        "Beside them:",
        (
            f"  every decision: what the answers are worth apart spreads by "
            f"{max(every['worth_spread'], 0.0) ** 0.5:.3f}, the head's difference follows it at "
            f"{every['head_correlation']:.2f} ± {every['head_correlation_error']:.2f}"
        ),
    ]
    for name, told in result.items():
        if not isinstance(told, dict) or name == "all":
            continue
        lines.append(
            f"  {name} ({told['decisions']:,}): cancelled {told['cancelled']:.3f} ± "
            f"{told['cancelled_error']:.3f}; a pair strays {told['play_noise']:.3f}, the head "
            f"{told['head_error']:.3f} (beyond {told['gap']:+.3f} ± {told['gap_error']:.3f}); "
            f"what the answers are worth apart spreads by {max(told['worth_spread'], 0.0) ** 0.5:.3f}, "
            f"the head's difference follows it at {told['head_correlation']:.2f} ± "
            f"{told['head_correlation_error']:.2f}"
        )
    lines.append(
        f"  a decision played on: {every['later']:.0f} of the network's decisions after it a play, "
        f"{every['share_later']:.0%} of its duel's"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    playing = commands.add_parser("play")
    playing.add_argument("checkpoint", type=Path)
    playing.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
    playing.add_argument("--output", type=Path, required=True)
    playing.add_argument("--duels", type=int, default=2000)
    playing.add_argument("--pairs", type=int, default=4)
    playing.add_argument("--away", type=int, default=20_000, help="updates away for the games")
    playing.add_argument(
        "--away-pairs", type=int, default=40_000, help="updates away for the pairs' draws"
    )
    playing.add_argument("--seed", type=int, default=2012, help="of the decisions drawn")
    playing.add_argument("--learners", default=None, help="comma-separated; the run's by default")
    playing.add_argument("--opponents", default=None, help="comma-separated; the run's by default")
    playing.add_argument("--threads", type=int, default=20, help="the pool's threads")
    playing.add_argument("--slots", type=int, default=24)
    playing.add_argument("--torch-threads", type=int, default=8)
    playing.add_argument(
        "--scratch-run",
        type=Path,
        default=ROOT / "runs" / "decision-pairing",
        help="the run the trainer is told of; nothing is written there",
    )
    reported = commands.add_parser("report")
    reported.add_argument("found", type=Path)
    reported.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "report":
        result = report(json.loads(args.found.read_text()), args.draws)
        print(shown(result))
        return
    found = play(args)
    args.output.write_text(json.dumps(found))
    print(shown(report(found)))


if __name__ == "__main__":
    main()
