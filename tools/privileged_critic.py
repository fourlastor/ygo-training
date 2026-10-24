"""A critic that sees the hidden cards, fitted alone: the first phase of
experiments/2026-10-22-privileged-critic.md.

    .venv/bin/python tools/privileged_critic.py fit CHECKPOINT --run-json RUN.json --output DIR
    .venv/bin/python tools/privileged_critic.py report DIR/fit.json

`fit` builds the trainer as the run's `run.json` says (as
`tools/update_signal.py` does), the policy as the checkpoint has it and
never changed, and plays the games of the run's next UPDATES updates with
each of the policy's decisions kept with nothing hidden as well
(`DuelPool.batch(whole=True)`).  After each update's games the critic
(`ygo_training/critic.py`), both its readers begun as the policy's network,
is taught the results of POSITIONS of each seat's transactions a duel,
drawn at random, in one pass of minibatches: a duel's result is one
outcome, and its positions tell it again and again (the value head taught
every one of them learnt the results by heart, taught four did not:
experiments/2026-10-08-value-head-and-board.md). On each update's games,
before it is taught them, what it explains of their results is kept beside
what the value head explains: games it has not seen.  Then it plays the
games of HELD updates far from those, which it never trains on, and keeps
of every transaction of them the critic's value, the value head's (the
estimate the game was played with), the seat's result, the turn and the
duel.  The critic is saved in DIR (`critic.pt`) for the second phase.

`report` takes the first reading as it was fixed on 2026-10-22, before
anything was built: the critic sees more when it explains at least 0.10 more
of a result than the value head over every decision of the held games, by
two standard errors (the duels drawn again).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from update_signal import explained, trainer_for, update_games

# The first reading's mark.
SEES_MORE = 0.10
ERRORS = 2.0
# The turns told apart beside it: their first turns.
TURNS = (1, 2, 3, 5, 7, 11)


def roots(
    trainer: Any, trajectories: list[dict[str, Any]]
) -> list[tuple[int, float, float, int, int]]:
    """The transactions an update trains on, by their first step: its row,
    its seat's result, the value head's estimate, its duel and its seat."""
    samples, _, _ = trainer.policy_optimizer.prepare_samples(trajectories)
    seen: set[object] = set()
    found = []
    for sample in samples:
        key = sample.step["_transaction_key"]
        if key in seen:
            continue
        seen.add(key)
        found.append(
            (
                int(sample.step["row"]),
                float(sample.terminal_return),
                float(sample.step["value"]),
                int(key[0]),
                int(sample.step["player"]),
            )
        )
    return found


def drawn(
    found: list[tuple[int, float, float, int, int]], positions: int, generator: random.Random
) -> list:
    """``positions`` of each seat's transactions in a duel, drawn at random."""
    by_seat: dict[tuple[int, int], list] = {}
    for root in found:
        by_seat.setdefault((root[3], root[4]), []).append(root)
    taken = []
    for seat in by_seat.values():
        taken.extend(seat if len(seat) <= positions else generator.sample(seat, positions))
    generator.shuffle(taken)
    return taken


def valued(trainer: Any, critic: Any, found: list, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """The critic's values of ``found`` and their turns, without a gradient."""
    import torch

    from ygo_training.schema import SCALARS

    turn = SCALARS.index("turn")
    values, turns = [], []
    with torch.no_grad():
        for start in range(0, len(found), chunk):
            rows = [root[0] for root in found[start : start + chunk]]
            seat = trainer._kept(rows)
            values.append(critic(seat, trainer._kept_whole(rows)).cpu().numpy())
            turns.append(np.rint(seat["scalars"][:, turn].cpu().numpy() * 100).astype(np.int64))
    return np.concatenate(values), np.concatenate(turns)


def fit(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from ygo_training.critic import SightedCritic

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    critic = SightedCritic.from_policy(trainer.model, warm=args.warm).to(trainer.device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    generator = random.Random(args.seed)
    curve = []
    started = time.perf_counter()
    for later in range(args.updates):
        seed, games = update_games(trainer, later)
        trajectories, _ = trainer._learner_games(
            games, seed, labels=False, whole=True, whole_deck=args.deck
        )
        taught = drawn(roots(trainer, trajectories), args.positions, generator)
        # What it makes of these games before it is taught them.
        critic.eval()
        before, _ = valued(trainer, critic, taught, args.chunk)
        results = np.array([root[1] for root in taught])
        heads = np.array([root[2] for root in taught])
        critic.train()
        losses = []
        for start in range(0, len(taught), args.minibatch):
            part = taught[start : start + args.minibatch]
            rows = [root[0] for root in part]
            value = critic(trainer._kept(rows), trainer._kept_whole(rows))
            target = torch.tensor(
                [root[1] for root in part], dtype=value.dtype, device=value.device
            )
            loss = torch.nn.functional.mse_loss(value, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append(
            {
                "update": later + 1,
                "positions": len(taught),
                "critic_before": explained(results, before),
                "head": explained(results, heads),
                "loss": float(np.mean(losses)),
            }
        )
        print(
            f"update {later + 1} of {args.updates}: {len(taught)} positions, before it is taught "
            f"them the critic explains {curve[-1]['critic_before']:+.3f}, the head "
            f"{curve[-1]['head']:+.3f}; {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "critic": critic.state_dict(),
            "checkpoint": str(args.checkpoint),
            "updates": args.updates,
            "deck": args.deck,
            "warm": args.warm,
        },
        args.output / "critic.pt",
    )
    critic.eval()
    held = []
    for place in range(args.held):
        seed, games = update_games(trainer, args.away + place)
        trajectories, _ = trainer._learner_games(
            games, seed, labels=False, whole=True, whole_deck=args.deck
        )
        found = roots(trainer, trajectories)
        values, turns = valued(trainer, critic, found, args.chunk)
        for root, value, turn in zip(found, values.tolist(), turns.tolist(), strict=True):
            held.append(
                {
                    "duel": place * 100_000 + root[3],
                    "result": root[1],
                    "head": root[2],
                    "critic": value,
                    "turn": turn,
                }
            )
        print(f"held update {place + 1} of {args.held}: {len(found)} transactions", flush=True)
    trainer.close()
    return {
        "checkpoint": str(args.checkpoint),
        "updates": args.updates,
        "positions": args.positions,
        "deck": args.deck,
        "warm": args.warm,
        "learning_rate": args.learning_rate,
        "held_updates": args.held,
        "seconds": time.perf_counter() - started,
        "curve": curve,
        "held": held,
    }


def gap(held: list[dict[str, Any]], draws: int = 1000, seed: int = 2012) -> dict[str, Any]:
    """What the critic and the value head explain of the held games' results
    over all their decisions, the critic's lead and its error, the duels
    drawn again; and both by turn."""
    duels = np.array([row["duel"] for row in held])
    results = np.array([row["result"] for row in held])
    heads = np.array([row["head"] for row in held])
    critics = np.array([row["critic"] for row in held])
    turns = np.array([row["turn"] for row in held])
    names, places = np.unique(duels, return_inverse=True)
    order = np.argsort(places, kind="stable")
    bounds = np.searchsorted(places[order], np.arange(len(names) + 1))
    rows_of = [order[bounds[index] : bounds[index + 1]] for index in range(len(names))]

    def measured(chosen: np.ndarray) -> tuple[float, float]:
        picked = np.concatenate([rows_of[index] for index in chosen])
        return explained(results[picked], critics[picked]), explained(
            results[picked], heads[picked]
        )

    critic, head = measured(np.arange(len(names)))
    generator = np.random.default_rng(seed)
    leads = []
    for _ in range(draws):
        one, other = measured(generator.integers(0, len(names), len(names)))
        leads.append(one - other)
    error = float(np.std(leads, ddof=1))
    by_turn = []
    for first, last in zip(TURNS, (*TURNS[1:], 10_000), strict=True):
        kept = (turns >= first) & (turns < last)
        if kept.sum() > 1:
            by_turn.append(
                {
                    "turns": f"{first}"
                    if last == first + 1
                    else f"{first}-{last - 1}"
                    if last < 10_000
                    else f"{first}+",
                    "transactions": int(kept.sum()),
                    "critic": explained(results[kept], critics[kept]),
                    "head": explained(results[kept], heads[kept]),
                }
            )
    return {
        "duels": len(names),
        "transactions": len(held),
        "critic": critic,
        "head": head,
        "lead": critic - head,
        "error": error,
        "by_turn": by_turn,
    }


def reading(lead: float, error: float) -> str:
    """The first reading: it sees more at 0.10 more explained, by two errors."""
    return "it sees more" if lead - ERRORS * error >= SEES_MORE else "not shown to see 0.10 more"


def report(found: dict[str, Any]) -> str:
    told = gap(found["held"])
    curve = found["curve"]
    lines = [
        (
            f"Fitted on {found['checkpoint']}, seeing {found.get('deck', 0)} of each Deck's next "
            f"cards, its value begun {'as the head' if found.get('warm') else 'anew'}: "
            f"{found['updates']} updates' games, {found['positions']} "
            f"positions of each seat a duel, at {found['learning_rate']}; read on {told['duels']:,} "
            f"held duels ({told['transactions']:,} transactions)."
        ),
        "On each update's games before it was taught them, what each explains of the results:",
    ]
    shown = sorted({*range(0, len(curve), max(1, len(curve) // 10)), len(curve) - 1})
    for point in (curve[index] for index in shown):
        lines.append(
            f"  update {point['update']:3d}: the critic {point['critic_before']:+.3f}, "
            f"the head {point['head']:+.3f} (loss {point['loss']:.3f})"
        )
    lines.append("On the held games, by turn:")
    for part in told["by_turn"]:
        lines.append(
            f"  turns {part['turns']:6} ({part['transactions']:7,} transactions): the critic "
            f"{part['critic']:+.3f}, the head {part['head']:+.3f}"
        )
    lines += [
        (
            f"Over every decision: the critic {told['critic']:+.3f}, the head {told['head']:+.3f}, "
            f"the critic's lead {told['lead']:+.3f} ± {told['error']:.3f}"
        ),
        f"The reading: {reading(told['lead'], told['error'])}",
    ]
    return "\n".join(lines)


def hidden(args: argparse.Namespace) -> str:
    """How much a critic that sees the hidden cards has to see: of the
    decisions of the held games, by turn, the other player's cards the seat
    does not see, in hand and face-down on the field, a decision."""
    import torch

    from ygo_training.schema import SCALARS

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    turn_at = SCALARS.index("turn")
    # The packed locations (`native/ygo-model-input`, `view::location`), and
    # the other player as a decision's own seat sees it.
    hand, monsters, spells, theirs = 2, 3, 4, 1
    counts = {first: np.zeros(6) for first in TURNS}
    for place in range(args.held):
        seed, games = update_games(trainer, args.away + place)
        trajectories, _ = trainer._learner_games(games, seed, labels=False)
        found = roots(trainer, trajectories)
        for start in range(0, len(found), args.chunk):
            seat = trainer._pool().batch([root[0] for root in found[start : start + args.chunk]])
            width = seat["state_card"].shape[1]
            present = np.arange(width)[None, :] < seat["state_lengths"][:, None]
            other = present & (seat["state_controller"] == theirs)
            unknown = other & (seat["state_card"] == 0)
            location = seat["state_location"]
            parts = [
                (other & (location == hand)).sum(1),
                (unknown & (location == hand)).sum(1),
                (unknown & (location == monsters)).sum(1),
                (unknown & (location == spells)).sum(1),
                unknown.any(1),
            ]
            turns = np.rint(seat["scalars"][:, turn_at] * 100).astype(np.int64)
            for first, last in zip(TURNS, (*TURNS[1:], 10_000), strict=True):
                kept = (turns >= first) & (turns < last)
                counts[first] += [kept.sum(), *(part[kept].sum() for part in parts)]
        print(f"held update {place + 1} of {args.held}: {len(found)} transactions", flush=True)
    trainer.close()
    lines = [
        (
            f"The held games of {args.checkpoint} ({args.held} updates, {args.away} away), by "
            "turn: of every decision, the other player's hand, and its cards the seat does not see"
        ),
        "  turns     decisions  their hand  unseen in hand  face-down monsters  face-down backrow  any",
    ]
    for first, last in zip(TURNS, (*TURNS[1:], 10_000), strict=True):
        decisions, size, in_hand, down, backrow, any_unseen = counts[first]
        named = (
            f"{first}"
            if last == first + 1
            else f"{first}-{last - 1}"
            if last < 10_000
            else f"{first}+"
        )
        lines.append(
            f"  {named:6} {int(decisions):12,} {size / decisions:11.2f} {in_hand / decisions:15.2f} "
            f"{down / decisions:19.2f} {backrow / decisions:18.2f} {any_unseen / decisions:5.0%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fitted = commands.add_parser("fit")
    fitted.add_argument("checkpoint", type=Path)
    fitted.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
    fitted.add_argument("--output", type=Path, required=True)
    fitted.add_argument("--updates", type=int, default=40)
    fitted.add_argument(
        "--positions", type=int, default=8, help="of each seat's transactions a duel"
    )
    fitted.add_argument("--held", type=int, default=4, help="updates' games read, never trained on")
    fitted.add_argument("--away", type=int, default=10_000, help="updates away for the held games")
    fitted.add_argument(
        "--deck", type=int, default=0, help="the next cards of each Deck it sees as well"
    )
    fitted.add_argument(
        "--warm", action="store_true", help="its value begun as the policy's value head"
    )
    fitted.add_argument("--learning-rate", type=float, default=3e-4)
    fitted.add_argument("--minibatch", type=int, default=256)
    fitted.add_argument("--chunk", type=int, default=1024, help="decisions valued at a time")
    fitted.add_argument("--seed", type=int, default=2012)
    fitted.add_argument("--learners", default=None, help="comma-separated; the run's by default")
    fitted.add_argument("--opponents", default=None, help="comma-separated; the run's by default")
    fitted.add_argument("--threads", type=int, default=20, help="the pool's threads")
    fitted.add_argument("--slots", type=int, default=16)
    fitted.add_argument("--torch-threads", type=int, default=8)
    fitted.add_argument("--device", default="cpu", help="where the networks run: cpu or cuda")
    fitted.add_argument(
        "--scratch-run",
        type=Path,
        default=ROOT / "runs" / "privileged-critic",
        help="the run the trainer is told of; nothing is written there",
    )
    reported = commands.add_parser("report")
    reported.add_argument("fit", type=Path)
    seen = commands.add_parser("hidden")
    seen.add_argument("checkpoint", type=Path)
    seen.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
    seen.add_argument("--held", type=int, default=4, help="updates' games read")
    seen.add_argument("--away", type=int, default=10_000, help="updates away for the held games")
    seen.add_argument("--chunk", type=int, default=1024, help="decisions read at a time")
    seen.add_argument("--learners", default=None, help="comma-separated; the run's by default")
    seen.add_argument("--opponents", default=None, help="comma-separated; the run's by default")
    seen.add_argument("--threads", type=int, default=20, help="the pool's threads")
    seen.add_argument("--slots", type=int, default=16)
    seen.add_argument("--torch-threads", type=int, default=8)
    seen.add_argument("--device", default="cpu", help="where the networks run: cpu or cuda")
    seen.add_argument(
        "--scratch-run",
        type=Path,
        default=ROOT / "runs" / "privileged-critic",
        help="the run the trainer is told of; nothing is written there",
    )
    args = parser.parse_args()
    if args.command == "report":
        print(report(json.loads(args.fit.read_text())))
        return
    if args.command == "hidden":
        print(hidden(args))
        return
    found = fit(args)
    (args.output / "fit.json").write_text(json.dumps(found))
    print(report(found))


if __name__ == "__main__":
    main()
