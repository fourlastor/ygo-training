"""A value head with more results: the first measurement of
experiments/2026-10-23-credit-offline.md.

    .venv/bin/python tools/value_phase.py fit CHECKPOINT --run-json RUN.json --output DIR
    .venv/bin/python tools/value_phase.py report DIR

`fit` builds the trainer as the run's `run.json` says (as
`tools/update_signal.py` does), the policy as the checkpoint has it and never
changed, and plays the games of the run's next UPDATES updates. Two copies of
its network, begun as it is and with its optimizer's state, are taught their
value alone on them:

- the control as the run's PPO taught its head: after each update's games,
  the returns of their transactions from its own estimates (the run's λ),
  every transaction, in the minibatches an update's pass is cut in, one
  pass (`objective`'s value term, as weighted in the run);
- the variant the same, and after every PHASE-th update a value phase as
  Phasic Policy Gradient has one (Cobbe et al. 2020): PASSES passes, in
  minibatches of the run's size, over the last PHASE updates' duels, each
  seat's result at POSITIONS of its transactions a duel, drawn at random
  (the way the head was found to learn best, taught more and it learnt the
  results by heart: experiments/2026-10-08-value-head-and-board.md).

On each update's games, before they are taught, what each explains of their
results is kept beside what the value head explains. Then it plays the
games of HELD updates AWAY from those, which neither is taught, and keeps at
every decision of them each copy's estimate, the value head's (the one the
game was played with), the seat's result, whether it opens a transaction,
and the duel (`held.npz`). The copies are saved in DIR for the second
reading.

`report` takes the first reading as it was fixed on 2026-10-23, before
anything was built: the variant predicts better when it explains at least
0.02 more of a result than the control over every decision of the held
games, by two standard errors, the duels drawn again.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from update_signal import explained, trainer_for, update_games

# The first reading's mark.
BETTER = 0.02
ERRORS = 2.0


def reading(gap: float, error: float) -> str:
    """The first reading: 0.02 more explained, by two errors."""
    return "it predicts better" if gap - ERRORS * error >= BETTER else "not shown to predict better"


def gap(
    duels: np.ndarray,
    results: np.ndarray,
    variant: np.ndarray,
    control: np.ndarray,
    draws: int = 1000,
    seed: int = 2012,
) -> tuple[float, float]:
    """What the variant explains of the results beyond the control, and its
    error, the duels drawn again."""
    names, places = np.unique(duels, return_inverse=True)
    order = np.argsort(places, kind="stable")
    bounds = np.searchsorted(places[order], np.arange(len(names) + 1))
    rows_of = [order[bounds[index] : bounds[index + 1]] for index in range(len(names))]

    def lead(picked: np.ndarray) -> float:
        return explained(results[picked], variant[picked]) - explained(
            results[picked], control[picked]
        )

    whole = lead(np.arange(len(results)))
    generator = np.random.default_rng(seed)
    leads = [
        lead(
            np.concatenate(
                [rows_of[index] for index in generator.integers(0, len(names), len(names))]
            )
        )
        for _ in range(draws)
    ]
    return whole, float(np.std(leads, ddof=1))


# Teaching a copy.


def estimates(optimizer: Any, model: Any, steps: list[dict[str, Any]], chunk: int) -> list[float]:
    """``model``'s value at each of ``steps``, read where they are kept."""
    import torch

    found: list[float] = []
    with torch.no_grad():
        for start in range(0, len(steps), chunk):
            _, values = optimizer.forward_steps(model, steps[start : start + chunk])
            found.extend(values.reshape(-1).tolist())
    return found


def valued(trajectories: list[dict[str, Any]], values: list[float]) -> list[dict[str, Any]]:
    """The trajectories with each step's estimate replaced by ``values``, in
    the order of their steps."""
    given = iter(values)
    return [
        {**trajectory, "steps": [{**step, "value": next(given)} for step in trajectory["steps"]]}
        for trajectory in trajectories
    ]


def taught(model: Any, adam: Any, loss: Any) -> None:
    """One step of ``model``'s optimizer on ``loss``, clipped as the run's."""
    import torch

    adam.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    adam.step()


def pass_as_the_run(
    optimizer: Any,
    model: Any,
    adam: Any,
    trajectories: list[dict[str, Any]],
    chunks: Any,
    chunk: int,
) -> tuple[list[Any], Any, float]:
    """``model`` taught an update's games as the run's PPO taught its head:
    the returns from its own estimates, the value term alone, one pass of
    the update's minibatches (``chunks``, cut once for both copies where
    given as None). Its samples and minibatches, and its mean loss."""
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]
    samples, advantages, _ = optimizer.prepare_samples(
        valued(trajectories, estimates(optimizer, model, steps, chunk))
    )
    if chunks is None:
        chunks = optimizer._chunks(samples, list(range(len(samples))), optimizer.config.minibatch)
    own, losses = optimizer.model, []
    optimizer.model = model
    try:
        for indices in chunks:
            objective = optimizer.objective(samples, advantages, indices)
            loss = objective.components["weighted_value"]
            taught(model, adam, loss)
            losses.append(float(loss.detach()))
    finally:
        optimizer.model = own
    return samples, chunks, float(np.mean(losses)) if losses else 0.0


def phase(
    optimizer: Any,
    model: Any,
    adam: Any,
    buffer: list[tuple[dict[str, Any], float]],
    passes: int,
    generator: np.random.Generator,
) -> float:
    """The variant's value phase: ``passes`` passes over ``buffer`` (steps and
    the results they are taught) in minibatches of the run's size, the value
    term weighted as the run's. Its mean loss."""
    import torch

    from ygo_training.ppo import PpoConfig

    size = optimizer.config.minibatch
    # The value term's weight in the run's objective.
    weight = PpoConfig().value_coefficient
    losses = []
    for _ in range(passes):
        order = generator.permutation(len(buffer))
        for start in range(0, len(order), size):
            picked = [buffer[index] for index in order[start : start + size]]
            _, values = optimizer.forward_steps(model, [step for step, _ in picked])
            targets = torch.tensor([target for _, target in picked], dtype=values.dtype)
            loss = weight * torch.nn.functional.mse_loss(values.reshape(-1), targets)
            taught(model, adam, loss)
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def roots(samples: list[Any]) -> list[Any]:
    """The first sample of each transaction."""
    seen: set[object] = set()
    found = []
    for sample in samples:
        key = sample.step["_transaction_key"]
        if key not in seen:
            seen.add(key)
            found.append(sample)
    return found


def fit(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from ygo_training.pool import sampled_games

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    optimizer = trainer.policy_optimizer
    config = trainer.config
    pool = trainer._pool()
    copies = {}
    for name in ("control", "variant"):
        model, adam = copy.deepcopy((trainer.model, trainer.optimizer))
        for group in adam.param_groups:
            group["lr"] = float(described["learning_rate"])
        copies[name] = (model, adam)
    generator = np.random.default_rng(args.seed)

    def games_of(later: int) -> list[dict[str, Any]]:
        seed, games = update_games(trainer, later)
        trajectories, _ = sampled_games(
            pool,
            trainer.model,
            trainer.device,
            learner=config.learner_deck,
            games=games,
            limit=config.limit,
            rollout_seed=seed,
            temperature=config.rollout_temperature,
        )
        return trajectories

    curve: list[dict[str, Any]] = []
    # The variant's results to come: steps kept in the pool, with the result
    # each is taught.
    buffer: list[tuple[dict[str, Any], float]] = []
    started = time.perf_counter()
    for update in range(args.updates):
        trajectories = games_of(update)
        point: dict[str, Any] = {"update": update}
        chunks = None
        for name, (model, adam) in copies.items():
            samples, chunks, loss = pass_as_the_run(
                optimizer, model, adam, trajectories, chunks, args.chunk
            )
            first = roots(samples)
            results = np.array([sample.terminal_return for sample in first])
            # What it gave before this update's pass: the estimates its
            # returns were made from.
            point[name] = explained(
                results, np.array([float(sample.step["value"]) for sample in first])
            )
            point[f"{name}_loss"] = loss
        played = [step for trajectory in trajectories for step in trajectory["steps"]]
        head_samples, _, _ = optimizer.prepare_samples(trajectories)
        head_first = roots(head_samples)
        point["head"] = explained(
            np.array([sample.terminal_return for sample in head_first]),
            np.array([float(sample.step["value"]) for sample in head_first]),
        )
        # Of each seat's transactions in a duel, POSITIONS drawn for the
        # variant's phase; the update's other decisions let go of.
        by_seat: dict[tuple[int, int], list[Any]] = {}
        for sample in head_first:
            key = sample.step["_transaction_key"]
            by_seat.setdefault((int(key[0]), int(key[1])), []).append(sample)
        kept: set[int] = set()
        for drawn in by_seat.values():
            for index in generator.permutation(len(drawn))[: args.positions]:
                sample = drawn[int(index)]
                buffer.append((sample.step, float(sample.terminal_return)))
                kept.add(int(sample.step["row"]))
        pool.forget_rows([int(step["row"]) for step in played if int(step["row"]) not in kept])
        if (update + 1) % args.phase == 0:
            model, adam = copies["variant"]
            point["phase_loss"] = phase(optimizer, model, adam, buffer, args.passes, generator)
            point["phase_positions"] = len(buffer)
            pool.forget_rows([int(step["row"]) for step, _ in buffer])
            buffer = []
        curve.append(point)
        print(
            f"update {update + 1} of {args.updates}: control {point['control']:+.3f}, variant "
            f"{point['variant']:+.3f}, head {point['head']:+.3f}"
            + (
                f"; a phase of {point['phase_positions']:,} positions"
                if "phase_loss" in point
                else ""
            )
            + f", {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    if buffer:
        pool.forget_rows([int(step["row"]) for step, _ in buffer])
    held: dict[str, list[float]] = {
        name: [] for name in ("duel", "result", "head", "control", "variant", "root")
    }
    for number in range(args.held):
        trajectories = games_of(args.away + number)
        samples, _, _ = optimizer.prepare_samples(trajectories)
        steps = [sample.step for sample in samples]
        seen: set[object] = set()
        for sample in samples:
            key = sample.step["_transaction_key"]
            held["duel"].append(number * 10_000 + int(key[0]))
            held["result"].append(float(sample.terminal_return))
            held["head"].append(float(sample.step["value"]))
            held["root"].append(float(key not in seen))
            seen.add(key)
        for name, (model, _) in copies.items():
            held[name].extend(estimates(optimizer, model, steps, args.chunk))
        pool.forget()
        print(
            f"held update {number + 1} of {args.held}, {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    trainer.close()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "held.npz", **{name: np.array(values) for name, values in held.items()}
    )
    for name, (model, _) in copies.items():
        torch.save(model.state_dict(), args.output / f"{name}.pt")
    found = {
        "checkpoint": str(args.checkpoint),
        "measured_from": trainer.measured_from,
        "updates": args.updates,
        "phase": args.phase,
        "passes": args.passes,
        "positions": args.positions,
        "held": args.held,
        "away": args.away,
        "seconds": time.perf_counter() - started,
        "curve": curve,
    }
    (args.output / "fit.json").write_text(json.dumps(found))
    return found


def report(directory: Path, draws: int = 1000) -> dict[str, Any]:
    found = json.loads((directory / "fit.json").read_text())
    held = dict(np.load(directory / "held.npz"))
    result: dict[str, Any] = {"fit": {name: found[name] for name in found if name != "curve"}}
    for name, rows in (
        ("every decision", np.ones(len(held["result"]), dtype=bool)),
        ("each transaction's first", held["root"] > 0),
    ):
        chosen = np.flatnonzero(rows)
        lead, error = gap(
            held["duel"][chosen],
            held["result"][chosen],
            held["variant"][chosen],
            held["control"][chosen],
            draws,
        )
        result[name] = {
            "decisions": len(chosen),
            "duels": len(np.unique(held["duel"][chosen])),
            "variant": explained(held["result"][chosen], held["variant"][chosen]),
            "control": explained(held["result"][chosen], held["control"][chosen]),
            "head": explained(held["result"][chosen], held["head"][chosen]),
            "gap": lead,
            "error": error,
        }
    every = result["every decision"]
    result["reading_1"] = reading(every["gap"], every["error"])
    result["curve"] = found["curve"]
    return result


def shown(result: dict[str, Any]) -> str:
    fit_ = result["fit"]
    every = result["every decision"]
    lines = [
        (
            f"Of {fit_['checkpoint']}: {fit_['updates']} updates' games taught, a phase every "
            f"{fit_['phase']} ({fit_['passes']} passes, {fit_['positions']} positions of each "
            f"seat a duel); read on {every['duels']:,} held duels."
        ),
        (
            f"1. It predicts better: {result['reading_1']}. Over every decision "
            f"({every['decisions']:,}) the variant explains {every['variant']:.3f} of a result, "
            f"the control {every['control']:.3f}: {every['gap']:+.3f} ± {every['error']:.3f}; "
            f"the head they began as {every['head']:.3f}."
        ),
    ]
    first = result["each transaction's first"]
    lines.append(
        f"  at each transaction's first decision ({first['decisions']:,}): the variant "
        f"{first['variant']:.3f}, the control {first['control']:.3f} ({first['gap']:+.3f} ± "
        f"{first['error']:.3f}), the head {first['head']:.3f}"
    )
    lines.append("  on each update's games before they were taught (variant, control, head):")
    curve = result["curve"]
    shown_at = sorted({*range(0, len(curve), max(1, len(curve) // 12)), len(curve) - 1})
    for point in (curve[index] for index in shown_at):
        lines.append(
            f"    update {point['update'] + 1:3d}: {point['variant']:+.3f}, {point['control']:+.3f}, "
            f"{point['head']:+.3f}"
            + (f" (then a phase, loss {point['phase_loss']:.3f})" if "phase_loss" in point else "")
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fitted = commands.add_parser("fit")
    fitted.add_argument("checkpoint", type=Path)
    fitted.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
    fitted.add_argument("--output", type=Path, required=True)
    fitted.add_argument("--updates", type=int, default=48)
    fitted.add_argument(
        "--phase", type=int, default=16, help="updates between the variant's phases"
    )
    fitted.add_argument("--passes", type=int, default=2)
    fitted.add_argument(
        "--positions", type=int, default=4, help="of each seat's transactions a duel"
    )
    fitted.add_argument("--held", type=int, default=8, help="updates' games read, never taught")
    fitted.add_argument("--away", type=int, default=10_000, help="updates away for the held games")
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
        default=ROOT / "runs" / "value-phase",
        help="the run the trainer is told of; nothing is written there",
    )
    reported = commands.add_parser("report")
    reported.add_argument("directory", type=Path)
    reported.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "report":
        print(shown(report(args.directory, args.draws)))
        return
    fit(args)
    print(shown(report(args.output)))


if __name__ == "__main__":
    main()
