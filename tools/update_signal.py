"""How strong the signal of an update is (experiments/2026-10-22-update-signal.md).

Two measurements of a run's checkpoint, with the run's own settings and
games, and no training: the trainer is built as the run's `run.json` says,
on this machine's processor, and nothing is written but OUTPUT.

    .venv/bin/python tools/update_signal.py noise CHECKPOINT --run-json RUN.json --output OUT.json
    .venv/bin/python tools/update_signal.py luck CHECKPOINT --run-json RUN.json --output OUT.json
    .venv/bin/python tools/update_signal.py report NOISE.json LUCK.json

`noise` plays the games of the run's next UPDATES updates, as the run would
have, and takes each update's gradient on groups of its duels (GROUPS an
update, each half against itself where the update is): the policy's part
and the value head's, as the update's own objective has them. Two groups'
gradients share the update's signal and stray from it by their own noise;
the noise scale is how many duels an update would need for its step to be
half signal and half noise (McCandlish et al. 2018, on an empirical model of
large-batch training), with its error by leaving a group out at a time.

`luck` plays DEALS of the games of the run's next update, each PLAYS times
with the network's answers drawn afresh (the same deal, other draws), and
tells how much of a result is the deal's, how much of it the value head
tells at a seat's first decision, and how much of the variance of the
advantages the trainer computes the result of another play of the same deal
accounts for: what training on the deal played twice would cancel. Errors
by drawing the deals again.

`report` takes the readings as fixed on 2026-10-22, before any run.
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
# An update of the runs read: its duels.
UPDATE_DUELS = 520
# The readings' marks (experiments/2026-10-22-update-signal.md).
NOISE_BOUND = 2 * UPDATE_DUELS
NOT_NOISE_BOUND = UPDATE_DUELS // 2
PAIRING_PAYS = 0.25
PAIRING_DOES_NOT = 0.10
ERRORS = 2.0


# The measures, on numbers alone.


def noise_scale(gram: np.ndarray, duels_per_group: float) -> float:
    """The noise scale, in duels, of gradients taken on separate groups of
    ``duels_per_group`` duels each: ``gram`` holds their dot products. What
    two groups share estimates the signal's squared length; what each has
    beyond it, the noise's over a group's duels. Infinite where no signal is
    seen."""
    count = len(gram)
    diagonal = float(np.trace(gram)) / count
    shared = (float(gram.sum()) - float(np.trace(gram))) / (count * (count - 1))
    if shared <= 0.0:
        return math.inf
    return duels_per_group * (diagonal - shared) / shared


def left_out(gram: np.ndarray, duels_per_group: float) -> tuple[float, float]:
    """The noise scale and its standard error by leaving one group out at a
    time (the jackknife)."""
    count = len(gram)
    whole = noise_scale(gram, duels_per_group)
    each = []
    for index in range(count):
        kept = [other for other in range(count) if other != index]
        each.append(noise_scale(gram[np.ix_(kept, kept)], duels_per_group))
    if not all(math.isfinite(value) for value in each):
        return whole, math.inf
    middle = sum(each) / count
    error = math.sqrt((count - 1) / count * sum((value - middle) ** 2 for value in each))
    return whole, error


def shared_signal(gram: np.ndarray, duels_per_group: float) -> tuple[float, float]:
    """How many standard errors (leaving a group out at a time) the signal's
    squared length is from none, and the noise scale were it two of them
    larger: the least the measure allows."""
    count = len(gram)

    def parts(kept: list[int]) -> tuple[float, float]:
        part = gram[np.ix_(kept, kept)]
        diagonal = float(np.trace(part)) / len(kept)
        shared = (float(part.sum()) - float(np.trace(part))) / (len(kept) * (len(kept) - 1))
        return diagonal, shared

    diagonal, shared = parts(list(range(count)))
    each = [parts([other for other in range(count) if other != index])[1] for index in range(count)]
    middle = sum(each) / count
    error = math.sqrt((count - 1) / count * sum((value - middle) ** 2 for value in each))
    larger = shared + ERRORS * error
    least = duels_per_group * (diagonal - shared) / larger if larger > 0.0 else math.inf
    return (shared / error if error > 0.0 else math.inf), least


def jackknifed(each: list[float]) -> float:
    """The standard error of a measure from its values with one part left out
    at a time."""
    count = len(each)
    if not all(math.isfinite(value) for value in each):
        return math.inf
    middle = sum(each) / count
    return math.sqrt((count - 1) / count * sum((value - middle) ** 2 for value in each))


def block_parts(
    gram: np.ndarray, squares: np.ndarray, counts: np.ndarray, kept: np.ndarray
) -> tuple[float, float]:
    """A group's mean squared length and the part two groups share, over the
    kept blocks of groups: ``gram`` holds the dot products of the blocks' sums
    of their groups' gradients, ``squares`` each block's sum of its groups'
    squared lengths, ``counts`` its groups."""
    together = float(gram[np.ix_(kept, kept)].sum())
    square = float(squares[kept].sum())
    groups = float(counts[kept].sum())
    return square / groups, (together - square) / (groups * (groups - 1))


def by_blocks(
    gram: np.ndarray, squares: np.ndarray, counts: np.ndarray, duels_per_group: float
) -> dict[str, float]:
    """The noise scale and the signal per duel (its inverse, which stays
    finite where no signal is seen) of gradients kept by blocks of groups,
    each with its error by leaving a block out at a time; how many errors the
    shared part is from none, and the noise scale were it two errors larger."""

    def measures(kept: np.ndarray) -> tuple[float, float, float, float]:
        diagonal, shared = block_parts(gram, squares, counts, kept)
        noise = duels_per_group * (diagonal - shared)
        scale = noise / shared if shared > 0.0 else math.inf
        return scale, shared / noise, shared, noise

    everything = np.arange(len(counts))
    scale, signal, shared, noise = measures(everything)
    each = [measures(np.delete(everything, block)) for block in everything]
    shared_error = jackknifed([value[2] for value in each])
    larger = shared + ERRORS * shared_error
    return {
        "noise_scale": scale,
        "error": jackknifed([value[0] for value in each]),
        "signal_per_duel": signal,
        "signal_error": jackknifed([value[1] for value in each]),
        "shared_z": shared / shared_error if shared_error > 0.0 else math.inf,
        "least": noise / larger if larger > 0.0 else math.inf,
    }


def signal_gap(
    one: dict[str, Any], other: dict[str, Any], duels_per_group: float
) -> tuple[float, float]:
    """The signal per duel of ``one`` less ``other``, measured on the same
    blocks of groups, and its error by leaving a block out at a time."""

    def signal(kept: np.ndarray, measured: dict[str, Any]) -> float:
        diagonal, shared = block_parts(
            np.array(measured["gram"]), np.array(measured["squares"]),
            np.array(measured["counts"]), kept,
        )  # fmt: skip
        return shared / (duels_per_group * (diagonal - shared))

    everything = np.arange(len(one["counts"]))
    gap = signal(everything, one) - signal(everything, other)
    each = [
        signal(np.delete(everything, block), one) - signal(np.delete(everything, block), other)
        for block in everything
    ]
    return gap, jackknifed(each)


def deal_share(results: np.ndarray) -> float:
    """Of a result's variance, the share that is the deal's: ``results`` holds
    each deal's plays in a row (one-way analysis of variance)."""
    plays = results.shape[1]
    within = float(results.var(axis=1, ddof=1).mean())
    between = float(results.mean(axis=1).var(ddof=1))
    deal = max(between - within / plays, 0.0)
    total = deal + within
    return deal / total if total > 0.0 else 0.0


def explained(targets: np.ndarray, estimates: np.ndarray) -> float:
    """The share of the targets' variance that subtracting the estimates takes
    away, as the trainer reads its value head (it can be under zero)."""
    variance = float(targets.var())
    if variance <= 0.0:
        return 0.0
    return 1.0 - float((targets - estimates).var()) / variance


def accounted(values: np.ndarray, other: np.ndarray) -> float:
    """The share of ``values``' variance that ``other`` accounts for with one
    coefficient: their squared correlation."""
    spread = float(values.std()) * float(other.std())
    if spread <= 0.0:
        return 0.0
    correlation = float(((values - values.mean()) * (other - other.mean())).mean()) / spread
    return correlation * correlation


def drawn_again(
    measure: Any, deals: int, *, draws: int = 1000, seed: int = 2012
) -> tuple[float, float]:
    """``measure(chosen)`` over the deals, and its standard error over the
    deals drawn again with replacement (``chosen`` holds deal indices)."""
    whole = measure(np.arange(deals))
    generator = np.random.default_rng(seed)
    each = [measure(generator.integers(0, deals, deals)) for _ in range(draws)]
    return whole, float(np.std(each, ddof=1))


# The readings.


def noise_reading(scale: float, error: float) -> str:
    if scale >= NOISE_BOUND and scale - UPDATE_DUELS > ERRORS * error:
        return "limited by noise"
    if scale <= NOT_NOISE_BOUND and UPDATE_DUELS - scale > ERRORS * error:
        return "not limited by noise"
    return "neither"


def pairing_reading(share: float, error: float) -> str:
    if share - ERRORS * error >= PAIRING_PAYS:
        return "pairing pays"
    if share + ERRORS * error <= PAIRING_DOES_NOT:
        return "pairing does not pay"
    return "not decided"


def signal_reading(new: float, old: float, error: float) -> str:
    """The third reading of the follow-up: the signal per duel at λ = 1
    (``new``) against λ = 0.95's (``old``), ``error`` that of their gap."""
    if new >= 2 * old and new - old > ERRORS * error:
        return "more signal"
    if new <= old / 2 and old - new > ERRORS * error:
        return "less signal"
    return "not told apart"


# The trainer, as the run made it.


def trainer_for(checkpoint: Path, described: dict[str, Any], args: argparse.Namespace) -> Any:
    """A trainer with the run's settings on the processor, the checkpoint
    loaded, at the update after it."""
    sys.path.insert(0, str(ROOT))
    from ygo_training.__main__ import trainer_config
    from ygo_training.checkpoints import load
    from ygo_training.cli_parser import build_parser
    from ygo_training.ppo_trainer import PpoTrainer

    handlers = {
        name: (lambda _: None)
        for name in (
            "train",
            "status",
            "evaluate-suite",
            "evaluate-search",
            "edopro",
            "edopro-review",
        )
    }
    learners = args.learners or ",".join(described["learners"])
    opponents = args.opponents or ",".join(described["opponents"])
    parsed = build_parser(handlers=handlers, default_learner=described["learner"]).parse_args(
        [
            "train",
            "--device", getattr(args, "device", "cpu"),
            "--workers", str(described["workers"]),
            "--pool-threads", str(args.threads),
            "--pool-slots", str(args.slots),
            "--run", str(args.scratch_run),
            "--learner", learners,
            "--opponents", opponents,
            "--duels", str(described["duels"]),
            "--self-play-duels", str(described["self_play_duels"]),
            "--rollout-estimates", described["rollout_estimates"],
            "--duel-gae-lambda", str(described["duel_gae_lambda"]),
            "--minibatch", str(described["minibatch"]),
            # Measured here, the run's games are played by the pool itself.
            "--duels-in-flight", "0",
            "--ppo-epochs", str(described["ppo_epochs"]),
            "--learning-rate", str(described["learning_rate"]),
            "--seed", str(described["seed"]),
            "--training-seed", str(described["training_seed"]),
            "--eval-every", "0",
        ]
    )  # fmt: skip
    config = trainer_config(parsed)
    for name in ("duel_gae_lambda", "clip_epsilon", "entropy_coefficient", "ppo_value_target",
                 "rollout_temperature", "duels", "self_play_duels", "minibatch"):  # fmt: skip
        if getattr(config, name) != described[name]:
            raise SystemExit(f"{name}: the run's {described[name]}, here {getattr(config, name)}")
    trainer = PpoTrainer(parsed.root, config)
    state = load(
        checkpoint,
        trainer.model,
        trainer.optimizer,
        map_location=trainer.device,
        expected_board_representation=config.board_representation,
        expected_leave_out=config.leave_out,
    )
    start = int(state["step"])
    trainer.set_rollout_index(start)
    trainer.set_training_schedule_index(start - int(described["schedule_origin"]))
    trainer.measured_from = start
    return trainer


def update_games(trainer: Any, later: int) -> tuple[int, list[dict[str, Any]]]:
    """The rollout seed and the games of the update ``later`` updates after
    the checkpoint's, as the run would have played them."""
    schedule = trainer.coordinator.schedule
    seed = schedule.rollout_seed(trainer.measured_from + later)
    return seed, trainer.coordinator._games(seed, imitation_only=False)


def flat(gradients: tuple[Any, ...], parameters: list[Any]) -> np.ndarray:
    import torch

    return (
        torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                for gradient, parameter in zip(gradients, parameters, strict=True)
            ]
        )
        .double()
        .cpu()
        .numpy()
    )


def group_gradients(
    trainer: Any,
    samples: list[Any],
    advantages: Any,
    indices: list[int],
    chunk: int,
    with_value: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The gradients of the update's policy and value terms over a group of
    its samples, as one minibatch of all of them would give them: the
    objective's mean over the group's transactions, taken a part at a time.
    Without ``with_value`` the value term's is left at nothing."""
    import torch

    optimizer = trainer.policy_optimizer
    parameters = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]
    keys = {samples[index].step["_transaction_key"] for index in indices}
    policy = np.zeros(sum(parameter.numel() for parameter in parameters))
    value = np.zeros_like(policy)
    for part in optimizer._chunks(samples, indices, chunk):
        share = len({samples[index].step["_transaction_key"] for index in part}) / len(keys)
        components = optimizer.objective(samples, advantages, part).components
        policy += flat(
            torch.autograd.grad(components["policy"] * share, parameters, retain_graph=with_value,
                                allow_unused=True),
            parameters,
        )  # fmt: skip
        if with_value:
            value += flat(
                torch.autograd.grad(components["value"] * share, parameters, allow_unused=True),
                parameters,
            )
    return policy, value, len(keys)


def groups_of(trajectories: list[dict[str, Any]], groups: int) -> list[list[int]]:
    """The update's duels dealt into ``groups``, each with as many of the
    duels against itself as can be."""
    order = sorted(
        range(len(trajectories)), key=lambda index: trajectories[index].get("other") != "self"
    )
    dealt: list[list[int]] = [[] for _ in range(groups)]
    for place, index in enumerate(order):
        dealt[place % groups].append(index)
    return dealt


def measure_noise(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    trainer.model.train()
    policy, value, sizes, duels = [], [], [], []
    started = time.perf_counter()
    for later in range(args.updates):
        seed, games = update_games(trainer, later)
        trajectories, _ = trainer._learner_games(games, seed, labels=False)
        trainer.policy_optimizer.begin_update(trainer.measured_from + later)
        samples, advantages, _ = trainer.policy_optimizer.prepare_samples(trajectories)
        by_duel: dict[int, list[int]] = {}
        for index, sample in enumerate(samples):
            by_duel.setdefault(int(sample.step["_transaction_key"][0]), []).append(index)
        for group in groups_of(trajectories, args.groups):
            indices = [index for duel in group for index in by_duel.get(duel, [])]
            taken = group_gradients(trainer, samples, advantages, indices, args.chunk)
            policy.append(taken[0])
            value.append(taken[1])
            sizes.append(taken[2])
            duels.append(len(group))
        print(
            f"update {later + 1} of {args.updates}: {len(trajectories)} duels, "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    found: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "measured_from": trainer.measured_from,
        "learners": args.learners or "the run's",
        "opponents": args.opponents or "the run's",
        "updates": args.updates,
        "groups": len(duels),
        "duels_per_group": float(np.mean(duels)),
        "transactions_per_group": float(np.mean(sizes)),
        "seconds": time.perf_counter() - started,
    }
    for name, gradients in (("policy", policy), ("value", value)):
        stacked = np.stack(gradients)
        gram = stacked @ stacked.T
        scale, error = left_out(gram, found["duels_per_group"])
        found[name] = {"noise_scale": scale, "error": error, "gram": gram.tolist()}
    trainer.close()
    return found


def at_lambda(optimizer: Any, value: float) -> None:
    """The optimizer computing its advantages at λ = ``value``."""
    from dataclasses import replace

    optimizer.config = replace(optimizer.config, duel_gae_lambda=value)


def measure_noise_lambdas(args: argparse.Namespace) -> dict[str, Any]:
    """`noise` with each update's gradients taken at every λ of LAMBDAS on the
    same games, kept by BLOCKS of updates: every group's gradient at three λ
    would not fit in memory."""
    import torch

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    lambdas = [float(value) for value in args.lambdas.split(",")]
    if args.updates % args.blocks:
        raise SystemExit("BLOCKS must divide UPDATES")
    trainer = trainer_for(args.checkpoint, described, args)
    trainer.model.train()
    optimizer = trainer.policy_optimizer
    run_lambda = optimizer.config.duel_gae_lambda
    size = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    sums = {
        (lam, term): np.zeros((args.blocks, size))
        for lam in lambdas
        for term in ("policy", "value")
    }
    squares = {key: np.zeros(args.blocks) for key in sums}
    counts = np.zeros(args.blocks)
    sizes, duels = [], []
    started = time.perf_counter()
    for later in range(args.updates):
        block = later * args.blocks // args.updates
        seed, games = update_games(trainer, later)
        trajectories, _ = trainer._learner_games(games, seed, labels=False)
        for lam in lambdas:
            at_lambda(optimizer, lam)
            optimizer.begin_update(trainer.measured_from + later)
            samples, advantages, _ = optimizer.prepare_samples(trajectories)
            by_duel: dict[int, list[int]] = {}
            for index, sample in enumerate(samples):
                by_duel.setdefault(int(sample.step["_transaction_key"][0]), []).append(index)
            for group in groups_of(trajectories, args.groups):
                indices = [index for duel in group for index in by_duel.get(duel, [])]
                policy, value, transactions = group_gradients(
                    trainer, samples, advantages, indices, args.chunk
                )
                for term, gradient in (("policy", policy), ("value", value)):
                    sums[(lam, term)][block] += gradient
                    squares[(lam, term)][block] += float(gradient @ gradient)
                if lam == lambdas[0]:
                    counts[block] += 1
                    sizes.append(transactions)
                    duels.append(len(group))
        at_lambda(optimizer, run_lambda)
        print(
            f"update {later + 1} of {args.updates}: {len(trajectories)} duels, "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    trainer.close()
    found: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "measured_from": trainer.measured_from,
        "learners": args.learners or "the run's",
        "opponents": args.opponents or "the run's",
        "updates": args.updates,
        "blocks": args.blocks,
        "groups": len(duels),
        "duels_per_group": float(np.mean(duels)),
        "transactions_per_group": float(np.mean(sizes)),
        "seconds": time.perf_counter() - started,
        "lambdas": {},
    }
    for lam in lambdas:
        found["lambdas"][str(lam)] = {}
        for term in ("policy", "value"):
            gram = sums[(lam, term)] @ sums[(lam, term)].T
            measured = {
                "gram": gram.tolist(),
                "squares": squares[(lam, term)].tolist(),
                "counts": counts.tolist(),
            }
            measured |= by_blocks(gram, squares[(lam, term)], counts, found["duels_per_group"])
            found["lambdas"][str(lam)][term] = measured
    return found


def measure_luck(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    _, games = update_games(trainer, 0)
    games = games[: args.deals]
    schedule = trainer.coordinator.schedule
    optimizer = trainer.policy_optimizer
    run_lambda = optimizer.config.duel_gae_lambda
    lambdas = [float(value) for value in args.lambdas.split(",")] if args.lambdas else []
    deals = len(games)
    # Each play's results by deal and seat, and its transactions' advantages:
    # at the run's λ, and at each of LAMBDAS on the same plays.
    results = np.full((deals, args.plays, 2), np.nan)
    learner_seat = np.zeros(deals, dtype=np.int64)
    against_itself = np.zeros(deals, dtype=bool)
    rows: list[tuple[int, int, int, float, float, bool]] = []
    at: dict[float, list[float]] = {lam: [] for lam in lambdas}
    started = time.perf_counter()
    for play in range(args.plays):
        seed = schedule.rollout_seed(trainer.measured_from + args.away + play)
        trajectories, _ = trainer._learner_games(games, seed, labels=False)
        keys = []
        for lam in [run_lambda, *lambdas]:
            at_lambda(optimizer, lam)
            samples, _, _ = optimizer.prepare_samples(trajectories)
            firsts: set[tuple[int, int]] = set()
            seen: set[object] = set()
            roots = []
            for sample in samples:
                key = sample.step["_transaction_key"]
                if key not in seen:
                    seen.add(key)
                    roots.append(sample)
            if lam != run_lambda or keys:
                if [sample.step["_transaction_key"] for sample in roots] != keys:
                    raise SystemExit("the advantages at another λ are of other transactions")
                at[lam].extend(sample.advantage for sample in roots)
                continue
            for sample in roots:
                key = sample.step["_transaction_key"]
                keys.append(key)
                duel, seat = int(key[0]), int(sample.step["player"])
                first = (duel, seat) not in firsts
                firsts.add((duel, seat))
                results[duel, play, seat] = sample.terminal_return
                rows.append(
                    (duel, play, seat, sample.advantage, float(sample.step["value"]), first)
                )
        at_lambda(optimizer, run_lambda)
        for duel, trajectory in enumerate(trajectories):
            learner_seat[duel] = int(trajectory["learner_player"])
            against_itself[duel] = trajectory.get("other") == "self"
        print(
            f"play {play + 1} of {args.plays}: {len(trajectories)} duels, "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    trainer.close()
    duel = np.array([row[0] for row in rows])
    play = np.array([row[1] for row in rows])
    seat = np.array([row[2] for row in rows])
    advantage = np.array([row[3] for row in rows])
    estimate = np.array([row[4] for row in rows])
    first = np.array([row[5] for row in rows])
    result = results[duel, play, seat]
    # Another play of the deal, for the same seat; and the others' mean.
    other = results[duel, (play + 1) % args.plays, seat]
    others = (np.nansum(results[duel, :, seat], axis=1) - result) / (args.plays - 1)
    learner = results[np.arange(deals), :, learner_seat]
    # Each deal's rows, and its rows at a seat's first decision.
    order = np.argsort(duel, kind="stable")
    bounds = np.searchsorted(duel[order], np.arange(deals + 1))
    of_deal = [order[bounds[deal] : bounds[deal + 1]] for deal in range(deals)]
    firsts_of_deal = [rows_of[first[rows_of]] for rows_of in of_deal]

    def by_deals(chosen: np.ndarray, firsts_only: bool) -> np.ndarray:
        """The rows of the chosen deals, with repeats, or their first decisions'."""
        lists = firsts_of_deal if firsts_only else of_deal
        return np.concatenate([lists[deal] for deal in chosen])

    found: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "measured_from": trainer.measured_from,
        "deals": deals,
        "plays": args.plays,
        "duels": deals * args.plays,
        "transactions": len(rows),
        "seconds": time.perf_counter() - started,
    }

    def measured(listed: np.ndarray) -> dict[str, Any]:
        """The measures over the deals ``listed``, each with its error."""

        def pairing(chosen: np.ndarray, with_others: bool = False) -> float:
            picked = by_deals(listed[chosen], firsts_only=False)
            return accounted(advantage[picked], (others if with_others else other)[picked])

        def head(chosen: np.ndarray) -> float:
            picked = by_deals(listed[chosen], firsts_only=True)
            return explained(result[picked], estimate[picked])

        def luck(chosen: np.ndarray) -> float:
            return deal_share(learner[listed[chosen]])

        share, error = drawn_again(pairing, len(listed), draws=args.draws)
        mean_share, mean_error = drawn_again(
            lambda chosen: pairing(chosen, with_others=True), len(listed), draws=args.draws
        )
        lucky, lucky_error = drawn_again(luck, len(listed), draws=args.draws)
        told, told_error = drawn_again(head, len(listed), draws=args.draws)
        return {
            "deals": len(listed),
            "pairing_share": share,
            "pairing_error": error,
            "pairing_share_with_the_others_mean": mean_share,
            "pairing_error_with_the_others_mean": mean_error,
            "deal_share": lucky,
            "deal_error": lucky_error,
            "head_at_first_decision": told,
            "head_error": told_error,
        }

    kinds = (
        ("all", np.ones(deals, dtype=bool)),
        ("against itself", against_itself),
        ("against the pilots", ~against_itself),
    )
    for kind, kept_deals in kinds:
        found[kind] = measured(np.flatnonzero(kept_deals))

    def at_one_lambda(listed: np.ndarray, values: np.ndarray) -> dict[str, Any]:
        """Of the advantages ``values``, over the deals ``listed``: what
        another play's result and the others' mean account for, their
        variance, and what the seat's own result accounts for."""

        def share(chosen: np.ndarray, against: np.ndarray) -> float:
            picked = by_deals(listed[chosen], firsts_only=False)
            return accounted(values[picked], against[picked])

        pairing, pairing_error = drawn_again(
            lambda chosen: share(chosen, other), len(listed), draws=args.draws
        )
        mean, mean_error = drawn_again(
            lambda chosen: share(chosen, others), len(listed), draws=args.draws
        )
        own, own_error = drawn_again(
            lambda chosen: share(chosen, result), len(listed), draws=args.draws
        )
        picked = by_deals(listed, firsts_only=False)
        return {
            "deals": len(listed),
            "pairing_share": pairing,
            "pairing_error": pairing_error,
            "pairing_share_with_the_others_mean": mean,
            "pairing_error_with_the_others_mean": mean_error,
            "advantage_variance": float(values[picked].var()),
            "own_result_share": own,
            "own_result_error": own_error,
        }

    if lambdas:
        found["lambdas"] = {
            str(lam): {
                kind: at_one_lambda(np.flatnonzero(kept_deals), np.array(at[lam]))
                for kind, kept_deals in kinds
            }
            for lam in lambdas
        }
    return found


def report(noise: dict[str, Any], luck: dict[str, Any]) -> str:
    scale, error = noise["policy"]["noise_scale"], noise["policy"]["error"]
    every = luck["all"]
    lines = [
        (
            f"Noise: {noise['checkpoint']}, {noise['groups']} groups of "
            f"{noise['duels_per_group']:.0f} duels ({noise['transactions_per_group']:.0f} "
            "transactions)."
        )
    ]
    for name, called in (("policy", "the policy's"), ("value", "the value head's")):
        part = noise[name]
        seen, least = shared_signal(np.array(part["gram"]), noise["duels_per_group"])
        lines.append(
            f"  {called} noise scale: {part['noise_scale']:,.0f} ± {part['error']:,.0f} duels "
            f"(an update: {UPDATE_DUELS}); the signal {seen:.1f} standard errors from none, and "
            f"with it two standard errors larger the noise scale {least:,.0f}"
        )
    lines.append(
        f"Luck: {luck['checkpoint']}, {luck['deals']} deals, {luck['plays']} plays each "
        f"({luck['transactions']:,} transactions)."
    )
    for kind in ("all", "against itself", "against the pilots"):
        part = luck[kind]
        lines.append(
            f"  {kind} ({part['deals']} deals): another play's result accounts for "
            f"{part['pairing_share']:.3f} ± {part['pairing_error']:.3f} of the advantages' variance, "
            f"the others' mean {part['pairing_share_with_the_others_mean']:.3f} ± "
            f"{part['pairing_error_with_the_others_mean']:.3f}; the deal's share of a result "
            f"{part['deal_share']:.3f} ± {part['deal_error']:.3f}; the value head at a seat's first "
            f"decision {part['head_at_first_decision']:.3f} ± {part['head_error']:.3f}"
        )
    lines += [
        "The readings:",
        f"  1. the update's step: {noise_reading(scale, error)}",
        f"  2. pairing deals: {pairing_reading(every['pairing_share'], every['pairing_error'])}",
    ]
    return "\n".join(lines)


def measure_noise_critic(args: argparse.Namespace) -> dict[str, Any]:
    """`noise-lambdas` with the update's advantages from the value head and
    from the critic that sees the hidden cards (CRITIC, saved by
    `tools/privileged_critic.py fit`), at the run's λ on the same games: the
    policy's term of the step with each, kept by BLOCKS of updates."""
    import torch

    from ygo_training.critic import SightedCritic

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    if args.updates % args.blocks:
        raise SystemExit("BLOCKS must divide UPDATES")
    trainer = trainer_for(args.checkpoint, described, args)
    trainer.model.train()
    saved = torch.load(args.critic, map_location=trainer.device)
    critic = SightedCritic.from_policy(trainer.model).to(trainer.device)
    critic.load_state_dict(saved["critic"])
    critic.eval()
    deck = int(saved.get("deck", 0))
    optimizer = trainer.policy_optimizer
    size = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    variants = ("head", "critic")
    sums = {variant: np.zeros((args.blocks, size)) for variant in variants}
    squares = {variant: np.zeros(args.blocks) for variant in variants}
    counts = np.zeros(args.blocks)
    sizes, duels = [], []
    started = time.perf_counter()
    for later in range(args.updates):
        block = later * args.blocks // args.updates
        seed, games = update_games(trainer, later)
        trajectories, _ = trainer._learner_games(
            games, seed, labels=False, whole=True, whole_deck=deck
        )
        steps = [step for game in trajectories for step in game["steps"] if "row" in step]
        heads = [float(step["value"]) for step in steps]
        critics: list[float] = []
        with torch.no_grad():
            for start in range(0, len(steps), args.chunk):
                rows = [int(step["row"]) for step in steps[start : start + args.chunk]]
                critics += critic(trainer._kept(rows), trainer._kept_whole(rows)).tolist()
        for variant, values in (("head", heads), ("critic", critics)):
            for step, value in zip(steps, values, strict=True):
                step["value"] = value
            optimizer.begin_update(trainer.measured_from + later)
            samples, advantages, _ = optimizer.prepare_samples(trajectories)
            by_duel: dict[int, list[int]] = {}
            for index, sample in enumerate(samples):
                by_duel.setdefault(int(sample.step["_transaction_key"][0]), []).append(index)
            for group in groups_of(trajectories, args.groups):
                indices = [index for duel in group for index in by_duel.get(duel, [])]
                policy, _, transactions = group_gradients(
                    trainer, samples, advantages, indices, args.chunk, with_value=False
                )
                sums[variant][block] += policy
                squares[variant][block] += float(policy @ policy)
                if variant == "head":
                    counts[block] += 1
                    sizes.append(transactions)
                    duels.append(len(group))
        for step, value in zip(steps, heads, strict=True):
            step["value"] = value
        print(
            f"update {later + 1} of {args.updates}: {len(trajectories)} duels, "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    trainer.close()
    found: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "critic": str(args.critic),
        "deck": deck,
        "measured_from": trainer.measured_from,
        "updates": args.updates,
        "blocks": args.blocks,
        "groups": len(duels),
        "duels_per_group": float(np.mean(duels)),
        "transactions_per_group": float(np.mean(sizes)),
        "seconds": time.perf_counter() - started,
        "variants": {},
    }
    for variant in variants:
        gram = sums[variant] @ sums[variant].T
        measured = {
            "gram": gram.tolist(),
            "squares": squares[variant].tolist(),
            "counts": counts.tolist(),
        }
        measured |= by_blocks(gram, squares[variant], counts, found["duels_per_group"])
        found["variants"][variant] = measured
    return found


# The second reading of experiments/2026-10-22-privileged-critic.md: the
# value head's signal per duel at λ = 0.95, as the follow-up measured it.
HEAD_SIGNAL = 5.0e-5


def critic_reading(critic: float, head: float, error: float) -> str:
    """A stronger signal at twice the value head's per duel, the gap more
    than two standard errors."""
    if critic >= 2 * head and critic - head > ERRORS * error:
        return "a stronger signal"
    return "not shown to be stronger"


def report_critic(noise: dict[str, Any]) -> str:
    """The second reading of the critic that sees the hidden cards, as it
    was fixed on 2026-10-22 before anything was built."""
    head, critic = noise["variants"]["head"], noise["variants"]["critic"]
    gap, error = signal_gap(critic, head, noise["duels_per_group"])
    lines = [
        (
            f"Noise: {noise['checkpoint']} with the critic {noise['critic']} (seeing {noise['deck']} "
            f"of each Deck's next cards), {noise['groups']} groups of {noise['duels_per_group']:.0f} "
            f"duels in {noise['blocks']} blocks."
        )
    ]
    for name, part in (("the value head's", head), ("the critic's", critic)):
        lines.append(
            f"  with {name} advantages: noise scale {part['noise_scale']:,.0f} ± {part['error']:,.0f} "
            f"duels, a signal per duel of {part['signal_per_duel']:.2e} ± {part['signal_error']:.2e}"
        )
    lines += [
        f"  the gap {gap:.2e} ± {error:.2e}; the value head's in the follow-up, {HEAD_SIGNAL:.1e}",
        f"The reading: {critic_reading(critic['signal_per_duel'], head['signal_per_duel'], error)}",
    ]
    return "\n".join(lines)


def keyed(by_lambda: dict[str, Any], value: float) -> Any:
    """What is kept under λ = ``value``."""
    (found,) = [found for lam, found in by_lambda.items() if float(lam) == value]
    return found


def report_lambdas(
    noise: dict[str, Any], luck: dict[str, Any], mirror: dict[str, Any] | None = None
) -> str:
    """The follow-up at three λ, with its readings as they were fixed on
    2026-10-22, before any run of it."""
    lines = []
    for name, measured in (("Noise", noise), ("The Blackwing mirror alone", mirror)):
        if measured is None:
            continue
        lines.append(
            f"{name}: {measured['checkpoint']}, {measured['groups']} groups of "
            f"{measured['duels_per_group']:.0f} duels in {measured['blocks']} blocks "
            f"({measured['transactions_per_group']:.0f} transactions a group)."
        )
        for lam, terms in measured["lambdas"].items():
            for term, called in (("policy", "the policy's"), ("value", "the value head's")):
                part = terms[term]
                lines.append(
                    f"  λ {lam}, {called}: noise scale {part['noise_scale']:,.0f} ± "
                    f"{part['error']:,.0f} duels, a signal per duel of {part['signal_per_duel']:.2e} "
                    f"± {part['signal_error']:.2e}; the shared part {part['shared_z']:.1f} errors "
                    f"from none, the noise scale with it two errors larger {part['least']:,.0f}"
                )
    lines.append(
        f"Luck: {luck['checkpoint']}, {luck['deals']} deals, {luck['plays']} plays each "
        f"({luck['transactions']:,} transactions)."
    )
    for lam, kinds in luck["lambdas"].items():
        for kind, part in kinds.items():
            lines.append(
                f"  λ {lam}, {kind}: another play's result accounts for {part['pairing_share']:.3f} "
                f"± {part['pairing_error']:.3f} of the advantages' variance, the others' mean "
                f"{part['pairing_share_with_the_others_mean']:.3f} ± "
                f"{part['pairing_error_with_the_others_mean']:.3f}; the advantages' variance "
                f"{part['advantage_variance']:.3f}, the seat's own result accounting for "
                f"{part['own_result_share']:.3f} ± {part['own_result_error']:.3f} of it"
            )
    step = keyed(noise["lambdas"], 0.95)["policy"]
    whole = keyed(noise["lambdas"], 1.0)["policy"]
    gap, gap_error = signal_gap(whole, step, noise["duels_per_group"])
    paired = keyed(luck["lambdas"], 1.0)["all"]
    paired_reading = pairing_reading(paired["pairing_share"], paired["pairing_error"])
    signal = signal_reading(whole["signal_per_duel"], step["signal_per_duel"], gap_error)
    lines += [
        "The readings:",
        f"  1. the step at λ 0.95: {noise_reading(step['noise_scale'], step['error'])}",
        f"  2. pairing deals at λ 1: {paired_reading}",
        (
            f"  3. λ 1 against 0.95: a signal per duel of {whole['signal_per_duel']:.2e} against "
            f"{step['signal_per_duel']:.2e}, the gap {gap:.2e} ± {gap_error:.2e}: {signal}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("noise", "noise-lambdas", "noise-critic", "luck"):
        command = commands.add_parser(name)
        command.add_argument("checkpoint", type=Path)
        command.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--learners", default=None, help="comma-separated; the run's by default"
        )
        command.add_argument(
            "--opponents", default=None, help="comma-separated; the run's by default"
        )
        command.add_argument("--threads", type=int, default=20, help="the pool's threads")
        command.add_argument("--slots", type=int, default=16)
        command.add_argument("--torch-threads", type=int, default=8)
        command.add_argument("--device", default="cpu", help="where the networks run: cpu or cuda")
        command.add_argument(
            "--scratch-run", type=Path, default=ROOT / "runs" / "update-signal",
            help="the run the trainer is told of; nothing is written there",
        )  # fmt: skip
    for name, updates in (("noise", 8), ("noise-lambdas", 32), ("noise-critic", 32)):
        noise = commands.choices[name]
        noise.add_argument("--updates", type=int, default=updates)
        noise.add_argument("--groups", type=int, default=8, help="groups of an update's duels")
        noise.add_argument("--chunk", type=int, default=1024, help="samples a part, for memory")
    lambdas = commands.choices["noise-lambdas"]
    lambdas.add_argument("--lambdas", default="0.95,0.98,1.0")
    lambdas.add_argument("--blocks", type=int, default=16, help="blocks of updates for the errors")
    critical = commands.choices["noise-critic"]
    critical.add_argument(
        "--critic", type=Path, required=True, help="the fitted critic's critic.pt"
    )
    critical.add_argument("--blocks", type=int, default=16, help="blocks of updates for the errors")
    luck = commands.choices["luck"]
    luck.add_argument("--deals", type=int, default=512)
    luck.add_argument("--plays", type=int, default=16)
    luck.add_argument("--away", type=int, default=1000, help="updates away for the plays' seeds")
    luck.add_argument("--draws", type=int, default=1000, help="draws of the deals for the errors")
    luck.add_argument("--lambdas", default=None, help="comma-separated λ to tell beside the run's")
    reported = commands.add_parser("report")
    reported.add_argument("noise", type=Path)
    reported.add_argument("luck", type=Path)
    reported_lambdas = commands.add_parser("report-lambdas")
    reported_lambdas.add_argument("noise", type=Path)
    reported_lambdas.add_argument("luck", type=Path)
    reported_lambdas.add_argument("--mirror", type=Path, default=None)
    reported_critic = commands.add_parser("report-critic")
    reported_critic.add_argument("noise", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        print(report(json.loads(args.noise.read_text()), json.loads(args.luck.read_text())))
        return
    if args.command == "report-critic":
        print(report_critic(json.loads(args.noise.read_text())))
        return
    if args.command == "report-lambdas":
        mirror = json.loads(args.mirror.read_text()) if args.mirror else None
        print(
            report_lambdas(
                json.loads(args.noise.read_text()), json.loads(args.luck.read_text()), mirror
            )
        )
        return
    measure = {
        "noise": measure_noise,
        "noise-lambdas": measure_noise_lambdas,
        "noise-critic": measure_noise_critic,
        "luck": measure_luck,
    }[args.command]
    found = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(found))
    print(f"{args.command}: {found['seconds']:.0f} s, into {args.output}")


if __name__ == "__main__":
    main()
