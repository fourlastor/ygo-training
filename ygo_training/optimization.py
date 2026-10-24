"""PyTorch optimization services for PPO and imitation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

import torch

from .card_head import card_loss, chosen_answers, labels_of, scaled
from .config import GAE_UNIT, TrainerConfig
from .ppo import (
    PpoConfig,
    joint_ppo_loss_components,
    transaction_advantages,
    transaction_upgo_advantages,
)
from .rewards import RewardModel


class PreparedSample(NamedTuple):
    step: dict[str, Any]
    advantage: float
    return_: float
    terminal_return: float = 0.0
    cutoff: bool = False


class ObjectiveBatch(NamedTuple):
    """One canonical evaluation of the production optimization objective."""

    components: dict[str, torch.Tensor]
    imitation_loss: torch.Tensor | None
    imitation_count: int
    logits: torch.Tensor
    values: torch.Tensor
    steps: list[dict[str, Any]]
    # The loss of the head that learns how cards work, where there is one
    # and the minibatch has labels, and what its parts came to.
    card_loss: torch.Tensor | None = None
    card_parts: dict[str, float] | None = None


# The parameters of the three heads: all others are read by every one of them.
HEAD_PARAMETERS = ("score.", "value.", "lead.")

# Rollout step metadata used only by ``decision_advantage_metrics``.
DIAGNOSTIC_STEP_KEYS = frozenset({"prompt_type", "action_kind"})

# (question, answer) of the rollout decisions whose credit is reported every
# update, in the duel environment's names.
DIAGNOSTIC_DECISIONS = {
    "chain_pass": ("Chain", "Pass"),
    "chain_activate": ("Chain", "Activate"),
}


def seat_deck(trajectory: dict[str, Any], step: dict[str, Any], learner: str) -> str:
    """The deck of the seat a step's decision was asked of: the learner's,
    ``learner`` in a game that names none, or the other seat's where the
    game has a network in both."""
    if "opponent_deck" in trajectory and int(step["player"]) != int(trajectory["learner_player"]):
        return str(trajectory["opponent_deck"])
    return str(trajectory.get("learner_deck", learner))


def decision_advantage_metrics(
    samples: list[PreparedSample], raw: torch.Tensor, normalized: torch.Tensor
) -> dict[str, float]:
    """Credit PPO assigns to each diagnostic decision group in one update.

    ``normalized_advantage_sum`` is the net push on that choice's
    probability; ``terminal_return_mean`` against ``value_mean`` shows
    whether the choice appears in games that end better than the critic
    expected, which the advantage then credits to it.
    """
    metrics: dict[str, float] = {}
    for name, (prompt, kind) in DIAGNOSTIC_DECISIONS.items():
        indices = [
            index
            for index, sample in enumerate(samples)
            if sample.step.get("prompt_type") == prompt
            and sample.step.get("action_kind") == kind
        ]
        metrics[f"{name}_samples"] = float(len(indices))
        if not indices:
            continue
        steps = [samples[index] for index in indices]
        metrics[f"{name}_raw_advantage_mean"] = float(raw[indices].mean())
        metrics[f"{name}_normalized_advantage_mean"] = float(normalized[indices].mean())
        metrics[f"{name}_normalized_advantage_sum"] = float(normalized[indices].sum())
        metrics[f"{name}_positive_advantage_fraction"] = float((raw[indices] > 0).float().mean())
        metrics[f"{name}_value_mean"] = sum(float(s.step["value"]) for s in steps) / len(steps)
        metrics[f"{name}_return_mean"] = sum(s.return_ for s in steps) / len(steps)
        metrics[f"{name}_terminal_return_mean"] = sum(s.terminal_return for s in steps) / len(
            steps
        )
        metrics[f"{name}_probability_mean"] = sum(
            math.exp(float(s.step["log_prob"])) for s in steps
        ) / len(steps)
    return metrics


class PpoOptimizer:
    """Own all gradient-producing work for a policy model."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        reward_model: RewardModel,
        device: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.reward_model = reward_model
        self.device = device
        self._shuffle_update = 0
        self._shuffle_sequence = 0
        # The model's input for decisions kept elsewhere (``pool.DuelPool``),
        # by their rows: what a step's ``row`` is read with.
        self.kept: Callable[[list[int]], dict[str, torch.Tensor]] | None = None
        # During a pass over such decisions: the minibatches another thread
        # is building or has built, by their rows, and that thread.
        self._ahead: dict[tuple[int, ...], Future] = {}
        self._builder: ThreadPoolExecutor | None = None
        # The networks the imitation-only updates imitate, by the deck each
        # teaches, when their games are theirs and not a pilot's: each is
        # asked again at the decisions of its deck's seats, for what it gives
        # every answer.
        self.teachers: dict[str, torch.nn.Module] = {}
        # The head that learns how cards work (`card_head.py`), its own
        # optimizer, and the weight of the gradient it gives the network
        # beside PPO's: where there is one, a minibatch's decisions come with
        # their labels.
        self.card_head: torch.nn.Module | None = None
        self.card_head_optimizer: torch.optim.Optimizer | None = None
        self.card_head_weight = 0.0

    @staticmethod
    def _transaction_key(trajectory_index: int, step_index: int, step: dict[str, Any]) -> object:
        """What a step is optimized with: the other steps of its transaction."""
        if "transaction" in step:
            return (trajectory_index, int(step["player"]), int(step["transaction"]))
        return (trajectory_index, "step", step_index)

    def begin_update(self, step: int) -> None:
        """Reset deterministic minibatch shuffling for one global update."""
        self._shuffle_update = int(step)
        self._shuffle_sequence = 0

    def _carrying(self, steps: list[dict[str, Any]], starts: list[int], positions: int) -> set[int]:
        """Of the transactions that start at ``starts`` among ``steps``, those
        that carry the value head's target in this update: all of them, or
        ``positions`` of each seat's in a duel, drawn for the update.

        A duel has one outcome, and every transaction more that is taught it
        tells the duel again where it should tell a position.
        """
        if not positions:
            return set(starts)
        by_seat: dict[tuple[object, int], list[int]] = {}
        for index in starts:
            step = steps[index]
            by_seat.setdefault((step["_transaction_key"][0], int(step["player"])), []).append(
                index
            )
        drawn = random.Random(
            (int(self.config.seed) * 1_000_003 + self._shuffle_update * 10_007) ^ 0x56414C55
        )
        return {
            index
            for found in by_seat.values()
            for index in (found if len(found) <= positions else drawn.sample(found, positions))
        }

    def update(
        self, trajectories: list[dict[str, Any]], teacher_imitation_weight: float = 0.0
    ) -> dict[str, float | str]:
        self.model.train()
        samples, advantages, metrics = self.prepare_samples(trajectories)
        # The transactions the value head is taught at: every one, or so many
        # of each seat's in a duel.  Their advantages are all used as they are.
        positions = int(self.config.ppo_value_positions)
        value_carrying: set[int] | None = None
        starts: list[int] = []
        opened: set[object] = set()
        for index, sample in enumerate(samples):
            if sample.step["_transaction_key"] not in opened:
                opened.add(sample.step["_transaction_key"])
                starts.append(index)
        if positions:
            value_carrying = self._carrying([sample.step for sample in samples], starts, positions)
        metrics["ppo_value_positions"] = float(positions)
        metrics["ppo_value_target"] = self.config.ppo_value_target
        metrics["ppo_value_targets"] = float(
            len(starts) if value_carrying is None else len(value_carrying)
        )
        observed: dict[str, list[float]] = defaultdict(list)
        imitation_samples = 0
        # KL-targeted early stopping: each minibatch's approximate KL against
        # the rollout policy is measured before its step; the update ends
        # before stepping once it exceeds 1.5 x the target.
        target_kl = self.config.target_kl
        stopped = False
        optimizer_steps = 0
        completed_epochs = 0.0
        epoch_kls: list[float] = []

        # Where the pool keeps the decisions, another thread builds the next
        # minibatch and sends it to the card while the card works on this
        # one: the same minibatches in the same order.  One thread, so that
        # the pool is asked by one at a time; nothing else asks it in a pass.
        builder = None
        if self.kept is not None and samples and "row" in samples[0].step:
            builder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minibatch")
        self._builder = builder
        try:
            for _ in range(self.config.ppo_epochs):
                epoch_kls = []
                chunks = self._chunks(samples, list(range(len(samples))), self.config.minibatch)
                rows = (
                    [tuple(int(samples[index].step["row"]) for index in chunk) for chunk in chunks]
                    if builder is not None
                    else []
                )
                for position, indices in enumerate(chunks):
                    # This minibatch, where a pass has only begun, and the next.
                    for coming in rows[position : position + 2]:
                        if coming not in self._ahead:
                            self._ahead[coming] = builder.submit(self.kept, list(coming))
                    objective = self.objective(
                        samples,
                        advantages,
                        indices,
                        teacher_imitation_weight=teacher_imitation_weight,
                        value_carrying=value_carrying,
                    )
                    components = objective.components
                    total_loss = torch.zeros((), device=self.device) + components["total"]
                    if objective.card_loss is not None:
                        # Its weight is that of the gradient it gives the
                        # network (`card_head.scaled`).
                        total_loss = total_loss + objective.card_loss
                        for name, value in (objective.card_parts or {}).items():
                            observed[name].append(value)
                    if objective.imitation_loss is not None:
                        total_loss = total_loss + teacher_imitation_weight * objective.imitation_loss
                        observed["teacher_imitation_loss"].append(
                            float(objective.imitation_loss.detach())
                        )
                        imitation_samples += objective.imitation_count
                    for name in (
                        "policy",
                        "value",
                        "entropy",
                        "approximate_kl",
                        "clip_fraction",
                        "total",
                    ):
                        observed[f"duel_{name}"].append(float(components[name].detach()))
                    kl = float(components["approximate_kl"].detach())
                    epoch_kls.append(kl)
                    if target_kl is not None and kl > 1.5 * target_kl:
                        stopped = True
                        completed_epochs += position / len(chunks)
                        break
                    self._step(total_loss)
                    optimizer_steps += 1
                    observed["optimization_loss"].append(float(total_loss.detach()))
                if stopped:
                    break
                completed_epochs += 1.0
        finally:
            # A minibatch built for a step that was not made is dropped.
            if builder is not None:
                builder.shutdown(wait=True)
            self._builder = None
            self._ahead = {}

        for name, values in observed.items():
            metrics[name] = sum(values) / max(1, len(values))
        # The mean over the last (possibly partial) epoch is the closest
        # available measure of how far this update moved the policy.
        metrics["duel_final_approximate_kl"] = sum(epoch_kls) / max(1, len(epoch_kls))
        metrics["ppo_target_kl"] = 0.0 if target_kl is None else float(target_kl)
        metrics["ppo_early_stopped"] = float(stopped)
        metrics["ppo_optimizer_steps"] = float(optimizer_steps)
        metrics["ppo_epochs_completed"] = completed_epochs
        totals = observed.get("duel_total", [])
        metrics["ppo_loss"] = sum(totals) / max(1, len(totals))
        metrics.setdefault("optimization_loss", 0.0)
        metrics.setdefault("teacher_imitation_loss", 0.0)
        metrics["teacher_imitation_samples"] = float(imitation_samples)
        return metrics

    def recompute_rollout_estimates(self, trajectories: list[dict[str, Any]]) -> int:
        """Canonicalize PPO baselines after timing-dependent rollout batching."""
        steps = [
            step
            for trajectory in trajectories
            for step in trajectory["steps"]
            if "log_prob" in step and "value" in step
        ]
        if not steps:
            return 0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(steps), self.config.minibatch):
                chunk = steps[start : start + self.config.minibatch]
                logits, values = self.forward_steps(self.model, chunk)
                chosen = torch.tensor(
                    [int(step["chosen"]) for step in chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                distribution = torch.distributions.Categorical(
                    logits=logits / self.config.rollout_temperature
                )
                log_probs = distribution.log_prob(chosen)
                # Read together: a number at a time is a wait for the card each.
                for step, log_prob, value in zip(
                    chunk, log_probs.tolist(), values.tolist(), strict=True
                ):
                    step["log_prob"] = log_prob
                    step["value"] = value
        return len(steps)

    def teacher_label_metrics(self, trajectories: list[dict[str, Any]]) -> dict[str, float]:
        """How teacher labels of the learner's own states compare with its choices.

        Argmax agreement is measured with the rollout model, before the update.
        """
        learner_steps = [
            step
            for trajectory in trajectories
            for step in trajectory["steps"]
            if "log_prob" in step and "value" in step
        ]
        labeled = [step for step in learner_steps if "teacher_chosen" in step]
        if not labeled:
            return {}
        matches = 0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(labeled), self.config.minibatch):
                chunk = labeled[start : start + self.config.minibatch]
                logits, _ = self.forward_steps(self.model, chunk)
                teacher = torch.tensor(
                    [int(step["teacher_chosen"]) for step in chunk], device=self.device
                )
                matches += int((logits.argmax(dim=-1) == teacher).sum())
        return {
            "teacher_label_coverage": len(labeled) / len(learner_steps),
            "teacher_sampled_agreement": sum(
                int(step["chosen"]) == int(step["teacher_chosen"]) for step in labeled
            )
            / len(labeled),
            "teacher_argmax_agreement": matches / len(labeled),
        }

    def prepare_samples(
        self, trajectories: list[dict[str, Any]]
    ) -> tuple[list[PreparedSample], torch.Tensor, dict[str, float | str]]:
        """Build returns and normalized advantages used by ``update``."""
        samples: list[PreparedSample] = []
        cutoff_losses = cutoff_loss_steps = 0
        win_rewards: list[float] = []
        # What UPGO added to each transaction's advantage, where it is on.
        upgo_parts: list[float] = []
        all_returns: list[float] = []
        all_values: list[float] = []
        all_outcomes: list[float] = []
        transaction_lengths: list[int] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            cutoff = False
            if not trajectory["terminal"]:
                cutoff = True
                cutoff_losses += 1
            # Transaction keys are optimizer-local annotations.  Copy each
            # step so repeated fixture objects or shared rollout views cannot
            # overwrite another trajectory's grouping key.
            steps = [dict(step) for step in trajectory["steps"]]
            for step_index, step in enumerate(steps):
                step["_transaction_key"] = self._transaction_key(
                    trajectory_index, step_index, step
                )
            grouped: dict[object, int] = defaultdict(int)
            for step in steps:
                grouped[step["_transaction_key"]] += 1
            transaction_lengths.extend(grouped.values())
            winner = trajectory["winner"]
            targets: dict[int, tuple[float, float, float, bool]] = {}
            # A duel cut short is the learner's seat's to answer for: both
            # seats', where it played against itself.
            both = not cutoff or trajectory.get("other") == "self"
            players = (0, 1) if both else (int(trajectory["learner_player"]),)
            for player in players:
                indexed = [
                    (index, step) for index, step in enumerate(steps) if step["player"] == player
                ]
                if not indexed:
                    continue
                units_by_key: dict[object, list[tuple[int, dict[str, Any]]]] = {}
                units: list[list[tuple[int, dict[str, Any]]]] = []
                for index, step in indexed:
                    key = step["_transaction_key"]
                    if key not in units_by_key:
                        units_by_key[key] = []
                        units.append(units_by_key[key])
                    units_by_key[key].append((index, step))
                terminal_reward = (
                    self.reward_model.cutoff()
                    if cutoff
                    else self.reward_model.terminal(winner, player)
                )
                # The seat's transactions: no reward but the last one's, which
                # ends the duel for it.
                rewards = [0.0] * len(units)
                rewards[-1] += terminal_reward
                if not cutoff and player == winner:
                    win_rewards.append(terminal_reward)
                dones = [0.0] * len(units)
                dones[-1] = 1.0
                estimates = [members[0][1]["value"] for members in units]
                computed_advantages, returns = transaction_advantages(
                    rewards,
                    estimates,
                    dones,
                    gae_lambda=self.config.duel_gae_lambda,
                )
                # Where a critic gives the credit (`TrainerConfig.critic`),
                # the advantages are measured against its values; the head is
                # still taught its own return.
                if "critic_value" in units[0][0][1]:
                    estimates = [members[0][1]["critic_value"] for members in units]
                    computed_advantages, _ = transaction_advantages(
                        rewards,
                        estimates,
                        dones,
                        gae_lambda=self.config.duel_gae_lambda,
                    )
                if self.config.upgo:
                    with_upgo = transaction_upgo_advantages(
                        computed_advantages, rewards, estimates, dones
                    )
                    upgo_parts.extend(
                        ours - theirs
                        for ours, theirs in zip(with_upgo, computed_advantages, strict=True)
                    )
                    computed_advantages = with_upgo
                for members, advantage, ret in zip(
                    units, computed_advantages, returns, strict=True
                ):
                    target = (advantage, ret, float(terminal_reward), cutoff)
                    for index, _ in members:
                        targets[index] = target
                    all_returns.append(ret)
                    all_values.append(float(members[0][1]["value"]))
                    all_outcomes.append(float(terminal_reward))
            for index, step in enumerate(steps):
                if index not in targets:
                    continue
                advantage, ret, terminal_return, sample_cutoff = targets[index]
                samples.append(PreparedSample(step, advantage, ret, terminal_return, sample_cutoff))
                if cutoff:
                    cutoff_loss_steps += 1
        if not samples:
            raise RuntimeError(
                "No terminal trajectories were collected; increase rollout "
                "limit or use a terminal deck pool."
            )

        advantages = torch.tensor([sample.advantage for sample in samples], device=self.device)
        # One advantage per transaction: its members share it.
        members_by_key: dict[object, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            members_by_key[sample.step["_transaction_key"]].append(index)
        root_indices = [members[0] for members in members_by_key.values()]
        raw_advantages = advantages.clone()
        normalized = self._normalize_advantages(advantages[root_indices])
        # Every member of a transaction gets its normalised advantage, in one
        # read by each decision's place among the transactions: an assignment
        # for every transaction is a small job on the card for each of them.
        place = {key: index for index, key in enumerate(members_by_key)}
        advantages = normalized[
            torch.tensor(
                [place[sample.step["_transaction_key"]] for sample in samples],
                dtype=torch.long,
                device=self.device,
            )
        ]

        raw = raw_advantages[root_indices]
        normalized = advantages[root_indices]
        metrics: dict[str, float | str] = {
            "samples": float(len(samples)),
            "duel_samples": float(len(samples)),
            "cutoff_loss_trajectories": float(cutoff_losses),
            "cutoff_loss_steps": float(cutoff_loss_steps),
            "duel_gae_lambda": self.config.duel_gae_lambda,
            "gae_unit": GAE_UNIT,
            "policy_transactions": float(len(transaction_lengths)),
            "multi_step_policy_transactions": float(
                sum(length > 1 for length in transaction_lengths)
            ),
            "max_transaction_length": float(max(transaction_lengths, default=0)),
            "policy_loss_weight": self.config.policy_loss_weight,
            "mean_win_reward": sum(win_rewards) / max(len(win_rewards), 1),
            "duel_raw_advantage_mean": float(raw.mean()),
            "duel_raw_advantage_std": float(raw.std(unbiased=False)),
            "duel_normalized_advantage_mean": float(normalized.mean()),
            "duel_normalized_advantage_std": float(normalized.std(unbiased=False)),
            "duel_normalized_advantage_positive_rate": float(
                (normalized > 0.0).to(dtype=torch.float32).mean()
            ),
            "duel_explained_variance": self._explained_variance(all_returns, all_values),
            # The head against how its seat's duel ended, at every
            # transaction of games it has not trained on: what the returns
            # above are only when nothing of them is the head's own estimate.
            "duel_outcome_explained_variance": self._explained_variance(
                all_outcomes, all_values
            ),
            "duel_outcome_correlation": self._correlation(all_outcomes, all_values),
        }
        if self.config.upgo:
            # What UPGO added, by transaction, before the advantages are
            # normalised: a run without it reports nothing of it.
            counted = max(len(upgo_parts), 1)
            metrics["upgo_advantage_mean"] = sum(upgo_parts) / counted
            positive = sum(part > 0.0 for part in upgo_parts)
            metrics["upgo_advantage_positive_rate"] = positive / counted
        metrics.update(decision_advantage_metrics(samples, raw_advantages, advantages))
        return samples, advantages, metrics

    def imitation_update(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        samples = []
        # Terminal payoff from each labelled step's own seat. With undiscounted,
        # terminal-only rewards this is the Monte Carlo return PPO bootstraps.
        value_targets: list[float] = []
        # The deck of the seat each step's decision was asked of.
        decks: list[str] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            for step_index, step in enumerate(trajectory["steps"]):
                if not step.get("teacher"):
                    continue
                step["_transaction_key"] = self._transaction_key(
                    trajectory_index, step_index, step
                )
                samples.append(step)
                decks.append(seat_deck(trajectory, step, self.config.learner_deck))
                value_targets.append(
                    self.reward_model.terminal(trajectory.get("winner"), int(step["player"]))
                    if trajectory.get("terminal")
                    else self.reward_model.cutoff()
                )
        if not samples:
            raise RuntimeError("The imitation-only update collected no teacher actions.")
        # The first step of every transaction: where the value head has a
        # target, here as in PPO, and so where it is measured.
        starts: list[int] = []
        opened: set[object] = set()
        for index, step in enumerate(samples):
            if step["_transaction_key"] not in opened:
                opened.add(step["_transaction_key"])
                starts.append(index)
        value_weight = float(self.config.imitation_value_weight)
        # The transactions whose first step carries the duel's outcome as the
        # head's target: all of them, or so many of each seat's in a duel. A
        # duel has one outcome, and every transaction more tells the duel
        # again where it should tell a position.
        positions = int(self.config.imitation_value_positions)
        carrying = self._carrying(samples, starts, positions)
        detach_value = bool(self.config.imitation_value_detach)
        # The leads a logged search measured for the answers it tried: the
        # lead head's targets, where the update's duels carry any.
        lead_weight = float(self.config.imitation_lead_weight)
        with_leads = any(step.get("lead_targets") for step in samples)
        # A teacher that is a network: what it gives every answer is told
        # beside the model's, and is the target when the model is to learn
        # its distribution and not its choice alone.
        teachers = self.teachers
        soft = bool(teachers) and self.config.imitation_teacher_targets == "distribution"

        def forward(
            chunk: list[dict[str, Any]], places: list[int]
        ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
            if teachers:
                # One batch for the networks.
                given = self._input(chunk)
                with torch.no_grad():
                    taught = self._taught(given, [decks[place] for place in places])
                logits, values, _ = self.model.evaluate(given, detach_value=detach_value)
                return logits, values, None, taught
            if with_leads or detach_value:
                logits, values, leads = self.evaluate_steps(
                    self.model,
                    chunk,
                    detach_lead=self.config.imitation_lead_detach,
                    detach_value=detach_value,
                )
                return logits, values, leads if with_leads else None, None
            return (*self.forward_steps(self.model, chunk), None, None)

        self.model.eval()
        agreed: list[bool] = []
        # Where the game imitated left its pilot's answer, a recorded duel
        # or a teacher's own: how many decisions, and at how many the
        # model's first choice is the answer the game played or still the
        # pilot's.
        recorded = any("pilot_chosen" in step for step in samples)
        departures = departures_followed = departures_refused = 0
        predicted_values: list[float] = []
        predicted_leads: list[float] = []
        measured_leads: list[float] = []
        # How far the model's answers are from the teacher's, a decision,
        # and how far the teacher is from sure of its own.
        divergences: list[float] = []
        entropies: list[float] = []
        with torch.no_grad():
            for start in range(0, len(samples), self.config.minibatch):
                chunk = samples[start : start + self.config.minibatch]
                logits, values, leads, taught = forward(
                    chunk, list(range(start, start + len(chunk)))
                )
                if taught is not None:
                    own = self._teacher_cross_entropy(taught, taught)
                    entropies.extend(own.tolist())
                    divergences.extend((self._teacher_cross_entropy(logits, taught) - own).tolist())
                for predicted, step in zip(logits.argmax(dim=-1).tolist(), chunk):
                    agreed.append(int(predicted) == int(step["chosen"]))
                    played = int(step.get("executed_chosen", step["chosen"]))
                    pilot = step.get("pilot_chosen", played)
                    if pilot != played:
                        departures += 1
                        departures_followed += int(predicted) == played
                        departures_refused += int(predicted) == pilot
                predicted_values.extend(values.tolist())
                picked = self._lead_targets(chunk) if with_leads else None
                if picked is not None:
                    rows, columns, measured = picked
                    predicted_leads.extend(leads[rows, columns].tolist())
                    measured_leads.extend(measured.tolist())
        self.model.train()
        losses: list[float] = []
        value_losses: list[float] = []
        lead_losses: list[float] = []
        gradient_norms: dict[str, float] = {}
        for _ in range(self.config.ppo_epochs):
            for indices in self._chunks(
                [
                    PreparedSample(step, 0.0, target, target, False)
                    for step, target in zip(samples, value_targets, strict=True)
                ],
                list(range(len(samples))),
                self.config.minibatch,
            ):
                chunk = [samples[index] for index in indices]
                logits, values, leads, taught = forward(chunk, indices)
                if soft:
                    loss = self._by_transaction(self._teacher_cross_entropy(logits, taught), chunk)
                else:
                    chosen = torch.tensor([step["chosen"] for step in chunk], device=self.device)
                    loss = self._joint_imitation_loss(logits, chosen, chunk)
                losses.append(float(loss.detach()))
                picked = self._lead_targets(chunk) if with_leads else None
                if picked is not None:
                    rows, columns, measured = picked
                    lead_loss = torch.nn.functional.mse_loss(leads[rows, columns], measured)
                    lead_losses.append(float(lead_loss.detach()))
                    if not gradient_norms:
                        # Once an update, before it has moved anything: how
                        # hard each of the two losses pulls on what they share.
                        gradient_norms = {
                            "imitation_shared_gradient_norm": self._shared_gradient_norm(loss),
                            "imitation_lead_shared_gradient_norm": self._shared_gradient_norm(
                                lead_loss
                            ),
                        }
                    if lead_weight > 0.0:
                        loss = loss + lead_weight * lead_loss
                if value_weight > 0.0:
                    # One value target per causal transaction, as in PPO.
                    _, roots, root_indices = self._joint_groups(
                        [
                            {"_transaction_key": step.get("_transaction_key", ("step", index))}
                            for step, index in zip(chunk, indices, strict=True)
                        ],
                        indices,
                    )
                    if positions:
                        roots = roots[
                            torch.tensor(
                                [index in carrying for index in root_indices.tolist()],
                                dtype=torch.bool,
                            )
                        ]
                    roots = roots.to(self.device)
                    targets = torch.tensor(
                        [value_targets[index] for index in indices],
                        dtype=values.dtype,
                        device=self.device,
                    )
                    # A batch may hold none of the transactions that carry it.
                    if len(roots):
                        value_loss = torch.nn.functional.mse_loss(values[roots], targets[roots])
                        value_losses.append(float(value_loss.detach()))
                        loss = loss + value_weight * value_loss
                self._step(loss)
        departure_metrics: dict[str, float] = {}
        if recorded:
            departure_metrics["imitation_departures"] = float(departures)
        if departures:
            # Measured before this update's optimization, like agreement.
            departure_metrics["imitation_model_departure_agreement"] = (
                departures_followed / departures
            )
            departure_metrics["imitation_model_departure_pilot_agreement"] = (
                departures_refused / departures
            )
        lead_metrics: dict[str, float] = {}
        if with_leads:
            lead_metrics = {
                "imitation_lead_weight": lead_weight,
                "imitation_lead_targets": float(len(measured_leads)),
                "imitation_lead_loss": sum(lead_losses) / max(1, len(lead_losses)),
                # Measured before this update's optimization, like agreement.
                "imitation_lead_explained_variance": self._explained_variance(
                    measured_leads, predicted_leads
                ),
                **gradient_norms,
            }
        # Deck by deck, where the learner's seat plays several: the same
        # agreement, measured before this update's optimization.
        deck_metrics: dict[str, Any] = {}
        if len(set(decks)) > 1:
            by_deck: dict[str, list[bool]] = defaultdict(list)
            for deck, same in zip(decks, agreed, strict=True):
                by_deck[str(deck)].append(same)
            deck_metrics["imitation_model_teacher_agreement_by_deck"] = {
                deck: sum(found) / len(found) for deck, found in sorted(by_deck.items())
            }
            deck_metrics["imitation_samples_by_deck"] = {
                deck: float(len(found)) for deck, found in sorted(by_deck.items())
            }
        teacher_metrics: dict[str, float] = {}
        if teachers:
            # Measured before this update's optimization, like agreement: in
            # nats a decision, the divergence of the teacher's answers from
            # the model's, and the entropy of the teacher's own.
            teacher_metrics = {
                "imitation_teacher_divergence": sum(divergences) / len(divergences),
                "imitation_teacher_entropy": sum(entropies) / len(entropies),
            }
        return {
            **departure_metrics,
            **lead_metrics,
            **teacher_metrics,
            "samples": float(len(samples)),
            "imitation_loss": sum(losses) / max(1, len(losses)),
            "imitation_value_weight": value_weight,
            "imitation_value_loss": sum(value_losses) / max(1, len(value_losses)),
            "imitation_transactions": float(len(starts)),
            # The transactions that carried the outcome as the head's target.
            "imitation_value_targets": float(len(carrying)) if value_weight > 0.0 else 0.0,
            # Beside the explained variance below, which a head too sure of
            # itself loses though it ranks the positions rightly.
            "imitation_value_correlation": self._correlation(
                [value_targets[index] for index in starts],
                [predicted_values[index] for index in starts],
            ),
            # Measured before this update's optimization, like agreement, and
            # at the transactions' first steps: the head is trained on no other.
            "imitation_value_explained_variance": self._explained_variance(
                [value_targets[index] for index in starts],
                [predicted_values[index] for index in starts],
            ),
            "imitation_teacher_agreement": (
                sum(
                    int(int(step.get("executed_chosen", step["chosen"])) == int(step["chosen"]))
                    for step in samples
                )
                / len(samples)
            ),
            "imitation_model_teacher_agreement": sum(agreed) / len(samples),
            **deck_metrics,
            "warmup": 1.0,
        }

    def objective(
        self,
        samples: list[PreparedSample],
        advantages: torch.Tensor,
        indices: list[int],
        *,
        teacher_imitation_weight: float = 0.0,
        value_carrying: set[int] | None = None,
    ) -> ObjectiveBatch:
        """Evaluate exactly the objective used by production and diagnostics.

        ``value_carrying`` holds the samples at which the value head is
        taught, among the first of every transaction: all of them when left
        out (``ppo_value_positions``).
        """
        chunk = [samples[index].step for index in indices]
        card = None
        if self.card_head is None:
            logits, values = self.forward_steps(self.model, chunk)
            actions = torch.tensor([step["chosen"] for step in chunk], device=self.device)
        else:
            # The network's own parts, which the head that learns how cards
            # work reads, from the same pass.
            batch = self._input(chunk)
            logits, values, state, tokens, action = self.model.parts(batch)
            actions = torch.tensor([step["chosen"] for step in chunk], device=self.device)
            labels = labels_of(batch)
            if labels is not None:
                weight = self.card_head_weight
                answers = scaled(chosen_answers(action, actions), weight)
                card = card_loss(
                    self.card_head, scaled(state, weight), scaled(tokens, weight), answers, labels
                )
        old = torch.tensor([step["log_prob"] for step in chunk], device=self.device)
        # What the value head is taught: PPO's return, or the duel's result.
        result = self.config.ppo_value_target == "result"
        returns = torch.tensor(
            [
                samples[index].terminal_return if result else samples[index].return_
                for index in indices
            ],
            device=self.device,
        )
        ppo_config = PpoConfig(
            clip_epsilon=self.config.clip_epsilon,
            entropy_coefficient=self.config.entropy_coefficient,
            rollout_temperature=self.config.rollout_temperature,
        )
        group_ids, root_positions, root_indices = self._joint_groups(chunk, indices)
        value_roots = None
        if value_carrying is not None:
            value_roots = torch.tensor(
                [
                    place
                    for place, index in enumerate(root_indices.tolist())
                    if index in value_carrying
                ],
                dtype=torch.long,
                device=self.device,
            )
        group_ids = group_ids.to(self.device)
        root_positions = root_positions.to(self.device)
        root_indices = root_indices.to(self.device)
        components = joint_ppo_loss_components(
            logits,
            actions,
            old,
            group_ids,
            root_positions,
            advantages[root_indices],
            returns[root_positions],
            values,
            ppo_config,
            value_roots=value_roots,
        )
        if self.config.policy_loss_weight != 1.0:
            components["total"] = (
                self.config.policy_loss_weight * components["policy"]
                + components["weighted_value"]
                + components["weighted_entropy"]
            )
        teacher_targets = [
            (position, step["teacher_chosen"])
            for position, step in enumerate(chunk)
            if "teacher_chosen" in step
        ]
        imitation_loss = None
        if teacher_targets and teacher_imitation_weight > 0.0:
            positions = torch.tensor([item[0] for item in teacher_targets], device=self.device)
            targets = torch.tensor([item[1] for item in teacher_targets], device=self.device)
            teacher_steps = [chunk[int(position)] for position in positions]
            imitation_loss = self._joint_imitation_loss(
                logits[positions], targets, teacher_steps
            )
        return ObjectiveBatch(
            components,
            imitation_loss,
            len(teacher_targets),
            logits,
            values,
            chunk,
            None if card is None else card[0],
            None if card is None else card[1],
        )

    def forward_steps(
        self,
        model: torch.nn.Module,
        steps: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The model's answers to ``steps``, read where their decisions are kept."""
        return model(self._input(steps))

    def evaluate_steps(
        self,
        model: torch.nn.Module,
        steps: list[dict[str, Any]],
        *,
        detach_lead: bool = False,
        detach_value: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same path, with the lead predicted for every answer."""
        return model.evaluate(
            self._input(steps), detach_lead=detach_lead, detach_value=detach_value
        )

    def _input(self, steps: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """The batch the model is given for ``steps``, built where their
        decisions are kept."""
        if self.kept is None:
            raise RuntimeError("these decisions are kept in a pool the optimizer was not given")
        rows = [int(step["row"]) for step in steps]
        ahead = self._ahead.pop(tuple(rows), None)
        if ahead is None and self._builder is not None:
            # In a pass the pool is asked by the one thread that builds ahead.
            ahead = self._builder.submit(self.kept, rows)
        return self.kept(rows) if ahead is None else ahead.result()

    def _lead_targets(
        self, chunk: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Where in a batch's answers a lead was measured, and the leads."""
        rows: list[int] = []
        columns: list[int] = []
        measured: list[float] = []
        for row, step in enumerate(chunk):
            for answer, lead in step.get("lead_targets", ()):
                rows.append(row)
                columns.append(int(answer))
                measured.append(float(lead))
        if not rows:
            return None
        return (
            torch.tensor(rows, dtype=torch.long, device=self.device),
            torch.tensor(columns, dtype=torch.long, device=self.device),
            torch.tensor(measured, dtype=torch.float32, device=self.device),
        )

    def _shared_gradient_norm(self, loss: torch.Tensor) -> float:
        """The size of a loss's gradient on the parameters every head reads through."""
        shared = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith(HEAD_PARAMETERS)
        ]
        gradients = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        return math.sqrt(
            sum(float((gradient**2).sum()) for gradient in gradients if gradient is not None)
        )

    @staticmethod
    def _joint_groups(
        chunk: list[dict[str, Any]], indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys: dict[object, int] = {}
        group_ids: list[int] = []
        roots: list[int] = []
        root_indices: list[int] = []
        for position, (step, sample_index) in enumerate(zip(chunk, indices)):
            key = step["_transaction_key"]
            group = keys.get(key)
            if group is None:
                group = len(keys)
                keys[key] = group
                roots.append(position)
                root_indices.append(sample_index)
            group_ids.append(group)
        return (
            torch.tensor(group_ids, dtype=torch.long),
            torch.tensor(roots, dtype=torch.long),
            torch.tensor(root_indices, dtype=torch.long),
        )

    @staticmethod
    def _joint_imitation_loss(
        logits: torch.Tensor, chosen: torch.Tensor, steps: list[dict[str, Any]]
    ) -> torch.Tensor:
        losses = torch.nn.functional.cross_entropy(logits, chosen, reduction="none")
        return PpoOptimizer._by_transaction(losses, steps)

    def _taught(self, given: dict[str, torch.Tensor], decks: list[str]) -> torch.Tensor:
        """What the teacher of each decision's deck gives its answers: a batch
        of one deck's decisions is given to its teacher whole."""
        named = sorted(set(decks))
        if len(named) == 1:
            taught, _ = self._teacher_of(named[0])(given)
            return taught
        found: torch.Tensor | None = None
        for deck in named:
            rows = torch.tensor(
                [row for row, own in enumerate(decks) if own == deck], device=self.device
            )
            part, _ = self._teacher_of(deck)({name: value[rows] for name, value in given.items()})
            if found is None:
                found = part.new_empty((len(decks), *part.shape[1:]))
            found[rows] = part
        assert found is not None
        return found

    def _teacher_of(self, deck: str) -> torch.nn.Module:
        if deck not in self.teachers:
            raise RuntimeError(f"No teacher plays {deck}")
        return self.teachers[deck]

    @staticmethod
    def _teacher_cross_entropy(logits: torch.Tensor, taught: torch.Tensor) -> torch.Tensor:
        """A decision's cross-entropy of the answers ``logits`` gives against
        the probabilities a teacher's ``taught`` gives them: its entropy when
        the two are the same, and more than it by their divergence."""
        offered = torch.isfinite(taught)
        log_probabilities = torch.log_softmax(logits, dim=-1).masked_fill(~offered, 0.0)
        return -(torch.softmax(taught, dim=-1) * log_probabilities).sum(dim=-1)

    @staticmethod
    def _by_transaction(losses: torch.Tensor, steps: list[dict[str, Any]]) -> torch.Tensor:
        """One loss for the steps of a batch: a transaction's is the sum of
        its steps', and the batch's the mean of its transactions'."""
        keys: dict[object, int] = {}
        group_ids: list[int] = []
        for index, step in enumerate(steps):
            key = step.get("_transaction_key", ("step", index))
            group_ids.append(keys.setdefault(key, len(keys)))
        ids = torch.tensor(group_ids, dtype=torch.long, device=losses.device)
        grouped = torch.zeros(len(keys), dtype=losses.dtype, device=losses.device)
        grouped.index_add_(0, ids, losses)
        return grouped.mean()

    def _chunks(
        self, samples: list[PreparedSample], indices: list[int], size: int
    ) -> list[list[int]]:
        units: dict[object, list[int]] = {}
        for index in indices:
            key = samples[index].step.get("_transaction_key", ("sample", index))
            units.setdefault(key, []).append(index)
        unit_values = list(units.values())
        generator = torch.Generator(device="cpu")
        shuffle_seed = (
            int(self.config.seed) * 1_000_003
            + self._shuffle_update * 10_007
            + self._shuffle_sequence
        ) % (2**63 - 1)
        self._shuffle_sequence += 1
        generator.manual_seed(shuffle_seed)
        order = torch.randperm(len(unit_values), generator=generator).tolist()
        shuffled = [unit_values[position] for position in order]
        chunks: list[list[int]] = []
        current: list[int] = []
        for unit in shuffled:
            if current and len(current) + len(unit) > size:
                chunks.append(current)
                current = []
            current.extend(unit)
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _normalize_advantages(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)

    @staticmethod
    def _explained_variance(returns: list[float], values: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        targets = torch.tensor(returns)
        predictions = torch.tensor(values)
        variance = targets.var(unbiased=False)
        if float(variance) <= 1e-12:
            return 0.0
        return float(1.0 - (targets - predictions).var(unbiased=False) / variance)

    @staticmethod
    def _correlation(returns: list[float], values: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        targets = torch.tensor(returns)
        predictions = torch.tensor(values)
        spread = targets.std(unbiased=False) * predictions.std(unbiased=False)
        if float(spread) <= 1e-12:
            return 0.0
        centred = (targets - targets.mean()) * (predictions - predictions.mean())
        return float(centred.mean() / spread)

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        if self.card_head_optimizer is not None:
            self.card_head_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        # The head's own parameters, where a minibatch taught it.
        if self.card_head_optimizer is not None and self.card_head is not None:
            taught = [p for p in self.card_head.parameters() if p.grad is not None]
            if taught:
                torch.nn.utils.clip_grad_norm_(taught, 1.0)
                self.card_head_optimizer.step()
