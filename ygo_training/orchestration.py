"""The games of an update: which are played, who sits where, what is told."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from .config import TrainerConfig
from .records import RecordedGame, update_games
from .rewards import RewardModel
from .schedules import (
    TrainingSchedule,
    allocate_balanced_matchups,
    allocate_balanced_opponents,
    allocate_episodes,
    allocate_matchup_seats,
    allocate_other_seats,
    episode_offsets,
    rollout_games,
)

# Who plays an update's games, given them: the pilot, or a network that is
# imitated (its trajectories and how it was asked), or the model, drawing
# its answers (given also the rollout's seed and whether the pilot's answers
# are kept beside its own); and recorded duels played again.
Played = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
Asked = Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, float]]]
Drawn = Callable[[list[dict[str, Any]], int, bool], tuple[list[dict[str, Any]], dict[str, float]]]
# Or the model, drawing its answers in duels kept in flight from one update
# to the next (`pool.Stream`): given what gives the games of a rollout and
# its seed, the rollout of the update, and whether the pilot's answers are
# kept beside its own.
Blocks = Callable[[int], tuple[list[dict[str, Any]], int]]
Streamed = Callable[[Blocks, int, bool], tuple[list[dict[str, Any]], dict[str, float]]]


def evaluation_assignments(
    opponents: list[str], games_per_opponent: int, seed: int
) -> list[dict[str, Any]]:
    """The games of an evaluation: against each opponent as many, the seats
    in turn, each opponent on seeds of its own."""
    assignments: list[dict[str, Any]] = []
    for opponent_index, opponent in enumerate(opponents):
        for game in range(games_per_opponent):
            assignments.append(
                {
                    "index": len(assignments),
                    "opponent": opponent,
                    "learner_player": game % 2,
                    "seed": seed + opponent_index * 10_000 + game,
                }
            )
    return assignments


class Coordinator:
    """The games of each update: which deals, which decks, who sits in the
    other seat, and what is told of them.

    It plays nothing itself: what it is given does, through the library's
    pool, and it owns no model optimization.
    """

    def __init__(
        self,
        *,
        config: TrainerConfig,
        reward_model: RewardModel,
        opponents: list[str],
        pilot_games: Played,
        learner_games: Drawn,
        teacher_games: Asked | None = None,
        replayed_games: Played | None = None,
        recorded: list[RecordedGame] | None = None,
        learner_stream: Streamed | None = None,
    ) -> None:
        self.config = config
        # The decks the learner plays: one, or several in equal shares.
        self.learners = config.learners
        # The decks the learner meets in rollouts, in equal shares.
        self.opponents = list(opponents)
        # Recorded duels the imitation-only updates play again, when given,
        # in place of duels the pilot plays now.
        self.recorded = recorded
        self.reward_model = reward_model
        self.schedule = TrainingSchedule(seed=config.seed)
        # The games of an imitation-only update that the pilot plays now
        # (``schedules.rollout_games``), or a network in the pilot's place,
        # the one imitated; the games of every other update, with the model
        # in the learner's seat; and recorded duels, where they are imitated.
        self.pilot_games = pilot_games
        self.teacher_games = teacher_games
        self.learner_games = learner_games
        self.replayed_games = replayed_games
        # Where duels are kept in flight from one update to the next, what
        # the model's updates are played with.
        self.learner_stream = learner_stream

    def set_rollout_index(self, index: int) -> None:
        self.schedule.set_rollout_index(index)

    def set_training_schedule_index(self, index: int) -> None:
        self.schedule.set_update_index(index)

    def collect(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rollout_index, rollout_seed, imitation_weight, imitation_only = self.schedule.next(
            teacher_weight=self.config.teacher_imitation_weight,
            teacher_decay_updates=self.config.teacher_imitation_decay_updates,
            warmup_updates=self.config.imitation_only_updates,
        )
        # The update's recorded duels, when its games are not played now.
        replayed: list[dict[str, Any]] | None = None
        first_recorded = 0
        if imitation_only and self.recorded is not None:
            first_recorded = (self.schedule.update_index - 1) * self.config.duels
            picks = update_games(
                len(self.recorded),
                self.config.duels,
                self.schedule.update_index - 1,
                self.config.seed,
            )
            replayed = [self.recorded[pick].replay(place) for place, pick in enumerate(picks)]
        started = time.perf_counter()
        asked = {"prompts": 0.0, "batches": 0.0, "max_batch": 0.0, "inference_seconds": 0.0}
        if replayed is not None:
            if self.replayed_games is None:
                raise ValueError("recorded duels are given and nothing plays them again")
            # Each in its place among the update's games and the share it is
            # dealt in, as the games played now are (``rollout_games``).
            counts = allocate_episodes(self.config.workers, len(replayed))
            for share, (offset, count) in enumerate(
                zip(episode_offsets(counts), counts, strict=True)
            ):
                for place in range(offset, offset + count):
                    replayed[place] = {**replayed[place], "episode_index": place, "worker": share}
            trajectories = self.replayed_games(replayed)
        elif not imitation_only and self.learner_stream is not None:
            labels = imitation_weight > 0.0

            def blocks(index: int) -> tuple[list[dict[str, Any]], int]:
                seed = self.schedule.rollout_seed(index)
                return self._games(seed, imitation_only=False), seed

            trajectories, asked = self.learner_stream(blocks, rollout_index, labels)
        else:
            games = self._games(rollout_seed, imitation_only=imitation_only)
            if imitation_only and self.teacher_games is not None:
                trajectories, asked = self.teacher_games(games)
            elif imitation_only:
                trajectories = self.pilot_games(games)
            else:
                labels = imitation_weight > 0.0
                trajectories, asked = self.learner_games(games, rollout_seed, labels)
        metrics = self._rollout_metrics(
            trajectories,
            int(asked["prompts"]),
            int(asked["batches"]),
            int(asked["max_batch"]),
            asked["inference_seconds"],
            time.perf_counter() - started,
            imitation_weight,
            self.schedule.update_index - 1,
        )
        # Where the time of the games went, beside the model's, and how many
        # decisions a held checkpoint answered.
        metrics.update(
            {
                name: value
                for name, value in asked.items()
                if name.startswith(("pool_", "stream_")) or name == "held_decisions"
            }
        )
        if self.config.self_play_weights is not None and not imitation_only:
            # The weights the duels against itself were shared out by.
            metrics["self_play_weights"] = dict(self.config.self_play_weights)
        if replayed is not None and self.recorded is not None:
            metrics["recorded_games"] = float(len(self.recorded))
            # How many times the games had all been gone through before.
            metrics["recorded_pass"] = float(first_recorded // len(self.recorded))
            metrics["recorded_departures"] = float(
                sum(int(trajectory["departures"]) for trajectory in trajectories)
            )
        return trajectories, metrics

    def _games(self, rollout_seed: int, *, imitation_only: bool) -> list[dict[str, Any]]:
        """The games of the rollout of ``rollout_seed``: which deals, which
        decks, who sits in the other seat."""
        # The deck the learner plays in each game, where it plays several,
        # and who sits in the other seat, where a network does in some.
        learner_assignments: list[str] | None = None
        other_seats: list[str] | None = None
        # Duels against itself and against a held checkpoint are of the
        # updates whose games the model plays; and against itself of those
        # a network teaches, which then plays both seats.
        taught = imitation_only and self.teacher_games is not None
        itself = self.config.self_play_duels if taught or not imitation_only else 0
        held = 0 if imitation_only else self.config.opponent_checkpoint_duels
        if itself and len(self.learners) > 1:
            # A learner of several decks against itself: any two of its
            # decks, one in each seat.
            learner_assignments, opponent_assignments, other_seats = allocate_matchup_seats(
                self.learners,
                self.opponents,
                self.config.duels,
                itself=itself,
                seed=rollout_seed ^ 0x4F50504F,
                pairs=self.config.self_play_pairs,
                weights=self.config.pair_weights,
            )
        elif itself or held:
            opponent_assignments, other_seats = allocate_other_seats(
                allocate_balanced_opponents(
                    self.opponents,
                    self.config.duels - itself - held,
                    seed=rollout_seed ^ 0x4F50504F,
                ),
                self.config.learner_deck,
                itself=itself,
                checkpoint=held,
                seed=rollout_seed ^ 0x4F50504F,
            )
        elif len(self.learners) > 1:
            matchups = allocate_balanced_matchups(
                self.learners,
                self.opponents,
                self.config.duels,
                seed=rollout_seed ^ 0x4F50504F,
            )
            learner_assignments = [learner for learner, _ in matchups]
            opponent_assignments = [opponent for _, opponent in matchups]
        else:
            opponent_assignments = allocate_balanced_opponents(
                self.opponents,
                self.config.duels,
                seed=rollout_seed ^ 0x4F50504F,
            )
        return rollout_games(
            rollout_seed,
            self.config.workers,
            opponent_assignments,
            learner_assignments,
            other_seats,
        )

    def _rollout_metrics(
        self,
        trajectories: list[dict[str, Any]],
        prompt_count: int,
        inference_batches: int,
        max_batch: int,
        inference_seconds: float,
        elapsed: float,
        imitation_weight: float,
        update_index: int,
    ) -> dict[str, Any]:
        finished = [item for item in trajectories if item["terminal"]]
        won = [item for item in finished if item["winner"] == item["learner_player"]]
        deckout_losses = sum(
            item["win_reason"] == 2 and item["winner"] != item["learner_player"]
            for item in finished
        )
        deckout_wins = sum(
            item["win_reason"] == 2 and item["winner"] == item["learner_player"]
            for item in finished
        )
        deckout_draws = sum(
            item["win_reason"] == 2 and item["winner"] not in (0, 1) for item in finished
        )
        rewards = [
            self.reward_model.terminal(trajectory["winner"], int(trajectory["learner_player"]))
            if trajectory["terminal"]
            else self.reward_model.cutoff()
            for trajectory in trajectories
        ]
        metrics = {
            "duels": float(len(trajectories)),
            "decisions": float(prompt_count),
            "seconds": elapsed,
            "duels_per_second": len(trajectories) / max(elapsed, 1e-9),
            "decisions_per_second": prompt_count / max(elapsed, 1e-9),
            "inference_batches": float(inference_batches),
            "mean_inference_batch": (prompt_count / max(inference_batches, 1)),
            "max_inference_batch": float(max_batch),
            "inference_seconds": inference_seconds,
            "learner_finished_duels": float(len(finished)),
            "learner_win_rate": (len(won) / max(len(finished), 1)),
            "learner_deckout_losses": float(deckout_losses),
            "learner_deckout_wins": float(deckout_wins),
            "learner_deckout_draws": float(deckout_draws),
            "mean_terminal_turns": (
                sum(item["turns"] for item in finished) / max(len(finished), 1)
            ),
            "mean_learner_win_turns": (sum(item["turns"] for item in won) / max(len(won), 1)),
            "learner_total_reward": sum(rewards),
            "learner_mean_reward": (sum(rewards) / max(len(rewards), 1)),
            "rewarded_learner_duels": float(len(rewards)),
            "teacher_imitation_weight": imitation_weight,
            "training_schedule_update": float(update_index),
        }
        opponent_counts = Counter(str(item["opponent_deck"]) for item in trajectories)
        metrics["opponent_deck_duels"] = {
            opponent: float(opponent_counts[opponent]) for opponent in sorted(opponent_counts)
        }
        # By who sat in the other seat, where a network did in some duels.
        # Against itself the learner's seat is one of its two, and says
        # nothing: what is told there is how often the seat that opens wins.
        if any("other" in item for item in trajectories):
            seated = Counter(str(item.get("other", "pilot")) for item in trajectories)
            ended_by = Counter(str(item.get("other", "pilot")) for item in finished)
            won_by = Counter(str(item.get("other", "pilot")) for item in won)
            metrics["other_seat_duels"] = {kind: float(seated[kind]) for kind in sorted(seated)}
            metrics["other_seat_win_rate"] = {
                kind: won_by[kind] / max(ended_by[kind], 1) for kind in sorted(seated)
            }
            mirrors = [item for item in finished if item.get("other") == "self"]
            if mirrors:
                metrics["self_play_first_seat_win_rate"] = sum(
                    item["winner"] == 0 for item in mirrors
                ) / len(mirrors)
                metrics["self_play_mean_turns"] = sum(item["turns"] for item in mirrors) / len(
                    mirrors
                )
        # Deck by deck, where the learner plays several.  Of the duels
        # against a pilot: in a duel against itself the learner's seat is
        # one of its two, and says nothing of the deck there.
        decks = {str(item.get("learner_deck")) for item in trajectories}
        if len(decks) > 1:

            def piloted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [item for item in items if item.get("other", "pilot") != "self"]

            played = Counter(str(item.get("learner_deck")) for item in piloted(trajectories))
            ended = Counter(str(item.get("learner_deck")) for item in piloted(finished))
            taken = Counter(str(item.get("learner_deck")) for item in piloted(won))
            metrics["learner_deck_duels"] = {deck: float(played[deck]) for deck in sorted(decks)}
            metrics["learner_deck_win_rate"] = {
                deck: taken[deck] / max(ended[deck], 1) for deck in sorted(decks)
            }
            # Against itself each duel has two of the learner's decks, one a
            # seat: the seats a deck took, and how many of them won.
            against_itself = [item for item in finished if item.get("other") == "self"]
            if against_itself:
                seats: Counter[str] = Counter()
                wins: Counter[str] = Counter()
                for item in against_itself:
                    own, other = str(item["learner_deck"]), str(item["opponent_deck"])
                    seats[own] += 1
                    seats[other] += 1
                    if item["winner"] in (0, 1):
                        wins[own if item["winner"] == item["learner_player"] else other] += 1
                metrics["self_play_deck_seats"] = {
                    deck: float(seats[deck]) for deck in sorted(seats)
                }
                metrics["self_play_deck_win_rate"] = {
                    deck: wins[deck] / seats[deck] for deck in sorted(seats)
                }
        return metrics
