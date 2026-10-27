"""The PPO trainer's composition root."""

from __future__ import annotations

import atexit
import copy
import os
import random
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from .card_head import PREFIX, CardHead
from .card_table import CardTable
from .champion_records import Recordings
from .checkpoints import (
    load_beside,
    load_inference_model,
    save,
    value_fingerprint,
    write,
    write_beside,
)
from .config import TrainerConfig
from .critic import SightedCritic, drawn, transaction_roots
from .environment import pool
from .graphs import Graphs
from .model import PolicyValueNet
from .optimization import DIAGNOSTIC_STEP_KEYS, PpoOptimizer
from .orchestration import Coordinator, evaluation_assignments
from .pool import (
    DuelPool,
    Locked,
    Stream,
    imitation_games,
    replayed_games,
    sampled_games,
    taught_games,
    watched_games,
)
from .pool import evaluate_games as pooled_games
from .records import RecordedGame, recorded_games
from .rewards import OutcomeRewardModel
from .trainer import Trainer

# The arrays of the model's input that hold cards by their codes.
CARD_ARRAYS = ("state_card", "context_card", "action_card", "action_member_card", "history_card")
# How many decisions the critic values at once, without a gradient.
CRITIC_CHUNK = 1024
# The head that learns how cards work begins anew, and learns at the rate
# the critic does, whatever the network's.
CARD_HEAD_LEARNING_RATE = 3e-4


def configure_reproducible_torch(requested_device: str) -> None:
    """Require deterministic optimization kernels for reproducible branches."""
    if requested_device == "cuda":
        # CUDA reads this when creating cuBLAS handles. Set it before the first
        # model or optimizer tensor reaches the device.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PpoTrainer(Trainer):
    """Compose the library's pool of duels, the games of each update and PPO
    optimization.  Every game, an update's and an evaluation's, is played
    through the pool."""

    def __init__(self, root: Path, config: TrainerConfig):
        self.root = root.resolve()
        self.config = config
        configure_reproducible_torch(config.device)
        self.device = (
            config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.model = PolicyValueNet(
            board_representation=config.board_representation,
            leave_out=config.leave_out,
            cards=None if config.card_table is None else CardTable.read(config.card_table),
            descriptions=config.card_descriptions,
            hide_identities=config.hide_identities,
        ).to(self.device)
        # Of a model that hides identities, in the update under way: the
        # cards of its table hidden, and those it has been trained on.
        self._hidden: np.ndarray | None = None
        self._trained: np.ndarray | None = None
        self._codes: np.ndarray | None = None
        if self.model.hides_identities:
            self._codes = self.model.card_codes.cpu().numpy()
        self.initial_model_fingerprint = value_fingerprint(self.model.state_dict())
        # How the model is asked in an update's games: what the run named,
        # and for the rest the way of the device the games are answered on.
        self.calls = config.calls(self.device)
        # The calls of them that are answered at sizes fixed in advance, from
        # CUDA graphs on a card.
        self.graphs: Graphs | None = None
        self._graphs_failure_told = False
        if self.calls.graphs:
            self.graphs = Graphs(self.model, self.device, largest=self.calls.graphs)
        # The checkpoints the imitation-only updates imitate in the pilot's
        # place, when they are named, one for each of the learner's decks,
        # and what a run says of them: of a learner of one deck, its
        # teacher's alone.
        self.teachers: dict[str, PolicyValueNet] = {}
        self.teacher_description: dict[str, Any] | None = None
        if config.teachers:
            # A network draws its first parameters when it is built: the
            # generator is left where the learner's left it, and the run goes
            # on as one without a teacher would.
            drawn = torch.get_rng_state()
            described: dict[str, dict[str, Any]] = {}
            for deck, named in config.teachers.items():
                path = Path(named).resolve()
                teacher, saved = load_inference_model(path, device=self.device)
                teacher.requires_grad_(False)
                self.teachers[deck] = teacher
                described[deck] = {
                    "path": str(path),
                    "step": saved.get("step"),
                    "fingerprint": value_fingerprint(teacher.state_dict()),
                    "board_representation": teacher.board_representation,
                    "targets": config.imitation_teacher_targets,
                }
            torch.set_rng_state(drawn)
            self.teacher_description = (
                described[config.learner_deck] if config.imitation_teacher else described
            )
        # The teachers' calls answered from CUDA graphs, and their duels taken
        # forward by the pool's groups, where the run names its graphs and its
        # groups: a copy keeps its earlier calls otherwise, as the teachers'
        # first choices can then go another way where two answers score the
        # same to their last digits.
        self.teacher_graphs: dict[str, Graphs] = {}
        if self.teachers and config.inference_graphs:
            self.teacher_graphs = {
                deck: Graphs(teacher, self.device, largest=config.inference_graphs)
                for deck, teacher in self.teachers.items()
            }
        self._teacher_graphs_failure_told = False
        # The checkpoint held as it is for the other seat, when one is named:
        # in duels of an update and in the cases of a suite that ask for it.
        self.held: PolicyValueNet | None = None
        self.held_description: dict[str, Any] | None = None
        if config.opponent_checkpoint:
            # As with a teacher: the generator is left where the learner's
            # left it.
            drawn = torch.get_rng_state()
            path = Path(config.opponent_checkpoint).resolve()
            # It plays the other seat: its decks are the opponents', the
            # other seat's the learner's.
            pairings = (
                None
                if config.opponent_decks is None
                else {
                    (theirs, ours) for theirs in config.opponent_decks for ours in config.learners
                }
            )
            self.held, saved = load_inference_model(path, device=self.device, pairings=pairings)
            torch.set_rng_state(drawn)
            self.held.requires_grad_(False)
            self.held_description = {
                "path": str(path),
                "step": saved.get("step"),
                "fingerprint": value_fingerprint(self.held.state_dict()),
                "board_representation": self.held.board_representation,
            }
            folded = saved.get("metadata", {}).get("folded_deck_lists")
            if folded is not None:
                self.held_description["folded_deck_lists"] = folded
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.reward_model = OutcomeRewardModel()
        self.policy_optimizer = PpoOptimizer(
            self.model,
            self.optimizer,
            config,
            self.reward_model,
            self.device,
        )
        # Policy id -> name of its deck list.
        self.decks = pool(self.root)
        for name in [*config.learners, *(config.opponent_decks or [])]:
            if name not in self.decks:
                raise ValueError(f"Unknown deck: {name}")
        # The decks the learner meets in rollouts: every other one unless told.
        self.opponents = config.opponent_decks or sorted(
            name for name in self.decks if name != config.learner_deck
        )
        # The recorded duels the imitation-only updates imitate, if any.
        self.recorded: list[RecordedGame] | None = None
        if config.imitation_records:
            self.recorded = recorded_games(
                config.imitation_records,
                config.imitation_game,
                learner=config.learner_deck,
                games=config.imitation_games,
            )
            for game in self.recorded:
                if game.opponent not in self.decks:
                    raise ValueError(f"Unknown deck in the recorded games: {game.opponent}")
        # The champions' recordings a copy plays its games again from, where
        # it does: every deck the learner plays has one.
        self.copy_records: Recordings | None = None
        if config.copy_records:
            self.copy_records = Recordings(Path(config.copy_records))
            self.copy_records.teachers(config.learners)
        self.coordinator = Coordinator(
            config=config,
            reward_model=self.reward_model,
            opponents=self.opponents,
            pilot_games=self._pilot_games,
            learner_games=self._learner_games,
            teacher_games=self._teacher_games if self.teachers else None,
            replayed_games=None if self.recorded is None else self._replayed_games,
            recorded=self.recorded,
            learner_stream=self._streamed_games if config.duels_in_flight else None,
            copy_records=self.copy_records,
            recorded_copy_games=None if self.copy_records is None else self._recorded_copy_games,
        )
        # The duels kept in flight from one update to the next, once the
        # model plays an update's games.
        self._stream: Stream | None = None
        self._rollout_index = 0
        self._schedule_index = 0
        self._closed = False
        # The library's pool, opened when games first go through it: its
        # calls made one at a time where the next update's games are played
        # beside the pass.
        self._duel_pool: DuelPool | Locked | None = None
        # The next update's games, played beside an update's pass where the
        # run says so (`TrainerConfig.overlap`): under way or played; the
        # copy of the model that plays them, with the weights from before
        # the pass, and the model the games being played are answered by;
        # the thread that plays them beside the pass, and on a card the
        # stream its calls go on; and when the last pass began and ended.
        self._ahead: Future[tuple[list[dict], dict[str, float], tuple[float, float]]] | None = None
        self._player: PolicyValueNet | None = None
        self._playing_with: PolicyValueNet | None = None
        self._beside = ThreadPoolExecutor(max_workers=1) if config.overlap == "thread" else None
        self._player_stream: torch.cuda.Stream | None = None
        self._last_pass: tuple[float, float] | None = None
        # How long an update's minibatches took to come from the pool, where
        # its decisions are kept.
        self._kept_seconds = 0.0
        self.policy_optimizer.kept = self._kept
        self.policy_optimizer.teachers = self.teachers
        # The critic that sees the hidden cards, where the run has one: made
        # once the model is as the run begins (`begin_critic`).
        self.critic: SightedCritic | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.critic_description: dict[str, Any] | None = None
        if config.critic:
            self.critic_description = {
                "deck": config.critic_deck,
                "positions": config.critic_positions,
                "learning_rate": config.critic_learning_rate,
                "minibatch": config.minibatch,
                "begun_as": "the value head",
            }
        # The head that learns how cards work, where the run has one: made
        # once the model is as the run begins (`begin_card_head`).
        self.card_head: CardHead | None = None
        self.card_head_description: dict[str, Any] | None = None
        if config.card_head:
            self.card_head_description = {
                "weight": config.card_head_weight,
                "learning_rate": CARD_HEAD_LEARNING_RATE,
                "begun": config.card_head_begun,
            }
        atexit.register(self.close)

    def close(self, force: bool = False) -> None:
        """Close the pool; ``force`` is kept for callers that stop on an
        interruption, which ends its duels all the same."""
        if self._closed:
            return
        # The next update's games, where they are under way, end before
        # their pool does.
        if self._ahead is not None:
            # Waited for, not taken: what they leave, games or the error that
            # stopped them, goes with the pool.
            self._ahead.exception()
            self._ahead = None
        if self._beside is not None:
            self._beside.shutdown(wait=True)
        self._closed = True
        if self._duel_pool is not None:
            self._duel_pool.close()
            self._duel_pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        self.close(force=exception_type is KeyboardInterrupt)

    def begin_critic(self, beside: Path | None = None) -> None:
        """Make the run's critic, where it has one (`TrainerConfig.critic`):
        begun as the model's value head as the model now is, or, for a run
        that goes on, as it was kept beside the checkpoint ``beside``."""
        if not self.config.critic:
            return
        # As with a teacher: building it draws parameters, and the generator
        # is left where the learner's left it.
        drawn_state = torch.get_rng_state()
        critic = SightedCritic.from_policy(self.model, warm=True).to(self.device)
        torch.set_rng_state(drawn_state)
        optimizer = torch.optim.Adam(critic.parameters(), lr=self.config.critic_learning_rate)
        if beside is not None:
            load_beside(beside, "critic", critic, optimizer, map_location=self.device)
        self.critic, self.critic_optimizer = critic, optimizer

    def begin_card_head(self, beside: Path | None = None) -> None:
        """Make the run's head that learns how cards work, where it has one
        (`TrainerConfig.card_head`): begun anew or as a probe left it
        (`card_head_begun`), or, for a run that goes on, as it was kept
        beside the checkpoint ``beside``.  From here an update's minibatches
        come with their labels."""
        if not self.config.card_head:
            return
        # Building it draws parameters: the generator is left where the
        # learner's left it.
        drawn_state = torch.get_rng_state()
        head = CardHead(self.model).to(self.device)
        torch.set_rng_state(drawn_state)
        optimizer = torch.optim.Adam(head.parameters(), lr=CARD_HEAD_LEARNING_RATE)
        if beside is not None:
            load_beside(beside, "card-head", head, optimizer, map_location=self.device)
        elif self.config.card_head_begun:
            begun = torch.load(
                self.config.card_head_begun, map_location=self.device, weights_only=False
            )
            head.load_state_dict(begun["card-head"])
        self.card_head = head
        self.policy_optimizer.card_head = head
        self.policy_optimizer.card_head_optimizer = optimizer
        self.policy_optimizer.card_head_weight = self.config.card_head_weight
        self.policy_optimizer.kept = self._kept_taught

    def _kept_taught(self, rows: list[int]) -> dict[str, torch.Tensor]:
        """Decisions kept in the pool as `_kept` gives them, with what became
        of their cards beside them under `card_head.PREFIX`: what the head
        that learns how cards work is taught, never what the network reads."""
        batch = self._kept(rows)
        started = time.perf_counter()
        labels = {
            f"{PREFIX}{name}": torch.from_numpy(array)
            for name, array in self._pool().labels(rows).items()
        }
        self._kept_seconds += time.perf_counter() - started
        if self.device != "cpu":
            labels = {
                name: value.to(self.device, non_blocking=True) for name, value in labels.items()
            }
        return {**batch, **labels}

    def _credit(self, trajectories: list[dict[str, Any]], step: int) -> dict[str, float]:
        """The critic's part of an update: its value at the first decision of
        every transaction the update trains on, told before it is taught
        these games, which the advantages are measured against in the value
        head's place (``critic_value``); then it is taught the results of so
        many of each seat's transactions a duel, drawn for the update, in one
        pass of minibatches."""
        assert self.critic is not None and self.critic_optimizer is not None
        started = time.perf_counter()
        found = transaction_roots(trajectories, self.reward_model)
        self.critic.eval()
        values: list[float] = []
        with torch.no_grad():
            for start in range(0, len(found), CRITIC_CHUNK):
                rows = [int(root.step["row"]) for root in found[start : start + CRITIC_CHUNK]]
                values += self.critic(self._kept(rows), self._kept_whole(rows)).tolist()
        for root, value in zip(found, values, strict=True):
            root.step["critic_value"] = value
        results = [root.result for root in found]
        heads = [float(root.step["value"]) for root in found]
        generator = random.Random(
            (int(self.config.seed) * 1_000_003 + int(step) * 10_007) ^ 0x43524954
        )
        taught = drawn(found, self.config.critic_positions, generator)
        self.critic.train()
        losses: list[torch.Tensor] = []
        for start in range(0, len(taught), self.config.minibatch):
            part = taught[start : start + self.config.minibatch]
            rows = [int(root.step["row"]) for root in part]
            value = self.critic(self._kept(rows), self._kept_whole(rows))
            target = torch.tensor(
                [root.result for root in part], dtype=value.dtype, device=value.device
            )
            loss = torch.nn.functional.mse_loss(value, target)
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            losses.append(loss.detach())
        explained = self.policy_optimizer._explained_variance
        return {
            "critic_transactions": float(len(found)),
            "critic_positions": float(len(taught)),
            # Of the update's games before the critic is taught them, what
            # each explains of the results: the critic, and the value head
            # at the same decisions.
            "critic_outcome_explained_variance": explained(results, values),
            "critic_head_outcome_explained_variance": explained(results, heads),
            "critic_loss": float(torch.stack(losses).mean()) if losses else 0.0,
            "critic_seconds": time.perf_counter() - started,
        }

    def _pool(self) -> DuelPool:
        if self._duel_pool is None:
            duels = DuelPool(
                self.root,
                threads=self.config.pool_threads,
                slots=self.config.pool_slots,
                shared=self.config.pool_shared,
                groups=self.calls.groups,
            )
            self._duel_pool = Locked(duels) if self.config.overlap else duels
        return self._duel_pool  # type: ignore[return-value]

    def failed_duels(self) -> list[dict[str, Any]]:
        """The duels the engine stopped since this was last asked
        (``DuelPool.failed``): none where no duel has been played."""
        return [] if self._duel_pool is None else self._duel_pool.failed()

    def _pilot_games(self, games: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """An update's games with the pilot in the learner's seat, their
        decisions kept in the pool until the next update's."""
        duels = self._pool()
        duels.forget()
        return imitation_games(
            duels, learner=self.config.learner_deck, games=games, limit=self.config.limit
        )

    def _replayed_games(self, games: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """An update's recorded duels played again, the record's answers the
        labels, their decisions kept in the pool until the next update's."""
        duels = self._pool()
        duels.forget()
        return replayed_games(
            duels,
            learner=self.config.learner_deck,
            games=games,
            limit=self.config.limit,
            label=self.config.imitation_label,
        )

    def _teacher_games(
        self, games: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """An update's games with the teachers in the learner's seat, each in
        its deck's, taking their first choices, their decisions kept in the
        pool until the next update's."""
        duels = self._pool()
        duels.forget()
        for graphs in self.teacher_graphs.values():
            graphs.begin_update()
        played = taught_games(
            duels,
            self.teachers,
            self.device,
            learner=self.config.learner_deck,
            games=games,
            limit=self.config.limit,
            groups_over=self.calls.groups_over,
            grouped=self.config.pool_groups is not None,
            graphs=self.teacher_graphs,
        )
        failed = [graphs.failed for graphs in self.teacher_graphs.values() if graphs.failed]
        if self.teacher_graphs:
            played[1]["pool_graph_failed"] = float(bool(failed))
            if failed and not self._teacher_graphs_failure_told:
                self._teacher_graphs_failure_told = True
                print(
                    f"A teacher's calls are answered as they are from here: {failed[0]}",
                    file=sys.stderr,
                    flush=True,
                )
        return played

    def _recorded_copy_games(
        self, games: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """An update's games of a copy played again from the champions'
        recordings (``champion_records``): the answers recorded in the
        teachers' place, each game held to the digest it was recorded with."""
        duels = self._pool()
        duels.forget()
        trajectories, asked = taught_games(
            duels,
            {},
            self.device,
            learner=self.config.learner_deck,
            games=games,
            limit=self.config.limit,
            groups_over=self.calls.groups_over,
            grouped=self.config.pool_groups is not None,
            played=[game["answers"] for game in games],
        )
        for game, trajectory in zip(games, trajectories, strict=True):
            if trajectory["digest"] != game["digest"]:
                raise ValueError(
                    f"the recorded game of seed {game['seed']} ({game['learner']} against "
                    f"{game['opponent']}) plays again with digest {trajectory['digest']}, not "
                    f"{game['digest']}: it was recorded with another engine, other scripts, "
                    "cards or pilots"
                )
        return trajectories, asked

    def _learner_games(
        self,
        games: list[dict[str, Any]],
        rollout_seed: int,
        labels: bool,
        whole: bool | None = None,
        whole_deck: int = 0,
        follow: bool | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """An update's games with the model in the learner's seat, drawing
        its answers, their decisions kept in the pool until the next update's.
        In the other seat of a game sits who the game says: a pilot, the
        model again, or the checkpoint that is held.  With ``whole`` each of
        the model's decisions is kept with nothing hidden as well
        (``_kept_whole``), for a critic that sees the hidden cards, and the
        next ``whole_deck`` cards of each Deck with it: unless told, as the
        run's critic sees them, where it has one.  With ``follow`` the cards
        of each transaction are followed to its horizon (`DuelPool.labels`):
        unless told, where the run has a head that learns how cards work."""
        if whole is None:
            whole, whole_deck = self.config.critic, self.config.critic_deck
        if follow is None:
            follow = self.config.card_head
        duels = self._pool()
        duels.forget()
        if self.graphs is not None:
            self.graphs.begin_update()
        return self._graphed(
            sampled_games(
                duels,
                self.model,
                self.device,
                learner=self.config.learner_deck,
                games=games,
                limit=self.config.limit,
                rollout_seed=rollout_seed,
                temperature=self.config.rollout_temperature,
                labels=labels,
                label_fraction=self.config.teacher_label_fraction,
                held=self.held,
                groups_over=self.calls.groups_over,
                graphs=self.graphs,
                whole=whole,
                whole_deck=whole_deck,
                follow=follow,
            )
        )

    def _streamed_games(
        self, blocks: Any, index: int, labels: bool
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """An update's games with the model in the learner's seat, from duels
        kept in flight from one update to the next (``pool.Stream``): the next
        ``duels`` of them to end."""
        if labels:
            raise ValueError("duels in flight keep no pilot's answer beside the model's")
        if self._stream is None:
            duels = self._pool()
            duels.forget()
            self._stream = Stream(
                duels,
                self.device,
                learner=self.config.learner_deck,
                limit=self.config.limit,
                temperature=self.config.rollout_temperature,
                graphs=self.graphs,
            )
        ahead = self.config.duels_in_flight // self.config.duels - 1
        model = self.model if self._playing_with is None else self._playing_with
        return self._graphed(self._stream.play(model, blocks, index, ahead, self.config.duels))

    def _graphed(
        self, played: tuple[list[dict[str, Any]], dict[str, float]]
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """An update's games, with whether the graphs their calls were to be
        answered from failed: told once where it happens, and in every
        update's figures from then on."""
        if self.graphs is not None:
            played[1]["pool_graph_failed"] = float(self.graphs.failed is not None)
            if self.graphs.failed is not None and not self._graphs_failure_told:
                self._graphs_failure_told = True
                print(
                    f"The calls are answered as they are from here: {self.graphs.failed}",
                    file=sys.stderr,
                    flush=True,
                )
        return played

    def _kept(self, rows: list[int]) -> dict[str, torch.Tensor]:
        """Decisions kept in the pool as the batch the model is given: what
        an update trains on, the cards in it shown with their identities kept
        among those trained on."""
        started = time.perf_counter()
        arrays = self._pool().batch(rows)
        if self._hidden is not None:
            self._shown(arrays)
        batch = {name: torch.from_numpy(array) for name, array in arrays.items()}
        self._kept_seconds += time.perf_counter() - started
        if self.device == "cpu":
            return batch
        return {name: value.to(self.device, non_blocking=True) for name, value in batch.items()}

    def _kept_whole(self, rows: list[int]) -> dict[str, torch.Tensor]:
        """The same decisions with nothing hidden, as `_kept` gives them:
        what a critic that sees the hidden cards reads, never the policy."""
        started = time.perf_counter()
        arrays = self._pool().batch(rows, whole=True)
        if self._hidden is not None:
            self._shown(arrays)
        batch = {name: torch.from_numpy(array) for name, array in arrays.items()}
        self._kept_seconds += time.perf_counter() - started
        if self.device == "cpu":
            return batch
        return {name: value.to(self.device, non_blocking=True) for name, value in batch.items()}

    def _hide(self, step: int) -> dict[str, float]:
        """Hide the identities of a share of the table's cards for update
        ``step``, drawn from the run's seed and the update, the same cards in
        its games and its training."""
        if self._codes is None:
            return {}
        count = len(self._codes)
        drawn = np.random.default_rng([int(self.config.seed), int(step)]).choice(
            count, size=round(self.model.hide_identities * count), replace=False
        )
        hidden = np.zeros(count, dtype=bool)
        hidden[drawn] = True
        self._hidden = hidden
        self._trained = self.model.card_trained.cpu().numpy().copy()
        self.model.hide(torch.from_numpy(hidden))
        return {"cards_hidden": float(hidden.sum())}

    def _shown(self, arrays: dict[str, np.ndarray]) -> None:
        """Keep the cards of the table these decisions show with their
        identities among those trained on."""
        assert self._codes is not None and self._hidden is not None and self._trained is not None
        codes = np.unique(np.concatenate([arrays[name].ravel() for name in CARD_ARRAYS]))
        place = np.searchsorted(self._codes, codes).clip(max=len(self._codes) - 1)
        place = place[self._codes[place] == codes]
        self._trained[place[~self._hidden[place]]] = True

    @contextmanager
    def _playing(self) -> Iterator[None]:
        """The model as it plays, for games outside an update: hiding the
        identities of the cards it was not trained on, then what it hid."""
        if not self.model.hides_identities:
            yield
            return
        hidden = self.model.card_hidden.clone()
        unknown = self.model.hides_unknown.clone()
        self.model.play()
        try:
            yield
        finally:
            self.model.card_hidden.copy_(hidden)
            self.model.hides_unknown.copy_(unknown)

    def set_rollout_index(self, index: int) -> None:
        self._rollout_index = index
        if hasattr(self, "coordinator"):
            self.coordinator.set_rollout_index(index)

    def set_training_schedule_index(self, index: int) -> None:
        self._schedule_index = index
        if hasattr(self, "coordinator"):
            self.coordinator.set_training_schedule_index(index)

    def collect(self) -> tuple[list[dict], dict[str, float]]:
        if self._closed:
            raise RuntimeError("This trainer has been closed.")
        return self.coordinator.collect()

    def _begin_ahead(self) -> dict[str, float]:
        """The next update's games begun, by a copy of the model with the
        weights from before this update's pass: beside the pass by the
        second thread, or in turn, played at once."""
        started = time.perf_counter()
        if self._player is None:
            self._player = copy.deepcopy(self.model)
            if self.device == "cuda":
                self._player_stream = torch.cuda.Stream()
        else:
            self._player.load_state_dict(self.model.state_dict())
        if self._player_stream is not None:
            # Its calls wait for the weights it was given.
            self._player_stream.wait_stream(torch.cuda.current_stream())
        self._playing_with = self._player
        copied = time.perf_counter() - started
        if self._beside is None:
            self._ahead = Future()
            self._ahead.set_result(self._played_ahead())
        else:
            self._ahead = self._beside.submit(self._played_ahead)
        return {"overlap_copy_seconds": copied}

    def _played_ahead(self) -> tuple[list[dict], dict[str, float], tuple[float, float]]:
        """The next update's games as `collect` plays them, by the copy of
        the model, its calls on a stream of their own on a card; and when
        they began and ended."""
        begun = time.perf_counter()
        if self._player_stream is None:
            trajectories, metrics = self.collect()
        else:
            with torch.cuda.stream(self._player_stream):
                trajectories, metrics = self.collect()
        pool = self._duel_pool
        metrics["overlap_games_lock_seconds"] = (
            pool.take_waited() if isinstance(pool, Locked) else 0.0
        )
        return trajectories, metrics, (begun, time.perf_counter())

    def evaluate_pool(
        self,
        learner: str,
        opponents: list[str],
        games_per_opponent: int,
        seed: int,
        *,
        pilot: bool = False,
        checkpoint: bool = False,
        learner_list: str | None = None,
    ) -> list[dict]:
        """Games of ``learner`` against each opponent deck's pilot, or with
        ``checkpoint`` against the checkpoint that is held, playing that deck.

        The live model plays the learner's seat by its first choice, or with
        ``pilot`` the pilot of the learner's deck plays it; with
        ``learner_list`` that seat plays the deck list of that file.
        """
        if self._closed:
            raise RuntimeError("This trainer has been closed.")
        for name in [learner, *opponents]:
            if name not in self.decks:
                raise ValueError(f"Unknown deck: {name}")
        if checkpoint:
            if self.held is None:
                raise ValueError("No checkpoint is held for the other seat: name one.")
            return self.evaluate_games(
                learner,
                [
                    {**game, "other": "checkpoint"}
                    for game in evaluation_assignments(opponents, games_per_opponent, seed)
                ],
                pilot=pilot,
                learner_list=learner_list,
            )
        return self.evaluate_games(
            learner,
            evaluation_assignments(opponents, games_per_opponent, seed),
            pilot=pilot,
            learner_list=learner_list,
        )

    def evaluate_games(
        self,
        learner: str,
        assignments: list[dict[str, Any]],
        *,
        pilot: bool = False,
        lead_margin: float | None = None,
        learner_list: str | None = None,
    ) -> list[dict]:
        """The listed games of ``learner``, each with its opponent, seat and
        seed, by their index.  Games that carry a record are played again as
        recorded with the live model watching (``records.RecordedGame``,
        ``pool.watched_games``).  With ``lead_margin`` the model plays by
        following its lead head.  With ``learner_list`` the learner's seat
        plays the deck list of that ``.ydk`` file (``pool.evaluate_games``).
        """
        if self._closed:
            raise RuntimeError("This trainer has been closed.")
        if self._stream is not None:
            raise ValueError(
                "a run's duels in flight hold its pool: it is not evaluated as it goes"
            )
        for name in [learner, *(str(item["opponent"]) for item in assignments)]:
            if name not in self.decks:
                raise ValueError(f"Unknown deck: {name}")
        watched = sum("record" in item for item in assignments)
        if watched and (
            watched < len(assignments)
            or pilot
            or lead_margin is not None
            or learner_list is not None
        ):
            raise ValueError(
                "Recorded duels are watched on their own: no pilot plays them, "
                "no lead head is followed in them, and they play the lists they were "
                "recorded with."
            )
        with self._playing():
            if watched:
                return watched_games(
                    self._pool(),
                    self.model,
                    self.device,
                    learner=learner,
                    assignments=assignments,
                    limit=self.config.limit,
                )
            return pooled_games(
                self._pool(),
                self.model,
                self.device,
                learner=learner,
                assignments=assignments,
                limit=self.config.limit,
                pilot=pilot,
                held=self.held,
                lead_margin=lead_margin,
                learner_list=learner_list,
            )

    def update(
        self,
        trajectories: list[dict],
        teacher_imitation_weight: float = 0.0,
    ) -> dict[str, float]:
        return self.policy_optimizer.update(trajectories, teacher_imitation_weight)

    def imitation_update(
        self,
        trajectories: list[dict],
    ) -> dict[str, Any]:
        return self.policy_optimizer.imitation_update(trajectories)

    def save_checkpoint(
        self,
        checkpoint: Path,
        step: int,
        metrics: dict[str, Any],
        *,
        finish: bool = True,
    ) -> None:
        """Save the current live model and optimizer with canonical metadata.

        Unfinished, the checkpoint is written and has yet to take its name,
        which ``checkpoints.finish`` gives it.  A run's critic is written
        beside it, and takes its name with it.
        """
        if self.critic is not None and self.critic_optimizer is not None:
            write_beside(checkpoint, "critic", self.critic, self.critic_optimizer, step)
        optimizer = self.policy_optimizer.card_head_optimizer
        if self.card_head is not None and optimizer is not None:
            write_beside(checkpoint, "card-head", self.card_head, optimizer, step)
        (save if finish else write)(
            checkpoint,
            self.model,
            self.optimizer,
            step,
            {
                **metrics,
                "device": self.device,
                "board_representation": self.config.board_representation,
            },
        )

    def train_update(
        self,
        checkpoint: Path | None = None,
        step: int = 1,
        *,
        finish: bool = True,
        ahead: bool = False,
    ) -> dict[str, float]:
        """An update: its games, then its pass. With ``ahead``, an update
        follows it, and where the run plays the next update's games beside
        the pass (`TrainerConfig.overlap`) they are begun before it."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._kept_seconds = 0.0
        hidden = self._hide(step)
        if self._ahead is not None:
            # Played beside the pass before, with the weights from before it.
            waited = time.perf_counter()
            trajectories, metrics, played = self._ahead.result()
            self._ahead = None
            metrics["overlap_wait_seconds"] = time.perf_counter() - waited
            metrics["overlap_lag"] = 1.0
            if self._last_pass is not None:
                begun, ended = self._last_pass
                metrics["overlap_beside_seconds"] = max(
                    0.0, min(ended, played[1]) - max(begun, played[0])
                )
        else:
            self._playing_with = None
            trajectories, metrics = self.collect()
        metrics.update(hidden)
        metrics["recomputed_rollout_estimates"] = float(
            self.policy_optimizer.recompute_rollout_estimates(trajectories)
            if self.config.rollout_estimates == "recomputed"
            else 0
        )
        metrics.update(self.policy_optimizer.teacher_label_metrics(trajectories))
        decision_records = [
            {
                "episode_index": trajectory.get("episode_index"),
                "winner": trajectory.get("winner"),
                "terminal": trajectory.get("terminal"),
                "learner_player": trajectory.get("learner_player"),
                "steps": [
                    {
                        key: step.get(key)
                        for key in ("player", "chosen", "teacher_chosen", "transaction")
                        if key in step
                    }
                    for step in trajectory["steps"]
                ],
            }
            for trajectory in trajectories
        ]
        training_records = [
            {
                "episode_index": trajectory.get("episode_index"),
                "winner": trajectory.get("winner"),
                "terminal": trajectory.get("terminal"),
                "learner_player": trajectory.get("learner_player"),
                "steps": [
                    {
                        key: value
                        for key, value in step.items()
                        # Where a decision's input is held says nothing of it.
                        if key != "row" and key not in DIAGNOSTIC_STEP_KEYS
                    }
                    for step in trajectory["steps"]
                ],
            }
            for trajectory in trajectories
        ]
        metrics["rollout_decision_fingerprint"] = value_fingerprint(decision_records)
        metrics["rollout_training_fingerprint"] = value_fingerprint(training_records)
        has_teacher_steps = any(
            sample.get("teacher") for trajectory in trajectories for sample in trajectory["steps"]
        )
        if self.critic is not None and not has_teacher_steps:
            metrics.update(self._credit(trajectories, step))
        if ahead and self.config.overlap:
            metrics.update(self._begin_ahead())
        update_started = time.perf_counter()
        self.policy_optimizer.begin_update(step)
        if has_teacher_steps:
            metrics.update(self.imitation_update(trajectories))
        else:
            metrics.update(self.update(trajectories, metrics["teacher_imitation_weight"]))
        metrics["board_representation"] = self.config.board_representation
        if self.config.leave_out:
            metrics["leave_out"] = list(self.config.leave_out)
        metrics["initial_model_fingerprint"] = self.initial_model_fingerprint
        metrics["model_parameter_count"] = float(
            sum(parameter.numel() for parameter in self.model.parameters())
        )
        metrics["board_attention_parameter_count"] = float(
            sum(
                parameter.numel()
                for name, parameter in self.model.named_parameters()
                if name.startswith("board_")
            )
        )
        metrics["peak_accelerator_memory_bytes"] = float(
            torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
        )
        self._last_pass = (update_started, time.perf_counter())
        metrics["ppo_update_seconds"] = self._last_pass[1] - update_started
        if self._stream is not None:
            self._stream.trained()
        if isinstance(self._duel_pool, Locked):
            metrics["overlap_pass_lock_seconds"] = self._duel_pool.take_waited()
        if self._trained is not None:
            self.model.trained_on(torch.from_numpy(self._trained))
            metrics["cards_trained"] = float(self._trained.sum())
        # Of the update and of what precedes it: the building of the
        # minibatches where the decisions are kept.
        metrics["pool_minibatch_seconds"] = self._kept_seconds
        if checkpoint:
            self.save_checkpoint(checkpoint, step, metrics, finish=finish)
        return metrics
