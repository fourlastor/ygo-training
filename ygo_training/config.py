"""Configuration objects shared by rollout and optimization components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .records import GAMES as RECORDED_GAMES
from .schedules import MAX_EPISODES_PER_ROLLOUT, MAX_SEEDED_WORKERS, SELF_PLAY_PAIRS


@dataclass(frozen=True)
class TrainingProtocolDefaults:
    """Defaults of the training CLI."""

    workers: int = 16
    duels: int = 512
    # One pass over an update's games: in four, PPO undoes a copy of the
    # pilot, and a value head taught results learns them by heart.
    ppo_epochs: int = 1
    minibatch: int = 256
    learning_rate: float = 3e-4
    rollout_temperature: float = 1.0
    duel_gae_lambda: float = 0.95
    eval_every: int = 5
    seed: int = 2011


GOLDEN_TRAINING_PROTOCOL = TrainingProtocolDefaults()
BOARD_REPRESENTATIONS = ("mean-pool", "individual-tokens")
# What an update of PPO trains with at each decision of its games
# (`TrainerConfig.rollout_estimates`).
ROLLOUT_ESTIMATES = ("recomputed", "kept")
# Parts of its input a model may leave out (`TrainerConfig.leave_out`): the
# seat's own earlier answers.  The two deck lists were parts too, until the
# input dropped them (format 15).
LEAVE_OUT = ("history",)


def left_out(parts: Iterable[str]) -> tuple[str, ...]:
    """Parts of LEAVE_OUT, in its order: one way of naming them for each set."""
    named = set(parts)
    return tuple(part for part in LEAVE_OUT if part in named)


# What a PPO update teaches the value head: see `ppo_value_target`.
PPO_VALUE_TARGETS = ("return", "result")
# The answer of a recorded duel that is imitated: see `TrainerConfig`.
IMITATION_LABELS = ("played", "pilot")
# What of a checkpoint that is imitated is the target: see `imitation_teacher`.
TEACHER_TARGETS = ("choice", "distribution")
# What an advantage belongs to: a command and the selections it leads to.
GAE_UNIT = "transactions"


# The decisions two groups of duels go on together from, where it is not
# named: what was measured with calls from graphs (`TrainerConfig.calls`).
GROUPS_OVER = 64
# What a run on a card keeps where its command line does not say, decided on
# 2026-10-23 after the run `fast`
# (experiments/2026-10-23-duels-in-flight.md) learned as `all71` did and
# played 1.46 times the duels an hour: so many updates' duels in flight, in
# minibatches of so many decisions.
IN_FLIGHT_UPDATES = 2
IN_FLIGHT_MINIBATCH = 512
# Who plays the next update's games beside an update's pass
# (`TrainerConfig.overlap`).
OVERLAPS = ("thread", "in-turn")


@dataclass(frozen=True)
class Calls:
    """How the model is asked in an update's games: the groups a thread's
    duels are shared out among, the decisions their rounds hold between them
    from which they go on together, and the decisions up to which a call is
    answered from a graph of fixed sizes (`TrainerConfig`)."""

    groups: int = 1
    groups_over: int = 0
    graphs: int = 0


@dataclass
class TrainerConfig:
    """Runtime configuration for :class:`PpoTrainer`.

    Keeping this data object separate means frontends and experiments can
    construct trainers without importing the implementation.
    """

    # The shares an update's duels are dealt in, each drawing the learner's
    # seat in its duels from a generator of its own (`schedules.rollout_games`),
    # as the worker processes that once played them did: the duels follow
    # from it.
    workers: int = 1
    duels: int = 2
    ppo_epochs: int = 2
    minibatch: int = 64
    limit: int = 4096
    # Threads of the library's pool that play every game, each with up to
    # `pool_slots` duels in flight.
    pool_threads: int = 1
    pool_slots: int = 16
    # Whether the pool's threads draw their duels from one queue, each taking
    # the next as a slot comes free, where a duel is otherwise dealt to the
    # thread of its number in advance.
    pool_shared: bool = False
    # How the model is asked in an update's games.  The three that follow are
    # worked out for the device the games are answered on where they are not
    # named (`calls`): on a card the way measured fastest there, on a
    # processor one group and every call as it is.
    #
    # Among how many groups a thread's duels are shared out, where the model
    # plays an update's games: the groups are taken forward one at a time,
    # the engines of one running while the model answers another's
    # decisions.  One group is every duel answered in one call a round.
    pool_groups: int | None = None
    # The groups go on together, every duel answered in one call a round,
    # once their last rounds hold this many decisions or fewer between them:
    # a call costs much the same whatever it holds, and a round split in two
    # pays only while it is large.  0: by groups to an update's last duel.
    pool_groups_over: int | None = None
    # Calls of the model of up to this many decisions, in an update's games,
    # are filled out to sizes fixed in advance and answered from CUDA graphs
    # on a card (graphs.Graphs): a call costs about the same whatever it
    # holds, and replayed from a graph a small one takes a third of it.  A
    # decision's scores then differ in their last digits from those of its
    # own batch.  0: every call as it is.
    inference_graphs: int | None = None
    seed: int = 2011
    device: str = "cpu"
    learning_rate: float = 3e-4
    # The deck the learner plays, and the decks it meets: every other deck of
    # the pool when none is named.
    learner_deck: str = "blackwing"
    opponent_decks: list[str] | None = None
    # The decks the learner plays when it plays several: each against every
    # opponent, in equal shares of an update's games, which then go through
    # the library's pool.  `learner_deck` is the first of them.
    learner_decks: list[str] | None = None
    entropy_coefficient: float = 0.01
    # PPO's clip on the ratio of an answer's chances to the rollout's.
    clip_epsilon: float = 0.2
    # UPGO, as ygo-agent's trainer adds it (`--upgo`): each transaction's
    # advantage gains the seat's return followed on for as long as the play
    # after it went at least as well as the head expected, less the head's
    # estimate there.  The head's target is the same either way.
    upgo: bool = False
    # Initial updates in which the deck's pilot plays and the learner imitates it.
    imitation_only_updates: int = 0
    # Rows of a `policy-bench search --record true` run whose duels those
    # updates play again and imitate, in place of duels the pilot plays now;
    # which of a row's two games (see `records.GAMES`); and how many of the
    # run's games, all of them when not said.
    imitation_records: str | None = None
    imitation_game: str = "search"
    imitation_games: int | None = None
    # Which answer of a recorded duel the policy imitates: the one played,
    # or the pilot's own, which a search leaves here and there.
    imitation_label: str = "played"
    # Weight of the lead head's squared error against the leads a logged
    # search measured, and whether the head only reads the policy's view of
    # an answer, with no gradient of its own into it.
    imitation_lead_weight: float = 0.0
    imitation_lead_detach: bool = False
    # A checkpoint those updates imitate in the pilot's place: it plays the
    # learner's seat by its first choice, through the library's pool, and its
    # answers are the ones imitated.  What of it is the target: its `choice`
    # at a decision, as a pilot's answer is, or its `distribution` over the
    # answers, which also tells what it nearly chose.
    imitation_teacher: str | None = None
    # For a learner of several decks, a checkpoint for each of its decks in
    # its place, by deck: each plays the seats of its deck, and its answers
    # there are the ones imitated (decided on 2026-10-24,
    # experiments/2026-10-24-specialist-distillation.md).
    imitation_teachers: dict[str, str] | None = None
    imitation_teacher_targets: str = "choice"
    # Weight of the pilot's labels on the learner's own decisions in PPO updates.
    teacher_imitation_weight: float = 0.0
    teacher_imitation_decay_updates: int = 0
    # Weight of the PPO policy term. Zero with teacher imitation trains the
    # policy on teacher labels of the learner's own states alone (DAgger).
    policy_loss_weight: float = 1.0
    teacher_label_fraction: float = 1.0
    rollout_temperature: float = 1.0
    duel_gae_lambda: float = 1.0
    board_representation: str = "mean-pool"
    # Parts of LEAVE_OUT the model does not read: each is given nothing where
    # it was, and the network keeps every weight it has, so that models that
    # read less begin as the same network under the same seed.  A checkpoint
    # keeps what its model left out.
    leave_out: tuple[str, ...] = ()
    # What the model is told of a card beside its input, from checkpoint
    # format 16 (`card_table`): the table of the cards' codes and
    # descriptions it reads (`data/card-descriptions.npz`, or a checkpoint
    # that carries one); whether it reads their descriptions; and the share
    # of the cards whose identities it hides in each update, the same cards
    # in its games and its training.  At play it hides the identities of the
    # cards it never trained on.
    card_table: str | None = None
    card_descriptions: bool = False
    hide_identities: float = 0.0
    # Value-head regression toward duel outcomes during imitation-only updates.
    imitation_value_weight: float = 0.0
    # Of a duel's transactions, how many carry its outcome as the head's
    # target there, drawn at random: every one of them when zero.  A duel has
    # one outcome and some seventy transactions to tell it by.
    imitation_value_positions: int = 0
    # Whether the head only reads the shared state in those updates, with no
    # gradient of its own into it.
    imitation_value_detach: bool = False
    # The same choice for the updates whose games the model plays: of each
    # seat's transactions in a duel, how many carry the value head's target,
    # drawn at random; every one of them when zero.  Every transaction still
    # has its advantage, measured against the head.
    ppo_value_positions: int = 0
    # What the head is taught in those updates.  `return` is PPO's own
    # target: the advantage added back to the head's estimate, which holds
    # the head's later estimates of the same duel whenever the advantages
    # are shortened (duel_gae_lambda under 1).  `result` is how the seat's
    # duel ended, as the imitating updates teach it; the advantages are
    # then still measured against the head, shortened or not.
    ppo_value_target: str = "return"
    # Stop an update's PPO epochs once minibatch approximate KL exceeds 1.5x this.
    target_kl: float | None = None
    # Of the duels of an update whose games the model plays, how many it
    # plays against itself: its own network in both seats, the learner's deck
    # on both sides, and every decision of the duel one it learns from.  The
    # others are against the pilots, as all of them are at zero.  In an
    # update that imitates a network (`imitation_teacher`) the network plays
    # both seats of them, and both are imitated.
    self_play_duels: int = 0
    # Which of a learner's decks meet in those duels where it has several:
    # any two of them, one in each seat (`all`), each deck against itself
    # alone (`mirrors`, decided on 2026-10-24,
    # experiments/2026-10-24-two-decks.md), or two different decks alone
    # (`cross`, decided the same day,
    # experiments/2026-10-24-specialist-distillation.md).  A learner of one
    # deck plays its mirror, and has no `cross`.
    self_play_pairs: str = "all"
    # Those duels shared out by a weight for each ordered pair of a learner's
    # decks, "OWN>OTHER" (`schedules.allocate_weighted_pairs`), in place of
    # every pair as often as any other: the weights a run is given, decided
    # on 2026-10-25 (experiments/2026-10-24-four-decks.md).
    self_play_weights: dict[str, float] | None = None
    # A checkpoint held as it is, which plays the learner's deck from the
    # other seat by its first choice: in so many of those duels, where only
    # the learner's own decisions are learned from, and in the cases of an
    # evaluation suite that ask for it.
    opponent_checkpoint: str | None = None
    opponent_checkpoint_duels: int = 0
    # What an update of PPO trains with at each decision of its games: those
    # the game was answered with (`kept`, the default since 2026-10-20); or
    # the chances and the value the model gives it when worked out again
    # over the update's decisions at once (`recomputed`, the trainer as it
    # was), which differ from a game's, answered in calls of other sizes, in
    # their last digits.
    rollout_estimates: str = "kept"
    # Duels kept in flight from one update of PPO to the next
    # (`pool.Stream`): so many at once, the games of the updates ahead under
    # way while an update's are, and an update made of the next `duels` to
    # end, each trained on once, whole.  0: every update's games played to
    # their end.
    duels_in_flight: int = 0
    # The next update's games played beside an update's pass, with the
    # weights from before it (experiments/2026-10-23-overlap.md): by a
    # second thread (`thread`), or by the same one before the pass
    # (`in-turn`), which gives the same updates and is what the first is
    # held to.  "": each update's games after the pass before it.
    overlap: str = ""
    # A critic that sees the hidden cards, in training only
    # (`ygo_training/critic.py`, experiments/2026-10-22-privileged-critic.md):
    # where it is on, the advantages of an update of PPO are measured
    # against its values in the value head's place, and the head is still
    # taught its own return.  It sees the next `critic_deck` cards of each
    # Deck as well.  After each update's games it is taught the results of
    # `critic_positions` of each seat's transactions a duel, drawn at random,
    # in one pass of minibatches at `critic_learning_rate`.
    critic: bool = False
    critic_deck: int = 0
    critic_positions: int = 8
    critic_learning_rate: float = 3e-4
    # A head that learns how cards work, in training only
    # (`ygo_training/card_head.py`, experiments/2026-09-08-engine-transition-
    # prediction.md): where it is on, the cards of the transactions of an
    # update's games are followed to their horizon, and the head is taught
    # in PPO's own minibatches, the gradient it gives the network at
    # `card_head_weight` beside PPO's (the head itself learns as much
    # whatever the weight).
    card_head: bool = False
    card_head_weight: float = 1.0
    # The head as a probe left it (`tools/card_head.py probe`), for a run
    # that begins one: anew where none is named.
    card_head_begun: str | None = None

    @property
    def learners(self) -> list[str]:
        """The decks the learner plays: one, unless several are named."""
        return list(self.learner_decks or [self.learner_deck])

    @property
    def pair_weights(self) -> dict[tuple[str, str], float] | None:
        """The weights of the duels against itself, by ordered pair of decks."""
        if self.self_play_weights is None:
            return None
        return {
            tuple(pair.split(">", 1)): float(weight)  # type: ignore[misc]
            for pair, weight in self.self_play_weights.items()
        }

    @property
    def teachers(self) -> dict[str, str]:
        """The checkpoint that teaches each of the learner's decks, by deck,
        where a network is imitated in the pilot's place."""
        if self.imitation_teacher:
            return {self.learner_deck: self.imitation_teacher}
        return dict(self.imitation_teachers or {})

    def calls(self, device: str) -> Calls:
        """How the model is asked in an update's games answered on
        ``device``: what was named, and for the rest the way of that device.

        On a card with a pool: calls of up to half the pool's duels from
        CUDA graphs, and with them two groups of duels that go on together
        from `GROUPS_OVER` decisions, where a thread's duels share out in
        two.  That is the way measured fastest on the card
        (docs/commands.md, its section on calls from CUDA graphs): half the
        pool's duels is the most a group of two asks at once.  On a
        processor, and where one group and every call as it is.
        """
        fast = str(device).startswith("cuda")
        graphs = self.inference_graphs
        if graphs is None:
            graphs = max(1, self.pool_threads * self.pool_slots // 2) if fast else 0
        groups = self.pool_groups
        if groups is None:
            halved = fast and graphs > 0 and not self.pool_shared and self.pool_slots % 2 == 0
            groups = 2 if halved else 1
        over = self.pool_groups_over
        if over is None:
            over = GROUPS_OVER if groups > 1 else 0
        return Calls(groups=groups, groups_over=over, graphs=graphs)

    def __post_init__(self) -> None:
        positive_integers = {
            "workers": self.workers,
            "duels": self.duels,
            "ppo_epochs": self.ppo_epochs,
            "minibatch": self.minibatch,
            "limit": self.limit,
            "pool_threads": self.pool_threads,
            "pool_slots": self.pool_slots,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"{invalid[0]} must be at least 1")
        if self.pool_groups is not None:
            if self.pool_groups < 1 or self.pool_slots % self.pool_groups:
                raise ValueError("pool_groups must share pool_slots out evenly")
            if self.pool_groups > 1 and self.pool_shared:
                raise ValueError("pool_groups and pool_shared do not go together")
        if self.pool_groups_over is not None and self.pool_groups_over < 0:
            raise ValueError("pool_groups_over must be non-negative")
        if self.inference_graphs is not None and self.inference_graphs < 0:
            raise ValueError("inference_graphs must be non-negative")
        if self.duels < self.workers:
            raise ValueError("duels must be at least workers")
        if self.workers > MAX_SEEDED_WORKERS:
            raise ValueError(f"workers must be at most {MAX_SEEDED_WORKERS}")
        if self.duels > MAX_EPISODES_PER_ROLLOUT:
            raise ValueError(f"duels must be at most {MAX_EPISODES_PER_ROLLOUT}")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.rollout_temperature <= 0.0:
            raise ValueError("rollout_temperature must be positive")
        if not self.learner_deck:
            raise ValueError("learner_deck must name a deck")
        if self.opponent_decks is not None and not self.opponent_decks:
            raise ValueError("opponent_decks must not be empty")
        if self.learner_decks is not None:
            if not self.learner_decks or len(set(self.learner_decks)) != len(self.learner_decks):
                raise ValueError("learner_decks must name decks, each once")
            if self.learner_decks[0] != self.learner_deck:
                raise ValueError("learner_deck must be the first of learner_decks")
        if len(self.learners) > 1:
            if self.opponent_decks is None:
                raise ValueError("several learner decks require opponent_decks to be named")
            if self.imitation_records:
                raise ValueError("imitation_records are the games of one learner deck")
            if self.imitation_teacher:
                raise ValueError(
                    "imitation_teacher plays one learner deck: imitation_teachers has one for each"
                )
        for name, value in {
            "teacher_imitation_weight": self.teacher_imitation_weight,
            "policy_loss_weight": self.policy_loss_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.teacher_imitation_decay_updates < 0:
            raise ValueError("teacher_imitation_decay_updates must be non-negative")
        if self.policy_loss_weight == 0.0 and not self.teacher_imitation_weight:
            raise ValueError("a zero policy_loss_weight requires teacher imitation")
        if not 0.0 < self.teacher_label_fraction <= 1.0:
            raise ValueError("teacher_label_fraction must be above 0 and at most 1")
        if self.teacher_label_fraction < 1.0 and not self.teacher_imitation_weight:
            raise ValueError("a teacher_label_fraction below 1 requires teacher imitation")
        if self.imitation_only_updates < 0:
            raise ValueError("imitation_only_updates must be non-negative")
        if self.imitation_only_updates and self.teacher_imitation_weight:
            raise ValueError("imitation-only updates and teacher imitation are mutually exclusive")
        if self.imitation_records and not self.imitation_only_updates:
            raise ValueError("imitation_records requires imitation-only updates")
        if self.imitation_game not in RECORDED_GAMES:
            raise ValueError("imitation_game must be search or baseline")
        if self.imitation_games is not None and not self.imitation_records:
            raise ValueError("imitation_games requires imitation_records")
        if self.imitation_games is not None and self.imitation_games < 1:
            raise ValueError("imitation_games must be at least 1")
        if self.imitation_label not in IMITATION_LABELS:
            raise ValueError("imitation_label must be played or pilot")
        if self.imitation_label != "played" and not self.imitation_records:
            raise ValueError("imitation_label requires imitation_records")
        if self.imitation_lead_weight < 0.0:
            raise ValueError("imitation_lead_weight must be non-negative")
        if self.imitation_lead_weight and not self.imitation_records:
            raise ValueError("imitation_lead_weight requires imitation_records")
        if self.imitation_lead_detach and not self.imitation_lead_weight:
            raise ValueError("imitation_lead_detach requires an imitation_lead_weight")
        if self.imitation_teachers is not None:
            if self.imitation_teacher:
                raise ValueError("imitation_teacher and imitation_teachers are mutually exclusive")
            if sorted(self.imitation_teachers) != sorted(self.learners) or not all(
                self.imitation_teachers.values()
            ):
                raise ValueError("imitation_teachers name a checkpoint for each learner deck")
        if self.teachers and not self.imitation_only_updates:
            raise ValueError("imitation_teacher requires imitation-only updates")
        if self.teachers and self.imitation_records:
            raise ValueError("imitation_teacher and imitation_records are mutually exclusive")
        if self.imitation_teacher_targets not in TEACHER_TARGETS:
            raise ValueError(f"imitation_teacher_targets must be one of {TEACHER_TARGETS}")
        if self.imitation_teacher_targets != "choice" and not self.teachers:
            raise ValueError("imitation_teacher_targets requires an imitation_teacher")
        if not 0.0 <= self.duel_gae_lambda <= 1.0:
            raise ValueError("duel_gae_lambda must be between 0 and 1")
        if self.board_representation not in BOARD_REPRESENTATIONS:
            raise ValueError("board_representation must be mean-pool or individual-tokens")
        if any(part not in LEAVE_OUT for part in self.leave_out) or len(set(self.leave_out)) != len(
            self.leave_out
        ):
            raise ValueError(f"leave_out names each of {', '.join(LEAVE_OUT)} once at most")
        self.leave_out = left_out(self.leave_out)
        if not 0.0 <= self.hide_identities <= 1.0:
            raise ValueError("hide_identities is a share of the cards, from 0 to 1")
        if (self.card_descriptions or self.hide_identities) and not self.card_table:
            raise ValueError("card_descriptions and hide_identities read a card_table")
        if self.card_table and not (self.card_descriptions or self.hide_identities):
            raise ValueError("a card_table is read for card_descriptions or hide_identities")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be non-negative")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be between 0 and 1")
        if self.target_kl is not None and not self.target_kl > 0.0:
            raise ValueError("target_kl must be positive")
        if self.imitation_value_weight < 0.0:
            raise ValueError("imitation_value_weight must be non-negative")
        if self.imitation_value_weight and not self.imitation_only_updates:
            raise ValueError("imitation_value_weight requires imitation-only updates")
        if self.imitation_value_positions < 0:
            raise ValueError("imitation_value_positions must be non-negative")
        if self.imitation_value_positions and not self.imitation_value_weight:
            raise ValueError("imitation_value_positions requires an imitation_value_weight")
        if self.ppo_value_positions < 0:
            raise ValueError("ppo_value_positions must be non-negative")
        if self.ppo_value_target not in PPO_VALUE_TARGETS:
            raise ValueError(f"ppo_value_target must be one of {PPO_VALUE_TARGETS}")
        if self.imitation_value_detach and not self.imitation_value_weight:
            raise ValueError("imitation_value_detach requires an imitation_value_weight")
        if self.self_play_duels < 0 or self.opponent_checkpoint_duels < 0:
            raise ValueError("the duels against a network must not be negative")
        if self.self_play_duels + self.opponent_checkpoint_duels > self.duels:
            raise ValueError("the duels against a network are of the update's duels: at most duels")
        if self.opponent_checkpoint_duels and not self.opponent_checkpoint:
            raise ValueError("opponent_checkpoint_duels requires an opponent_checkpoint")
        if self.self_play_pairs not in SELF_PLAY_PAIRS:
            raise ValueError(f"self_play_pairs must be one of {SELF_PLAY_PAIRS}")
        if self.self_play_pairs != "all" and not self.self_play_duels:
            raise ValueError(
                "self_play_pairs names the decks of duels against itself: self_play_duels"
            )
        if self.self_play_pairs == "cross" and len(self.learners) < 2:
            raise ValueError("self_play_pairs cross meets two different decks of the learner's")
        if self.self_play_weights is not None:
            if not self.self_play_duels or len(self.learners) < 2:
                raise ValueError(
                    "self_play_weights share out the duels against itself of several decks"
                )
            if self.self_play_pairs != "all":
                raise ValueError("self_play_weights and self_play_pairs are one or the other")
            for pair, weight in self.self_play_weights.items():
                own, sign, other = pair.partition(">")
                if not sign or own not in self.learners or other not in self.learners:
                    raise ValueError(
                        f"self_play_weights names pairs of the learner's decks: {pair}"
                    )
                if float(weight) < 0:
                    raise ValueError("self_play_weights are not below zero")
            if not any(float(weight) > 0 for weight in self.self_play_weights.values()):
                raise ValueError("self_play_weights are not all zero")
        # Against itself a learner of several decks plays any two of them; a
        # checkpoint held in the other seat plays one deck.
        if self.opponent_checkpoint and len(self.learners) > 1:
            raise ValueError("a checkpoint in the other seat plays one learner deck")
        if self.rollout_estimates not in ROLLOUT_ESTIMATES:
            raise ValueError(f"rollout_estimates must be one of {ROLLOUT_ESTIMATES}")
        if self.duels_in_flight:
            if self.duels_in_flight % self.duels or self.duels_in_flight < 2 * self.duels:
                raise ValueError(
                    "duels_in_flight is a multiple of duels, twice them at least: "
                    "an update's games and those of the updates ahead"
                )
            if self.rollout_estimates != "kept":
                raise ValueError(
                    "duels in flight are trained with what their games were answered with: "
                    "rollout_estimates kept"
                )
            if self.pool_threads * self.pool_slots < self.duels_in_flight:
                raise ValueError("duels_in_flight needs as many slots in the pool")
            if self.opponent_checkpoint_duels or self.teacher_imitation_weight:
                raise ValueError(
                    "duels in flight are played against the pilots and the model itself alone"
                )
        if self.overlap:
            if self.overlap not in OVERLAPS:
                raise ValueError(f"overlap must be one of {OVERLAPS}")
            if not self.duels_in_flight:
                raise ValueError("the next update's games beside the pass are duels in flight")
            if self.imitation_only_updates or self.hide_identities:
                raise ValueError(
                    "the next update's games beside the pass are of updates of PPO, by a model "
                    "that hides nothing"
                )
            if self.inference_graphs:
                raise ValueError(
                    "the games beside the pass are answered by a copy of the model, as it is"
                )
        if self.critic_deck < 0 or self.critic_positions < 1:
            raise ValueError("critic_deck must be non-negative, and critic_positions at least 1")
        if self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive")
        if self.critic_deck and not self.critic:
            raise ValueError("critic_deck is what a critic sees: it requires the critic")
        if self.critic:
            # Its games are an update's whole, and the cards it reads count
            # for nothing a model hides.
            if self.duels_in_flight:
                raise ValueError("a critic reads an update's games played to their end")
            if self.imitation_only_updates or self.teacher_imitation_weight:
                raise ValueError("a critic gives the credit of updates of PPO alone")
            if self.hide_identities:
                raise ValueError("a critic does not go with a model that hides identities")
        if self.card_head_weight < 0.0:
            raise ValueError("card_head_weight must be non-negative")
        if self.card_head_begun and not self.card_head:
            raise ValueError("card_head_begun is the head a run begins with: it requires the head")
        if self.card_head:
            if self.duels_in_flight:
                raise ValueError("the head's labels come of an update's games played to their end")
            if self.imitation_only_updates or self.teacher_imitation_weight:
                raise ValueError("the head that learns how cards work is taught in updates of PPO")
