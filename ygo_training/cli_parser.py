"""Argument parser construction, separate from command execution."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import (
    BOARD_REPRESENTATIONS,
    GOLDEN_TRAINING_PROTOCOL,
    GROUPS_OVER,
    IMITATION_LABELS,
    IN_FLIGHT_MINIBATCH,
    IN_FLIGHT_UPDATES,
    LEAVE_OUT,
    OVERLAPS,
    PPO_VALUE_TARGETS,
    ROLLOUT_ESTIMATES,
    SELF_PLAY_PAIRS,
    TEACHER_TARGETS,
    left_out,
)
from .records import GAMES as RECORDED_GAMES

Command = Callable[[argparse.Namespace], None]


def _runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    workers_default: int = 4,
) -> None:
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=workers_default,
        help="the shares an update's duels are dealt in, each drawing the learner's seat in its "
        "duels from a generator of its own, as the worker processes that once played them did: "
        "the duels follow from it",
    )
    parser.add_argument(
        "--pool-threads",
        type=int,
        default=4,
        help="threads of the library's pool, which plays every game",
    )
    parser.add_argument(
        "--pool-slots", type=int, default=16, help="duels in flight on each of those threads"
    )
    parser.add_argument(
        "--pool-groups",
        type=int,
        default=None,
        help="groups a thread's duels are shared out among where the model plays an update's "
        "games: one group's engines run while the model answers another's decisions; where it "
        "is not given, two on a card whose calls are answered from graphs or that keeps duels "
        "in flight, and one otherwise",
    )
    parser.add_argument(
        "--pool-groups-over",
        type=int,
        default=None,
        help="the groups go on together once their rounds hold this many decisions or fewer "
        f"between them; 0: by groups to the last duel; where it is not given, {GROUPS_OVER} for "
        "more groups than one",
    )
    parser.add_argument(
        "--inference-graphs",
        type=int,
        default=None,
        help="calls of the model of up to this many decisions, in an update's games, are answered "
        "at sizes fixed in advance, from CUDA graphs on a card; 0: none; where it is not given, "
        "half the pool's duels on a card that keeps no duels in flight, and none otherwise",
    )
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=GOLDEN_TRAINING_PROTOCOL.seed)
    parser.add_argument(
        "--board-representation",
        choices=BOARD_REPRESENTATIONS,
        default="mean-pool",
        help="how the cards on the field reach each action: pooled, or one token each",
    )


def _opponent_checkpoint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help=(
            "a checkpoint held as it is, which plays the learner's deck from the other "
            "seat by its first choice, in the cases of the suite whose opponent_policy is "
            "`checkpoint`"
        ),
    )


def _resume_arguments(parser: argparse.ArgumentParser) -> None:
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=Path)
    resume.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="resume the latest model-*.pt under --run/checkpoints",
    )
    parser.add_argument(
        "--new-pins",
        action="store_true",
        help="with --continue: go on although the policies' pins (duel_revisions) moved "
        "since the run was last made, the move noted in run.json (pin_moves); for a move "
        "whose duels are the old ones (docs/commands.md, 'Moving the pin')",
    )
    parser.add_argument("--allow-overwrite", action="store_true")


def _parts_left_out(given: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in given.split(",") if part.strip())
    unknown = [part for part in parts if part not in LEAVE_OUT]
    if unknown or len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError(
            f"name each of {', '.join(LEAVE_OUT)} once at most, not {given!r}"
        )
    return left_out(parts)


def _train_parser(commands: Any, command: Command, default_learner: str) -> None:
    parser = commands.add_parser("train", help="train one learner deck against opponent decks")
    _runtime_arguments(parser, workers_default=GOLDEN_TRAINING_PROTOCOL.workers)
    parser.add_argument(
        "--training-seed",
        type=int,
        help=(
            "seed for training randomness only (rollout duels, sampling, minibatches); "
            "--seed still sets the evaluation games (default: --seed)"
        ),
    )
    parser.add_argument("--run", type=Path, default=Path("runs/training"))
    parser.add_argument(
        "--leave-out",
        type=_parts_left_out,
        default=(),
        help=(
            f"parts of its input the model does not read, comma-separated, of {', '.join(LEAVE_OUT)}: "
            "each is given nothing where it was, the network the same; its checkpoints keep it"
        ),
    )
    parser.add_argument(
        "--card-table",
        type=Path,
        help=(
            "the cards' codes and descriptions the model reads, from checkpoint format 16 "
            "(data/card-descriptions.npz): for --card-descriptions and --hide-identities"
        ),
    )
    parser.add_argument(
        "--card-descriptions",
        action="store_true",
        help="the model reads each card's description, looked up by its code, beside its identity",
    )
    parser.add_argument(
        "--hide-identities",
        type=float,
        default=0.0,
        help=(
            "the share of the cards whose identities the model is not shown in each update, "
            "drawn again each update, the same in its games and its training; at play the "
            "identities of the cards it never trained on are hidden (default: 0)"
        ),
    )
    parser.add_argument(
        "--learner",
        default=default_learner,
        help=(
            "the deck the learner plays; or several, comma-separated, each against every "
            "opponent in equal shares of an update's games (needs --opponents)"
        ),
    )
    parser.add_argument(
        "--opponents",
        help="comma-separated deck IDs (default: every deck of the pool but the learner's)",
    )
    parser.add_argument(
        "--duels",
        type=int,
        default=GOLDEN_TRAINING_PROTOCOL.duels,
        help="total complete duels per update, distributed evenly across workers",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=GOLDEN_TRAINING_PROTOCOL.rollout_temperature,
        help="PPO rollout sampling temperature",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=100,
        help=(
            "number of optimizer updates to perform in this invocation; when resuming, "
            "these are additional updates rather than a target checkpoint version"
        ),
    )
    parser.add_argument("--ppo-epochs", type=int, default=GOLDEN_TRAINING_PROTOCOL.ppo_epochs)
    parser.add_argument(
        "--minibatch",
        type=int,
        default=None,
        help=(
            f"decisions a minibatch of PPO takes; where it is not given, {IN_FLIGHT_MINIBATCH} "
            f"with duels kept in flight and {GOLDEN_TRAINING_PROTOCOL.minibatch} otherwise, or "
            "what a run that goes on began with"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=GOLDEN_TRAINING_PROTOCOL.learning_rate,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=GOLDEN_TRAINING_PROTOCOL.eval_every,
        help="run held-out evaluation every N updates; 0 disables it",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=1,
        help=(
            "keep the checkpoint of every Nth update and the latest, removing "
            "the others of this invocation as it goes (default 1: keep all)"
        ),
    )
    parser.add_argument(
        "--eval-suite",
        type=Path,
        help="held-out evaluation suite JSON (default: the learner deck's screen)",
    )
    parser.add_argument(
        "--imitation-only-updates",
        type=int,
        default=0,
        help=(
            "number of initial updates in which the deck's pilot plays the "
            "learner's seat and the learner imitates it"
        ),
    )
    parser.add_argument(
        "--imitation-records",
        type=Path,
        help=(
            "rows of a `policy-bench search --record true` run: the imitation-only "
            "updates play these duels again and imitate the answers they hold, "
            "in place of duels the pilot plays now"
        ),
    )
    parser.add_argument(
        "--imitation-game",
        choices=RECORDED_GAMES,
        default="search",
        help=(
            "which game of each row to imitate: the pilot's with the search "
            "(default), or the pilot's own on the same deal"
        ),
    )
    parser.add_argument(
        "--imitation-games",
        type=int,
        help=(
            "imitate this many of the recorded games, an equal share of every "
            "opponent (default: all); a smaller number is part of a larger one"
        ),
    )
    parser.add_argument(
        "--imitation-label",
        choices=IMITATION_LABELS,
        default="played",
        help=(
            "which answer of a recorded game the policy imitates: the one played "
            "(default), or the pilot's own where a search played another"
        ),
    )
    parser.add_argument(
        "--imitation-lead-weight",
        type=float,
        default=0.0,
        help=(
            "weight of the lead head's squared error against the leads a search "
            "logged with --log true or full for the answers it tried (default 0: "
            "the head is not trained)"
        ),
    )
    parser.add_argument(
        "--imitation-lead-detach",
        action="store_true",
        help=(
            "the lead head reads the policy's view of an answer and sends no "
            "gradient of its own into it"
        ),
    )
    parser.add_argument(
        "--imitation-teacher",
        action="append",
        metavar="[DECK=]CHECKPOINT",
        help=(
            "a checkpoint the imitation-only updates imitate in the pilot's place: it "
            "plays the learner's seat by its first choice and its answers are the ones "
            "imitated; for a learner of several decks, DECK=CHECKPOINT once for each, "
            "each playing the seats of its deck"
        ),
    )
    parser.add_argument(
        "--imitation-teacher-targets",
        choices=TEACHER_TARGETS,
        default="choice",
        help=(
            "what of --imitation-teacher is imitated: its choice at each decision, as a "
            "pilot's answer is (default), or its distribution over the answers"
        ),
    )
    parser.add_argument(
        "--teacher-imitation-weight",
        type=float,
        default=0.0,
        help=(
            "weight of the pilot's answers to the learner's own decisions, "
            "added to the PPO objective"
        ),
    )
    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=0.01,
        help="weight of the PPO entropy bonus (default: 0.01)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.2,
        help="PPO's clip on the ratio of an answer's chances to the rollout's (default: 0.2)",
    )
    parser.add_argument(
        "--upgo",
        action="store_true",
        help=(
            "add UPGO to each transaction's advantage, as ygo-agent's trainer does: the "
            "seat's return followed on while the play after it went at least as well as "
            "the value head expected, less the head's estimate"
        ),
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help=(
            "stop an update's remaining PPO epochs once a minibatch's approximate "
            "KL to the rollout policy exceeds 1.5x this value (default: off)"
        ),
    )
    parser.add_argument(
        "--imitation-value-weight",
        type=float,
        default=0.0,
        help=(
            "value-head regression toward duel outcomes during imitation-only "
            "updates (default 0: the policy alone is trained)"
        ),
    )
    parser.add_argument(
        "--imitation-value-positions",
        type=int,
        default=0,
        help=(
            "how many of a duel's transactions, drawn at random, carry its "
            "outcome as the value head's target in an imitation-only update "
            "(default 0: every one)"
        ),
    )
    parser.add_argument(
        "--ppo-value-positions",
        type=int,
        default=0,
        help=(
            "how many transactions of each seat of a duel, drawn at random, carry "
            "the value head's target in an update whose games the model plays "
            "(default 0: every one)"
        ),
    )
    parser.add_argument(
        "--ppo-value-target",
        choices=PPO_VALUE_TARGETS,
        default="return",
        help=(
            "what the value head is taught in those updates: `return`, the advantage "
            "added back to its estimate (the default), or `result`, how the seat's duel "
            "ended, whatever --duel-gae-lambda the advantages use"
        ),
    )
    parser.add_argument(
        "--imitation-value-detach",
        action="store_true",
        help=(
            "in imitation-only updates the value head reads the shared state "
            "and sends no gradient of its own into it"
        ),
    )
    parser.add_argument("--teacher-imitation-decay-updates", type=int, default=0)
    parser.add_argument(
        "--duel-gae-lambda",
        type=float,
        default=GOLDEN_TRAINING_PROTOCOL.duel_gae_lambda,
        help="GAE trace decay",
    )
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=1.0,
        help=(
            "weight of the PPO policy term; 0 with --teacher-imitation-weight "
            "trains on teacher labels of the learner's own states alone (DAgger)"
        ),
    )
    parser.add_argument(
        "--teacher-label-fraction",
        type=float,
        default=1.0,
        help=(
            "share of the learner's decisions that keep their teacher label for "
            "--teacher-imitation-weight (default 1: all of them); the games are "
            "the same whatever the share"
        ),
    )
    parser.add_argument(
        "--rollout-estimates",
        choices=ROLLOUT_ESTIMATES,
        default="kept",
        help=(
            "what an update of PPO trains with at each decision: the chances and value its game "
            "was answered with (kept, the default), or those worked out again over the update's "
            "decisions at once (recomputed, the trainer as it was before 2026-10-20)"
        ),
    )
    parser.add_argument(
        "--duels-in-flight",
        type=int,
        default=None,
        help=(
            "duels kept in flight from one update of PPO to the next, a multiple of --duels: "
            "the games of the updates ahead under way while an update's are, an update made of "
            "the next --duels to end, each trained on once (needs --rollout-estimates kept); "
            "0 plays every update's games to their end. Where it is not given: on a card, "
            f"{IN_FLIGHT_UPDATES} times --duels for a run whose updates can keep them (of PPO "
            "from its start, not evaluated as it goes, --eval-every 0, with no critic, card "
            "head, held checkpoint's duels or teacher), the pool's slots raised to hold them "
            "where they do not; 0 "
            "otherwise; or what a run that goes on began with"
        ),
    )
    parser.add_argument(
        "--overlap",
        choices=OVERLAPS,
        default="",
        help=(
            "play the next update's games beside an update's pass, with the weights from "
            "before it, a duel's answers one update older (needs duels in flight; "
            "experiments/2026-10-23-overlap.md): by a second thread, or, the same updates, "
            "in turn by one thread before the pass, what the second thread is held to"
        ),
    )
    parser.add_argument(
        "--self-play-duels",
        type=int,
        default=0,
        help=(
            "of an update's duels, how many the learner plays against itself: its own "
            "network in both seats with the learner's deck, every decision of the duel "
            "learned from; the others are against the pilots. In updates that imitate "
            "--imitation-teacher, the teacher plays both seats and both are imitated"
        ),
    )
    parser.add_argument(
        "--self-play-pairs",
        choices=SELF_PLAY_PAIRS,
        default="all",
        help=(
            "which of a learner's decks meet in its duels against itself: any two of them, "
            "one in each seat (all), each deck against itself alone (mirrors), or two "
            "different decks alone (cross)"
        ),
    )
    parser.add_argument(
        "--self-play-weights",
        type=Path,
        help=(
            "a JSON file of a weight for each ordered pair of a learner's decks, "
            '{"OWN>OTHER": weight}: its duels against itself shared out by them, read '
            "when the run starts or goes on"
        ),
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        help=(
            "measure the advantages of an update of PPO against a critic that sees the hidden "
            "cards, in training only (ygo_training/critic.py), begun as the value head and kept "
            "beside each checkpoint; the head is still taught its own return, and nothing that "
            "plays holds the critic"
        ),
    )
    parser.add_argument(
        "--critic-deck",
        type=int,
        default=0,
        help="how many of the next cards of each Deck the critic sees as well (default 0)",
    )
    parser.add_argument(
        "--critic-positions",
        type=int,
        default=8,
        help=(
            "of each seat's transactions in a duel, how many the critic is taught the result "
            "of after an update's games, drawn at random (default 8)"
        ),
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=3e-4,
        help="the critic's own learning rate (default 3e-4)",
    )
    parser.add_argument(
        "--card-head",
        action="store_true",
        help=(
            "teach a head that learns how cards work beside PPO (ygo_training/card_head.py): "
            "where the cards of each transaction are at its horizon and what changed, its "
            "gradient reaching the network; kept beside each checkpoint, read by nothing that "
            "plays"
        ),
    )
    parser.add_argument(
        "--card-head-weight",
        type=float,
        default=1.0,
        help=(
            "the weight of the gradient the head gives the network beside PPO's; the head "
            "itself learns as much whatever it is (default 1)"
        ),
    )
    parser.add_argument(
        "--card-head-begun",
        type=Path,
        help="the head a run begins with, as a probe left it (tools/card_head.py); anew if none",
    )
    _opponent_checkpoint_argument(parser)
    parser.add_argument(
        "--opponent-checkpoint-duels",
        type=int,
        default=0,
        help=(
            "of an update's duels, how many are against --opponent-checkpoint, of which "
            "only the learner's own decisions are learned from"
        ),
    )
    _resume_arguments(parser)
    parser.add_argument("--resume-learning-rate", type=float)
    parser.set_defaults(func=command)


def _status_parser(commands: Any, command: Command) -> None:
    parser = commands.add_parser("status", help="print training progress")
    parser.add_argument(
        "metrics", type=Path, nargs="?", default=Path("runs/training/metrics.jsonl")
    )
    parser.add_argument("--eval-only", action="store_true")
    view = parser.add_mutually_exclusive_group()
    view.add_argument(
        "--optimization",
        "--ppo",
        dest="optimization",
        action="store_true",
        help=(
            "show per-update optimization health: value explained variance, "
            "approximate KL, clip fraction, entropy, and imitation fit"
        ),
    )
    view.add_argument(
        "--evaluation-suite",
        "--eval-suite",
        dest="evaluation_suite",
        action="store_true",
        help="show pooled and per-matchup evaluation-suite win rates",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(func=command)


def _evaluate_suite_parser(commands: Any, command: Command) -> None:
    parser = commands.add_parser(
        "evaluate-suite",
        help="run a held-out evaluation suite for checkpoints, or for the decks' pilots",
    )
    _runtime_arguments(parser)
    parser.add_argument("checkpoints", type=Path, nargs="*")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "instead of checkpoints, let the pilot of each learner deck play its "
            "seat: the reference a model is measured against, on the same deals"
        ),
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--learner-list",
        type=Path,
        help=(
            "play the learner's seat with the deck list of this .ydk file in place "
            "of its deck's own: a list no pilot was written for, which the pilot of "
            "the learner's deck plays with --pilot"
        ),
    )
    _opponent_checkpoint_argument(parser)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(func=command)


def _evaluate_search_parser(commands: Any, command: Command) -> None:
    parser = commands.add_parser(
        "evaluate-search",
        help=(
            "measure checkpoints on the duels of a search run: their games on its "
            "deals, and their answers where the search left or kept its pilot's"
        ),
    )
    _runtime_arguments(parser)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="rows of a `policy-bench search --record true` run the models did not train on",
    )
    parser.add_argument(
        "--allow-seen-deals",
        action="store_true",
        help=(
            "measure even on deals a checkpoint's run imitated: a check of what it "
            "memorised, not an evaluation (refused otherwise)"
        ),
    )
    parser.add_argument(
        "--lead-margin",
        type=float,
        help=(
            "also play the run's deals by following the lead head: the policy's "
            "first choice, unless another answer is predicted to lead it by this "
            "much (in units of win rate), at the decisions a search examines"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.set_defaults(func=command)


def _edopro_parser(commands: Any, command: Command, default_learner: str) -> None:
    parser = commands.add_parser(
        "edopro",
        help="seat a checkpoint in a room hosted in EDOPro (LAN mode), against a player",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--learner", default=default_learner, help="the deck the checkpoint plays, as it trained"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7911)
    parser.add_argument("--password", default="")
    parser.add_argument("--room-id", type=int, default=0)
    parser.add_argument(
        "--name", help='the player\'s name in the room (default "[AI] <learner> model")'
    )
    parser.add_argument(
        "--version", help="EDOPro's ClientVersion in hex (default: client 41.0, core 11.0)"
    )
    parser.add_argument(
        "--duels", type=int, default=1, help="duels to play, joining the room again for each"
    )
    parser.add_argument(
        "--go-second",
        action="store_true",
        help="take the second turn when the bot wins rock-paper-scissors",
    )
    parser.add_argument(
        "--wait", type=int, default=60, help="seconds to keep trying to reach the room"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("edopro-logs"),
        help="where each duel's .log (the duel in words) and .trace (its messages) are written",
    )
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--replay",
        type=Path,
        help="join nothing: feed a .trace to the seat again and ask the model what it was asked",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="say each decision: the answer, how sure, and the pilot's where it differs",
    )
    parser.set_defaults(func=command)


def _edopro_review_parser(commands: Any, command: Command, default_learner: str) -> None:
    parser = commands.add_parser(
        "edopro-review",
        help=(
            "read a duel EDOPro kept again: each player's moves beside the pilot's of "
            "their deck, and beside a checkpoint's"
        ),
    )
    parser.add_argument(
        "replay",
        type=Path,
        help="replay/_LastReplay.yrp of EDOPro's folder, or a replay it saved (.yrpX)",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="a checkpoint to ask as well, at the moves of whoever played its deck",
    )
    parser.add_argument(
        "--learner", default=default_learner, help="the deck the checkpoint plays, as it trained"
    )
    parser.add_argument(
        "--player",
        help=(
            "whose moves are listed, by a part of their name (default: the player whose "
            "name does not begin with [AI])"
        ),
    )
    parser.add_argument(
        "--decks",
        help=(
            "the deck of the player who went first, then the other's, by their ids in the "
            "pool (default: told from the cards each played)"
        ),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help=(
            "the bot's own trace of the duel (edopro-logs/*.trace): its decisions as the room "
            "showed them are set beside the same decisions of the duel played again here"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/edopro"),
        help="where the review, the duel as a record and its decisions are written",
    )
    parser.set_defaults(func=command)


def build_parser(
    *,
    handlers: dict[str, Command],
    default_learner: str,
) -> argparse.ArgumentParser:
    """Build the public CLI without importing command implementations."""
    parser = argparse.ArgumentParser(
        prog="python -m ygo_training", description="Training on OCGCore duels"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _train_parser(commands, handlers["train"], default_learner)
    _status_parser(commands, handlers["status"])
    if "evaluate-suite" in handlers:
        _evaluate_suite_parser(commands, handlers["evaluate-suite"])
    if "evaluate-search" in handlers:
        _evaluate_search_parser(commands, handlers["evaluate-search"])
    if "edopro" in handlers:
        _edopro_parser(commands, handlers["edopro"], default_learner)
    if "edopro-review" in handlers:
        _edopro_review_parser(commands, handlers["edopro-review"], default_learner)
    return parser
