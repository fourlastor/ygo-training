"""Training, evaluation and progress-report CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import card_options, discard, finish, folded_pairing, load, saved_card_options
from .cli_parser import build_parser
from .config import (
    GAE_UNIT,
    GOLDEN_TRAINING_PROTOCOL,
    IN_FLIGHT_MINIBATCH,
    IN_FLIGHT_UPDATES,
    TrainerConfig,
)
from .edopro import play as play_at_a_table
from .edopro_review import review as review_a_duel
from .evaluation import summarize_answer_key, summarize_deals, summarize_results
from .evaluation_suite import (
    against_checkpoint,
    joined_evaluation_suites,
    load_evaluation_suite,
)
from .metrics import (
    evaluation_for,
    format_evaluation_suite_status,
    format_optimization_status,
    format_search_evaluation,
    format_status,
    print_evaluation_suite_status,
    print_optimization_status,
    print_rich_status,
    read,
)
from .ppo_trainer import PpoTrainer
from .records import RecordedGame, fingerprint, recorded_games, shared_deals
from .revisions import duel_revisions, trainer_revision
from .run_repository import RunRepository, checkpoint_version
from .training_logging import CompositeLogger, JsonlLogger, StdoutLogger

DEFAULT_LEARNER = "blackwing"
EVALUATION_SUITES = Path("eval_suites")


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the latest numbered model checkpoint in one run directory."""
    return RunRepository(checkpoint_dir.parent).latest_checkpoint()


def _checkpoint_version(path: Path) -> int:
    return checkpoint_version(path)


def _resume_path(args: argparse.Namespace, checkpoint_dir: Path) -> Path | None:
    if args.continue_run:
        return _latest_checkpoint(checkpoint_dir)
    return args.resume


def _decks(trainer: PpoTrainer, identifiers: str) -> list[str]:
    """The decks a comma-separated list names, each checked against the pool."""
    names = [name.strip() for name in identifiers.split(",") if name.strip()]
    missing = sorted(set(names) - set(trainer.decks))
    if missing:
        raise ValueError(f"Unknown deck(s): {', '.join(missing)}")
    return names


def _suite_path(root: Path, requested: Path | None, learner: str) -> Path:
    """The requested suite, or the screen of the learner's deck."""
    if requested is not None:
        return requested.resolve()
    return (root.resolve() / EVALUATION_SUITES / f"ygo-policies-{learner}-screen.json").resolve()


def _training_suite(
    root: Path, requested: Path | None, learners: list[str]
) -> tuple[dict[str, Any], str]:
    """The suite a run is evaluated on, and where it is from: the requested
    one, or the screen of the learner's deck.  For a learner of several decks
    it is the screens of them all, each deck's games the ones its own screen
    deals."""
    if requested is not None or len(learners) == 1:
        path = _suite_path(root, requested, learners[0])
        return load_evaluation_suite(path), str(path)
    paths = [_suite_path(root, None, learner) for learner in learners]
    joined = joined_evaluation_suites([load_evaluation_suite(path) for path in paths])
    return joined, ", ".join(str(path) for path in paths)


def _run_evaluation_suite(
    trainer: PpoTrainer,
    suite: dict[str, Any],
    *,
    seed: int,
    pilot: bool = False,
    continue_on_case_error: bool = False,
    outcomes: bool = False,
    learner_list: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Play every case of a suite: the live model's games, or its deck's
    pilot's, against the pilot of the opponent's deck or, where the case says
    so, against the checkpoint the trainer holds.  A case may name several
    opponent decks: its games are so many against each, on seeds of their
    own, and all of them are played at once.  With ``outcomes`` a case also
    tells how each of its games ended.  With ``learner_list`` the learner's
    seat plays the deck list of that file in place of its deck's own."""
    if trainer.held is None and any(against_checkpoint(case) for case in suite["cases"]):
        raise ValueError(
            "The suite has a case against a checkpoint: name it with --opponent-checkpoint."
        )
    evaluations: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(suite["cases"]):
        learner = _decks(trainer, str(case["learner"]))
        opponents = _decks(trainer, str(case["opponent"]))
        if len(learner) != 1 or not opponents:
            raise ValueError(f"Evaluation case {case['name']!r} needs one learner and an opponent")
        try:
            results = trainer.evaluate_pool(
                learner[0],
                opponents,
                int(case["games"]),
                seed + int(case.get("seed_index", index)) * 10_000_000,
                pilot=pilot,
                checkpoint=against_checkpoint(case),
                learner_list=learner_list,
            )
        except Exception as exc:
            name = str(case["name"])
            if not continue_on_case_error:
                raise RuntimeError(f"Evaluation case {name!r} failed") from exc
            evaluations[name] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "games_requested": int(case["games"]),
            }
            continue
        evaluations[str(case["name"])] = summarize_results(results, outcomes=outcomes)
    return evaluations


def _deck_list(identifiers: str | None) -> list[str] | None:
    if not identifiers:
        return None
    return [name.strip() for name in identifiers.split(",") if name.strip()]


def measured_as_ever(args: argparse.Namespace) -> dict[str, int]:
    """How a measuring command asks its model, whatever a run takes on a
    card: every duel together and every call as it is, unless it names
    another way itself."""
    return {
        "pool_groups": args.pool_groups or 1,
        "pool_groups_over": args.pool_groups_over or 0,
        "inference_graphs": args.inference_graphs or 0,
    }


def _teachers(named: list[str] | None) -> dict[str, Any]:
    """What `--imitation-teacher` names: one checkpoint, for a learner of one
    deck, or DECK=CHECKPOINT for each deck of a learner of several."""
    if not named:
        return {"imitation_teacher": None}
    split = [item.partition("=") for item in named]
    if not any(sign for _, sign, _ in split):
        if len(named) > 1:
            raise SystemExit("--imitation-teacher names one checkpoint, or one for each deck")
        return {"imitation_teacher": named[0]}
    teachers = {deck: checkpoint for deck, sign, checkpoint in split if sign}
    if len(teachers) != len(named):
        raise SystemExit("--imitation-teacher names a deck for each checkpoint, each deck once")
    return {"imitation_teachers": teachers}


def _keeps_duels_in_flight(args: argparse.Namespace) -> bool:
    """Whether a run's updates can keep duels in flight (`TrainerConfig`) as
    `fast` did: they are of PPO from the run's start, it is not evaluated as
    it goes, its estimates are kept, and its duels have no critic, card head,
    held checkpoint or teacher. A copy keeps the protocol's settings."""
    return (
        not args.imitation_only_updates
        and not args.eval_every
        and args.rollout_estimates == "kept"
        and not args.critic
        and not args.card_head
        and not args.opponent_checkpoint_duels
        and not args.teacher_imitation_weight
    )


def run_defaults(args: argparse.Namespace, described: dict[str, Any] | None = None) -> None:
    """Fill in what a ``train`` command line leaves to the run. A run that
    goes on (``described``, its ``run.json``) keeps the duels in flight and
    the minibatches it began with. A new one on a card keeps
    ``IN_FLIGHT_UPDATES`` updates' duels in flight where its updates can,
    the pool's slots raised to hold them, in minibatches of
    ``IN_FLIGHT_MINIBATCH``; otherwise none, in minibatches of the protocol's.
    Duels in flight on a card have their model's calls answered as they are,
    in two groups: the settings of the run `fast`
    (experiments/2026-10-23-duels-in-flight.md)."""
    card = str(args.device).startswith("cuda")
    if args.duels_in_flight is None:
        if described:
            args.duels_in_flight = int(described.get("duels_in_flight", 0))
        elif card and _keeps_duels_in_flight(args):
            args.duels_in_flight = IN_FLIGHT_UPDATES * args.duels
            groups = args.pool_groups or 2
            slots = -(-args.duels_in_flight // args.pool_threads)
            args.pool_slots = max(args.pool_slots, -(-slots // groups) * groups)
        else:
            args.duels_in_flight = 0
    if args.minibatch is None:
        if described:
            args.minibatch = int(described["minibatch"])
        elif args.duels_in_flight:
            args.minibatch = IN_FLIGHT_MINIBATCH
        else:
            args.minibatch = GOLDEN_TRAINING_PROTOCOL.minibatch
    if args.duels_in_flight and card:
        if args.inference_graphs is None:
            args.inference_graphs = 0
        if args.pool_groups is None:
            args.pool_groups = 2


def trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """The trainer configuration of a parsed ``train`` command line, what it
    leaves to the run filled in as for a new run (`run_defaults`)."""
    run_defaults(args)
    # One deck, or several the learner plays in equal shares.
    learners = _deck_list(args.learner) or [args.learner]
    return TrainerConfig(
        workers=args.workers,
        duels=args.duels,
        pool_threads=args.pool_threads,
        pool_slots=args.pool_slots,
        pool_groups=args.pool_groups,
        pool_groups_over=args.pool_groups_over,
        inference_graphs=args.inference_graphs,
        ppo_epochs=args.ppo_epochs,
        minibatch=args.minibatch,
        limit=args.limit,
        learning_rate=(
            args.resume_learning_rate
            if args.resume_learning_rate is not None
            else args.learning_rate
        ),
        seed=args.seed if args.training_seed is None else args.training_seed,
        device=args.device,
        learner_deck=learners[0],
        learner_decks=learners if len(learners) > 1 else None,
        opponent_decks=_deck_list(args.opponents),
        imitation_only_updates=args.imitation_only_updates,
        imitation_records=None if args.imitation_records is None else str(args.imitation_records),
        copy_records=None if args.copy_records is None else str(args.copy_records),
        imitation_game=args.imitation_game,
        imitation_games=args.imitation_games,
        imitation_label=args.imitation_label,
        imitation_lead_weight=args.imitation_lead_weight,
        imitation_lead_detach=args.imitation_lead_detach,
        **_teachers(args.imitation_teacher),
        imitation_teacher_targets=args.imitation_teacher_targets,
        teacher_imitation_weight=args.teacher_imitation_weight,
        teacher_imitation_decay_updates=args.teacher_imitation_decay_updates,
        policy_loss_weight=args.policy_loss_weight,
        teacher_label_fraction=args.teacher_label_fraction,
        imitation_value_weight=args.imitation_value_weight,
        imitation_value_positions=args.imitation_value_positions,
        ppo_value_positions=args.ppo_value_positions,
        ppo_value_target=args.ppo_value_target,
        imitation_value_detach=args.imitation_value_detach,
        target_kl=args.target_kl,
        entropy_coefficient=args.entropy_coefficient,
        clip_epsilon=args.clip,
        upgo=args.upgo,
        rollout_temperature=args.rollout_temperature,
        duel_gae_lambda=args.duel_gae_lambda,
        board_representation=args.board_representation,
        leave_out=args.leave_out,
        card_table=None if args.card_table is None else str(args.card_table),
        card_descriptions=args.card_descriptions,
        hide_identities=args.hide_identities,
        rollout_estimates=args.rollout_estimates,
        duels_in_flight=args.duels_in_flight,
        overlap=args.overlap,
        self_play_duels=args.self_play_duels,
        self_play_weights=(
            None
            if args.self_play_weights is None
            else json.loads(args.self_play_weights.read_text(encoding="utf8"))
        ),
        self_play_pairs=args.self_play_pairs,
        opponent_checkpoint=(
            None if args.opponent_checkpoint is None else str(args.opponent_checkpoint)
        ),
        opponent_checkpoint_duels=args.opponent_checkpoint_duels,
        critic=args.critic,
        critic_deck=args.critic_deck,
        critic_positions=args.critic_positions,
        critic_learning_rate=args.critic_learning_rate,
        card_head=args.card_head,
        card_head_weight=args.card_head_weight,
        card_head_begun=None if args.card_head_begun is None else str(args.card_head_begun),
    )


def train(args: argparse.Namespace) -> None:
    repository = RunRepository(args.run)
    checkpoint_dir = repository.checkpoint_dir
    metrics_path = repository.metrics_path
    learning_rate = (
        args.resume_learning_rate if args.resume_learning_rate is not None else args.learning_rate
    )
    # What the run is described as, where it goes on: what it began with
    # stands for what its command line leaves out.
    described = (
        json.loads(repository.config_path.read_text(encoding="utf8"))
        if args.continue_run and repository.config_path.exists()
        else {}
    )
    run_defaults(args, described)
    config = trainer_config(args)
    if config.duels_in_flight and args.eval_every:
        raise ValueError(
            "a run with duels in flight is not evaluated as it goes (--eval-every 0): "
            "its duels hold the pool"
        )
    trainer = PpoTrainer(args.root, config)
    learner = config.learner_deck
    evaluation_suite: dict[str, Any] | None = None
    evaluation_suite_path: str | None = None
    if args.eval_every:
        evaluation_suite, evaluation_suite_path = _training_suite(
            args.root, args.eval_suite, config.learners
        )
        for case in evaluation_suite["cases"]:
            _decks(trainer, str(case["learner"]))
            _decks(trainer, str(case["opponent"]))
            if against_checkpoint(case) and trainer.held is None:
                raise ValueError(
                    f"Evaluation case {case['name']!r} is against a checkpoint: "
                    "name it with --opponent-checkpoint."
                )
    start = 0
    # The version the run's imitation-only and imitation-decay schedule counts from.
    schedule_origin = 0
    resume = _resume_path(args, checkpoint_dir)
    if args.resume_learning_rate is not None and resume is None:
        raise ValueError("--resume-learning-rate requires --resume or --continue.")
    if resume:
        start = int(
            load(
                resume,
                trainer.model,
                trainer.optimizer,
                map_location=trainer.device,
                expected_board_representation=config.board_representation,
                expected_leave_out=config.leave_out,
            )["step"]
        )
        # The optimizer state carries its saved LR.  A branch can deliberately
        # override it without changing the established --continue behavior.
        if args.resume_learning_rate is not None:
            for group in trainer.optimizer.param_groups:
                group["lr"] = learning_rate
        # A checkpoint step equals the number of completed rollout batches.
        # Preserve that offset so `--continue` does not replay initial deck /
        # shuffle seeds from update zero.
        trainer.set_rollout_index(start)
        # A new --resume branch starts its imitation-only and imitation-decay
        # schedule at the resumed checkpoint. A continuation goes on from
        # wherever its run began, which for a branch is not version zero.
        schedule_origin = repository.schedule_origin() if args.continue_run else start
        if args.continue_run:
            trainer.set_training_schedule_index(start - schedule_origin)
    # A run's critic, where it has one: begun as the value head of the model
    # as it now is, or as it was kept beside the checkpoint a run goes on from.
    trainer.begin_critic(resume if args.continue_run else None)
    trainer.begin_card_head(resume if args.continue_run else None)
    repository.ensure_versions_available(
        range(start, start + args.updates),
        allow_overwrite=args.allow_overwrite,
    )
    effective_learning_rate = float(trainer.optimizer.param_groups[0]["lr"])
    run_config = {
        "device": trainer.device,
        "workers": args.workers,
        "duels": args.duels,
        "decision_limit": args.limit,
        "ppo_epochs": args.ppo_epochs,
        "minibatch": args.minibatch,
        # How the model is asked in an update's games: the groups a thread's
        # duels are shared out among, and the calls answered at fixed sizes
        # from graphs, which move a decision's scores in their last digits.
        "model_calls": {
            "pool_groups": trainer.calls.groups,
            "pool_groups_over": trainer.calls.groups_over,
            "inference_graphs": trainer.calls.graphs,
        },
        "entropy_coefficient": config.entropy_coefficient,
        "clip_epsilon": config.clip_epsilon,
        "upgo": config.upgo,
        "learner": learner,
        # Every deck the learner plays: more than that one where it plays several.
        "learners": config.learners,
        "opponents": trainer.opponents,
        "imitation_only_updates": args.imitation_only_updates,
        # The recorded duels the imitation-only updates imitate, when they
        # do not imitate duels the pilot plays now.
        "imitation_records": None
        if trainer.recorded is None
        else {
            "path": str(args.imitation_records.resolve()),
            "game": config.imitation_game,
            "games": len(trainer.recorded),
            "fingerprint": fingerprint(trainer.recorded),
        },
        # The champions' recordings those updates play again, when they do,
        # and the champion each deck's was recorded with.
        "copy_records": None
        if trainer.copy_records is None
        else {
            "path": str(args.copy_records.resolve()),
            "teachers": trainer.copy_records.teachers(config.learners),
        },
        # The checkpoint those updates imitate, when it is not the pilot.
        "imitation_teacher": trainer.teacher_description,
        # Who sits in the other seat where it is not a pilot: the learner
        # itself in so many of an update's duels, and a checkpoint held as
        # it is, in so many more and in the suite's cases against it.
        "self_play_duels": config.self_play_duels,
        "self_play_pairs": config.self_play_pairs,
        # The file the weights of those duels are read from, which a run may
        # be given again changed as it goes on: each update's metrics say
        # the weights it was played with.
        "self_play_weights": None
        if args.self_play_weights is None
        else str(args.self_play_weights.resolve()),
        "opponent_checkpoint": trainer.held_description,
        "opponent_checkpoint_duels": config.opponent_checkpoint_duels,
        "teacher_imitation_weight": args.teacher_imitation_weight,
        "teacher_imitation_decay_updates": args.teacher_imitation_decay_updates,
        "policy_loss_weight": args.policy_loss_weight,
        "teacher_label_fraction": args.teacher_label_fraction,
        "imitation_value_weight": args.imitation_value_weight,
        "imitation_value_positions": config.imitation_value_positions,
        "ppo_value_positions": config.ppo_value_positions,
        "ppo_value_target": config.ppo_value_target,
        "imitation_value_detach": config.imitation_value_detach,
        "imitation_label": config.imitation_label,
        "imitation_lead_weight": config.imitation_lead_weight,
        "imitation_lead_detach": config.imitation_lead_detach,
        "target_kl": args.target_kl,
        "rollout_temperature": args.rollout_temperature,
        "duel_gae_lambda": args.duel_gae_lambda,
        "board_representation": config.board_representation,
        "leave_out": list(config.leave_out),
        # What the model is told of a card beside its input (format 16).
        "card_table": None
        if config.card_table is None
        else {"path": config.card_table, "fingerprint": card_options(trainer.model)["card_table"]},
        "card_descriptions": config.card_descriptions,
        "hide_identities": config.hide_identities,
        # What PPO trains with at a decision, and whether duels are kept in
        # flight from one update to the next.
        "rollout_estimates": config.rollout_estimates,
        "duels_in_flight": config.duels_in_flight,
        # Whether the next update's games are played beside the pass.
        "overlap": config.overlap,
        # The critic that sees the hidden cards and gives the credit, where
        # there is one.
        "critic": trainer.critic_description,
        # The head that learns how cards work, where there is one.
        "card_head": trainer.card_head_description,
        "initial_model_fingerprint": trainer.initial_model_fingerprint,
        "gae_unit": GAE_UNIT,
        "learning_rate": effective_learning_rate,
        "updates_per_invocation": args.updates,
        "evaluation_suite": None
        if evaluation_suite is None
        else {
            "path": evaluation_suite_path,
            "name": evaluation_suite.get("name"),
            "primary": evaluation_suite["primary"],
            "cases": evaluation_suite["cases"],
            "seed": args.seed,
            "every_updates": args.eval_every,
        },
        "seed": args.seed,
        "training_seed": config.seed,
        "schedule_origin": schedule_origin,
        # The pilots, the engine, the scripts and the cards the duels run
        # on: a run is not continued across a change of them.
        "duel_revisions": duel_revisions(args.root),
        "trainer_revision": trainer_revision(args.root),
    }
    # A run goes on with the model asked another way than it began with: it
    # is the same model that plays, and the description says the last way.
    mutable = {"updates_per_invocation", "trainer_revision", "model_calls"}
    if args.resume_learning_rate is not None:
        mutable.add("learning_rate")
    # A run that has only imitated so far may go on imitating for longer
    # than it set out to.
    if args.continue_run and repository.may_imitate_longer(
        start - schedule_origin, args.imitation_only_updates
    ):
        mutable.add("imitation_only_updates")
    # A run goes on across a move of the pins only when told to, for a move
    # whose duels are the old ones; the moves are kept with the run.
    run_config["pin_moves"] = list(described.get("pin_moves", []))
    if args.new_pins:
        if not args.continue_run:
            raise ValueError("--new-pins goes with --continue.")
        before = described.get("duel_revisions")
        if before != run_config["duel_revisions"]:
            run_config["pin_moves"].append(
                {"version": start, "from": before, "to": run_config["duel_revisions"]}
            )
            mutable |= {"duel_revisions", "pin_moves"}
    repository.initialize(run_config, continue_run=args.continue_run, mutable_keys=mutable)
    logger = CompositeLogger((JsonlLogger(metrics_path), StdoutLogger()))
    interrupted = False
    # What an update's games are played with: the checkpoint the run goes on
    # from, then each one an update leaves.
    played_with = resume
    try:
        for version in range(start, start + args.updates):
            wall_started = time.perf_counter()
            checkpoint = repository.ensure_checkpoint_available(
                version, allow_overwrite=args.allow_overwrite
            )
            # The checkpoint is written as the update leaves the model, and
            # takes its name when the update is evaluated and about to be
            # logged. A run stopped before that goes on from the checkpoint
            # before and makes the update again: none is there without its
            # row and the evaluation due to it.
            metrics = trainer.train_update(
                checkpoint, version + 1, finish=False, ahead=version + 1 < start + args.updates
            )
            failed = [
                {**report, "played_with": None if played_with is None else str(played_with)}
                for report in trainer.failed_duels()
            ]
            record = {
                "version": version,
                "checkpoint": str(checkpoint),
                "learner": learner,
                **metrics,
            }
            if evaluation_suite is not None and (version + 1) % args.eval_every == 0:
                # Evaluation must be held constant between checkpoints: changing
                # evaluation seeds would make a win-rate curve incomparable.
                record["evaluation_seed"] = args.seed
                evaluation_started = time.perf_counter()
                try:
                    evaluations = _run_evaluation_suite(trainer, evaluation_suite, seed=args.seed)
                except BaseException:
                    discard(checkpoint)
                    raise
                record["evaluation_suite"] = evaluations
                record["evaluation_suite_name"] = evaluation_suite.get("name")
                record["evaluation"] = evaluations[evaluation_suite["primary"]]
                record["evaluation_seconds"] = time.perf_counter() - evaluation_started
                failed += [
                    {**report, "evaluated": True, "played_with": str(checkpoint)}
                    for report in trainer.failed_duels()
                ]
            record["failed_duels"] = len(failed)
            _note_failed_duels(repository.run_dir / FAILED_DUELS, version, failed)
            record["wall_seconds"] = time.perf_counter() - wall_started
            finish(checkpoint)
            logger.log(record)
            # The checkpoint a failed duel was played with is kept, to play it again.
            repository.thin(
                version,
                every=args.keep_checkpoints,
                first=start,
                keep=any(not report.get("evaluated") for report in failed),
            )
            played_with = checkpoint
    except KeyboardInterrupt:
        interrupted = True
        trainer.close(force=True)
        raise
    finally:
        try:
            trainer.close(force=interrupted)
        finally:
            logger.finish()


# Where a run keeps the duels the engine stopped, one line each.
FAILED_DUELS = "failed-duels.jsonl"


def _note_failed_duels(path: Path, version: int, failed: list[dict[str, Any]]) -> None:
    """Keep the duels the engine stopped in an update (``DuelPool.failed``),
    and say so: each with the update, the checkpoint it was played with
    (``played_with``), what it was submitted as (the game, its deal and
    rollout seeds) and its record so far, which ``policy-bench replay
    --input`` plays again once written to a file of its own.  Each was ended
    as one the limit stops."""
    if not failed:
        return
    with path.open("a", encoding="utf8") as kept:
        for report in failed:
            kept.write(json.dumps({"version": version, **report}, default=str) + "\n")
            print(
                f"update {version}: the engine stopped duel {report['duel']} "
                f"({report['failure'].splitlines()[0]}); kept in {path}",
                file=sys.stderr,
                flush=True,
            )


def _with_failed_duels(record: dict[str, Any], failed: list[dict[str, Any]]) -> dict[str, Any]:
    """A suite's record, with the duels the engine stopped in it where it
    stopped any (``DuelPool.failed``), each said on standard error too: each
    ended as one the limit stops, and is told with its game and its record."""
    for report in failed:
        print(
            f"{record['checkpoint_label']}: the engine stopped duel {report['duel']} "
            f"({report['failure'].splitlines()[0]})",
            file=sys.stderr,
            flush=True,
        )
    return {**record, "failed_duels": failed} if failed else record


def _told_as(checkpoints: list[Path]) -> dict[str, Any]:
    """The trainer's model built as the checkpoints' were, for them to load
    into it: told of a card what the first one's was (from format 16), its
    table the one that checkpoint carries.  Checkpoints measured together
    are told the same: `checkpoints.load` refuses one that is not."""
    if not checkpoints or not checkpoints[0].is_file():
        return {}
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    told = saved_card_options(payload.get("metadata", {}))
    if not (told["card_descriptions"] or told["hide_identities"]):
        return {}
    return {
        "card_table": str(checkpoints[0]),
        "card_descriptions": told["card_descriptions"],
        "hide_identities": told["hide_identities"],
    }


def evaluate_suite_checkpoints(args: argparse.Namespace) -> None:
    """Play a suite with each checkpoint, or with the decks' own pilots."""
    if args.pilot == bool(args.checkpoints):
        raise ValueError(
            "Give checkpoints to evaluate, or --pilot for the reference: one of the two."
        )
    suite_path = args.suite.resolve()
    suite = load_evaluation_suite(suite_path)
    first_case = suite["cases"][0]
    learner_list = None
    if args.learner_list is not None:
        learner_list = args.learner_list.resolve()
        if not learner_list.is_file():
            raise FileNotFoundError(f"No deck list at {learner_list}")
    played_list = None if learner_list is None else str(learner_list)
    config = TrainerConfig(
        workers=args.workers,
        duels=args.workers,
        pool_threads=args.pool_threads,
        pool_slots=args.pool_slots,
        **measured_as_ever(args),
        limit=args.limit,
        seed=args.seed,
        device=args.device,
        learner_deck=str(first_case["learner"]),
        opponent_decks=_deck_list(str(first_case["opponent"])),
        board_representation=args.board_representation,
        opponent_checkpoint=(
            None if args.opponent_checkpoint is None else str(args.opponent_checkpoint)
        ),
        **_told_as(args.checkpoints),
    )
    if args.opponent_checkpoint is None and any(
        against_checkpoint(case) for case in suite["cases"]
    ):
        raise ValueError(
            "The suite has a case against a checkpoint: name it with --opponent-checkpoint."
        )
    # The decks of the learner's seat and of the other, in every case: a
    # checkpoint folded for one pairing plays only that one, and with its
    # deck's own list.
    played_pairings = (
        None
        if learner_list is not None
        else {
            (str(case["learner"]), opponent)
            for case in suite["cases"]
            for opponent in _deck_list(str(case["opponent"])) or []
        }
    )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    held: dict[str, Any] | None = None
    with PpoTrainer(args.root, config) as trainer:
        held = trainer.held_description
        folded = None if held is None else folded_pairing(held)
        if folded is not None and {
            (opponent, str(case["learner"]))
            for case in suite["cases"]
            if against_checkpoint(case)
            for opponent in _deck_list(str(case["opponent"])) or []
        } != {folded}:
            raise ValueError(
                f"The checkpoint held was folded for {folded[0]} against {folded[1]} "
                "(tools/fold_deck_lists.py): the suite's cases against it play other decks"
            )
        if args.pilot:
            evaluations = _run_evaluation_suite(
                trainer, suite, seed=args.seed, pilot=True, outcomes=True, learner_list=played_list
            )
            records.append(
                _with_failed_duels(
                    {
                        "checkpoint_label": "pilot",
                        "evaluation_suite_name": suite.get("name"),
                        "evaluation_suite": evaluations,
                        "evaluation": evaluations[suite["primary"]],
                    },
                    trainer.failed_duels(),
                )
            )
        for checkpoint in args.checkpoints:
            if checkpoint.suffix != ".pt":
                raise ValueError(f"Evaluation checkpoint must use the .pt extension: {checkpoint}")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
        for checkpoint in args.checkpoints:
            payload = load(
                checkpoint,
                trainer.model,
                map_location=trainer.device,
                expected_board_representation=config.board_representation,
                pairings=played_pairings,
            )
            trainer.model.eval()
            try:
                version = _checkpoint_version(checkpoint)
            except ValueError:
                version = None
            evaluations = _run_evaluation_suite(
                trainer,
                suite,
                seed=args.seed,
                continue_on_case_error=True,
                outcomes=True,
                learner_list=played_list,
            )
            records.append(
                _with_failed_duels(
                    {
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_label": checkpoint.stem,
                        "checkpoint_step": payload.get("step"),
                        "version": version,
                        "evaluation_suite_name": suite.get("name"),
                        "evaluation_suite": evaluations,
                        "evaluation": evaluations[suite["primary"]],
                    },
                    trainer.failed_duels(),
                )
            )
    result = {
        "suite": str(suite_path),
        "seed": args.seed,
        "duel_revisions": duel_revisions(args.root),
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    # The checkpoint in the other seat of the cases against one.
    if held is not None:
        result["opponent_checkpoint"] = held
    # The list the learner's seat played in place of its deck's own.
    if learner_list is not None:
        result["learner_list"] = {
            "path": str(learner_list),
            "sha256": hashlib.sha256(learner_list.read_bytes()).hexdigest(),
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    print(json.dumps(result, sort_keys=True), flush=True)


def _imitated_games(checkpoint: Path) -> tuple[dict[str, Any], list[RecordedGame] | None] | None:
    """The recorded games the run of a checkpoint imitated, by its `run.json`.

    None when the run imitated none or left no description; the games are
    None when their rows are no longer where the run read them.
    """
    described_at = checkpoint.resolve().parent.parent / "run.json"
    if not described_at.is_file():
        return None
    described = json.loads(described_at.read_text(encoding="utf8")).get("imitation_records")
    if not described:
        return None
    if not Path(described["path"]).is_file():
        return described, None
    return described, recorded_games(
        described["path"], described["game"], games=int(described["games"])
    )


def evaluate_search(args: argparse.Namespace) -> None:
    """Measure checkpoints on the duels of a search run they did not train on.

    Each checkpoint plays the run's deals itself, which sets it beside the
    pilot and the search deal by deal, and is asked what it would answer at
    every decision of the searched duels: the search's answer where it left
    the pilot's, the pilot's where it kept it.
    """
    searched = recorded_games(args.records, "search")
    piloted = recorded_games(args.records, "baseline")
    learner = searched[0].learner
    # Deals of the run that each checkpoint's own run imitated: none, for
    # the run to tell anything about games the model has not seen.
    seen: dict[Path, int | None] = {}
    for checkpoint in args.checkpoints:
        if checkpoint.suffix != ".pt":
            raise ValueError(f"Evaluation checkpoint must use the .pt extension: {checkpoint}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
        imitated = _imitated_games(checkpoint)
        if imitated is None:
            seen[checkpoint] = 0
            continue
        described, games = imitated
        if games is None:
            seen[checkpoint] = None
            print(
                f"{checkpoint}: the games its run imitated ({described['path']}) are not "
                "there to compare with this run's deals",
                file=sys.stderr,
            )
            continue
        seen[checkpoint] = shared_deals(searched, games)
        if seen[checkpoint] and not args.allow_seen_deals:
            raise ValueError(
                f"{seen[checkpoint]} of the run's {len(searched)} deals are among the games "
                f"the run of {checkpoint} imitated ({described['path']}). Measure on a search "
                "run whose seed is further from that one's than its games per matchup, or "
                "pass --allow-seen-deals to see what was memorised."
            )
    config = TrainerConfig(
        workers=args.workers,
        duels=args.workers,
        pool_threads=args.pool_threads,
        pool_slots=args.pool_slots,
        **measured_as_ever(args),
        limit=args.limit,
        seed=args.seed,
        device=args.device,
        learner_deck=learner,
        opponent_decks=sorted({game.opponent for game in searched}),
        board_representation=args.board_representation,
        **_told_as(args.checkpoints),
    )
    references = {
        "pilot": [game.outcome for game in piloted],
        "search": [game.outcome for game in searched],
    }
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with PpoTrainer(args.root, config) as trainer:
        for checkpoint in args.checkpoints:
            payload = load(
                checkpoint,
                trainer.model,
                map_location=trainer.device,
                expected_board_representation=config.board_representation,
            )
            trainer.model.eval()
            try:
                version = _checkpoint_version(checkpoint)
            except ValueError:
                version = None
            played = trainer.evaluate_games(
                learner, [game.deal(index) for index, game in enumerate(searched)]
            )
            watched = trainer.evaluate_games(
                learner, [game.replay(index) for index, game in enumerate(searched)]
            )
            answer_key = summarize_answer_key(watched)
            otherwise = (answer_key.get("tried") or {}).get("examined_otherwise")
            if otherwise:
                raise ValueError(
                    f"{otherwise} decisions of {args.records} are examined by the search and "
                    "not by the trainer's reading of a decision, or the other way round: a "
                    "lead head would be asked where it was not taught"
                )
            led = None
            if args.lead_margin is not None:
                games = trainer.evaluate_games(
                    learner,
                    [game.deal(index) for index, game in enumerate(searched)],
                    lead_margin=args.lead_margin,
                )
                led = {
                    "margin": args.lead_margin,
                    # First choices left for an answer the lead head led to.
                    "changed": sum(int(game["led"]) for game in games),
                    **summarize_deals(games, references),
                }
            # Checkpoints of several runs are compared: name the run too.
            folder = checkpoint.resolve().parent
            label = (
                f"{folder.parent.name}/{checkpoint.stem}"
                if folder.name == "checkpoints"
                else checkpoint.stem
            )
            records.append(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_label": label,
                    "checkpoint_step": payload.get("step"),
                    "version": version,
                    # Deals its run imitated; None when that could not be told.
                    "seen_deals": seen[checkpoint],
                    "deals": summarize_deals(played, references),
                    # The same deals played by following the lead head.
                    "led_deals": led,
                    "answer_key": answer_key,
                }
            )
    result = {
        "records": str(args.records.resolve()),
        "fingerprint": fingerprint(searched),
        "learner": learner,
        "games": len(searched),
        "duel_revisions": duel_revisions(args.root),
        "checkpoints": records,
        "seconds": time.perf_counter() - started,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    for record in records:
        print(format_search_evaluation(record), flush=True)


def status(args: argparse.Namespace) -> None:
    records = read(args.metrics)
    if args.eval_only:
        records = [
            record for record in records if evaluation_for(record) or record.get("evaluation_suite")
        ]
    if args.optimization and sys.stdout.isatty():
        print_optimization_status(records)
    elif args.optimization:
        print(format_optimization_status(records))
    elif args.evaluation_suite and sys.stdout.isatty():
        print_evaluation_suite_status(records)
    elif args.evaluation_suite:
        print(format_evaluation_suite_status(records))
    elif sys.stdout.isatty():
        print_rich_status(records, verbose=args.verbose)
    else:
        print(format_status(records, verbose=args.verbose))


def main() -> None:
    parser = build_parser(
        handlers={
            "train": train,
            "status": status,
            "evaluate-suite": evaluate_suite_checkpoints,
            "evaluate-search": evaluate_search,
            "edopro": play_at_a_table,
            "edopro-review": review_a_duel,
        },
        default_learner=DEFAULT_LEARNER,
    )
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted; training stopped.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
