"""Many duels at once: the library's pool, and every game played through it.

The library's pool (``crates/ygo-env``, ``pool.rs``) holds many duels on its
own threads and takes them forward in rounds: every duel that waits is
answered, each runs to its next decision or to its end, and the decisions
come back together as the batch the model reads, built by the library
(``native/ygo-model-input``).

A round is the same whatever the threads' pace: its decisions are in the
order of their duels, and a duel's course depends on its setup and its
answers alone.

Training reads a decision again after its duel.  The decisions of duels
submitted to be kept stay in the library, each under a row, and any of them
come back as a batch (``DuelPool.batch``) until they are forgotten: the
update's minibatches are built there, from decisions Python never held.
What is read of a decision beside the model's input (a search's log, a
lead head followed) is read from its view, which a duel submitted to show
its decisions hands over with them.
"""

from __future__ import annotations

import ctypes
import json
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .environment import (
    CARD_SCRIPTS,
    CARDS_DATABASE,
    DECKS_DIRECTORY,
    DuelError,
    Environment,
    _load,
    library_path,
)
from .schedules import OTHER_SEATS, duel_sample_seed, keeps_label
from .searched import Reading, check_replay, probabilities, recorded_setup

# What tells the decisions of a round apart, beside the model's input, and
# what is told of the decisions of observed seats.
META = "meta_"
SAMPLE = "sample_"


@dataclass
class Round:
    """What a round leaves."""

    # The waiting decisions as the model's input, in the order of their duels.
    arrays: dict[str, np.ndarray]
    # For each of them: its duel, the answer to imitate among its choices (-1
    # when there is none), the seat, the transaction and the place of the
    # decision among the duel's engine answers.
    duel: np.ndarray
    label: np.ndarray
    player: np.ndarray
    transaction: np.ndarray
    answer: np.ndarray
    # The row each is kept under, -1 where its duel is not kept.
    row: np.ndarray
    # The decisions of observed seats answered in the round, of the duels
    # that are kept, in the order of their duels and of play: under ``row``,
    # ``duel``, ``label``, ``pilot`` (the accompanying pilot's own answer),
    # ``chosen`` (the answer played), ``player``, ``transaction`` and
    # ``answer``, a number for each.
    samples: dict[str, np.ndarray]
    # The views the waiting decisions and the observed ones were packed
    # from, in the same orders, where their duels show them (``show``), and
    # None for the others.
    views: list[dict[str, Any] | None]
    sample_views: list[dict[str, Any] | None]
    # The duels that ended, each with its number and how it ended.
    finished: list[dict[str, Any]]
    # Duels submitted and not yet ended.
    in_flight: int
    # How long the round took: the library's ``threads`` (the engines, the
    # pilots and the packing of each decision), its building of the
    # ``batch``, and the ``copy`` of the arrays to this side.
    seconds: dict[str, float]

    def __len__(self) -> int:
        return len(self.duel)


class DuelPool:
    """Duels spread over the library's threads and taken forward together."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        threads: int,
        slots: int,
        shared: bool = False,
        groups: int = 1,
    ):
        """``shared``: the threads draw their duels from one queue, each
        taking the next as a slot of its own comes free, where a duel is
        otherwise dealt to the thread of its number in advance.

        ``groups``: every thread's duels are shared out evenly among so many
        groups, which :meth:`begin` and :meth:`finish` take forward one at a
        time, so that the engines of one run while the decisions of another
        are answered.  :meth:`step` takes all of them forward together, as
        it does a pool of one group."""
        root = Path(root).resolve()
        self._lib = _load(str(library_path(root)))
        self._handle = self._lib.ygo_pool_open(
            str(root / CARDS_DATABASE).encode(),
            str(root / CARD_SCRIPTS).encode(),
            str(root / DECKS_DIRECTORY).encode(),
            int(threads),
            int(slots),
            int(shared),
            int(groups),
        )
        if not self._handle:
            raise self._error("cannot open the pool")
        self.capacity = int(threads) * int(slots)
        self.groups = int(groups)
        # What each duel in flight was submitted as, for a report of it if
        # the engine stops it, and the reports not yet taken (`failed`).
        self._about: dict[int, dict[str, Any]] = {}
        self._failed: list[dict[str, Any]] = []
        # The names behind the numbers of a decision, for reports, and a
        # search's reading of one.
        with Environment(root) as environment:
            vocabulary = environment.vocabulary
        self.reading = Reading(vocabulary)
        self.decision_names = self.reading.decision_names
        self.choice_names = self.reading.choice_names

    def _error(self, what: str) -> DuelError:
        message = self._lib.ygo_env_last_error()
        return DuelError(f"{what}: {message.decode() if message else 'unknown error'}")

    def submit(
        self,
        setups: list[dict[str, Any]],
        *,
        keep: bool = False,
        show: bool = False,
        about: list[dict[str, Any]] | None = None,
    ) -> int:
        """Queue duels (see ``Setup`` in ``crates/ygo-env``): the number of the
        first, the others following it.  They start as slots are free.  With
        ``keep`` their decisions stay in the library, for :meth:`batch`; with
        ``show`` each comes in its round with the view it was packed from
        (``Round.views``).  ``about`` says of each duel what the report of
        it says if the engine stops it (:meth:`failed`): the game it is, and
        the seeds that play it again."""
        first = self._lib.ygo_pool_submit(
            self._handle, json.dumps(setups).encode(), int(keep), int(show)
        )
        if first < 0:
            raise self._error("cannot queue the duels")
        for offset, setup in enumerate(setups):
            told = {} if about is None else dict(about[offset])
            # A recorded duel's record is in the report, as played so far.
            told["setup"] = {name: value for name, value in setup.items() if name != "record"}
            self._about[int(first) + offset] = told
        return int(first)

    def failed(self) -> list[dict[str, Any]]:
        """The duels the engine stopped since this was last asked, a card's
        script failing in them or their arena found full: each ended as one
        the limit stops, and is reported with what it was submitted as
        (``about``, ``setup``), its number, what the engine said
        (``failure``), its answers and turns, and its record so far, which
        ``policy-bench replay`` plays again."""
        failed, self._failed = self._failed, []
        return failed

    def _arrays(self, layout: list[dict[str, Any]], read: Any) -> dict[str, np.ndarray]:
        """The arrays of a layout, copied: the library keeps the numbers only
        as long as what ``read`` reads from."""
        arrays: dict[str, np.ndarray] = {}
        for index, entry in enumerate(layout):
            kind = np.dtype(entry["dtype"])
            count = int(np.prod(entry["shape"], dtype=np.int64))
            if count:
                data = read(index)
                if not data:
                    raise self._error(f"cannot read {entry['name']}")
                borrowed = (ctypes.c_char * (count * kind.itemsize)).from_address(data)
                flat = np.frombuffer(borrowed, dtype=kind).copy()
            else:
                flat = np.zeros(0, dtype=kind)
            arrays[entry["name"]] = flat.reshape(entry["shape"])
        return arrays

    def batch(self, rows: Any, *, whole: bool = False) -> dict[str, np.ndarray]:
        """The kept decisions of ``rows`` as the batch the model is given, in
        that order.  With ``whole``, the same decisions with nothing hidden
        but the Decks' order, of duels submitted to keep them (``_setups``):
        what a critic that sees the hidden cards reads in training, never a
        seat that plays."""
        wanted = np.ascontiguousarray(rows, dtype=np.uint64)
        read = self._lib.ygo_pool_batch_whole if whole else self._lib.ygo_pool_batch
        batch = read(
            self._handle, wanted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(wanted)
        )
        if not batch:
            raise self._error("cannot build the batch")
        try:
            layout = json.loads(self._lib.ygo_env_batch_layout(batch))
            return self._arrays(layout, lambda index: self._lib.ygo_env_batch_data(batch, index))
        finally:
            self._lib.ygo_env_batch_free(batch)

    def labels(self, rows: Any) -> dict[str, np.ndarray]:
        """What became of the cards of the kept decisions of ``rows`` by their
        horizons, in that order (``Pool::batch_labels``): of duels submitted
        to follow them (``_setups``), and of each transaction its last
        decision; ``labelled`` is false for the others.  The targets of a head
        that learns how cards work, never what the network reads."""
        wanted = np.ascontiguousarray(rows, dtype=np.uint64)
        batch = self._lib.ygo_pool_batch_labels(
            self._handle, wanted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(wanted)
        )
        if not batch:
            raise self._error("cannot build the labels")
        try:
            layout = json.loads(self._lib.ygo_env_batch_layout(batch))
            return self._arrays(layout, lambda index: self._lib.ygo_env_batch_data(batch, index))
        finally:
            self._lib.ygo_env_batch_free(batch)

    @property
    def kept(self) -> int:
        """How many decisions are kept."""
        return int(self._lib.ygo_pool_kept(self._handle))

    def forget(self) -> None:
        """Let go of the decisions kept.  Their rows are not given again."""
        if self._lib.ygo_pool_forget(self._handle) != 0:
            raise self._error("cannot forget the decisions")

    def forget_rows(self, rows: Any) -> None:
        """Let go of the decisions kept under ``rows``, each kept and named
        once, while duels are in flight or not: all of them, or none."""
        wanted = np.ascontiguousarray(rows, dtype=np.uint64)
        if (
            self._lib.ygo_pool_forget_rows(
                self._handle, wanted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(wanted)
            )
            != 0
        ):
            raise self._error("cannot forget those decisions")

    def step(self, answers: Any = ()) -> Round:
        """Answer the last round's decisions, each with one of its choices and
        in the order of that round, and take every duel to its next decision
        or to its end."""
        given = np.ascontiguousarray(answers, dtype=np.uint32)
        started = time.perf_counter()
        stepped = self._lib.ygo_pool_step(
            self._handle, given.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(given)
        )
        return self._round(stepped, started)

    def begin(self, group: int, answers: Any = ()) -> None:
        """Answer the decisions of the last round of ``group``, in the order
        of that round, and set its duels on their way.  It does not wait for
        them: :meth:`finish` does, and other groups can be begun and finished
        meanwhile."""
        given = np.ascontiguousarray(answers, dtype=np.uint32)
        begun = self._lib.ygo_pool_begin(
            self._handle,
            int(group),
            given.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(given),
        )
        if begun != 0:
            raise self._error(f"group {group} could not be begun")

    def finish(self, group: int) -> Round:
        """Wait for the duels of ``group``, which was begun: its round, as
        :meth:`step` gives the round of all the duels.  Its ``threads``
        seconds are those the slowest thread took over it, and ``wait`` those
        this call stood waiting for the threads."""
        started = time.perf_counter()
        return self._round(self._lib.ygo_pool_finish(self._handle, int(group)), started, True)

    def _round(self, stepped: Any, started: float, grouped: bool = False) -> Round:
        """What the library hands over of a round, copied to this side."""
        if not stepped:
            raise self._error("the round failed")
        returned = time.perf_counter()
        try:
            layout = json.loads(self._lib.ygo_pool_round_layout(stepped))
            arrays = self._arrays(
                layout["arrays"], lambda index: self._lib.ygo_pool_round_data(stepped, index)
            )
        finally:
            self._lib.ygo_pool_round_free(stepped)
        for over in layout["finished"]:
            about = self._about.pop(int(over["duel"]), {})
            if over.get("failure") is not None:
                self._failed.append(
                    {
                        **about,
                        "duel": int(over["duel"]),
                        "failure": over["failure"],
                        "decisions": int(over["decisions"]),
                        "turns": int(over["turns"]),
                        "record": over["record"],
                    }
                )
        none = np.zeros(0, dtype=np.int64)
        seconds = {name: float(value) for name, value in layout["seconds"].items()}
        if grouped:
            # The threads ran while something else was done here: what this
            # side stood waiting for them, and its own copy.
            seconds["copy"] = time.perf_counter() - returned
            seconds["wait"] = max(0.0, returned - started - seconds["batch"])
        else:
            seconds["copy"] = time.perf_counter() - started - sum(seconds.values())
        return Round(
            arrays={
                name: array for name, array in arrays.items() if not name.startswith((META, SAMPLE))
            },
            duel=arrays.get(f"{META}duel", none),
            label=arrays.get(f"{META}label", none),
            player=arrays.get(f"{META}player", none),
            transaction=arrays.get(f"{META}transaction", none),
            answer=arrays.get(f"{META}answer", none),
            row=arrays.get(f"{META}row", none),
            samples={
                name.removeprefix(SAMPLE): array
                for name, array in arrays.items()
                if name.startswith(SAMPLE)
            },
            views=layout["views"],
            sample_views=layout["sample_views"],
            finished=layout["finished"],
            in_flight=int(layout["in_flight"]),
            seconds=seconds,
        )

    def close(self) -> None:
        if self._handle is not None:
            self._lib.ygo_pool_close(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self.close()


def _other_seat(game: dict[str, Any]) -> dict[str, Any]:
    """The seat the learner does not take in ``game``: the pilot of the deck
    there, or one that is asked, where a network sits in it
    (``schedules.OTHER_SEATS``)."""
    other = str(game.get("other", "pilot"))
    if other not in OTHER_SEATS:
        raise ValueError(f"Unknown other seat: {other}")
    return {"control": "pilot" if other == "pilot" else "caller"}


def _held_rows(games: list[dict[str, Any]], places: list[int], players: list[int]) -> np.ndarray:
    """Which decisions of a round are a held checkpoint's to answer: those of
    the seat that is not the learner's, in the duels it sits in."""
    return np.fromiter(
        (
            games[place].get("other") == "checkpoint"
            and player != int(games[place]["learner_player"])
            for place, player in zip(places, players, strict=True)
        ),
        dtype=bool,
        count=len(places),
    )


def _seat_decks(
    games: list[dict[str, Any]], places: list[int], players: list[int], learner: str
) -> list[str]:
    """The deck of the seat that asks each decision of a round: the
    learner's, or the other seat's where a network sits in both."""
    return [
        str(games[place].get("learner", learner))
        if player == int(games[place]["learner_player"])
        else str(games[place]["opponent"])
        for place, player in zip(places, players, strict=True)
    ]


def _first_choices(
    network: Any, arrays: dict[str, np.ndarray], rows: np.ndarray, device: str
) -> np.ndarray:
    """The first choice of ``network`` at the decisions of ``rows`` of a
    round's arrays."""
    import torch

    if network is None:
        raise DuelError("a duel has a checkpoint in the other seat, and none is held")
    tensor = {name: torch.from_numpy(array[rows]).to(device) for name, array in arrays.items()}
    with torch.no_grad():
        logits, _ = network(tensor)
    return logits.argmax(-1).cpu().numpy()


def _setups(
    learner: str,
    games: list[dict[str, Any]],
    seat: dict[str, Any],
    limit: int,
    *,
    whole: bool = False,
    whole_deck: int = 0,
    follow: bool = False,
) -> list[dict[str, Any]]:
    """The duels of ``games``: ``seat`` for the learner and, for the other,
    a pilot, or a seat that is asked where the game has a network there.  The
    learner plays ``learner`` in a game that names no deck for it.  With
    ``whole`` the pool keeps beside each of the network's decisions the same
    decision with nothing hidden (``DuelPool.batch(whole=True)``), and the
    next ``whole_deck`` cards of each Deck as well.  With ``follow`` it keeps
    beside each transaction's last decision what became of its cards by its
    horizon (``DuelPool.labels``)."""
    setups = []
    for game in games:
        # The deck the learner plays in this game, where it plays several.
        own = str(game.get("learner", learner))
        opponent = str(game["opponent"])
        player = int(game["learner_player"])
        seats: list[dict[str, Any]] = [_other_seat(game), _other_seat(game)]
        seats[player] = seat
        setups.append(
            {
                "seed": int(game["seed"]),
                "decks": [own, opponent] if player == 0 else [opponent, own],
                "seats": seats,
                "limit": limit,
                **({"whole": True, "whole_deck": int(whole_deck)} if whole else {}),
                **({"follow": True} if follow else {}),
            }
        )
    return setups


def _trajectories(
    learner: str,
    games: list[dict[str, Any]],
    steps: list[list[dict[str, Any]]],
    ended: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """What ``workers.collect_worker`` sends of each game, in the order given."""
    trajectories = []
    for place, game in enumerate(games):
        over = ended[place]
        trajectory = {
            "kind": "trajectory",
            "worker": int(game["worker"]),
            "winner": over["winner"],
            "steps": steps[place],
            "terminal": not over["limit"],
            "learner_player": int(game["learner_player"]),
            "win_reason": over["reason"],
            "turns": over["turns"],
            "learner_deck": str(game.get("learner", learner)),
            "opponent_deck": str(game["opponent"]),
            "episode_index": int(game["episode_index"]),
        }
        # Who sat in the other seat, where it is not the pilot in every game.
        if "other" in game:
            trajectory["other"] = str(game["other"])
        trajectories.append(trajectory)
    return trajectories


def sampled_games(
    pool: DuelPool,
    model: Any,
    device: str,
    *,
    learner: str,
    games: list[dict[str, Any]],
    limit: int,
    rollout_seed: int,
    temperature: float,
    labels: bool = False,
    label_fraction: float = 1.0,
    held: Any = None,
    groups_over: int = 0,
    graphs: Any = None,
    whole: bool = False,
    whole_deck: int = 0,
    follow: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """The listed games (``schedules.rollout_games``) with the model in the
    learner's seat, an answer drawn at each of its decisions: the
    trajectories ``workers.collect_worker`` sends for them, in the order
    given, and how the model was asked (``prompts``, ``batches``,
    ``max_batch``, ``inference_seconds``) with where the pool's own time
    went (``pool_threads_seconds``, ``pool_batch_seconds``,
    ``pool_copy_seconds``: see ``Round.seconds``).

    A pool of several groups (``DuelPool(groups=...)``) is taken forward a
    group at a time: the engines of one group run while the model answers
    the decisions of another.  The duels and their draws are the same, and
    the batches the model answers in are other batches; what the threads
    took is then their own time over a group's round, and
    ``pool_wait_seconds`` what this side stood waiting for them.  A call of
    the model costs much the same whatever it holds, so that a round split
    in two pays only while it is large: once the groups' last rounds hold
    ``groups_over`` decisions or fewer between them, every duel is taken
    forward together for the rest of the games.  0: by groups to the last
    duel.  ``pool_grouped_batches`` says how many of the calls answered a
    group.

    With ``graphs`` (``graphs.Graphs``) the calls it takes are answered at
    sizes fixed in advance, from CUDA graphs on a card, and the others as
    they are; ``pool_graph_batches`` says how many it took.  A decision's
    scores then differ in their last digits from those of its own batch.

    A game says who sits in the other seat (``other``) where that is not the
    pilot of the deck there.  Against ``self`` the model answers both seats,
    drawing at each, and the steps of the game are those of both.  Against
    ``checkpoint`` the network ``held`` answers the other seat by its first
    choice: those decisions are no steps, and are counted
    (``held_decisions``).

    A step holds the row its decision is kept under in ``pool`` where a
    worker's holds the decision's arrays.  The draw at a decision is seeded
    by its duel and its place in it (``schedules.duel_sample_seed``), not by
    a worker's count of requests: the same deals as the workers play, and
    other draws.  With ``labels`` the pilot's answer is kept beside the
    model's, at the fraction of the decisions ``label_fraction`` says.  With
    ``whole`` and ``follow`` the pool keeps what ``_setups`` says beside the
    model's decisions; neither changes the games.
    """
    import torch

    from .inference import _sample_with_seeds

    first = pool.submit(
        _setups(
            learner,
            games,
            {"control": "caller"},
            limit,
            whole=whole,
            whole_deck=whole_deck,
            follow=follow,
        ),
        keep=True,
        about=[{**game, "rollout_seed": rollout_seed} for game in games],
    )
    steps: list[list[dict[str, Any]]] = [[] for _ in games]
    ended: dict[int, dict[str, Any]] = {}
    asked = {"prompts": 0.0, "batches": 0.0, "max_batch": 0.0, "inference_seconds": 0.0}
    asked.update({f"pool_{part}_seconds": 0.0 for part in ("threads", "batch", "copy")})
    if graphs is not None:
        asked["pool_graph_batches"] = 0.0

    def noted(round_: Round) -> Round:
        """What a round took and which duels ended in it."""
        for part, seconds in round_.seconds.items():
            asked[f"pool_{part}_seconds"] = asked.get(f"pool_{part}_seconds", 0.0) + seconds
        for over in round_.finished:
            ended[int(over["duel"]) - first] = over
        return round_

    def answered(round_: Round) -> Any:
        """The answers to a round's decisions, and their steps kept."""
        count = len(round_)
        started = time.perf_counter()
        places = (round_.duel - first).tolist()
        # The decisions a held checkpoint answers, and the model's own.
        theirs = _held_rows(games, places, round_.player.tolist())
        mine = np.flatnonzero(~theirs)
        chosen = np.zeros(count, dtype=np.int64)
        if len(mine) < count:
            chosen[theirs] = _first_choices(held, round_.arrays, theirs, device)
            asked["held_decisions"] = asked.get("held_decisions", 0.0) + float(count - len(mine))
        if not len(mine):
            asked["inference_seconds"] += time.perf_counter() - started
            return chosen
        seeds = [
            duel_sample_seed(
                rollout_seed, int(games[places[row]]["episode_index"]), len(steps[places[row]])
            )
            for row in mine.tolist()
        ]
        # A round with none of a checkpoint's decisions is given whole.
        given = (
            round_.arrays
            if len(mine) == count
            else {name: array[mine] for name, array in round_.arrays.items()}
        )
        # From a graph of fixed sizes where the call is one of those: the
        # answer is read before the next call, which is written over it.
        made = None if graphs is None else graphs.answer(given)
        with torch.no_grad():
            if made is None:
                tensor = {name: torch.from_numpy(array).to(device) for name, array in given.items()}
                logits, values = model(tensor)
            else:
                logits, values = made
            drawn = _sample_with_seeds(logits, seeds, temperature)
            log_probs = torch.distributions.Categorical(logits=logits / temperature).log_prob(drawn)
        chosen[mine] = drawn.cpu().numpy()
        asked["inference_seconds"] += time.perf_counter() - started
        asked["prompts"] += len(mine)
        asked["batches"] += 1
        asked["max_batch"] = max(asked["max_batch"], float(len(mine)))
        if graphs is not None:
            asked["pool_graph_batches"] += float(made is not None)
        kinds = round_.arrays["action_kind"][mine, chosen[mine]]
        for place, row, answer, log_prob, value, player, transaction, label, question, kind in zip(
            np.asarray(places)[mine].tolist(),
            round_.row[mine].tolist(),
            chosen[mine].tolist(),
            log_probs.tolist(),
            values.tolist(),
            round_.player[mine].tolist(),
            round_.transaction[mine].tolist(),
            round_.label[mine].tolist(),
            round_.arrays["decision_kind"][mine].tolist(),
            kinds.tolist(),
            strict=True,
        ):
            taken = steps[place]
            step = {
                "row": row,
                "chosen": answer,
                "log_prob": log_prob,
                "value": value,
                "player": player,
                "transaction": transaction,
                # DIAGNOSTIC_STEP_KEYS: never tensorized, trained on or
                # fingerprinted.
                "prompt_type": pool.decision_names[question],
                "action_kind": pool.choice_names[kind],
            }
            if (
                labels
                and label >= 0
                and keeps_label(int(games[place]["seed"]), len(taken), label_fraction)
            ):
                step["teacher_chosen"] = label
            taken.append(step)
        return chosen

    def together(answers: Any) -> int:
        """Every duel taken forward a round at a time, to the last: how many
        are left when none asks any more."""
        while True:
            round_ = noted(pool.step(answers))
            if not len(round_):
                return round_.in_flight
            answers = answered(round_)

    def in_turn() -> int:
        """Every group is set on its way; then each in turn is waited for and
        answered, the others' engines running meanwhile.  A group whose round
        comes back empty has no duel left to play or to start.  Once the
        groups' last rounds hold ``groups_over`` decisions or fewer between
        them, the others are waited for and all go on together."""
        left = 0
        going = list(range(pool.groups))
        for group in going:
            pool.begin(group)
        sizes: dict[int, int] = {}
        while going:
            for group in list(going):
                round_ = noted(pool.finish(group))
                if not len(round_):
                    going.remove(group)
                    sizes.pop(group, None)
                    left = round_.in_flight
                    continue
                waiting = {group: answered(round_)}
                sizes[group] = len(round_)
                if len(sizes) < len(going) or sum(sizes.values()) > groups_over:
                    pool.begin(group, waiting[group])
                    continue
                for other in going:
                    if other != group:
                        round_ = noted(pool.finish(other))
                        if len(round_):
                            waiting[other] = answered(round_)
                asked["pool_grouped_batches"] = asked["batches"]
                return together(np.concatenate([waiting[key] for key in sorted(waiting)]))
        asked["pool_grouped_batches"] = asked["batches"]
        return left

    left = together(()) if pool.groups == 1 else in_turn()
    if left:
        raise DuelError(f"{left} duels neither ended nor asked")
    return _trajectories(learner, games, steps, ended), asked


class Locked:
    """A pool whose calls are made one at a time, whichever thread makes
    them: the next update's games played beside an update's pass (`train
    --overlap`), which takes its minibatches from the same pool.  The
    library's calls let go of Python's lock, not of this one."""

    def __init__(self, pool: DuelPool) -> None:
        self._pool = pool
        self._lock = threading.Lock()
        # How long each thread waited for the other's calls, since it last
        # asked (`take_waited`).
        self._waited: dict[int, float] = {}

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._pool, name)
        if not callable(found):
            return found

        def locked(*args: Any, **kwargs: Any) -> Any:
            asked = time.perf_counter()
            with self._lock:
                thread = threading.get_ident()
                self._waited[thread] = self._waited.get(thread, 0.0) + time.perf_counter() - asked
                return found(*args, **kwargs)

        return locked

    def take_waited(self) -> float:
        """How long the calling thread waited for the other's calls since it
        last asked."""
        return self._waited.pop(threading.get_ident(), 0.0)


class Stream:
    """Duels kept in flight from one update to the next (`train
    --duels-in-flight`).

    The games of the updates ahead are under way while an update's are, so
    that the model's calls stay full, where a pool that plays an update's
    games to their end answers fewer and fewer decisions a call as its duels
    end (the tail of an update).  An update is made of the next duels to
    end, each of them trained on once, whole, when it has ended: a duel is
    answered by the weights the model has when it asks, and what the model
    gave at each of its decisions, its chances and its value, is what the
    update trains with.  Between two updates every group of the pool waits,
    its last round held to be answered by the next weights.

    The duels end in the same order however fast the threads are, so that
    an update is the same duels played the same way.  A duel's draws are
    seeded by its own game, as in `sampled_games`; the duels of an update,
    and the weights each decision is answered by, are others than those of
    updates played to their end.  Neither a checkpoint in the other seat nor
    the pilot's answers kept beside the model's are taken.
    """

    def __init__(
        self,
        pool: DuelPool,
        device: str,
        *,
        learner: str,
        limit: int,
        temperature: float,
        graphs: Any = None,
    ) -> None:
        self.pool = pool
        self.device = device
        self.learner = learner
        self.limit = limit
        self.temperature = temperature
        self.graphs = graphs
        # Each duel in flight, or ended and not yet handed over: its game,
        # with the seed of its rollout and the update it began in, its steps,
        # and every row it is kept under.
        self.games: dict[int, dict[str, Any]] = {}
        self.steps: dict[int, list[dict[str, Any]]] = {}
        self.rows: dict[int, list[int]] = {}
        # The duels that ended and are not yet handed over, as they ended.
        self.ended: list[tuple[int, dict[str, Any]]] = []
        # Each group's last round, held over an update; and the rows of the
        # duels handed over, an update's at a time, each let go of once
        # trained on: with the next update's games played beside the pass,
        # those are handed over before these are trained on.
        self.held: dict[int, Round] = {}
        self.handed: deque[list[int]] = deque()
        # The last rollout whose games are submitted, and the updates played.
        self.until: int | None = None
        self.update = 0

    def play(
        self,
        model: Any,
        blocks: Any,
        index: int,
        ahead: int,
        duels: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """The duels of the next update: the games of rollout ``index`` and
        of the ``ahead`` after it under way (``blocks`` gives a rollout's
        games and its seed), and the duels taken forward, ``model``
        answering, until ``duels`` have ended since the last update; those
        of them, as `sampled_games` gives an update's games, in the order
        they ended, and how the model was asked."""
        self.update += 1
        if self.graphs is not None:
            self.graphs.begin_update()
        first = index if self.until is None else self.until + 1
        for block in range(first, index + ahead + 1):
            games, rollout_seed = blocks(block)
            start = self.pool.submit(
                _setups(self.learner, games, {"control": "caller"}, self.limit),
                keep=True,
                about=[
                    {**game, "rollout_seed": rollout_seed, "begun_in": self.update}
                    for game in games
                ],
            )
            for offset, game in enumerate(games):
                duel = start + offset
                self.games[duel] = {**game, "rollout_seed": rollout_seed, "begun_in": self.update}
                self.steps[duel] = []
                self.rows[duel] = []
            self.until = block
        asked = {"prompts": 0.0, "batches": 0.0, "max_batch": 0.0, "inference_seconds": 0.0}
        asked.update({f"pool_{part}_seconds": 0.0 for part in ("threads", "batch", "copy", "wait")})
        if self.graphs is not None:
            asked["pool_graph_batches"] = 0.0
        groups = self.pool.groups
        # Every group on its way: what it held over the last update is
        # answered by the weights the model has now.
        for group in range(groups):
            held = self.held.pop(group, None)
            self.pool.begin(group, () if held is None else self._answered(held, model, asked))
        sizes: dict[int, int] = {}
        while True:
            for group in range(groups):
                round_ = self._noted(self.pool.finish(group), asked)
                sizes[group] = len(round_)
                spent = not any(sizes.values()) and len(sizes) == groups and not round_.in_flight
                if len(self.ended) < duels and not spent:
                    self.pool.begin(group, self._answered(round_, model, asked))
                    continue
                # Enough have ended: every group waits, its round held.
                self.held[group] = round_
                for other in [*range(group + 1, groups), *range(group)]:
                    self.held[other] = round_ = self._noted(self.pool.finish(other), asked)
                asked["pool_grouped_batches"] = asked["batches"]
                asked["stream_in_flight"] = float(round_.in_flight)
                trajectories, ages = self._handed(duels)
                asked["stream_carried"] = float(len(self.ended))
                asked["stream_oldest"] = float(max(ages, default=0))
                asked["stream_age"] = sum(ages) / max(1, len(ages))
                return trajectories, asked

    def trained(self) -> None:
        """Let go of the decisions of the duels of the first update handed
        over and not yet trained on: the update has trained on them."""
        if self.handed:
            rows = self.handed.popleft()
            if rows:
                self.pool.forget_rows(rows)

    def _noted(self, round_: Round, asked: dict[str, float]) -> Round:
        """What a round took, which duels ended in it, and the rows its
        decisions are kept under, the observed seats' and those that wait,
        let go of with their duel."""
        for part, seconds in round_.seconds.items():
            asked[f"pool_{part}_seconds"] = asked.get(f"pool_{part}_seconds", 0.0) + seconds
        for over in round_.finished:
            self.ended.append((int(over["duel"]), over))
        kept = [(round_.samples.get("duel"), round_.samples.get("row")), (round_.duel, round_.row)]
        for duels, rows in kept:
            if rows is not None:
                for duel, row in zip(duels.tolist(), rows.tolist(), strict=True):
                    if row >= 0:
                        self.rows[duel].append(row)
        return round_

    def _answered(self, round_: Round, model: Any, asked: dict[str, float]) -> Any:
        """The answers to a round's decisions, drawn as `sampled_games`
        draws them, and their steps kept."""
        import torch

        from .inference import _sample_with_seeds

        count = len(round_)
        if not count:
            return np.zeros(0, dtype=np.int64)
        started = time.perf_counter()
        duels = round_.duel.tolist()
        seeds = [
            duel_sample_seed(
                int(self.games[duel]["rollout_seed"]),
                int(self.games[duel]["episode_index"]),
                len(self.steps[duel]),
            )
            for duel in duels
        ]
        made = None if self.graphs is None else self.graphs.answer(round_.arrays)
        with torch.no_grad():
            if made is None:
                tensor = {
                    name: torch.from_numpy(array).to(self.device)
                    for name, array in round_.arrays.items()
                }
                logits, values = model(tensor)
            else:
                logits, values = made
            drawn = _sample_with_seeds(logits, seeds, self.temperature)
            log_probs = torch.distributions.Categorical(logits=logits / self.temperature).log_prob(
                drawn
            )
        chosen = drawn.cpu().numpy()
        asked["inference_seconds"] += time.perf_counter() - started
        asked["prompts"] += count
        asked["batches"] += 1
        asked["max_batch"] = max(asked["max_batch"], float(count))
        if self.graphs is not None:
            asked["pool_graph_batches"] += float(made is not None)
        kinds = round_.arrays["action_kind"][np.arange(count), chosen]
        for duel, row, answer, log_prob, value, player, transaction, question, kind in zip(
            duels,
            round_.row.tolist(),
            chosen.tolist(),
            log_probs.tolist(),
            values.tolist(),
            round_.player.tolist(),
            round_.transaction.tolist(),
            round_.arrays["decision_kind"].tolist(),
            kinds.tolist(),
            strict=True,
        ):
            self.steps[duel].append(
                {
                    "row": row,
                    "chosen": answer,
                    "log_prob": log_prob,
                    "value": value,
                    "player": player,
                    "transaction": transaction,
                    # DIAGNOSTIC_STEP_KEYS, as in `sampled_games`.
                    "prompt_type": self.pool.decision_names[question],
                    "action_kind": self.pool.choice_names[kind],
                }
            )
        return chosen

    def _handed(self, duels: int) -> tuple[list[dict[str, Any]], list[int]]:
        """The first ``duels`` of those that ended, as trajectories, and how
        many updates ago each began; their rows are let go of once trained
        on (`trained`)."""
        taken, self.ended = self.ended[:duels], self.ended[duels:]
        games = [self.games.pop(duel) for duel, _ in taken]
        steps = [self.steps.pop(duel) for duel, _ in taken]
        self.handed.append([row for duel, _ in taken for row in self.rows.pop(duel)])
        ended = {place: over for place, (_, over) in enumerate(taken)}
        ages = [self.update - int(game["begun_in"]) for game in games]
        return _trajectories(self.learner, games, steps, ended), ages


def imitation_games(
    pool: DuelPool, *, learner: str, games: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """The listed games (``schedules.rollout_games``) with the pilot of
    ``learner`` in its seat, watched: the trajectories
    ``workers.collect_worker`` sends for them, in the order given.

    A step holds the row its decision is kept under in ``pool`` where a
    worker's holds the decision's arrays.
    """
    watched = {"control": "pilot", "observe": True}
    first = pool.submit(_setups(learner, games, watched, limit), keep=True, about=games)
    # No seat is the caller's: one round takes every duel to its end.
    played = pool.step()
    if len(played) or played.in_flight:
        raise DuelError(f"{played.in_flight} duels the pilots play did not end")
    steps: list[list[dict[str, Any]]] = [[] for _ in games]
    told = played.samples
    if told:
        for row, duel, label, chosen, player, transaction in zip(
            told["row"].tolist(),
            told["duel"].tolist(),
            told["label"].tolist(),
            told["chosen"].tolist(),
            told["player"].tolist(),
            told["transaction"].tolist(),
            strict=True,
        ):
            steps[duel - first].append(
                {
                    "row": row,
                    "chosen": label,
                    "executed_chosen": chosen,
                    "player": player,
                    "teacher": True,
                    "transaction": transaction,
                }
            )
    ended = {int(over["duel"]) - first: over for over in played.finished}
    return _trajectories(learner, games, steps, ended)


# Recorded duels played again in one round, at most: what is shown of their
# decisions is held until the round's duels are read.
REPLAYED_AT_ONCE = 32


def _recorded_rounds(
    pool: DuelPool,
    learner: str,
    games: list[dict[str, Any]],
    limit: int,
    *,
    show: bool,
    forget: bool,
) -> Any:
    """Recorded duels played again (``records.RecordedGame.replay``), the
    learner's seat watched with its pilot asked alongside, so many at a
    time: for each of them in turn, the game, its observed decisions (rows,
    numbers and, with ``show``, views, in the order of play) and how it
    ended, checked to be the duel recorded (``searched.check_replay``).
    With ``forget`` the decisions kept of a round are let go once its duels
    are read, where they are otherwise kept for the update."""
    for start in range(0, len(games), REPLAYED_AT_ONCE):
        chunk = games[start : start + REPLAYED_AT_ONCE]
        setups = []
        for game in chunk:
            opponent = str(game["opponent"])
            names = [learner, opponent] if int(game["learner_player"]) == 0 else [opponent, learner]
            setups.append(recorded_setup(game, names, limit))
        first = pool.submit(
            setups,
            keep=True,
            show=show,
            about=[
                {name: value for name, value in game.items() if name != "record"} for game in chunk
            ],
        )
        # No seat is the caller's: one round takes every duel to its end.
        played = pool.step()
        if len(played) or played.in_flight:
            raise DuelError(f"{played.in_flight} recorded duels did not end")
        told = played.samples
        columns = (
            {name: told[name].tolist() for name in told}
            if told
            else {name: [] for name in ("row", "duel")}
        )
        of: list[list[dict[str, Any]]] = [[] for _ in chunk]
        for place, duel in enumerate(columns["duel"]):
            of[duel - first].append(
                {
                    **{name: values[place] for name, values in columns.items()},
                    "view": played.sample_views[place],
                }
            )
        ended = {int(over["duel"]) - first: over for over in played.finished}
        for place, game in enumerate(chunk):
            over = ended[place]
            try:
                check_replay(game, over)
            except ValueError as error:
                raise ValueError(
                    f"recorded duel {game.get('index')} (seed {game['seed']}, against "
                    f"{game['opponent']}): {error}"
                ) from error
            yield game, of[place], over
        if forget:
            pool.forget()


def _logged(game: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """What a search's log says it tried at each answer of a game, by the
    answer's place: nothing for a game without a log."""
    return {int(entry["answer"]): entry for entry in json.loads(game.get("tried") or "[]")}


def replayed_games(
    pool: DuelPool,
    *,
    learner: str,
    games: list[dict[str, Any]],
    limit: int,
    label: str = "played",
) -> list[dict[str, Any]]:
    """Recorded duels played again for an imitation-only update: the
    trajectories :func:`imitation_games` returns for the games the pilot
    plays now, in the order given, the answers of the record as the labels.

    A game is a recorded duel (``records.RecordedGame.replay``) with its
    place in the update's games (``episode_index``) and the share it was
    dealt in (``worker``).  With ``label`` ``pilot`` the label is the
    pilot's own answer, and a decision the pilot was not asked (a later step
    of a selection the record answers another way) is no step.  A step holds
    the pilot's own answer beside it (``pilot_chosen``) wherever the pilot
    was asked, and where the search's log says what it tried, the lead it
    measured for each alternative (``lead_targets``): the lead head's
    targets, read from the decision's view (``searched.Reading``).
    """
    pilot_labels = label == "pilot"
    reading = pool.reading
    trajectories = []
    shown = any(game.get("tried") for game in games)
    for game, samples, over in _recorded_rounds(
        pool, learner, games, limit, show=shown, forget=False
    ):
        logged = _logged(game)
        steps = []
        for sample in samples:
            pilot = int(sample["pilot"])
            if pilot_labels and pilot < 0:
                continue
            step = {
                "row": int(sample["row"]),
                "chosen": pilot if pilot_labels else int(sample["label"]),
                "executed_chosen": int(sample["chosen"]),
                "player": int(sample["player"]),
                "teacher": True,
                "transaction": int(sample["transaction"]),
            }
            if pilot >= 0:
                # The pilot's own answer, which a searched duel leaves here
                # and there.
                step["pilot_chosen"] = pilot
            entry = logged.pop(int(sample["answer"]), None) if logged else None
            if entry is not None:
                # The lead the search measured for each answer it tried
                # here, the pilot's own first, with 0.
                step["lead_targets"] = reading.targets(sample["view"], entry)
            steps.append(step)
        if logged:
            raise ValueError(
                f"recorded duel {game.get('index')}: the search's log holds {len(logged)} "
                "answers the seat was not asked"
            )
        trajectories.append(
            {
                "kind": "trajectory",
                "worker": int(game["worker"]),
                "winner": over["winner"],
                "steps": steps,
                "terminal": not over["limit"],
                "learner_player": int(game["learner_player"]),
                "win_reason": over["reason"],
                "turns": over["turns"],
                "learner_deck": learner,
                "opponent_deck": str(game["opponent"]),
                "episode_index": int(game["episode_index"]),
                "departures": int(over["departures"]),
            }
        )
    return trajectories


def watched_games(
    pool: DuelPool,
    model: Any,
    device: str,
    *,
    learner: str,
    assignments: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Recorded duels played again, the model asked at every decision of the
    watched seat what it would have answered: by their index.

    Nothing the model says is played: the record is.  Each decision is read
    against the answer that was played and the pilot's own, which differ
    where a search left the pilot.  Where the search's log says what it tried
    at a decision, the model's answer is also set among those alternatives
    (``tried``), with the weight the model gives each, the lead its head
    predicts for each and its value of the decision.  A duel's decisions
    are asked in one call of the model.
    """
    import torch

    reading = pool.reading
    results = []
    for game, samples, _ in _recorded_rounds(
        pool, learner, assignments, limit, show=True, forget=True
    ):
        firsts: list[int] = []
        scores: list[list[float]] = []
        worths: list[float] = []
        heads: list[list[float]] | None = None
        if samples:
            tensor = {
                name: torch.from_numpy(array).to(device)
                for name, array in pool.batch([int(sample["row"]) for sample in samples]).items()
            }
            with torch.no_grad():
                logits, values, leads = model.evaluate(tensor)
            firsts = logits.argmax(-1).tolist()
            scores, worths = logits.tolist(), values.tolist()
            heads = None if leads is None else leads.tolist()
        examined = None if game.get("examined") is None else set(game["examined"])
        with_log = game.get("tried") is not None
        logged = _logged(game)
        questions: dict[str, Counter[str]] = {}
        deviations: list[dict[str, Any]] = []
        tried: list[dict[str, Any]] = []
        examined_otherwise = 0
        opened: set[int] = set()
        for position, sample in enumerate(samples):
            view = sample["view"]
            start = int(sample["transaction"]) not in opened
            opened.add(int(sample["transaction"]))
            offered = len(view["choices"])
            first, said = int(firsts[position]), scores[position][:offered]
            played = int(sample["label"])
            question, played_kind = reading.kinds(view, played)
            counts = questions.setdefault(question, Counter())
            entry = logged.pop(int(sample["answer"]), None)
            if with_log and (entry is not None) != reading.examined(view):
                # A lead head is asked at the decisions a search examines,
                # told from the decision alone: here the log says otherwise.
                examined_otherwise += 1
            if entry is not None:
                tried.append(
                    reading.tried(
                        view,
                        entry,
                        first,
                        said,
                        question,
                        None if heads is None else heads[position][:offered],
                        value=float(worths[position]),
                        start=start,
                    )
                )
            pilot = int(sample["pilot"])
            if pilot < 0:
                # A later step of a selection the record answers another
                # way: the pilot was not asked this.
                counts["unread"] += 1
                continue
            if played == pilot:
                reading_of = "pilot" if first == played else "other"
                counts[f"kept_{reading_of}"] += 1
                if examined is not None and int(sample["answer"]) in examined:
                    counts[f"examined_{reading_of}"] += 1
                continue
            reading_of = "search" if first == played else "pilot" if first == pilot else "other"
            counts[f"left_{reading_of}"] += 1
            weights = probabilities(said)
            deviations.append(
                {
                    "game": int(game["index"]),
                    "opponent": str(game["opponent"]),
                    "answer": int(sample["answer"]),
                    "turn": int(view["turn"]),
                    "question": question,
                    "pilot": reading.kinds(view, pilot)[1],
                    "search": played_kind,
                    "model": reading.kinds(view, first)[1],
                    "reading": reading_of,
                    "search_probability": weights[played],
                    "pilot_probability": weights[pilot],
                }
            )
        results.append(
            {
                "index": int(game["index"]),
                "seed": int(game["seed"]),
                "opponent": str(game["opponent"]),
                "learner_player": int(game["learner_player"]),
                "decisions": len(samples),
                "recorded_deviations": len(game["deviations"]),
                "examined_known": examined is not None,
                "questions": {question: dict(counts) for question, counts in questions.items()},
                "deviations": deviations,
                "tried": tried,
                # Answers the search examined that the seat was not asked.
                "tried_unread": len(logged),
                # Decisions the log holds and `searched.Reading.examined` does
                # not take a search to examine, or the other way round.
                "examined_otherwise": examined_otherwise,
            }
        )
    return sorted(results, key=lambda result: int(result["index"]))


def taught_games(
    pool: DuelPool,
    teachers: dict[str, Any],
    device: str,
    *,
    learner: str,
    games: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """The listed games (``schedules.rollout_games``) with a teacher, a
    network, in the learner's seat, the one of the deck it plays there
    (``teachers``, by deck), taking its first choice at every decision: the
    trajectories :func:`imitation_games` returns for the pilot's games, in
    the order given, and how the networks were asked, as
    :func:`sampled_games` tells it.  Against ``self`` the teacher of the
    other seat's deck plays that seat too, and the steps of the game are
    those of both.

    A step is labelled with the teacher's answer, as a step of the pilot's
    game is with the pilot's.  The deck's pilot is asked alongside, and its
    own answer is held beside the teacher's (``pilot_chosen``): where the
    two differ is what the teacher has that its pilot has not.
    """
    import torch

    first = pool.submit(
        _setups(learner, games, {"control": "caller"}, limit), keep=True, about=games
    )
    steps: list[list[dict[str, Any]]] = [[] for _ in games]
    ended: dict[int, dict[str, Any]] = {}
    asked = {"prompts": 0.0, "batches": 0.0, "max_batch": 0.0, "inference_seconds": 0.0}
    asked.update({f"pool_{part}_seconds": 0.0 for part in ("threads", "batch", "copy")})
    answers: Any = ()
    while True:
        round_ = pool.step(answers)
        for part, seconds in round_.seconds.items():
            asked[f"pool_{part}_seconds"] += seconds
        for over in round_.finished:
            ended[int(over["duel"]) - first] = over
        count = len(round_)
        if not count:
            if round_.in_flight:
                raise DuelError(f"{round_.in_flight} duels neither ended nor asked")
            break
        started = time.perf_counter()
        decks = np.asarray(
            _seat_decks(games, (round_.duel - first).tolist(), round_.player.tolist(), learner)
        )
        chosen = np.zeros(count, dtype=np.int64)
        for deck in sorted(set(decks.tolist())):
            if deck not in teachers:
                raise DuelError(f"no teacher plays {deck}")
            rows = np.flatnonzero(decks == deck)
            asked["batches"] += 1
            asked["max_batch"] = max(asked["max_batch"], float(len(rows)))
            if len(rows) < count:
                chosen[rows] = _first_choices(teachers[deck], round_.arrays, rows, device)
                continue
            # A round of one deck's decisions is given whole.
            tensor = {
                name: torch.from_numpy(array).to(device) for name, array in round_.arrays.items()
            }
            with torch.no_grad():
                logits, _ = teachers[deck](tensor)
                chosen = logits.argmax(-1).cpu().numpy()
        asked["inference_seconds"] += time.perf_counter() - started
        asked["prompts"] += count
        for duel, row, answer, player, transaction, label in zip(
            round_.duel.tolist(),
            round_.row.tolist(),
            chosen.tolist(),
            round_.player.tolist(),
            round_.transaction.tolist(),
            round_.label.tolist(),
            strict=True,
        ):
            step = {
                "row": row,
                "chosen": answer,
                "executed_chosen": answer,
                "player": player,
                "teacher": True,
                "transaction": transaction,
            }
            if label >= 0:
                step["pilot_chosen"] = label
            steps[duel - first].append(step)
        answers = chosen
    return _trajectories(learner, games, steps, ended), asked


def evaluate_games(
    pool: DuelPool,
    model: Any,
    device: str,
    *,
    learner: str,
    assignments: list[dict[str, Any]],
    limit: int,
    pilot: bool = False,
    held: Any = None,
    lead_margin: float | None = None,
    learner_list: str | None = None,
    observe: Callable[[Round, np.ndarray, np.ndarray], None] | None = None,
    forbid: Callable[[Round, np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """The listed games of ``learner``, each with its opponent, seat and seed,
    by their index, played through ``pool``.

    The learner's seat takes the model's first choice at every decision, or
    with ``pilot`` is played by the pilot of its deck.  With ``lead_margin``
    the model follows its lead head where it predicts that much of a lead
    (``searched.Reading.led``), read from the view of each decision, and
    ``led`` says how often that left its first choice.  The other seat is the
    pilot of the deck there, or in a game that says so (``other`` is
    ``checkpoint``) the network ``held``, by its first choice too: what is
    told of a game's decisions is of the learner's seat alone.  With
    ``learner_list`` the learner's seat plays the deck list of that ``.ydk``
    file in place of its deck's own, and its deck's pilot is the one that
    plays it or is asked alongside.  ``observe`` is shown each round once it
    is answered: the round, the rows of the learner's seat, and the answer
    of every row.  ``forbid`` bars the model from answers: given the round
    and the rows of the learner's seat, it tells which answers of each row
    the model may not take (where a row offers nothing else, the row is
    left whole), and ``barred`` says how often that changed its answer.
    """
    import torch

    if forbid is not None and lead_margin is not None:
        raise ValueError("Answers are barred to a model that takes its first choice.")
    games = []
    setups = []
    for assignment in assignments:
        opponent = str(assignment["opponent"])
        seat = int(assignment["learner_player"])
        if str(assignment.get("other", "pilot")) not in ("pilot", "checkpoint"):
            raise ValueError(f"A game is measured against a pilot or a checkpoint: {assignment}")
        seats: list[dict[str, Any]] = [_other_seat(assignment), _other_seat(assignment)]
        seats[seat] = {"control": "pilot" if pilot else "caller"}
        setup = {
            "seed": int(assignment["seed"]),
            "decks": [learner, opponent] if seat == 0 else [opponent, learner],
            "seats": seats,
            "limit": limit,
        }
        if learner_list is not None:
            setup["lists"] = [learner_list, None] if seat == 0 else [None, learner_list]
        setups.append(setup)
        games.append(
            {
                "decisions": 0,
                "labelled": 0,
                "agreed": 0,
                "led": 0,
                "answers": Counter(),
                "barred": 0,
                "over": None,
            }
        )
    first = pool.submit(
        setups,
        show=lead_margin is not None,
        about=[{**assignment, "learner": learner} for assignment in assignments],
    )
    answers: Any = ()
    while True:
        round_ = pool.step(answers)
        for ended in round_.finished:
            games[int(ended["duel"]) - first]["over"] = ended
        if not len(round_):
            if round_.in_flight:
                raise DuelError(f"{round_.in_flight} duels neither ended nor asked")
            break
        count = len(round_)
        places = (round_.duel - first).tolist()
        # The decisions a held checkpoint answers, and the model's own.
        theirs = _held_rows(assignments, places, round_.player.tolist())
        mine = np.flatnonzero(~theirs)
        chosen = np.zeros(count, dtype=np.int64)
        if len(mine) < count:
            chosen[theirs] = _first_choices(held, round_.arrays, theirs, device)
        if len(mine):
            # A round with none of a checkpoint's decisions is given whole.
            given = (
                round_.arrays
                if len(mine) == count
                else {name: array[mine] for name, array in round_.arrays.items()}
            )
            tensor = {name: torch.from_numpy(array).to(device) for name, array in given.items()}
            with torch.no_grad():
                if lead_margin is None:
                    logits, _ = model(tensor)
                else:
                    logits, _, leads = model.evaluate(tensor)
                if forbid is not None:
                    first_choices = logits.argmax(-1)
                    logits = _barred(logits, tensor["action_mask"], forbid(round_, mine))
                    changed = (logits.argmax(-1) != first_choices).cpu().numpy()
                    for place, row in enumerate(mine.tolist()):
                        games[places[row]]["barred"] += int(changed[place])
                chosen[mine] = logits.argmax(-1).cpu().numpy()
            if lead_margin is not None:
                predicted = leads.tolist()
                for place, row in enumerate(mine.tolist()):
                    view = round_.views[row]
                    assert view is not None
                    led = pool.reading.led(
                        view,
                        int(chosen[row]),
                        predicted[place][: len(view["choices"])],
                        float(lead_margin),
                    )
                    games[places[row]]["led"] += int(led != chosen[row])
                    chosen[row] = led
        kinds = round_.arrays["action_kind"][mine, chosen[mine]]
        for kind, row in zip(kinds.tolist(), mine.tolist(), strict=True):
            game = games[places[row]]
            game["decisions"] += 1
            question = pool.decision_names[int(round_.arrays["decision_kind"][row])]
            game["answers"][f"{question}:{pool.choice_names[int(kind)]}"] += 1
            if round_.label[row] >= 0:
                game["labelled"] += 1
                game["agreed"] += int(chosen[row] == round_.label[row])
        if observe is not None:
            observe(round_, mine, chosen)
        answers = chosen
    results = []
    for assignment, game in zip(assignments, games, strict=True):
        over = game["over"]
        seat = int(assignment["learner_player"])
        winner = over["winner"]
        results.append(
            {
                "index": int(assignment["index"]),
                "seed": int(assignment["seed"]),
                "opponent": str(assignment["opponent"]),
                "learner_player": seat,
                "outcome": "win" if winner == seat else "loss" if winner == 1 - seat else "draw",
                "terminal": not over["limit"],
                "win_reason": over["reason"],
                "turns": over["turns"],
                "decisions": game["decisions"],
                "pilot_labels": game["labelled"],
                "pilot_agreements": game["agreed"],
                # First choices left for an answer the lead head leads to.
                "led": game["led"],
                "learner_answers": dict(game["answers"]),
            }
        )
        if forbid is not None:
            results[-1]["barred"] = game["barred"]
    return sorted(results, key=lambda result: int(result["index"]))


def _barred(logits: Any, offered: Any, barred: np.ndarray) -> Any:
    """The scores of the answers ``barred`` put out of reach, but in a row
    that offers nothing else."""
    import torch

    barred_ = torch.from_numpy(barred).to(logits.device) & offered
    left = (offered & ~barred_).any(-1, keepdim=True)
    return logits.masked_fill(barred_ & left, float("-inf"))
