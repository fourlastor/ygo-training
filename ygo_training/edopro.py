"""A checkpoint at an EDOPro table, against a player.

The library joins a room hosted in EDOPro with the client of
``ygo-policies-edopro`` and seats the caller of an episode there
(``crates/ygo-env/src/edopro.rs``).  A decision comes as an episode gives
it, and is packed by the library and answered here as an evaluation answers
it: by the model's first choice.  What a network is not asked in a duel of
the trainer is not asked here either: the deck's pilot places the cards, and
a single answer is taken.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_inference_model
from .environment import BUILD_COMMAND, CARDS_DATABASE, DuelError, Environment
from .inference import scored

# What the library calls with what it tells or asks: see `Asked` in
# `crates/ygo-env/src/ffi.rs`.
ASKED = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_char_p)
# The threads PyTorch scores one decision with on a processor: more are
# slower (docs/commands.md, `evaluate-suite`).
PROCESSOR_THREADS = 4


def sit(
    environment: Environment,
    deck: str,
    answer: Callable[[dict[str, Any]], int],
    *,
    room: dict[str, Any] | None = None,
    trace: str | Path | None = None,
    told: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Seat ``answer`` with ``deck`` in a room of EDOPro, or before a trace.

    ``room`` is a ``Room`` of ``crates/ygo-env`` (host, port, duels, ...).
    ``answer`` is given each decision as :meth:`Episode.advance` returns one
    and says which of its choices to take; ``told`` hears how the session
    goes on.  What either raises ends the session and is raised again here,
    an interruption from the keyboard among them.  With ``trace``, a trace
    of ``edopro-bot`` is fed to the seat again and nothing is joined.
    """
    if (room is None) == (trace is None):
        raise ValueError("A table is a room or a trace: one of the two.")
    library = environment._lib
    try:
        seat = library.ygo_env_edopro
    except AttributeError:
        raise DuelError(
            f"the environment was built before it could join a room: {BUILD_COMMAND}"
        ) from None
    seat.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ASKED]
    seat.restype = ctypes.c_char_p
    if environment._handle is None:
        raise DuelError("the environment is closed")
    raised: list[BaseException] = []
    interrupted: list[bool] = []

    def asked(text: bytes | None) -> int:
        # The library cannot carry what is raised here: it is kept, the
        # session is ended, and it is raised again when the library returns.
        try:
            if interrupted:
                raise KeyboardInterrupt
            if text is None:
                return 0
            event = json.loads(text)
            if "note" in event:
                told(str(event["note"]))
                return 0
            return int(answer(event["decision"]))
        except BaseException as error:  # noqa: BLE001
            raised.append(error)
            return -1

    table: dict[str, Any] = {"deck": deck}
    if room is not None:
        table["room"] = room
    else:
        table["trace"] = str(trace)
    # An interruption from the keyboard reaches Python when the library next
    # calls back, anywhere in the call: raised there, before anything can
    # catch it, it would be lost and the session go on.  It is noted while
    # the library runs, and raised when it has returned.
    try:
        before = signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
    except ValueError:
        # Not the main thread: an interruption is not this call's to take.
        before = None
    # Kept alive by its name until the library has returned.
    callback = ASKED(asked)
    try:
        done = seat(environment._handle, json.dumps(table).encode(), callback)
    finally:
        if before is not None:
            signal.signal(signal.SIGINT, before)
    if interrupted and not raised:
        raise KeyboardInterrupt
    if raised:
        raise raised[0]
    if not done:
        raise environment._error("the session ended")
    return json.loads(done)


class FirstChoice:
    """A model's first choice at each decision of a duel, as an evaluation
    takes it, and what it took beside its pilot's own answer."""

    def __init__(
        self,
        model: torch.nn.Module,
        environment: Environment,
        *,
        device: str = "cpu",
        said: Callable[[dict[str, Any], int, float], None] | None = None,
    ) -> None:
        self.model = model
        self.environment = environment
        self.device = device
        self.said = said
        self.decisions = 0
        self.labelled = 0
        self.agreed = 0

    def __call__(self, view: dict[str, Any]) -> int:
        logits, _ = scored(self.model, self.environment, [view], self.device)
        first = logits.argmax(-1)
        chosen = int(first[0])
        self.decisions += 1
        if view.get("label") is not None:
            self.labelled += 1
            self.agreed += int(chosen == int(view["label"]))
        if self.said is not None:
            sure = torch.distributions.Categorical(logits=logits).log_prob(first)
            self.said(view, chosen, math.exp(float(sure[0])))
        return chosen


def describer(environment: Environment) -> Callable[[dict[str, Any], int], str]:
    """Words for an answer to a decision: the question, the answer, its card."""
    decisions = {number: name for name, number in environment.vocabulary["decision"].items()}
    choices = {number: name for name, number in environment.vocabulary["choice"].items()}
    names: dict[int, str] = {}
    database = environment.root / CARDS_DATABASE
    if database.is_file():
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as texts:
            names = {
                int(code): str(name) for code, name in texts.execute("select id, name from texts")
            }

    def describe(view: dict[str, Any], answer: int) -> str:
        choice = view["choices"][answer]
        about = [choice["card"]] if choice["card"] else choice["members"]
        named = ", ".join(
            names.get(int(card["code"]), "a card")
            + (" (theirs)" if int(card["controller"]) else "")
            for card in about
        )
        said = f"{decisions[int(view['question']['kind'])]}: {choices[int(choice['kind'])]}"
        return f"{said} {named}" if named else said

    return describe


def play(args: argparse.Namespace) -> None:
    """The ``edopro`` command: a checkpoint joins a room, or reads a trace."""
    if args.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(PROCESSOR_THREADS)
    model, payload = load_inference_model(args.checkpoint, device=args.device)
    with Environment(args.root) as environment:
        if args.learner not in environment.decks:
            raise ValueError(f"Unknown deck: {args.learner}")
        describe = describer(environment)

        def said(view: dict[str, Any], chosen: int, probability: float) -> None:
            line = f"  {describe(view, chosen)} ({probability:.0%})"
            label = view.get("label")
            if label is not None and int(label) != chosen:
                line += f"; its pilot: {describe(view, int(label))}"
            print(line, flush=True)

        answer = FirstChoice(
            model,
            environment,
            device=args.device,
            said=said if args.verbose else None,
        )
        print(
            f"{args.checkpoint.name} (after {payload.get('step')} updates) plays {args.learner}",
            flush=True,
        )

        def told(note: str) -> None:
            print(note, flush=True)

        if args.replay is not None:
            done = sit(environment, args.learner, answer, trace=args.replay, told=told)
            found = done["replayed"]
            print(
                f"{found['answers']} selections answered, {found['decisions']} of them the "
                f"model's to decide; {found['differ']} answers are not the trace's"
            )
        else:
            room = {
                "host": args.host,
                "port": args.port,
                "password": args.password,
                "room_id": args.room_id,
                "name": args.name or f"[AI] {args.learner} model",
                "duels": args.duels,
                "go_first": not args.go_second,
                "wait": args.wait,
                "log_dir": None if args.no_log else str(args.log_dir),
                "tag": f"{args.learner}-{args.checkpoint.stem}",
            }
            if args.version is not None:
                room["version"] = int(args.version, 16)
            stopped = False
            try:
                done = sit(environment, args.learner, answer, room=room, told=told)
            except DuelError as error:
                raise SystemExit(str(error)) from None
            except KeyboardInterrupt:
                # A session is ended from the keyboard: what was played of
                # it is still said.
                stopped, done = True, {"played": []}
                print("\nstopped")
            for number, duel in enumerate(done["played"], start=1):
                log = f"; {duel['log']}.log" if duel["log"] else ""
                print(f"duel {number}: {duel['result']}, {duel['decisions']} decisions{log}")
        if answer.labelled:
            print(
                f"the model took its pilot's answer at {answer.agreed} of {answer.labelled} "
                f"decisions ({answer.agreed / answer.labelled:.1%})"
            )
        if args.replay is None and stopped:
            raise SystemExit(130)
