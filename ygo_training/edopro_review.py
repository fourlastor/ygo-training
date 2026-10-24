"""A duel played in EDOPro, read again: each player's moves beside the pilot's
of their deck, and beside a checkpoint's.

EDOPro keeps every duel it plays: `replay/_LastReplay.yrp` holds the last
one, and a replay it saves under its time (a `.yrpX`) holds the same inside.
What is kept is how the duel began (the engine's seed, the rules, both
players' cards in the order the engine was given them) and every answer
either player gave.  Played again on the engine and the card scripts of this
checkout it is the same duel, with nothing hidden, and the library tells who
gave each answer (``Environment::kept`` in ``crates/ygo-env/src/edopro.rs``).

It is then played once more as a recorded duel of the trainer, the pilot of
each player's deck asked alongside at every decision a network would be
asked, and a checkpoint asked too where the deck is the one it plays.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import lzma
import math
import struct
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_inference_model
from .edopro import PROCESSOR_THREADS, describer, sit
from .environment import BUILD_COMMAND, POLICIES_DIRECTORY, DuelError, Environment
from .inference import scored

# A replay's header says what it holds (`ReplayHeader` in EDOPro's replay.h).
COMPRESSED, TEAMS, PUZZLE, COUNTED, WIDE_FLAGS, EXTENDED = 0x1, 0x2, 0x8, 0x20, 0x100, 0x200
# The packet of a `.yrpX` that holds the duel as a `.yrp` does.
KEPT_INSIDE = 231
# How much of a player's Main Deck has to be a deck of the pool's for that
# deck's pilot to be asked about their moves.
SHARED = 0.75
# The bench of the policies, which tells a recorded duel with nothing hidden.
BENCH = f"{POLICIES_DIRECTORY}/target/release/policy-bench"


class _Bytes:
    """A replay's bytes, read in order."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.at = 0

    def take(self, form: str) -> tuple[Any, ...]:
        try:
            values = struct.unpack_from(form, self.data, self.at)
        except struct.error:
            raise ValueError("the replay is cut short") from None
        self.at += struct.calcsize(form)
        return values

    def number(self, form: str = "<I") -> int:
        return int(self.take(form)[0])

    def name(self) -> str:
        raw = self.data[self.at : self.at + 40]
        self.at += 40
        return raw.decode("utf-16le", errors="replace").split("\0")[0]

    def left(self) -> int:
        return len(self.data) - self.at


def _opened(image: bytes) -> tuple[bytes, int, int, list[int] | None, _Bytes]:
    """A replay's kind, flags, time and seed, and what follows its header."""
    if len(image) < 32 or image[:4] not in (b"yrp1", b"yrpX"):
        raise ValueError("not a replay of EDOPro")
    _, flags, time, size = struct.unpack_from("<4I", image, 4)
    start, seed = 32, None
    if flags & EXTENDED:
        seed = list(struct.unpack_from("<4Q", image, 40))
        start = 72
    body = image[start:]
    if flags & COMPRESSED:
        # The header holds LZMA's properties and the size to expect: the
        # stream itself carries no end.
        alone = image[24:29] + struct.pack("<Q", size) + body
        body = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(alone)
        if len(body) != size:
            raise ValueError("the replay is cut short")
    return image[:4], flags, time, seed, _Bytes(body)


def read_replay(image: bytes) -> dict[str, Any]:
    """A duel as a `.yrp` or `.yrpX` of EDOPro keeps it: when it was played
    (``time``), the ``players`` (who went first, then the other), and what
    ``Kept`` of ``crates/ygo-env`` is made of."""
    kind, flags, time, seed, data = _opened(image)
    if flags & (TEAMS | PUZZLE):
        raise ValueError("a duel of two players is read, not one of teams or a puzzle")
    counts = []
    players = []
    for _ in range(2):
        counts.append(data.number() if flags & COUNTED else 1)
        players += [data.name() for _ in range(counts[-1])]
    if counts != [1, 1]:
        raise ValueError("a duel of two players is read, not one of teams")
    if kind == b"yrpX":
        # What the duel showed, message by message, and at its end the duel
        # itself as a `.yrp` keeps one.
        data.number("<Q")
        while data.left() >= 5:
            packet, length = data.take("<BI")
            if packet == KEPT_INSIDE:
                return read_replay(data.data[data.at : data.at + length])
            data.at += length
        raise ValueError("the replay holds what the duel showed and not the duel itself")
    if seed is None:
        raise ValueError("the replay is of an EDOPro that kept its seed otherwise")
    start_lp, start_hand, draw_count = data.take("<3I")
    duel_flags = data.number("<Q" if flags & WIDE_FLAGS else "<I")
    decks = []
    for _ in range(2):
        main = list(data.take(f"<{data.number()}I"))
        extra = list(data.take(f"<{data.number()}I"))
        decks.append({"main": main, "extra": extra})
    # Cards that set rules of their own: none in a duel of two decks.
    data.take(f"<{data.number()}I")
    responses = []
    while data.left():
        length = data.number("<B")
        if length > data.left():
            raise ValueError("the replay is cut short")
        responses.append(data.data[data.at : data.at + length].hex())
        data.at += length
    return {
        "time": time,
        "players": players,
        "seed": seed,
        "flags": duel_flags,
        "start_lp": start_lp,
        "start_hand": start_hand,
        "draw_count": draw_count,
        "decks": decks,
        "responses": responses,
    }


def reread(environment: Environment, kept: dict[str, Any]) -> dict[str, Any]:
    """The kept duel played again: its ``record`` as the policies keep one,
    each answer with who gave it, the ``winner`` (``None`` when the replay
    stops before the duel does) and its ``turns``."""
    try:
        again = environment._lib.ygo_env_kept
    except AttributeError:
        raise DuelError(
            f"the environment was built before it could read a replay: {BUILD_COMMAND}"
        ) from None
    again.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    again.restype = ctypes.c_char_p
    if environment._handle is None:
        raise DuelError("the environment is closed")
    fields = ("seed", "flags", "start_lp", "start_hand", "draw_count", "decks", "responses")
    done = again(environment._handle, json.dumps({key: kept[key] for key in fields}).encode())
    if not done:
        raise environment._error("the replay is not played again")
    return json.loads(done)


def matched(environment: Environment, main: list[int]) -> tuple[str | None, float]:
    """The deck of the pool a Main Deck is, and the share of its cards that
    are that deck's: no deck where too few are."""
    cards = Counter(main)
    best, shared = None, 0.0
    for name in sorted(environment.decks):
        theirs = Counter(environment.decks[name]["main"])
        common = sum((cards & theirs).values()) / max(len(main), 1)
        if common > shared:
            best, shared = name, common
    return (best if shared >= SHARED else None), shared


def _signature(choice: dict[str, Any]) -> tuple[Any, ...]:
    """What tells two answers apart: two copies of a card where they lie
    alike, or two cards the seat cannot see, are one answer."""
    cards = [choice["card"]] if choice["card"] else []
    named = [
        (card["code"], card["controller"], card["location"])
        for card in [*cards, *choice["members"]]
    ]
    return (choice["kind"], choice["description"], tuple(sorted(named)))


def alike(view: dict[str, Any], first: int | None, second: int | None) -> bool:
    if first is None or second is None:
        return False
    return first == second or _signature(view["choices"][first]) == _signature(
        view["choices"][second]
    )


def decisions(
    environment: Environment,
    record: dict[str, Any],
    decks: list[str | None],
    *,
    stop: int | None = None,
    asked: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    shown: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The duel played once more as a recorded one: every decision either
    player took that a network would be asked, with the answer played
    (``chosen``), the pilot's of that player's deck where it has one
    (``pilot``), and what ``asked`` adds to it.  ``stop`` ends a duel that
    was not played to its end after its last answer.  The decisions as the
    seats are shown them are added to ``shown``, in the same order."""
    describe = describer(environment)
    phases = {number: name for name, number in environment.vocabulary["phase"].items()}
    # A seat needs the name of a deck of the pool; its cards are the record's.
    known = min(environment.decks)
    setup = {
        "seed": int(record["seed"][0]) % 2**32,
        "decks": [deck or known for deck in decks],
        "seats": [{"control": "record", "observe": True, "alone": deck is None} for deck in decks],
        "record": record,
    }
    if stop is not None:
        setup["limit"] = stop
    rows = []
    with environment.start(setup) as episode:
        while True:
            event = episode.advance()
            for view in event["samples"]:
                if shown is not None:
                    shown.append(view)
                row = {
                    "player": int(view["player"]),
                    "turn": int(view["turn"]),
                    "phase": phases.get(int(view["phase"]), "?"),
                    "said": [describe(view, index) for index in range(len(view["choices"]))],
                    "chosen": int(view["chosen"]),
                    "pilot": view["pilot"],
                }
                row["kind"] = row["said"][0].split(":")[0]
                row["with_pilot"] = alike(view, row["chosen"], row["pilot"])
                if asked is not None:
                    row.update(asked(view))
                    if "model" in row:
                        row["with_model"] = alike(view, row["chosen"], row["model"])
                        row["pilot_with_model"] = alike(view, row["pilot"], row["model"])
                rows.append(row)
            if event["over"] is not None:
                return rows


def checkpoint_asked(
    model: torch.nn.Module,
    environment: Environment,
    seats: list[bool],
    *,
    device: str = "cpu",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """What a checkpoint says of a decision at the seats of ``seats``: its
    first choice, how sure it is of every answer, and how it rates the
    seat's chances."""

    def said(view: dict[str, Any]) -> dict[str, Any]:
        player = int(view["player"])
        if not seats[player]:
            return {}
        scores, values = scored(model, environment, [view], device)
        logits = [float(value) for value in scores[0].cpu().tolist()]
        top = max(logits)
        weights = [math.exp(value - top) for value in logits]
        return {
            "model": int(scores.argmax(-1)[0]),
            "probabilities": [weight / sum(weights) for weight in weights],
            "chances": (float(values[0]) + 1.0) / 2.0,
        }

    return said


# What a decision holds that a room and a duel played again each have their
# own way of: the place of the answer among the duel's, and the answers a
# duel played again knows of.
OWN = ("answer", "label", "pilot", "chosen")
# A probability that moves by more than this is told.
MOVED = 0.01


def shown_in_a_room(
    environment: Environment, trace: Path, deck: str, played: list[list[int]]
) -> list[dict[str, Any]]:
    """The decisions of a bot's seat as the room showed them: its trace is
    fed to the seat again (``deck`` is the deck it played).

    What a seat is shown follows from what it answered before, so each
    decision is given the answer the duel was played with: ``played`` holds
    them for either seat, in order, as the duel played again tells them.
    """
    views: list[dict[str, Any]] = []

    def answer(view: dict[str, Any]) -> int:
        mine = played[int(view["player"])]
        place = len(views)
        views.append(view)
        given = mine[place] if place < len(mine) else 0
        # A decision that is not the one played leaves the two apart: it is
        # answered somehow, and what was shown is set beside all the same.
        return given if given < len(view["choices"]) else 0

    sit(environment, deck, answer, trace=trace, told=lambda note: None)
    return views


def _leaves(first: Any, second: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Where two decisions differ, field by field."""
    if isinstance(first, dict) and isinstance(second, dict):
        return [
            found
            for key in sorted(set(first) | set(second))
            if not (path == "" and key in OWN)
            for found in _leaves(first.get(key), second.get(key), f"{path}.{key}")
        ]
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        return [
            found
            for place, (one, other) in enumerate(zip(first, second, strict=True))
            for found in _leaves(one, other, f"{path}[{place}]")
        ]
    return [] if first == second else [(path, first, second)]


def beside(
    describe: Callable[[dict[str, Any], int], str],
    room: list[dict[str, Any]],
    here: list[dict[str, Any]],
    asked: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A seat's decisions as a room showed them beside the same decisions of
    the duel played again here: how many are the same in every field, what
    differs in the others, and with ``asked`` where a checkpoint answers
    them otherwise."""
    found: dict[str, Any] = {"room": len(room), "here": len(here)}
    if len(room) != len(here):
        return found
    same = changed = moved = 0
    fields: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    for there, mine in zip(room, here, strict=True):
        differing = _leaves(there, mine)
        same += not differing
        for path, _, _ in differing:
            field = path.rsplit(".", 1)[-1]
            fields[field] += 1
            if path.startswith(".choices[") and len(there["choices"]) == len(mine["choices"]):
                answers[describe(mine, int(path[len(".choices[") : path.index("]")]))] += 1
        if asked is not None and differing:
            first, second = asked(there), asked(mine)
            if "model" in first and "model" in second:
                changed += not alike(mine, first["model"], second["model"])
                gaps = zip(first["probabilities"], second["probabilities"], strict=False)
                moved += max(abs(one - other) for one, other in gaps) > MOVED
    found.update(same=same, fields=dict(fields), answers=dict(answers))
    if asked is not None:
        found.update(first_choice_changed=changed, probabilities_moved=moved)
    return found


def _fidelity(player: str, trace: Path, found: dict[str, Any]) -> list[str]:
    """What :func:`beside` found, in words."""
    if found["room"] != found["here"]:
        return [
            (
                f"{trace.name} holds {found['room']} decisions and {player} took "
                f"{found['here']} in this duel: it is not the trace of this duel"
            )
        ]
    lines = [
        (
            f"{player}'s {found['here']} decisions as the room showed them ({trace.name}) "
            "beside the same decisions of the duel played again here:"
        ),
        f"  the same in every field: {found['same']}",
    ]
    if found["fields"]:
        lines.append(
            "  what differs: "
            + ", ".join(f"{field} ({count})" for field, count in sorted(found["fields"].items()))
        )
    if found["answers"]:
        most = sorted(found["answers"].items(), key=lambda item: (-item[1], item[0]))
        lines.append("  at the answers: " + ", ".join(f"{said} ({count})" for said, count in most))
    if "first_choice_changed" in found:
        lines.append(
            f"  the checkpoint's first choice is another at {found['first_choice_changed']} of "
            f"them, and its probabilities move by more than a point at "
            f"{found['probabilities_moved']}"
        )
    if found["same"] != found["here"]:
        lines.append(
            "  A room plays EDOPro's card scripts and its card database, and the trainer its "
            "own: where the two differ, a decision is not shown alike."
        )
    return lines


def _said(row: dict[str, Any], answer: int) -> str:
    """An answer in words, with how sure the checkpoint is of it."""
    sure = row.get("probabilities")
    return row["said"][answer] + (f" ({sure[answer]:.0%})" if sure else "")


def _share(count: int, total: int) -> str:
    return "-" if not total else f"{count} ({count / total:.0%})"


def report(
    players: list[str], rows: list[dict[str, Any]], reviewed: int, heading: list[str]
) -> list[str]:
    """The review in words: how often each player gave the pilot's answer
    and the checkpoint's, and the reviewed player's decisions where either
    would have answered otherwise."""
    lines = [*heading, ""]
    width = max(len(name) for name in players) + 2
    lines.append(f"{'':<{width}}{'decisions':>10}{'the pilot':>14}{'the checkpoint':>17}")
    for seat, name in enumerate(players):
        mine = [row for row in rows if row["player"] == seat]
        piloted = [row for row in mine if row["pilot"] is not None]
        modelled = [row for row in mine if "model" in row]
        lines.append(
            f"{name:<{width}}{len(mine):>10}"
            f"{_share(sum(row['with_pilot'] for row in piloted), len(piloted)):>14}"
            f"{_share(sum(row['with_model'] for row in modelled), len(modelled)):>17}"
        )
    mine = [row for row in rows if row["player"] == reviewed]
    other = [
        row
        for row in mine
        if (row["pilot"] is not None and not row["with_pilot"])
        or ("model" in row and not row["with_model"])
    ]
    name = players[reviewed]
    lines.append("")
    if not other:
        lines.append(
            f"{name} answered as the pilot and the checkpoint would, where they were asked."
        )
    else:
        modelled = any("model" in row for row in other)
        lines.append(
            f"{name}'s decisions where the pilot or the checkpoint would answer otherwise"
            + (" (in brackets, how sure the checkpoint is of an answer):" if modelled else ":")
        )
    for row in other:
        lines.append(
            f"turn {row['turn']:>2} {row['phase']:<12} played: {_said(row, row['chosen'])}"
        )
        if "model" in row and not row["with_model"]:
            lines.append(f"{'':>21}checkpoint: {_said(row, row['model'])}")
        if row["pilot"] is not None and not row["with_pilot"]:
            same = " (as the checkpoint)" if row.get("pilot_with_model") else ""
            lines.append(f"{'':>21}pilot: {row['said'][row['pilot']]}{same}")
    both = [
        row
        for row in mine
        if row["pilot"] is not None
        and "model" in row
        and not row["with_pilot"]
        and not row["with_model"]
    ]
    if both:
        agreed = sum(row["pilot_with_model"] for row in both)
        lines.append("")
        lines.append(
            f"At {len(both)} of them both would answer otherwise, and at {agreed} they name "
            "the same answer."
        )
    rated = [row for row in mine if "chances" in row]
    if rated:
        turns = sorted({row["turn"] for row in rated})
        by_turn = ", ".join(
            f"{turn}: {sum(r['chances'] for r in rated if r['turn'] == turn) / sum(r['turn'] == turn for r in rated):.0%}"
            for turn in turns
        )
        lines.append("")
        lines.append(
            f"The checkpoint's rating of {name}'s chances at their decisions, by turn: {by_turn}"
        )
    return lines


def story(root: Path, row: Path, output: Path) -> bool:
    """The duel in words with nothing hidden, by the policies' bench where
    it is built: both hands, every activation, and where the pilot asked
    alongside would answer otherwise."""
    bench = root / BENCH
    if not bench.is_file():
        return False
    told = subprocess.run(
        [
            str(bench),
            "replay",
            "--input",
            str(row.resolve()),
            "--recheck",
            "true",
            "--output",
            str(output.resolve()),
        ],
        cwd=root / POLICIES_DIRECTORY,
        capture_output=True,
        text=True,
        check=False,
    )
    if told.returncode:
        print(
            f"the bench did not tell the duel: {(told.stderr or told.stdout).strip()}",
            file=sys.stderr,
        )
    return told.returncode == 0


def review(args: argparse.Namespace) -> None:
    """The ``edopro-review`` command."""
    for what, path in (
        ("replay", args.replay),
        ("checkpoint", args.checkpoint),
        ("trace", args.trace),
    ):
        if path is not None and not Path(path).is_file():
            raise SystemExit(
                f"No {what} at {path}"
                + (": EDOPro keeps its duels in replay/ of its folder" if what == "replay" else "")
            )
    kept = read_replay(args.replay.read_bytes())
    players = kept["players"]
    with Environment(args.root) as environment:
        try:
            again = reread(environment, kept)
        except DuelError as error:
            # The players may be kept in another order than they played in.
            turned = {**kept, "decks": kept["decks"][::-1]}
            try:
                again = reread(environment, turned)
            except DuelError:
                raise SystemExit(
                    f"{error}\nThe duel is the same only on the engine and the card scripts "
                    "it was played on: EDOPro's may be others than this checkout's."
                ) from None
            kept, players = turned, players[::-1]
        record = {**again["record"], "players": players}
        decks = [matched(environment, cards["main"]) for cards in kept["decks"]]
        if args.decks:
            named = [name.strip() for name in args.decks.split(",")]
            if len(named) != 2 or any(name not in environment.decks for name in named):
                raise SystemExit(
                    "--decks names the deck of the player who went first, then the other's"
                )
            decks = [(name, shared) for name, (_, shared) in zip(named, decks, strict=True)]
        names = [name for name, _ in decks]
        heading = [
            f"{args.replay.name}: {players[0]} went first against {players[1]}, {again['turns']} turns, "
            + (
                "not played to its end"
                if again["winner"] is None
                else "a draw"
                if again["winner"] == 2
                else f"won by {players[again['winner']]}"
            )
        ]
        for player, (name, shared) in zip(players, decks, strict=True):
            heading.append(
                f"{player} plays {name}: {shared:.0%} of the Main Deck is that list's"
                if name
                else f"{player} plays no deck of the pool ({shared:.0%} of the nearest): no pilot is asked"
            )
        asked = None
        if args.checkpoint is not None:
            if args.device == "cpu":
                torch.set_num_threads(PROCESSOR_THREADS)
            model, payload = load_inference_model(args.checkpoint, device=args.device)
            # The checkpoint is asked at the seats that play its own deck.
            seats = [name == args.learner for name in names]
            asked = checkpoint_asked(model, environment, seats, device=args.device)
            heading.append(
                f"{args.checkpoint.name} (after {payload.get('step')} updates) is asked where {args.learner} is played"
            )
        stop = None if again["winner"] is not None else len(record["responses"])
        shown: list[dict[str, Any]] = []
        rows = decisions(environment, record, names, stop=stop, asked=asked, shown=shown)
        fidelity: list[str] = []
        if args.trace is not None:
            # The bot's own seat: its trace says which, and plays its deck.
            played = [[row["chosen"] for row in rows if row["player"] == seat] for seat in (0, 1)]
            room = shown_in_a_room(environment, args.trace, args.learner, played)
            seat = int(room[0]["player"]) if room else 0
            mine = [view for view in shown if int(view["player"]) == seat]
            found = beside(describer(environment), room, mine, asked)
            fidelity = ["", *_fidelity(players[seat], args.trace, found)]
    wanted = args.player.lower() if args.player else None
    reviewed = next(
        (
            seat
            for seat, player in enumerate(players)
            if (wanted in player.lower() if wanted else not player.startswith("[AI]"))
        ),
        0,
    )
    lines = [*report(players, rows, reviewed, heading), *fidelity]
    print("\n".join(lines))
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"duel-{kept['time']}"
    stem.with_suffix(".record.json").write_text(json.dumps(record), encoding="utf8")
    stem.with_suffix(".decisions.json").write_text(json.dumps(rows), encoding="utf8")
    stem.with_suffix(".review.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    kept_as = [f"{stem}.review.txt"]
    if names[reviewed] is not None and again["winner"] is not None:
        row = stem.with_suffix(".row.jsonl")
        row.write_text(
            json.dumps(
                {
                    "a": names[reviewed],
                    "seat": reviewed,
                    "game": {"record": record, "winner": again["winner"]},
                }
            )
            + "\n",
            encoding="utf8",
        )
        if story(Path(args.root).resolve(), row, stem.with_suffix(".story.txt")):
            kept_as.append(f"{stem}.story.txt (the duel in words, nothing hidden)")
    print()
    print("kept: " + ", ".join(kept_as))
