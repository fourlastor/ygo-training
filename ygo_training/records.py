"""Recorded duels of a ``policy-bench search --record true`` run.

Every row of such a run is one deal played twice by the policy ``a``, in seat
``seat``, against the pilot of ``b``: by the pilot alone (``baseline``) and by
the pilot with the search (``search``).  Each of the two games comes as a
record the duel environment plays again with that seat watched, so a learner
can imitate either, and a model can be asked what it would have answered.

This module reads the rows, puts the games in an order that does not depend
on how the run was scheduled, and deals them out to the updates that imitate
them.  It needs neither the engine nor Torch.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

# The two games of a row: the pilot with the search, and the pilot alone.
GAMES = ("search", "baseline")


@dataclass(frozen=True)
class RecordedGame:
    """One recorded duel, and what it comes to when it is played again."""

    learner: str
    opponent: str
    learner_player: int
    seed: int
    winner: int | None
    # The bench's digest of every question and answer of the duel.
    digest: str
    # Where the search left its pilot: the answers' places among all the
    # answers of the duel, both seats'.  Empty for a pilot's own game.
    deviations: tuple[int, ...]
    # The answers the search examined, when the run logged them.
    examined: tuple[int, ...] | None
    # The record, as the JSON text the environment reads.
    record: str
    # What the search tried at each answer it examined, as JSON text: see
    # `_tried`.  None when the run did not log it.
    tried: str | None = None

    @property
    def outcome(self) -> str:
        """How the duel ended for the watched seat."""
        if self.winner == self.learner_player:
            return "win"
        return "loss" if self.winner == 1 - self.learner_player else "draw"

    def deal(self, index: int) -> dict[str, Any]:
        """The same deal as an evaluation game for whoever sits in the seat."""
        return {
            "index": index,
            "opponent": self.opponent,
            "learner_player": self.learner_player,
            "seed": self.seed,
        }

    def replay(self, index: int) -> dict[str, Any]:
        """What a worker needs to play the duel again as it was recorded."""
        return {
            **self.deal(index),
            "record": self.record,
            "digest": self.digest,
            "deviations": list(self.deviations),
            "examined": None if self.examined is None else list(self.examined),
            "tried": self.tried,
        }


PAYOFF = {"+": 1.0, "-": -1.0, "0": 0.0}


def alternative(label: dict[str, Any]) -> list[Any]:
    """An answer as a search's log names it: its kind, by the name the
    environment's vocabulary gives it, and its card's code (0 without one,
    or when the seat does not see which card it is)."""
    kind = label["kind"]
    if isinstance(kind, dict):
        # The one kind with something to it: a battle position.
        position = kind["Position"]
        kind = (
            "Position"
            + ("FaceUp" if position["face_up"] else "FaceDown")
            + ("Attack" if position["attack"] else "Defense")
        )
    return [str(kind), int(label.get("code") or 0)]


def _tried(entry: dict[str, Any]) -> dict[str, Any]:
    """One answer a search examined, as its log holds it: what it tried, the
    pilot's answer first, and how each did in every world it was played in."""
    return {
        "answer": int(entry["decision"]),
        "labels": [alternative(label) for label in entry["labels"]],
        "played": entry["played"],
        # The alternative the search played: 0 is the pilot's answer.
        "taken": int(entry["chosen"]),
        "outright": bool(entry["skipped"]),
    }


def _leads(tried: str, first: int) -> str:
    """What a search tried at each answer of a game, with the lead of each
    alternative over the pilot's answer in the first ``first`` worlds.

    The search plays the pilot's answer and the alternatives out in the same
    worlds, in stages.  Every alternative it tries goes through the whole
    first stage, whatever comes of it, so a lead over those worlds does not
    depend on how the alternative did: `leads` is, for each alternative, how
    much more often it won there than the pilot's answer (a draw counts as
    half a win), and None where it was not played.  Where the pilot's answer
    wins every one of them nothing else is tried (`outright`), and nothing
    could be ahead.

    `rate` is what the pilot's answer is worth there by the same worlds: its
    mean result over them, a win 1, a loss -1 and a draw 0, and `worlds` how
    many those were.  They are dealt again from what the seat sees, so the
    rate estimates the value of the decision to a seat that goes on as its
    pilot does.
    """
    entries = json.loads(tried)
    for entry in entries:
        played = [[PAYOFF.get(mark) for mark in worlds[:first]] for worlds in entry.pop("played")]
        own = [result for result in played[0] if result is not None]
        entry["rate"] = sum(own) / len(own) if own else None
        entry["worlds"] = len(own)
        leads: list[float | None] = [0.0]
        for worlds in played[1:]:
            both = [(a, b) for a, b in zip(worlds, played[0]) if a is not None and b is not None]
            leads.append(sum(a - b for a, b in both) / (2 * len(both)) if both else None)
        entry["leads"] = leads
    return json.dumps(entries, separators=(",", ":"))


def _game(row: dict[str, Any], game: str, line: int) -> RecordedGame:
    played = row.get(game)
    if not isinstance(played, dict) or not isinstance(played.get("record"), dict):
        raise ValueError(  # noqa: TRY004 - the run lacks an option; no value has a wrong type
            f"Row {line} holds no record of its {game} game: "
            "run policy-bench search with --record true"
        )
    logged = played.get("examined") or []
    return RecordedGame(
        learner=str(row["a"]),
        opponent=str(row["b"]),
        learner_player=int(row["seat"]),
        seed=int(row["seed"]),
        winner=None if played.get("winner") is None else int(played["winner"]),
        digest=str(played["digest"]),
        deviations=tuple(int(item["decision"]) for item in played.get("deviations", [])),
        # A run that logs nothing writes an empty list, as a logged game
        # does in which nothing was examined: the whole run tells which.
        examined=tuple(int(item["decision"]) for item in logged) if logged else None,
        record=json.dumps(played["record"], separators=(",", ":")),
        # An older log names what was examined without the worlds' results.
        tried=(
            json.dumps([_tried(item) for item in logged], separators=(",", ":"))
            if logged and all("played" in item for item in logged)
            else None
        ),
    )


def recorded_games(
    path: str | Path,
    game: str = "search",
    *,
    learner: str | None = None,
    games: int | None = None,
) -> list[RecordedGame]:
    """The ``game`` of every row of a search run, in a settled order.

    The bench writes rows as they finish.  Here they are ordered by the place
    of each game among its opponent's (by seed), opponents taking turns: the
    first ``games`` of that order are then an equal share of every opponent,
    as many going first as second, and a smaller set lies within a larger.
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game {game!r}; a row holds {' and '.join(GAMES)}")
    by_opponent: dict[str, list[RecordedGame]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    with Path(path).open(encoding="utf8") as rows:
        for line, text in enumerate(rows, start=1):
            if not text.strip():
                continue
            item = _game(json.loads(text), game, line)
            if learner is not None and item.learner != learner:
                raise ValueError(
                    f"Row {line} records {item.learner!r}, not the learner's deck {learner!r}"
                )
            learner = item.learner
            key = (item.opponent, item.seed, item.learner_player)
            if key in seen:
                raise ValueError(f"Row {line} repeats the game {key}")
            seen.add(key)
            by_opponent[item.opponent].append(item)
    ordered = sorted(
        (
            (place, opponent, item)
            for opponent, items in by_opponent.items()
            for place, item in enumerate(
                sorted(items, key=lambda item: (item.seed, item.learner_player))
            )
        ),
        key=lambda entry: entry[:2],
    )
    selected = [item for _, _, item in ordered]
    if not selected:
        raise ValueError(f"{path} holds no recorded game")
    if any(item.examined is not None for item in selected):
        # The run logged what it examined: a game without any examined none.
        selected = [
            item if item.examined is not None else replace(item, examined=()) for item in selected
        ]
    if any(item.tried is not None for item in selected):
        # The first stage is as long as the shortest run of worlds in the
        # log: where the pilot's answer wins them all the search stops there.
        first = min(
            len(entry["played"][0])
            for item in selected
            if item.tried is not None
            for entry in json.loads(item.tried)
        )
        selected = [replace(item, tried=_leads(item.tried or "[]", first)) for item in selected]
    if games is not None:
        if not 0 < games <= len(selected):
            raise ValueError(f"{path} holds {len(selected)} games; {games} were asked for")
        selected = selected[:games]
    return selected


def shared_deals(first: list[RecordedGame], second: list[RecordedGame]) -> int:
    """How many deals of ``first`` are also among ``second``.

    A deal is an opponent and a seed, whichever seat the learner took.  The
    bench numbers a matchup's games from the run's seed on, so two runs share
    deals unless their seeds lie further apart than their games per matchup.
    """
    known = {(game.opponent, game.seed) for game in second}
    return sum((game.opponent, game.seed) in known for game in first)


def fingerprint(games: list[RecordedGame]) -> str:
    """What tells one set of recorded games from another."""
    digest = hashlib.sha256()
    for game in games:
        digest.update(
            f"{game.learner}|{game.opponent}|{game.learner_player}|{game.seed}|{game.digest}\n".encode()
        )
    return digest.hexdigest()[:16]


@lru_cache(maxsize=4)
def _pass_order(count: int, seed: int, number: int) -> tuple[int, ...]:
    order = list(range(count))
    random.Random((int(seed) << 24) ^ number).shuffle(order)
    return tuple(order)


def update_games(count: int, duels: int, update: int, seed: int) -> list[int]:
    """Which of ``count`` games an update imitates.

    The games are gone through in passes, each in an order of its own, and
    every update takes the next ``duels`` of them.  A run that is continued
    goes on where it stood, and two runs with the same seed see the same
    games in the same updates.
    """
    if count < 1 or duels < 1 or update < 0:
        raise ValueError("count and duels must be positive and update non-negative")
    first = update * duels
    return [
        _pass_order(count, seed, position // count)[position % count]
        for position in range(first, first + duels)
    ]
