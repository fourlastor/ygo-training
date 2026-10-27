"""A copy's games recorded once and played again: each deck's champion in
its own mirror, both seats its own, and against each of the thirteen decks'
pilots (`tools/record_champion.py`), kept as the answers the champion gave,
in the order its decisions came. A copy plays them again with those answers
in the teachers' place (`train --copy-records`, `pool.taught_games` with
``played``): the same decisions, labelled the same, with no network asked.
A game counts only where it ends with the digest it was recorded with, of
every question and answer: another engine, other scripts, cards or pilots
play other games.

A deck's recording is a folder of its own: `KIND/CHUNK.jsonl.gz`, KIND
`mirror` or `against-DECK`, `CHUNK` games each, a line a game, and
`about.json`, what made them. The games of a chunk are a rollout's of the
training schedule of `SEED` (`schedules.TrainingSchedule`), at an index of
the deck's, the kind's and the chunk's own, so no two chunks of the pool's
decks share a deal; none of them is a deal of the suites measured at seed
2012.

A copy deals an update's games as it would play them (`orchestration`):
which deck, against whom, the mirrors among them. Each is then the next game
of its deck's recording of that kind not yet given, in the recording's
order, so a copy goes through a recording once, and two copies of the same
decks and seed through the same games.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from .schedules import MAX_EPISODES_PER_ROLLOUT, TrainingSchedule, rollout_games

# The training seed of every recording's deals: the first from 7,100 whose
# chunks, 64 of each deck's mirror and 8 against each of the thirteen decks'
# pilots, hold none of the suites' deals.
SEED = 7_105
# The games of a chunk, a rollout's.
CHUNK = 1_000
# The rollout indices of a deck's chunks, and of a kind's within them.
DECK_STRIDE = 5_000
KIND_STRIDE = 64
MIRROR = "mirror"
# The deals of the suites measured at seed 2012: that seed, and so many
# million more for each case, and as many games as a suite holds of one.
SUITE_SEED = 2012
SUITE_CASES = 256
SUITE_GAMES = 2048
SUITE_STRIDE = 10_000_000

assert CHUNK <= MAX_EPISODES_PER_ROLLOUT


def kind_of(game: dict[str, Any]) -> str:
    """A copy's game is its deck's mirror, both seats the champion's, or
    against a deck's pilot."""
    if str(game.get("other", "pilot")) == "self":
        return MIRROR
    return f"against-{game['opponent']}"


def rollout_index(decks: list[str], deck: str, kind: str, chunk: int) -> int:
    """The rollout of the training schedule of `SEED` a chunk's games are
    dealt from: one of the deck's, the kind's and the chunk's own, by the
    place of each deck among ``decks``, the pool's."""
    if not 0 <= chunk < KIND_STRIDE:
        raise ValueError(f"a kind's games are in at most {KIND_STRIDE} chunks, not {chunk + 1}")
    against = 0 if kind == MIRROR else 1 + decks.index(kind.removeprefix("against-"))
    if (against + 1) * KIND_STRIDE > DECK_STRIDE:
        raise ValueError("more decks than a deck's rollouts hold kinds of")
    return decks.index(deck) * DECK_STRIDE + against * KIND_STRIDE + chunk


def held_out(seed: int) -> bool:
    """Whether a deal is one of the suites measured at seed 2012."""
    case, game = divmod(seed - SUITE_SEED, SUITE_STRIDE)
    return seed >= SUITE_SEED and case < SUITE_CASES and game < SUITE_GAMES


def chunk_games(
    decks: list[str], deck: str, kind: str, chunk: int, *, workers: int
) -> list[dict[str, Any]]:
    """A chunk's games (``schedules.rollout_games``): ``deck`` in the
    learner's seat and the kind's in the other, the champion of the deck
    there too in its mirror."""
    rollout_seed = TrainingSchedule(seed=SEED).rollout_seed(rollout_index(decks, deck, kind, chunk))
    opponent = deck if kind == MIRROR else kind.removeprefix("against-")
    other = "self" if kind == MIRROR else "pilot"
    games = rollout_games(
        rollout_seed, workers, [opponent] * CHUNK, [deck] * CHUNK, [other] * CHUNK
    )
    for game in games:
        if held_out(int(game["seed"])):
            raise ValueError(f"the deal of seed {game['seed']} is one of the suites measured")
    return games


def recorded(game: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    """A game as a recording keeps it: its deal and seats, how it ended, its
    digest and the answers its teachers gave, in the order they came."""
    return {
        "learner": str(game["learner"]),
        "opponent": str(game["opponent"]),
        "other": str(game.get("other", "pilot")),
        "learner_player": int(game["learner_player"]),
        "seed": int(game["seed"]),
        "winner": trajectory["winner"],
        "turns": int(trajectory["turns"]),
        "terminal": bool(trajectory["terminal"]),
        "digest": str(trajectory["digest"]),
        "answers": [int(step["chosen"]) for step in trajectory["steps"]],
    }


def chunk_path(folder: Path, deck: str, kind: str, chunk: int) -> Path:
    return Path(folder) / deck / kind / f"{chunk:05d}.jsonl.gz"


def write_chunk(path: Path, games: list[dict[str, Any]]) -> None:
    """A chunk's games, written whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    with gzip.open(part, "wt", encoding="utf8") as written:
        for game in games:
            written.write(json.dumps(game, separators=(",", ":")) + "\n")
    os.replace(part, path)


def read_chunk(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf8") as read:
        return [json.loads(line) for line in read if line.strip()]


class Recordings:
    """The decks' recordings in ``folder``, a folder each, as a copy deals
    from them: each game of an update the next of its deck and kind."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise ValueError(f"No recordings in {self.folder}")
        # How many games of each deck's kind are given so far.
        self.given: dict[tuple[str, str], int] = {}
        # The chunk of each deck's kind last read.
        self._read: dict[tuple[str, str], tuple[int, list[dict[str, Any]]]] = {}

    def teachers(self, decks: list[str]) -> dict[str, str]:
        """The champion each deck's recording was made with, by the
        fingerprint of its weights: a deck without one is refused."""
        found = {}
        for deck in decks:
            about = self.folder / deck / "about.json"
            if not about.is_file():
                raise ValueError(f"No recording of {deck} in {self.folder}")
            found[deck] = str(
                json.loads(about.read_text(encoding="utf8"))["teacher"]["fingerprint"]
            )
        return found

    def passed(self, games: list[dict[str, Any]]) -> None:
        """An update's games gone through, as an update a copy made before it
        went on from where it stood."""
        for game in games:
            key = (str(game["learner"]), kind_of(game))
            self.given[key] = self.given.get(key, 0) + 1

    def dealt(self, games: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each of an update's games as a copy deals them, in its place among
        them, the next recorded game of its deck and kind: its deal, seats,
        answers and digest."""
        found = []
        for game in games:
            deck, kind = str(game["learner"]), kind_of(game)
            place = self.given.get((deck, kind), 0)
            line = self._game(deck, kind, place)
            self.given[(deck, kind)] = place + 1
            found.append(
                {
                    **game,
                    "seed": int(line["seed"]),
                    "learner_player": int(line["learner_player"]),
                    "answers": list(line["answers"]),
                    "digest": str(line["digest"]),
                }
            )
        return found

    def _game(self, deck: str, kind: str, place: int) -> dict[str, Any]:
        chunk, at = divmod(place, CHUNK)
        read = self._read.get((deck, kind))
        if read is None or read[0] != chunk:
            path = chunk_path(self.folder, deck, kind, chunk)
            if not path.is_file():
                raise ValueError(
                    f"{deck}'s recording holds {chunk * CHUNK} games of {kind}, and a copy asks "
                    f"for more: record more (tools/record_champion.py)"
                )
            read = (chunk, read_chunk(path))
            self._read[(deck, kind)] = read
        return read[1][at]
