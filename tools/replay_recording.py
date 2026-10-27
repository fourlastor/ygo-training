"""A champion's recording played again here, every game held to the digest
it was recorded with: whether this machine, its engine, scripts, cards and
pilots play the games another machine recorded (ygo_training/champion_records.py).

    uv run python tools/replay_recording.py records/champions DECK [--chunks 2]
    uv run python tools/replay_recording.py records/champions/DECK.tar [--chunks 2]

The deck's folder, or its archive, which is read where it is. Each of the
first CHUNKS chunks of every kind is played again with the recorded answers
in the champion's place, no network asked (`pool.taught_games` with
``played``), and told: its games, those that end with another digest, how
they ended against how they were recorded, and the games a second. It ends
in failure where one game is another.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.champion_records import kind_of
from ygo_training.pool import DuelPool, taught_games


def chunks(source: Path, deck: str | None) -> tuple[str, dict[str, list[list[dict[str, Any]]]]]:
    """The deck and its chunks by kind, in their order, from its folder or
    its archive."""
    import gzip

    found: dict[str, dict[int, list[dict[str, Any]]]] = {}
    if source.suffix == ".tar":
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if len(parts) == 3 and parts[2].endswith(".jsonl.gz"):
                    deck = parts[0]
                    data = archive.extractfile(member)
                    assert data is not None
                    with gzip.open(io.BytesIO(data.read()), "rt", encoding="utf8") as read:
                        lines = [json.loads(line) for line in read if line.strip()]
                    found.setdefault(parts[1], {})[int(parts[2].split(".")[0])] = lines
    else:
        if deck is None:
            raise SystemExit("A recordings folder needs the deck")
        for path in sorted((source / deck).glob("*/*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf8") as read:
                lines = [json.loads(line) for line in read if line.strip()]
            found.setdefault(path.parent.name, {})[int(path.name.split(".")[0])] = lines
    if deck is None or not found:
        raise SystemExit(f"No recording in {source}")
    return deck, {kind: [kept[index] for index in sorted(kept)] for kind, kept in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="the recordings' folder, or a deck's archive")
    parser.add_argument("deck", nargs="?")
    parser.add_argument("--chunks", type=int, default=1, help="of each kind, from the first")
    parser.add_argument("--pool-threads", type=int, default=12)
    parser.add_argument("--pool-slots", type=int, default=84)
    args = parser.parse_args()
    deck, kinds = chunks(args.source, args.deck)
    other = 0
    with DuelPool(ROOT, threads=args.pool_threads, slots=args.pool_slots) as pool:
        for kind, kept in sorted(kinds.items()):
            for place, lines in enumerate(kept[: args.chunks]):
                games = [
                    {**line, "worker": 0, "episode_index": index}
                    for index, line in enumerate(lines)
                ]
                if any(kind_of(game) != kind for game in games):
                    raise SystemExit(f"{deck} {kind} {place}: a game of another kind")
                pool.forget()
                started = time.perf_counter()
                trajectories, _ = taught_games(
                    pool,
                    {},
                    "cpu",
                    learner=deck,
                    games=games,
                    limit=4096,
                    grouped=False,
                    played=[line["answers"] for line in lines],
                )
                seconds = time.perf_counter() - started
                apart = sum(
                    trajectory["digest"] != line["digest"]
                    for trajectory, line in zip(trajectories, lines, strict=True)
                )
                ended = sum(
                    (trajectory["winner"], trajectory["turns"]) == (line["winner"], line["turns"])
                    for trajectory, line in zip(trajectories, lines, strict=True)
                )
                other += apart
                print(
                    f"{deck} {kind} {place}: {len(lines)} games, {apart} with another digest, "
                    f"{ended} ended as recorded, {len(lines) / seconds:.0f} games a second",
                    flush=True,
                )
    if other:
        raise SystemExit(f"{other} games of {deck}'s recording are other games here")
    print(f"{deck}: every game played again is the game recorded")


if __name__ == "__main__":
    main()
