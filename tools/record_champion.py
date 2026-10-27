"""A champion's games for a copy, recorded: its mirror, both seats its own,
and its games against each of the thirteen decks' pilots, played by its
first choices and kept as the answers it gave (ygo_training/champion_records.py),
a chunk of games at a time; then the deck's folder in one archive.

    OMP_NUM_THREADS=8 uv run python tools/record_champion.py DECK CHECKPOINT \\
        --out records/champions [--mirrors 44] [--against 4]

The deck's recording is OUT/DECK: MIRRORS chunks of its mirror and AGAINST
chunks against each of the opponents' pilots, `CHUNK` games each, and
`about.json`, what made them; then OUT/DECK.tar holds the folder, its chunks
already compressed. A chunk already there is kept: a recording stopped goes
on from where it stood, and one asked for more chunks adds them. 44 and 4
are a copy of 160 updates a deck and some over: 41,600 games of the mirror
and 3,200 against each pilot, the most a copy of the deck draws on, give or
take a few dozen (its updates deal the games to within one).

A recording binds the games to the engine, the scripts, the card database
and the pilots it was made with: played again with others, a game ends with
another digest, and a copy refuses it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training import champion_records
from ygo_training.champion_records import (
    MIRROR,
    SEED,
    chunk_games,
    chunk_path,
    recorded,
    write_chunk,
)
from ygo_training.checkpoints import load_inference_model, value_fingerprint
from ygo_training.environment import Environment
from ygo_training.pool import DuelPool, taught_games

THIRTEEN = (
    "blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,"
    "morphtronic,rock-block,spellcaster"
)
# The deals' seats follow from a rollout's workers, as many as the CUDA
# machine's training takes.
WORKERS = 24


def revisions() -> dict[str, str]:
    """The checkout and its submodules where the games were recorded."""
    found = {
        "checkout": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    }
    status = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in status.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            found[parts[1]] = parts[0].lstrip("+-U")
    return found


def record(
    deck: str,
    checkpoint: Path,
    out: Path,
    *,
    mirrors: int,
    against: int,
    opponents: list[str],
    device: str = "cpu",
    threads: int = 12,
    slots: int = 84,
    archive: bool = True,
) -> Path:
    """The deck's recording in OUT/DECK, the chunks not there yet played,
    and its archive OUT/DECK.tar: the folder."""
    kinds = {MIRROR: mirrors} | {f"against-{opponent}": against for opponent in opponents}
    teacher, saved = load_inference_model(checkpoint, device=device)
    teacher.requires_grad_(False)
    with Environment(ROOT) as environment:
        decks = sorted(environment.decks)
    for name in [deck, *opponents]:
        if name not in decks:
            raise SystemExit(f"Unknown deck: {name}")
    folder = out / deck
    folder.mkdir(parents=True, exist_ok=True)
    about = {
        "deck": deck,
        "teacher": {
            "path": str(checkpoint.resolve()),
            "step": saved.get("step"),
            "fingerprint": value_fingerprint(teacher.state_dict()),
        },
        "seed": SEED,
        "chunk": champion_records.CHUNK,
        "workers": WORKERS,
        "kinds": kinds,
        "revisions": revisions(),
    }
    known = folder / "about.json"
    if known.is_file():
        before = json.loads(known.read_text(encoding="utf8"))
        for part in ("teacher", "revisions"):
            if part == "teacher" and before[part]["fingerprint"] != about[part]["fingerprint"]:
                raise SystemExit(f"{folder} was recorded with another champion")
            if part == "revisions" and before[part] != about[part]:
                raise SystemExit(f"{folder} was recorded at other revisions: {before[part]}")
        # More chunks of a kind add to those there; fewer keep them.
        about["kinds"] = {
            kind: max(count, int(before["kinds"].get(kind, 0))) for kind, count in kinds.items()
        } | {kind: count for kind, count in before["kinds"].items() if kind not in kinds}
    known.write_text(json.dumps(about, indent=2) + "\n", encoding="utf8")
    started = time.perf_counter()
    played = 0
    with DuelPool(ROOT, threads=threads, slots=slots) as pool:
        for kind, chunks in kinds.items():
            for chunk in range(chunks):
                path = chunk_path(out, deck, kind, chunk)
                if path.is_file():
                    continue
                games = chunk_games(decks, deck, kind, chunk, workers=WORKERS)
                pool.forget()
                begun = time.perf_counter()
                trajectories, _ = taught_games(
                    pool,
                    {deck: teacher},
                    device,
                    learner=deck,
                    games=games,
                    limit=4096,
                    grouped=False,
                )
                write_chunk(
                    path,
                    [
                        recorded(game, trajectory)
                        for game, trajectory in zip(games, trajectories, strict=True)
                    ],
                )
                played += len(games)
                seconds = time.perf_counter() - begun
                print(
                    f"{time.strftime('%H:%M')} {deck} {kind} {chunk + 1}/{chunks}: "
                    f"{len(games) / seconds:.1f} games a second",
                    flush=True,
                )
    seconds = time.perf_counter() - started
    print(f"{deck}: {played} games in {seconds / 3600:.2f} h", flush=True)
    kept = out / f"{deck}.tar"
    if archive:
        part = kept.with_name(f"{kept.name}.part")
        with tarfile.open(part, "w") as written:
            written.add(folder, arcname=deck)
        part.replace(kept)
        print(f"{kept}: {kept.stat().st_size / 2**20:.0f} MiB", flush=True)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("deck")
    parser.add_argument("checkpoint", type=Path, help="the deck's champion")
    parser.add_argument("--out", type=Path, default=ROOT / "records/champions")
    parser.add_argument("--mirrors", type=int, default=44, help="chunks of the mirror")
    parser.add_argument("--against", type=int, default=4, help="chunks against each pilot")
    parser.add_argument("--opponents", default=THIRTEEN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool-threads", type=int, default=12)
    parser.add_argument(
        "--pool-slots", type=int, default=84, help="duels a thread holds: a chunk's at once"
    )
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    record(
        args.deck,
        args.checkpoint,
        args.out,
        mirrors=args.mirrors,
        against=args.against,
        opponents=[deck for deck in args.opponents.split(",") if deck],
        device=args.device,
        threads=args.pool_threads,
        slots=args.pool_slots,
        archive=not args.no_archive,
    )


if __name__ == "__main__":
    main()
