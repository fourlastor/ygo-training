"""Duels of the pool's decks with seats that answer at random among the legal
answers, through the pool, kept where the engine stops one: a card's script
that fails, an arena found full, a loop the engine guards against.  In each
duel a seat answers at random and the other is its deck's pilot, or, in a
third of them, both seats answer at random: what pilots never do finds what
the scripts and the engine do not expect (docs/commands.md, "Duels the
engine stops").

    uv run python tools/fuzz_duels.py --games 50 --seed 101 --threads 5 --output analysis/fuzz/101

Each pair of the pool's decks ``--games`` times, the seats taking turns;
OUTPUT/failed.jsonl gets each duel the engine stopped (``DuelPool.failed``:
the game, the setup, what the engine said and the record, which
``policy-bench replay --input`` plays again), OUTPUT/summary.json the counts.
With ``--per-duel`` an answer is drawn from the duel and its decision alone,
and each duel's digest goes to OUTPUT/duels.jsonl: two builds' games can be
set side by side, a change to one duel leaving the others as they are.
Several processes of a few threads each make more duels an hour than one of
many (on the development machine four of five threads, about 240 a second).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ygo_training.environment import pool as deck_pool
from ygo_training.pool import DuelPool

ROOT = Path(__file__).resolve().parents[1]
RANDOM = {"control": "caller", "alone": True}
PILOT = {"control": "pilot"}


def games(decks: list[str], count: int, seed: int) -> list[tuple[dict, dict]]:
    """Each pair of decks ``count`` times: (the setup, what a report says)."""
    made = []
    index = 0
    for first in decks:
        for second in decks:
            for game in range(count):
                kind = ("random-pilot", "pilot-random", "random-random")[index % 3]
                seats = {
                    "random-pilot": [RANDOM, PILOT],
                    "pilot-random": [PILOT, RANDOM],
                    "random-random": [RANDOM, RANDOM],
                }[kind]
                setup = {
                    "seed": seed * 1_000_003 + index,
                    "decks": [first, second],
                    "seats": seats,
                    "limit": 4096,
                }
                made.append((setup, {"index": index, "kind": kind, "decks": [first, second]}))
                index += 1
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=2, help="duels a pair of decks")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=4000, help="duels submitted at a time")
    parser.add_argument("--decks", default=None, help="comma-separated; the whole pool by default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-duel",
        action="store_true",
        help="answers drawn from the duel and its decision alone, and each duel's digest kept "
        "(OUTPUT/duels.jsonl): two builds' games can be set side by side",
    )
    args = parser.parse_args()
    decks = args.decks.split(",") if args.decks else sorted(deck_pool(ROOT))
    planned = games(decks, args.games, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    failed_path = args.output / "failed.jsonl"
    rng = np.random.default_rng(args.seed)
    asked: Counter[int] = Counter()
    duels_path = (args.output / "duels.jsonl").open("w") if args.per_duel else None
    started = reported = time.perf_counter()
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    decisions = 0
    with (
        DuelPool(ROOT, threads=args.threads, slots=args.slots) as duels,
        failed_path.open("a") as kept,
    ):
        submitted = 0
        round_ = None
        answers: np.ndarray | tuple = ()
        while True:
            # Keep the pool fed: the next chunk goes in as the last one drains.
            if submitted < len(planned) and (
                round_ is None or round_.in_flight < duels.capacity * 2
            ):
                chunk = planned[submitted : submitted + args.chunk]
                duels.submit([setup for setup, _ in chunk], about=[about for _, about in chunk])
                submitted += len(chunk)
            round_ = duels.step(answers)
            for over in round_.finished:
                counts["duels"] += 1
                counts["limit" if over["limit"] else "ended"] += 1
                if duels_path is not None:
                    keep = (
                        "duel",
                        "winner",
                        "reason",
                        "turns",
                        "decisions",
                        "limit",
                        "digest",
                        "failure",
                    )
                    duels_path.write(json.dumps({key: over.get(key) for key in keep}) + "\n")
            for report in duels.failed():
                counts["failed"] += 1
                failures[report["failure"].splitlines()[0]] += 1
                kept.write(json.dumps(report, default=str) + "\n")
                kept.flush()
            if len(round_):
                mask = round_.arrays["action_mask"].astype(bool)
                if args.per_duel:
                    # A draw of the duel and of its decision's place in it.
                    rows = []
                    for duel in round_.duel.tolist():
                        rows.append(
                            np.random.default_rng([args.seed, duel, asked[duel]]).random(
                                mask.shape[1]
                            )
                        )
                        asked[duel] += 1
                    scores = np.array(rows)
                else:
                    scores = rng.random(mask.shape)
                scores[~mask] = -1.0
                answers = scores.argmax(axis=1).astype(np.uint32)
                decisions += len(round_)
            else:
                answers = ()
            if submitted >= len(planned) and round_.in_flight == 0 and not len(round_):
                break
            if time.perf_counter() - reported > 60:
                reported = time.perf_counter()
                print(
                    f"{counts['duels']} duels, {counts['failed']} failed, {reported - started:.0f} s",
                    flush=True,
                )
    if duels_path is not None:
        duels_path.close()
    seconds = time.perf_counter() - started
    summary = {
        "decks": len(decks),
        "games_a_pair": args.games,
        "seed": args.seed,
        "counts": dict(counts),
        "random_decisions": decisions,
        "seconds": seconds,
        "failures": dict(failures.most_common()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
