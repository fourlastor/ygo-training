"""A duel a person played on the web client, read again: each decision of
the network's seat beside another checkpoint's and the pilot's answers, or
the duel played on from each of the person's decisions where the checkpoint
would have answered otherwise.

    uv run python tools/web_client_reading.py beside CHECKPOINT SAVED.json... [--other CHECKPOINT]
    uv run python tools/web_client_reading.py what-if CHECKPOINT SAVED.json... [--playouts 64] [--every-kind]

A file is a duel as `tools/web_client_duels.py` reads one; check first that
the page played it as the trainer's seat plays it.

`beside`: the duel read again as a recorded one, the network's seat asked
CHECKPOINT (the one the page played) and `--other` beside it, the pilot of
each deck asked alongside. For each decision of the network's seat: its
answer, how sure it was, its chances by its value head, and the other
checkpoint's and the pilot's answers where they are others; the person's
answers between, but for their passes. A selection the pilot answers as
recorded is walked in the pilot's order of picks, so at such a step an
answer said to be another may be the same cards picked in another order.

`what-if`: the person's seat is their record handed over to a caller where
it is cut (`handover` of a seat's setup), the network's seat a caller
throughout, its pilot made with the page's seed, as the page seated it: up
to the cut the duel is the person's, their zones and copies as they gave
them. The duel is first played whole this way and compared with the record.
Then at each of the person's decisions (Main and Battle Phase commands, or
every kind with `--every-kind`) where CHECKPOINT's first choice is another
answer, it is played on twice, with the person's answer and with the
checkpoint's, the checkpoint then answering both seats: once by its first
choices and PLAYOUTS times sampled from its probabilities. The checkpoint
plays the person's deck after a cut, so it should be one that plays both
decks. Each branch tells the person's wins of each answer's playouts, and
is marked where they are a sixth of the playouts apart or more.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from web_client_duels import kept, read

from ygo_training.checkpoints import load_inference_model
from ygo_training.edopro import PROCESSOR_THREADS, describer
from ygo_training.edopro_review import alike, checkpoint_asked, decisions, matched, reread
from ygo_training.environment import Environment
from ygo_training.inference import scored

COMMANDS = ("Idle", "Battle")
# Two answers' playouts this far apart, of every so many, are marked.
APART = 6


def recorded(
    environment: Environment, saved: dict[str, Any]
) -> tuple[dict[str, Any], list[str | None]]:
    """The saved duel played again as the engine played it, and its decks."""
    replay = kept(saved)
    record = reread(environment, replay)
    names = [matched(environment, cards["main"])[0] for cards in replay["decks"]]
    return record, names


def first_choice(model: torch.nn.Module, environment: Environment, view: dict[str, Any]) -> int:
    count = len(view["choices"])
    return int(scored(model, environment, [view], "cpu")[0][0, :count].argmax())


def sampled_choice(
    model: torch.nn.Module, environment: Environment, view: dict[str, Any], rng: random.Random
) -> int:
    count = len(view["choices"])
    scores = scored(model, environment, [view], "cpu")[0][0, :count]
    weights = torch.softmax(scores.double(), -1).tolist()
    return rng.choices(range(count), weights)[0]


def played(
    environment: Environment,
    model: torch.nn.Module,
    saved: dict[str, Any],
    *,
    cut: int | None = None,
    answer: str | None = None,
    sampled: int | None = None,
    shown: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The duel with the person's record cut before the engine's answer
    `cut`, or whole to its last answer kept. After the cut the person's
    first question is answered as `answer` says ("person": as recorded,
    with `shown` the view of it read whole; "network": the checkpoint's
    first choice), and every one after it by the checkpoint, at both seats:
    by its first choice, or with `sampled` drawn from its probabilities by
    that seed. The person's views while recorded, and the end."""
    record, names = recorded(environment, saved)
    human = int(saved["duel"]["human"])
    network = 1 - human
    kept_record = dict(record["record"])
    responses = kept_record["responses"]
    if cut is not None:
        kept_record["responses"] = [
            given for at, given in enumerate(responses) if int(given[0]) != human or at < cut
        ]
    seats: list[dict[str, Any]] = [{}, {}]
    seats[human] = {"control": "record", "handover": True, "observe": True}
    seats[network] = {"control": "caller"}
    setup: dict[str, Any] = {
        # The page makes its pilot with its policy seed and one more; an
        # episode makes a seat's pilot with this seed and the seat's number.
        "seed": (int(saved["policy_seed"]) + 1 - network) % 2**64,
        # A seat needs the name of a deck of the pool; its cards are the record's.
        "decks": [name or min(environment.decks) for name in names],
        "seats": seats,
        "record": kept_record,
        "keep_record": True,
    }
    if cut is None and record["winner"] is None:
        # A duel saved before its end is played to its last answer kept.
        setup["limit"] = len(responses)
    rng = random.Random(sampled)
    views: list[dict[str, Any]] = []
    handed = False
    with environment.start(setup) as episode:
        while True:
            event = episode.advance()
            views.extend(view for view in event["samples"] if int(view["player"]) == human)
            view = event["decision"]
            if view is not None:
                person = int(view["player"]) == human
                if person and not handed:
                    handed = True
                    if answer == "person":
                        if shown is None:
                            raise ValueError("the person's answer is named by the view read whole")
                        if [c["kind"] for c in view["choices"]] != [
                            c["kind"] for c in shown["choices"]
                        ]:
                            raise ValueError(
                                f"the person is asked another decision at answer {cut}"
                            )
                        choice = int(shown["chosen"])
                    else:
                        choice = first_choice(model, environment, view)
                elif sampled is None or not handed:
                    choice = first_choice(model, environment, view)
                else:
                    choice = sampled_choice(model, environment, view, rng)
                episode.choose(choice)
            if event["over"] is not None:
                return views, event["over"]


def departures(
    environment: Environment,
    model: torch.nn.Module,
    views: list[dict[str, Any]],
    every_kind: bool = False,
) -> list[dict[str, Any]]:
    """The person's decisions, read whole, where the checkpoint's first
    choice is another answer: the first step of a selection, a command but
    with `every_kind`."""
    describe = describer(environment)
    found = []
    for view in views:
        said = [describe(view, index) for index in range(len(view["choices"]))]
        if view["step"] != 0 or (not every_kind and said[0].split(":")[0] not in COMMANDS):
            continue
        first = first_choice(model, environment, view)
        if alike(view, int(view["chosen"]), first):
            continue
        found.append(
            {
                "answer": int(view["answer"]),
                "turn": int(view["turn"]),
                "theirs": said[int(view["chosen"])],
                "its": said[first],
                "view": view,
            }
        )
    return found


WORKER: dict[str, Any] = {}


def _start(checkpoint: str) -> None:
    torch.set_num_threads(1)
    WORKER["environment"] = Environment(ROOT)
    WORKER["model"] = load_inference_model(Path(checkpoint), device="cpu")[0]


def _branch(task: tuple[Path, int, str, int | None, dict[str, Any]]) -> tuple[tuple, bool]:
    path, cut, answer, sampled, view = task
    saved = read(path)
    _, over = played(
        WORKER["environment"],
        WORKER["model"],
        saved,
        cut=cut,
        answer=answer,
        sampled=sampled,
        shown=view,
    )
    return (str(path), cut, answer, sampled), over["winner"] == int(saved["duel"]["human"])


def what_if(args: argparse.Namespace) -> None:
    torch.set_num_threads(PROCESSOR_THREADS)
    model, _ = load_inference_model(args.checkpoint, device="cpu")
    tasks = []
    branches: list[tuple[Path, dict[str, Any]]] = []
    with Environment(ROOT) as environment:
        for path in args.saved:
            saved = read(path)
            record, _ = recorded(environment, saved)
            views, over = played(environment, model, saved)
            same = over["record"]["responses"] == record["record"]["responses"]
            print(
                f"{path.name}: played whole, {'the duel recorded' if same else 'NOT the duel recorded'} "
                f"({over['decisions']} answers, {len(views)} decisions of the person's)"
            )
            if not same:
                continue
            for found in departures(environment, model, views, args.every_kind):
                branches.append((path, found))
                for answer in ("person", "network"):
                    for seed in [None, *range(args.playouts)]:
                        tasks.append((path, found["answer"], answer, seed, found["view"]))
    if not tasks:
        return
    with Pool(args.workers, initializer=_start, initargs=(str(args.checkpoint),)) as pool:
        won = dict(pool.map(_branch, tasks, chunksize=2))
    for path, found in branches:
        key = (str(path), found["answer"])
        shares = {
            answer: sum(won[(*key, answer, seed)] for seed in range(args.playouts))
            for answer in ("person", "network")
        }
        first = {answer: "won" if won[(*key, answer, None)] else "lost" for answer in shares}
        line = (
            f"{path.name} turn {found['turn']}: you {found['theirs']}; it {found['its']} | "
            f"yours {shares['person']}/{args.playouts} ({first['person']} by first choices), "
            f"its {shares['network']}/{args.playouts} ({first['network']})"
        )
        if abs(shares["person"] - shares["network"]) * APART >= args.playouts:
            line += "  <-- " + (
                "yours better" if shares["person"] > shares["network"] else "its better"
            )
        print(line)


def beside(args: argparse.Namespace) -> None:
    torch.set_num_threads(PROCESSOR_THREADS)
    model, _ = load_inference_model(args.checkpoint, device="cpu")
    other = load_inference_model(args.other, device="cpu")[0] if args.other else None
    with Environment(ROOT) as environment:
        for path in args.saved:
            saved = read(path)
            human = int(saved["duel"]["human"])
            record, names = recorded(environment, saved)
            seats = [seat != human for seat in (0, 1)]
            first = checkpoint_asked(model, environment, seats)
            second = checkpoint_asked(other, environment, seats) if other is not None else None

            def asked(view: dict[str, Any], first=first, second=second) -> dict[str, Any]:
                found = first(view)
                if found and second is not None:
                    found["other"] = second(view)["model"]
                return found

            shown: list[dict[str, Any]] = []
            # A duel saved before its end is read to its last answer kept.
            stop = len(record["record"]["responses"]) if record["winner"] is None else None
            rows = decisions(
                environment, record["record"], names, stop=stop, asked=asked, shown=shown
            )
            won = "the person" if record["winner"] == human else "the network"
            print(f"== {path.name}: {won} won, {record['turns']} turns")
            told = with_other = with_pilot = 0
            for row, view in zip(rows, shown, strict=True):
                said = row["said"][row["chosen"]]
                if row["player"] == human:
                    if len(row["said"]) > 1 and not said.startswith("Chain: Pass"):
                        print(f"t{row['turn']:>2} {row['phase']:<11} PERSON {said}")
                    continue
                told += 1
                line = (
                    f"t{row['turn']:>2} {row['phase']:<11} NET [{len(row['said'])}] {said} "
                    f"p={row['probabilities'][row['chosen']]:.2f} ch={row['chances']:.2f} "
                    f"lp={view['life_points']}"
                )
                if "other" in row and not alike(view, row["chosen"], row["other"]):
                    line += f" | other: {row['said'][row['other']]}"
                else:
                    with_other += 1
                if row["pilot"] is not None and not alike(view, row["chosen"], row["pilot"]):
                    line += f" | pilot: {row['said'][row['pilot']]}"
                else:
                    with_pilot += 1
                print(line)
            summary = f"   {told} decisions of the network; the pilot's answer in {with_pilot}"
            if other is not None:
                summary += f"; the other checkpoint's in {with_other}"
            print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    read_again = commands.add_parser(
        "beside", help="each decision beside another checkpoint's and the pilot's"
    )
    read_again.add_argument("checkpoint", type=Path, help="the checkpoint the page played")
    read_again.add_argument("saved", type=Path, nargs="+")
    read_again.add_argument("--other", type=Path, help="a checkpoint asked beside it")
    branch = commands.add_parser("what-if", help="the duel played on from the person's decisions")
    branch.add_argument(
        "checkpoint", type=Path, help="the checkpoint the page played, which plays on"
    )
    branch.add_argument("saved", type=Path, nargs="+")
    branch.add_argument("--playouts", type=int, default=64)
    branch.add_argument(
        "--every-kind", action="store_true", help="every kind of decision, not commands alone"
    )
    branch.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    args = parser.parse_args()
    for path in [
        args.checkpoint,
        *args.saved,
        *([args.other] if getattr(args, "other", None) else []),
    ]:
        if not path.is_file():
            raise SystemExit(f"No file at {path}")
    {"beside": beside, "what-if": what_if}[args.command](args)


if __name__ == "__main__":
    main()
