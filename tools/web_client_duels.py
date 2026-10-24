"""Duels the web client played with a checkpoint seated, played again here.

    uv run python tools/web_client_duels.py runs/model-02279.pt SAVED.json...

The web client (`beat-claudio-yu-gi-oh`, a repository of its own) seats a
checkpoint with the crates of `native/ygo-model-input`, around its own build
of the engine, its own room and a forward pass of its own.  A file is a duel
as it saves one while the duel is played: the engine's seed, both players'
cards as they were dealt, every answer either player gave, and the seed its
pilot was made with.  The page keeps it in the tab's session storage
(`duelInProgress`, whose `saved` is this; either is read), and its engine
gives it as `LocalGame::save`.  A finished duel is the same thing: the page
keeps the last fifteen whole and exports one from its history as a log,
whose `replay` is this, and such a log is read too.

Each duel is played again on this checkout's engine, scripts and card data,
with the checkpoint in the trainer's own seat, answering with its first
choice, against the other player's answers as they were kept.  The page
seats the checkpoint as the trainer does when every answer of that seat is
the one this seat gives, byte for byte, and the other player's answers fit
to the last.  A duel is kept up to the last question its player was asked,
and is compared that far.

- The pilot asked alongside places the cards, and breaks its ties by its
  seed: it is made here with the seed the page made it with.  Zones are no
  part of the network's input.
- A deck of the player's that is no list of the pool is told to the network
  here in the order it was dealt, and on the page in the order of its list:
  a deck is read as the mean of its cards, so the two differ in the last
  digit of a score at most.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.checkpoints import load_inference_model
from ygo_training.edopro import PROCESSOR_THREADS
from ygo_training.edopro_review import checkpoint_asked, matched, reread
from ygo_training.environment import DuelError, Environment

# The rules the page hosts every duel with: OCGCore's flags of Master Rule 1,
# 8000 Life Points, a hand of five, one card drawn a turn.
MASTER_RULE_1 = 0xD0700
LIFE_POINTS, HAND, DRAWN = 8000, 5, 1


def read(path: Path) -> dict[str, Any]:
    """A duel the page saved: as its engine gives it, as the tab keeps one in
    progress (`saved`), or as its history exports a finished one (`replay`)."""
    found = json.loads(path.read_text(encoding="utf8"))
    for inside in ("saved", "replay"):
        if isinstance(found, dict) and isinstance(found.get(inside), str):
            found = json.loads(found[inside])
    if not isinstance(found, dict) or "duel" not in found or "policy_seed" not in found:
        raise SystemExit(f"{path}: not a duel the web client saved")
    return found


def kept(saved: dict[str, Any], flags: int = MASTER_RULE_1) -> dict[str, Any]:
    """The saved duel as a replay holds one (`Kept` of `crates/ygo-env`)."""
    duel = saved["duel"]
    return {
        "seed": duel["seed"],
        "flags": flags,
        "start_lp": LIFE_POINTS,
        "start_hand": HAND,
        "draw_count": DRAWN,
        # The page's room gives the engine the last card of a list first, and
        # a replay lists the cards as the engine was given them.
        "decks": [{pile: deck[pile][::-1] for pile in ("main", "extra")} for deck in duel["decks"]],
        "responses": [answer for _, answer in duel["responses"]],
    }


def again(
    environment: Environment,
    model: torch.nn.Module,
    saved: dict[str, Any],
    *,
    learner: str = "blackwing",
    flags: int = MASTER_RULE_1,
) -> dict[str, Any]:
    """The saved duel with the checkpoint in the trainer's seat.

    ``answers`` of the network's seat were kept, ``decisions`` of them are
    the network's here, and ``differing`` holds the places of those this
    seat answers otherwise.  ``left`` says why, where the duel is not played
    to the last answer kept.
    """
    duel = saved["duel"]
    network = 1 - int(duel["human"])
    told = [[int(player), answer] for player, answer in duel["responses"]]
    replay = kept(saved, flags)
    try:
        record = reread(environment, replay)["record"]
    except DuelError as error:
        return {"left": f"it is not played again here ({error}): another engine, scripts or cards"}
    if [[int(player), answer] for player, answer in record["responses"]] != told:
        return {"left": "its answers are given by another player here than on the page"}
    names = [matched(environment, cards["main"])[0] for cards in replay["decks"]]
    if names[network] != learner:
        return {
            "left": f"the network's seat plays {names[network] or 'no deck of the pool'}, not {learner}"
        }
    asked = checkpoint_asked(model, environment, [seat == network for seat in (0, 1)])
    setup = {
        # The page makes its pilot with its policy seed and one more; an
        # episode makes a seat's pilot with this seed and the seat's number.
        "seed": (int(saved["policy_seed"]) + 1 - network) % 2**64,
        # A seat needs the name of a deck of the pool; its cards are the record's.
        "decks": [name or min(environment.decks) for name in names],
        "seats": [
            {"control": "caller"} if seat == network else {"control": "record", "alone": True}
            for seat in (0, 1)
        ],
        "record": record,
        "keep_record": True,
        "limit": len(told),
    }
    found: dict[str, Any] = {"network": network, "decks": names, "kept": len(told), "decisions": 0}
    try:
        with environment.start(setup) as episode:
            while True:
                event = episode.advance()
                if event["decision"] is not None:
                    found["decisions"] += 1
                    episode.choose(asked(event["decision"])["model"])
                if event["over"] is not None:
                    break
    except DuelError as error:
        found["left"] = (
            f"it leaves the page's duel after {found['decisions']} decisions of the network ({error})"
        )
        return found
    here = [[int(player), answer] for player, answer in event["over"]["record"]["responses"]]
    own = [place for place, (player, _) in enumerate(told) if player == network]
    found["answers"] = len(own)
    found["differing"] = [
        (place, here[place][1] if place < len(here) else None, told[place][1])
        for place in own
        if place >= len(here) or here[place] != told[place]
    ]
    return found


def placed(answer: str | None) -> bool:
    """An answer that names a zone: the player, the location, the place in it."""
    return answer is not None and len(answer) == 6 and answer[2:4] in ("04", "08")


def said(name: str, found: dict[str, Any]) -> list[str]:
    """What :func:`again` found, in words."""
    if "answers" not in found:
        return [f"{name}: {found['left']}"]
    other = found["decks"][1 - found["network"]] or "a deck that is no list of the pool"
    lines = [
        (
            f"{name}: {other} against the network, which went "
            f"{'first' if found['network'] == 0 else 'second'}; {found['kept']} answers kept, "
            f"{found['answers']} of them its seat's, {found['decisions']} of those its own decisions"
        )
    ]
    differing = found["differing"]
    if not differing:
        lines.append("  every answer of its seat is this seat's, byte for byte")
        return lines
    lines.append(
        f"  answers of its seat that are others here: {len(differing)} of {found['answers']}"
    )
    for place, here, there in differing[:5]:
        lines.append(f"    answer {place}: here {here}, the page {there}")
    if all(placed(here) and placed(there) for _, here, there in differing):
        lines.append(
            "  All of them place a card in another zone, which is the pilot's choice and no "
            "part of the network's input: the pilot here is not made with the page's seed."
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("saved", type=Path, nargs="+", help="duels as the web client saved them")
    parser.add_argument("--learner", default="blackwing", help="the deck the checkpoint plays")
    parser.add_argument(
        "--flags",
        type=lambda text: int(text, 0),
        default=MASTER_RULE_1,
        help="OCGCore's duel flags of the page's rooms (Master Rule 1 unless said)",
    )
    args = parser.parse_args()
    for path in (args.checkpoint, *args.saved):
        if not path.is_file():
            raise SystemExit(f"No file at {path}")
    torch.set_num_threads(PROCESSOR_THREADS)
    model, payload = load_inference_model(args.checkpoint, device="cpu")
    answers = same = decisions = apart = 0
    with Environment(ROOT) as environment:
        for path in args.saved:
            found = again(environment, model, read(path), learner=args.learner, flags=args.flags)
            print("\n".join(said(str(path), found)))
            if "answers" not in found or found["differing"]:
                apart += 1
            answers += found.get("answers", 0)
            decisions += found.get("decisions", 0) if "answers" in found else 0
            same += found.get("answers", 0) - len(found.get("differing", []))
    print(
        f"{args.checkpoint.name} (after {payload.get('step')} updates) in the trainer's seat: "
        f"{same} of {answers} answers of the network's seat are the page's "
        f"({decisions} decisions of the network), over {len(args.saved)} "
        + ("duel" if len(args.saved) == 1 else "duels")
        + (f"; {apart} not the page's to the last answer" if apart else "")
    )
    raise SystemExit(1 if apart else 0)


if __name__ == "__main__":
    main()
