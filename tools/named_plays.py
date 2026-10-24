"""The named plays, counted in the games of Blackwing checkpoints.

    uv run python tools/named_plays.py copy=runs/a/model-00359.pt long=runs/b/model-02279.pt \\
        --keep analysis/named-plays

A stronger player than the pilots beat the Blackwing checkpoints and named
three plays (experiments/2026-10-09-ppo-with-a-head-that-predicts.md,
the duels on the page): Allure of Darkness activated with no DARK monster in
hand before the Spells and Traps beside it are set; Trap Hole at the first
Summon that allows it; Threatening Roar shown before the other player
attacks.  None of them costs anything against a pilot, which plays on the
same whatever it is shown, so what a checkpoint wins against pilots does not
see them.  This counts them.

Each checkpoint plays the same games on the processor: `--games` against the
pilot of each of `--opponents`, by its first choice, with every decision of
its seat kept beside its pilot's answer, and `--idle-games` against a player
who does nothing but enter the Battle Phase and leave it.  With `--keep` the
decisions are kept under each checkpoint's name and read again the next
time, where the checkpoint and the games asked for are the same.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEARNER = "blackwing"
# The pilots a Blackwing learner trains against, and the deck the other seat played.
OPPONENTS = (
    "blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,"
    "monarch,morphtronic,rock-block,spellcaster,machina"
)
IDLE_OPPONENTS = "blackwing,monarch"
ALLURE = "Allure of Darkness"
ROAR = "Threatening Roar"
TRAPS = ("Trap Hole", "Bottomless Trap Hole", "Dimensional Prison")
# Where a card lies and how, as the seat is shown it; its own cards are 0's.
HAND, MONSTERS = 2, 3
FACE_UP_ATTACK = 1
BATTLE = ("BattleStart", "BattleStep")
# The kinds of hand Allure of Darkness is held in, the card itself aside.
DARK, ALONE, SPELLS, OTHER = "dark", "alone", "spells", "other"


def idle(said: list[str]) -> int:
    """The answer of a player who plays nothing: into the Battle Phase and
    out of it, every turn."""
    for wanted in (
        "Idle: EnterBattle",
        "Battle: EndTurn",
        "Battle: EnterMain2",
        "Idle: EndTurn",
        "Chain: Pass",
    ):
        if wanted in said:
            return said.index(wanted)
    return 0


def played(
    environment: Any,
    model: Any,
    device: str,
    *,
    opponents: list[str],
    games: int,
    seed: int,
    against_idle: bool = False,
) -> Iterator[dict[str, Any]]:
    """Games of the checkpoint against each opponent, the seats in turn: a
    row for every decision of its seat, and one for how each game ended."""
    from ygo_training.edopro import describer
    from ygo_training.inference import scored

    describe = describer(environment)
    phases = {number: name for name, number in environment.vocabulary["phase"].items()}
    for opponent in opponents:
        for index in range(games):
            seat = index % 2
            other = {"control": "caller" if against_idle else "pilot"}
            setup = {
                "seed": seed + index,
                "decks": [LEARNER, opponent] if seat == 0 else [opponent, LEARNER],
                "seats": [{"control": "caller"} if player == seat else other for player in (0, 1)],
            }
            game = f"{opponent}-{index}"
            with environment.start(setup) as episode:
                while True:
                    event = episode.advance()
                    view = event["decision"]
                    if view is not None:
                        said = [describe(view, place) for place in range(len(view["choices"]))]
                        if int(view["player"]) != seat:
                            episode.choose(idle(said))
                            continue
                        logits, _ = scored(model, environment, [view], device)
                        chosen = int(logits[0, : len(said)].argmax())
                        yield {
                            "game": game,
                            "opponent": opponent,
                            "seat": seat,
                            "turn": view["turn"],
                            "turn_player": view["turn_player"],
                            "phase": phases.get(int(view["phase"]), "?"),
                            "said": said,
                            "chosen": chosen,
                            "pilot": view["pilot"],
                            "cards": [
                                [
                                    card["controller"],
                                    card["location"],
                                    card["code"],
                                    card["position"],
                                    card["attack"],
                                ]
                                for card in view["cards"]
                                if card["location"] in (HAND, MONSTERS)
                            ],
                            "events": [
                                {"controller": event_.get("controller"), "code": event_.get("code")}
                                for event_ in view["events"]
                            ],
                            "chain": [
                                {"controller": link.get("controller"), "code": link.get("code")}
                                for link in view["chain"]
                            ],
                            # Where the card of each answer lies.
                            "from": [
                                (choice["card"] or {}).get("location", 0)
                                for choice in view["choices"]
                            ],
                        }
                        episode.choose(chosen)
                    if event["over"] is not None:
                        over = event["over"]
                        yield {
                            "game": game,
                            "opponent": opponent,
                            "seat": seat,
                            "over": {key: over[key] for key in ("winner", "turns", "limit")},
                        }
                        break


def by_game(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict]]:
    """The decisions of each game in the order they were taken, and how each
    game ended."""
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    over: dict[str, dict] = {}
    for row in rows:
        if "over" in row:
            over[row["game"]] = row
        else:
            games[row["game"]].append(row)
    return games, over


def mine(row: dict[str, Any], location: int) -> list[int]:
    return [
        code
        for controller, where, code, _, _ in row["cards"]
        if controller == 0 and where == location
    ]


def theirs(row: dict[str, Any]) -> list[tuple[int, int, int]]:
    """The other player's monsters: the card, how it lies, its ATK."""
    return [
        (code, position, attack)
        for controller, where, code, position, attack in row["cards"]
        if controller == 1 and where == MONSTERS
    ]


def took(row: dict[str, Any], said: str) -> bool:
    return row["said"][row["chosen"]] == said


def pilot_took(row: dict[str, Any], said: str) -> bool:
    return row["pilot"] is not None and row["said"][row["pilot"]] == said


def settable(row: dict[str, Any]) -> int:
    """How many Spells and Traps the decision offers to set, Allure of
    Darkness itself aside."""
    return sum(
        1 for said in row["said"] if said.startswith("Idle: SetSpellTrap") and ALLURE not in said
    )


def allure(
    games: dict[str, list[dict[str, Any]]], name: Callable[[int], str], kind: Callable[[int], str]
) -> dict[str, Any]:
    """Allure of Darkness, by the turns it could be activated from the hand.

    A turn is told by what else was in hand when the turn first offered it
    (``kind`` says of a card whether it is a DARK monster, another monster,
    or a Spell or Trap): a DARK monster, nothing, or Spells and Traps alone.
    Activated with only those beside it, the cards the same decision still
    offered to set are what the play risks."""
    activate = f"Idle: Activate {ALLURE}"
    turns: Counter[str] = Counter()
    activated: Counter[str] = Counter()
    unset = lost = lost_with_cards = without_dark = 0
    for rows in games.values():
        offered: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for place, row in enumerate(rows):
            if activate in row["said"] and row["from"][row["said"].index(activate)] == HAND:
                offered[int(row["turn"])].append((place, row))
        for offers in offered.values():
            beside = [kind(code) for code in mine(offers[0][1], HAND) if name(code) != ALLURE]
            hand = (
                DARK
                if "dark" in beside
                else ALONE
                if not beside
                else SPELLS
                if set(beside) == {"spell"}
                else OTHER
            )
            turns[hand] += 1
            taken = [(place, row) for place, row in offers if took(row, activate)]
            if not taken:
                continue
            activated[hand] += 1
            if hand == SPELLS and settable(taken[0][1]):
                unset += 1
        # What came of each activation with no DARK monster in hand: the hand
        # at the seat's next decision of that turn with nothing on the chain.
        for place, row in enumerate(rows):
            if not (activate in row["said"] and took(row, activate)):
                continue
            if row["from"][row["said"].index(activate)] != HAND:
                continue
            beside = [code for code in mine(row, HAND) if name(code) != ALLURE]
            if any(kind(code) == "dark" for code in beside):
                continue
            without_dark += 1
            after = next(
                (
                    later
                    for later in rows[place + 1 :]
                    if later["turn"] == row["turn"] and not later["chain"]
                ),
                None,
            )
            if after is not None and not mine(after, HAND):
                lost += 1
                lost_with_cards += bool(beside)
    return {
        "turns": dict(turns),
        "activated": dict(activated),
        "with_cards_still_to_set": unset,
        "activations_without_a_dark_monster": without_dark,
        "hand_lost": lost,
        "hand_lost_with_other_cards": lost_with_cards,
    }


def traps(games: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The Traps that answer a Summon or an attack: of the windows that
    offer each, how many the checkpoint takes and how many its pilot would;
    and for Trap Hole, the weakest monsters it can answer and the first
    window of a game."""
    found: dict[str, Any] = {}
    for card in TRAPS:
        said = f"Chain: Activate {card}"
        windows = [row for rows in games.values() for row in rows if said in row["said"]]
        found[card] = {
            "windows": len(windows),
            "taken": sum(took(row, said) for row in windows),
            "pilot": sum(pilot_took(row, said) for row in windows),
        }
    said = f"Chain: Activate {TRAPS[0]}"
    weak = {"windows": 0, "taken": 0, "pilot": 0}
    first = {"windows": 0, "taken": 0, "pilot": 0}
    for rows in games.values():
        seen = False
        for row in rows:
            if said not in row["said"]:
                continue
            summoned = {event["code"] for event in row["events"] if event.get("controller") == 1}
            attacks = [attack for code, _, attack in theirs(row) if code in summoned]
            for counted in ([weak] if attacks and max(attacks) < 1500 else []) + (
                [] if seen else [first]
            ):
                counted["windows"] += 1
                counted["taken"] += took(row, said)
                counted["pilot"] += pilot_took(row, said)
            seen = True
    found["Trap Hole, 1,000 to 1,499 ATK"] = weak
    found["Trap Hole, first window of a game"] = first
    return found


def roar(games: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Threatening Roar: where it is activated, and what it answers there.

    A window is told by the other player's turn and phase and, at their
    Battle Phase, by what they have to attack with.  An activation answers
    a card of theirs on the chain or one that has just arrived, or nothing;
    spent early is one that answers nothing while they have no monster in
    Attack Position."""
    said = f"Chain: Activate {ROAR}"
    windows: Counter[str] = Counter()
    taken: Counter[str] = Counter()
    pilot: Counter[str] = Counter()
    answering: Counter[str] = Counter()
    activations = early = 0
    for rows in games.values():
        for row in rows:
            if said not in row["said"]:
                continue
            monsters = theirs(row)
            attackers = [1 for _, position, _ in monsters if position == FACE_UP_ATTACK]
            if row["turn_player"] != 1:
                where = "own turn"
            elif row["phase"] in BATTLE:
                where = (
                    "battle, attackers"
                    if attackers
                    else "battle, no monster"
                    if not monsters
                    else "battle, none attacking"
                )
            else:
                where = str(row["phase"])
            windows[where] += 1
            pilot[where] += pilot_took(row, said)
            if not took(row, said):
                continue
            activations += 1
            taken[where] += 1
            link = row["chain"][-1] if row["chain"] else None
            if link is not None and link.get("controller") == 1:
                answered = "chain"
            elif any(event.get("controller") == 1 for event in row["events"]):
                answered = "arrival"
            else:
                answered = "nothing"
            answering[f"{where}: {answered}"] += 1
            early += answered == "nothing" and not attackers
    return {
        "activations": activations,
        "windows": dict(windows),
        "taken": dict(taken),
        "pilot": dict(pilot),
        "answering": dict(answering),
        "spent_early": early,
    }


def counted(
    rows: list[dict[str, Any]],
    idle_rows: list[dict[str, Any]],
    name: Callable[[int], str],
    kind: Callable[[int], str],
) -> dict[str, Any]:
    """Everything the table tells of one checkpoint's games."""
    games, over = by_game(rows)
    won: Counter[str] = Counter()
    ended: Counter[str] = Counter()
    for row in over.values():
        ended[row["opponent"]] += 1
        won[row["opponent"]] += row["over"]["winner"] == row["seat"]
    decisions = [row for found in games.values() for row in found]
    # The pilot's answer in its words: the other copy of a card is the same answer.
    agreed = sum(
        row["pilot"] is not None and took(row, row["said"][row["pilot"]]) for row in decisions
    )
    _, idle_over = by_game(idle_rows)
    idle_won = [row for row in idle_over.values() if row["over"]["winner"] == row["seat"]]
    return {
        "games": sum(ended.values()),
        "won": sum(won.values()),
        "by_opponent": {deck: [won[deck], ended[deck]] for deck in sorted(ended)},
        "decisions": len(decisions),
        "pilot_answers": agreed,
        "allure": allure(games, name, kind),
        "traps": traps(games),
        "roar": roar(games),
        "idle": {
            "games": len(idle_over),
            "won": len(idle_won),
            "turns": sum(row["over"]["turns"] for row in idle_won) / max(len(idle_won), 1),
        },
    }


def of(count: int, total: int) -> str:
    return "-" if not total else f"{count} of {total} ({count / total:.0%})"


def lines(found: dict[str, Any]) -> list[tuple[str, str]]:
    """The rows of the table for one checkpoint: what is told, and its figure."""
    cards, held, said = found["allure"], found["traps"], found["roar"]
    turns, activated = cards["turns"], cards["activated"]

    def window(counts: dict[str, int]) -> str:
        share = (
            f"; its pilot {counts['pilot'] / counts['windows']:.0%}" if counts["windows"] else ""
        )
        return of(counts["taken"], counts["windows"]) + share

    def where(key: str) -> str:
        return of(said["taken"].get(key, 0), said["windows"].get(key, 0))

    told = [
        ("games won", of(found["won"], found["games"])),
        ("its pilot's answer", of(found["pilot_answers"], found["decisions"])),
    ]
    told += [
        (f"won against {deck}", of(*found["by_opponent"][deck]))
        for deck in (LEARNER, "machina")
        if deck in found["by_opponent"]
    ]
    told += [
        (
            f"{ALLURE}, turns with a DARK monster in hand: activated",
            of(activated.get(DARK, 0), turns.get(DARK, 0)),
        ),
        ("... with nothing in hand but it", of(activated.get(ALONE, 0), turns.get(ALONE, 0))),
        (
            "... with only Spells and Traps beside it",
            of(activated.get(SPELLS, 0), turns.get(SPELLS, 0)),
        ),
        ("... of those, with cards still to set", str(cards["with_cards_still_to_set"])),
        (
            "activated with no DARK monster in hand: the hand lost",
            of(cards["hand_lost"], cards["activations_without_a_dark_monster"]),
        ),
        ("... a hand with other cards in it", str(cards["hand_lost_with_other_cards"])),
    ]
    told += [(f"{card} at a window that offers it", window(held[card])) for card in TRAPS]
    told += [
        (
            "Trap Hole, a monster of 1,000 to 1,499 ATK",
            window(held["Trap Hole, 1,000 to 1,499 ATK"]),
        ),
        (
            "Trap Hole at the first window of a game",
            window(held["Trap Hole, first window of a game"]),
        ),
        (f"{ROAR}: activations", str(said["activations"])),
        ("... at their Battle Phase, monsters in Attack Position", where("battle, attackers")),
        ("... at their Battle Phase, no monster of theirs", where("battle, no monster")),
        ("... in their Main Phase 1", where("Main1")),
        (
            "...... of those, in answer to a card on the chain",
            str(said["answering"].get("Main1: chain", 0)),
        ),
        ("... in their Draw Phase", where("Draw")),
        ("... spent before they had an attacker, answering nothing", str(said["spent_early"])),
        (
            "against a player who does nothing: won",
            of(found["idle"]["won"], found["idle"]["games"]),
        ),
        (
            "... in turns, on the average",
            f"{found['idle']['turns']:.1f}" if found["idle"]["won"] else "-",
        ),
    ]
    return told


def table(columns: list[tuple[str, dict[str, Any]]]) -> str:
    """The checkpoints beside each other, a row for each thing told."""
    told = [(name, lines(found)) for name, found in columns]
    labels = [label for label, _ in told[0][1]]
    width = max(len(label) for label in labels) + 2
    wide = max(24, *(len(figure) + 2 for _, rows in told for _, figure in rows))
    text = [f"{'':<{width}}" + "".join(f"{name:>{wide}}" for name, _ in told)]
    for place, label in enumerate(labels):
        text.append(f"{label:<{width}}" + "".join(f"{rows[place][1]:>{wide}}" for _, rows in told))
    return "\n".join(text)


def card_data(root: Path) -> tuple[Callable[[int], str], Callable[[int], str]]:
    """The name of a card, and whether it is a DARK monster (``dark``),
    another monster (``monster``) or a Spell or Trap (``spell``)."""
    from ygo_training.environment import CARDS_DATABASE

    with closing(sqlite3.connect(f"file:{root / CARDS_DATABASE}?mode=ro", uri=True)) as cards:
        names = {int(code): str(name) for code, name in cards.execute("select id, name from texts")}
        printed = {
            int(code): (int(kind), int(attribute))
            for code, kind, attribute in cards.execute("select id, type, attribute from datas")
        }

    def kind(code: int) -> str:
        # A monster's type has its first bit, and DARK is the attribute 0x20.
        card_type, attribute = printed.get(int(code), (0, 0))
        if not card_type & 0x1:
            return "spell"
        return "dark" if attribute == 0x20 else "monster"

    return (lambda code: names.get(int(code), "")), kind


def decisions_of(
    checkpoint: Path, kept: Path | None, name: str, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The decisions of a checkpoint's games, played now or read again where
    they were kept for this checkpoint and these games."""
    from ygo_training.checkpoints import load_inference_model, value_fingerprint
    from ygo_training.environment import Environment
    from ygo_training.revisions import duel_revisions

    model, payload = load_inference_model(checkpoint, device=args.device)
    described = {
        "checkpoint": value_fingerprint(model.state_dict()),
        "step": payload.get("step"),
        "opponents": args.opponents,
        "games": args.games,
        "idle_opponents": args.idle_opponents,
        "idle_games": args.idle_games,
        "seed": args.seed,
        "duel_revisions": duel_revisions(args.root),
    }
    where = None if kept is None else kept / f"{name}.jsonl"
    if where is not None and where.is_file():
        with where.open(encoding="utf8") as handle:
            if json.loads(handle.readline()) != {"described": described}:
                raise SystemExit(
                    f"{where} is of another checkpoint or of other games: move it away"
                )
            rows = [json.loads(line) for line in handle]
        return (
            [row for row in rows if "idle" not in row],
            [
                {key: value for key, value in row.items() if key != "idle"}
                for row in rows
                if "idle" in row
            ],
        )
    with Environment(args.root) as environment:
        plays = list(
            played(
                environment,
                model,
                args.device,
                opponents=args.opponents.split(","),
                games=args.games,
                seed=args.seed,
            )
        )
        idles = list(
            played(
                environment,
                model,
                args.device,
                opponents=args.idle_opponents.split(",") if args.idle_games else [],
                games=args.idle_games,
                seed=args.seed,
                against_idle=True,
            )
        )
    if where is not None:
        where.parent.mkdir(parents=True, exist_ok=True)
        with where.open("w", encoding="utf8") as handle:
            handle.write(json.dumps({"described": described}) + "\n")
            for row in plays:
                handle.write(json.dumps(row) + "\n")
            for row in idles:
                handle.write(json.dumps({**row, "idle": True}) + "\n")
    return plays, idles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", metavar="NAME=CHECKPOINT")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponents", default=OPPONENTS)
    parser.add_argument("--games", type=int, default=160, help="games against each opponent")
    parser.add_argument("--idle-opponents", default=IDLE_OPPONENTS)
    parser.add_argument(
        "--idle-games",
        type=int,
        default=60,
        help="games against a player who does nothing, each deck",
    )
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--keep", type=Path, help="a folder for each checkpoint's decisions")
    parser.add_argument("--output", type=Path, help="the counts, as JSON")
    args = parser.parse_args()
    if args.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
        import torch

        from ygo_training.edopro import PROCESSOR_THREADS

        torch.set_num_threads(PROCESSOR_THREADS)
    name_of, kind_of = card_data(args.root)
    columns = []
    for given in args.checkpoints:
        name, _, path = given.partition("=")
        if not path:
            raise SystemExit(f"{given}: a checkpoint is given as NAME=CHECKPOINT")
        plays, idles = decisions_of(Path(path), args.keep, name, args)
        columns.append((name, counted(plays, idles, name_of, kind_of)))
        print(f"{name}: {columns[-1][1]['games']} games", file=sys.stderr, flush=True)
    print(table(columns))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(columns), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
