"""What a network does with each card of the list it plays: at the decisions
of its seat that offer an answer with the card (a summon, a set, an
activation and so on, by its kind), how often it takes one, beside how often
the pilot asked alongside it would (experiments/2026-10-19-darker-than-black.md).

    uv run python tools/card_use.py CHECKPOINT \\
        --suite eval_suites/ygo-policies-blackwing-wide.json \\
        --learner-list data/decks/darker-than-black.ydk \\
        --output analysis/darker-than-black/card-use/half.json

The games are a suite's, dealt as `evaluate-suite` deals them on the same
seed, the network taking its first choice. Each card of the list is told by
what it is to the network: a card of its deck's own list, one of another deck
it trained on (`--trained`), or one it never trained on.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.environment import CARDS_DATABASE, pool
from ygo_training.evaluation_suite import load_evaluation_suite
from ygo_training.orchestration import evaluation_assignments
from ygo_training.pool import DuelPool, Round, evaluate_games

DECKS = ROOT / "native" / "ygo-policies" / "decks"
THIRTEEN = (
    "blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,"
    "morphtronic,rock-block,spellcaster"
)
# What a card of the list is to the network.
OWN, TRAINED, NEVER = "its deck's list", "another deck trained on", "never trained on"
# The answers that put a card to use, at the decisions of a turn.
USES = ("NormalSummon", "SpecialSummon", "SetMonster", "SetSpellTrap", "Activate")
QUESTIONS = ("Idle", "Battle", "Chain")


def listed(path: Path) -> list[int]:
    """The codes of a `.ydk` list, main and extra deck."""
    codes, part = [], False
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if line in ("#main", "#extra"):
            part = True
        elif line.startswith("!"):
            part = False
        elif part and line.isdigit():
            codes.append(int(line))
    return codes


class Names:
    """A card's name by its code, an alternate artwork by its original's."""

    def __init__(self, database: Path) -> None:
        connection = sqlite3.connect(database)
        self.names = dict(connection.execute("select id, name from texts"))
        self.alias = dict(connection.execute("select id, alias from datas"))

    def __call__(self, code: int) -> str:
        return self.names.get(self.alias.get(code) or code, str(code))


def groups(played: list[int], learner: str, trained: list[str], name: Names) -> dict[str, str]:
    """What each card of the list played is to the network, by its name."""
    decks = pool(ROOT)
    own = {name(code) for code in listed(DECKS / f"{decks[learner]}.ydk")}
    seen = {name(code) for deck in trained for code in listed(DECKS / f"{decks[deck]}.ydk")}
    found = {}
    for code in played:
        card = name(code)
        found[card] = OWN if card in own else TRAINED if card in seen else NEVER
    return found


class Counts:
    """At each decision of the learner's seat, for each card of the list
    that an answer of it uses, by the decision's kind and the answer's: that
    it was offered, and whether the network took it and the pilot would."""

    def __init__(self, pool_: DuelPool, cards: dict[str, str], name: Names) -> None:
        self.pool = pool_
        self.cards = cards
        self.name = name
        self.counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    def __call__(self, round_: Round, mine: np.ndarray, chosen: np.ndarray) -> None:
        codes = round_.arrays["action_card"]
        kinds = round_.arrays["action_kind"]
        questions = round_.arrays["decision_kind"].reshape(-1)
        for row in mine.tolist():
            question = self.pool.decision_names[int(questions[row])]
            offered = {}
            for answer, (code, kind) in enumerate(zip(codes[row], kinds[row], strict=True)):
                card = self.name(int(code)) if code else None
                if kind > 0 and card in self.cards:
                    offered.setdefault((card, self.pool.choice_names[int(kind)]), []).append(answer)
            label = int(round_.label[row])
            for (card, kind), answers in offered.items():
                count = self.counts[(question, kind, card)]
                count[0] += 1
                count[1] += int(int(chosen[row]) in answers)
                if label >= 0:
                    count[2] += 1
                    count[3] += int(label in answers)

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "question": question,
                "kind": kind,
                "card": card,
                "group": self.cards[card],
                "offered": offered,
                "network": network,
                "labelled": labelled,
                "pilot": pilot,
            }
            for (question, kind, card), (offered, network, labelled, pilot) in sorted(
                self.counts.items()
            )
        ]


def measure(
    model: Any,
    suite: dict[str, Any],
    *,
    learner_list: Path | None,
    trained: list[str],
    seed: int,
    device: str,
    threads: int,
    slots: int,
    limit: int = 4096,
) -> dict[str, Any]:
    name = Names(CARDS_DATABASE if Path(CARDS_DATABASE).is_absolute() else ROOT / CARDS_DATABASE)
    learner = str(suite["cases"][0]["learner"])
    played = (
        listed(learner_list)
        if learner_list is not None
        else listed(DECKS / f"{pool(ROOT)[learner]}.ydk")
    )
    cards = groups(played, learner, trained, name)
    games = 0
    with DuelPool(ROOT, threads=threads, slots=slots) as duels:
        counts = Counts(duels, cards, name)
        for index, case in enumerate(suite["cases"]):
            if str(case["learner"]) != learner:
                raise SystemExit(f"{case['name']}: another learner than {learner}")
            assignments = evaluation_assignments(
                [str(case["opponent"])],
                int(case["games"]),
                seed + int(case.get("seed_index", index)) * 10_000_000,
            )
            evaluate_games(
                duels,
                model,
                device,
                learner=learner,
                assignments=assignments,
                limit=limit,
                learner_list=None if learner_list is None else str(learner_list),
                observe=counts,
            )
            games += len(assignments)
    return {"games": games, "cards": cards, "counts": counts.rows()}


def summary(result: dict[str, Any]) -> str:
    """Each card's uses at the decisions of a turn, the cards of a group together."""
    by_card: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for row in result["counts"]:
        if row["question"] in QUESTIONS and row["kind"] in USES:
            count = by_card[row["card"]]
            for place, key in enumerate(("offered", "network", "labelled", "pilot")):
                count[place] += row[key]
    lines = [
        "At the decisions of a turn that offer to summon, set or activate a card of the list:",
        f"  {'':44}{'offered':>9}{'network':>9}{'pilot':>9}",
    ]
    for group in (OWN, TRAINED, NEVER):
        cards = sorted(card for card, told in result["cards"].items() if told == group)
        total = [0, 0, 0, 0]
        lines.append(f"  {group}:")
        for card in cards:
            offered, network, labelled, pilot = by_card.get(card, [0, 0, 0, 0])
            total = [a + b for a, b in zip(total, (offered, network, labelled, pilot), strict=True)]
            lines.append(
                f"    {card[:42]:42}{offered:>9}{rate(network, offered):>9}{rate(pilot, labelled):>9}"
            )
        offered, network, labelled, pilot = total
        lines.append(
            f"    {'together':42}{offered:>9}{rate(network, offered):>9}{rate(pilot, labelled):>9}"
        )
    return "\n".join(lines)


def rate(part: int, whole: int) -> str:
    return "-" if not whole else f"{100 * part / whole:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--learner-list", type=Path)
    parser.add_argument(
        "--trained", default=THIRTEEN, help="the decks the network trained on, by policy id"
    )
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool-threads", type=int, default=4)
    parser.add_argument("--pool-slots", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from ygo_training.checkpoints import load_inference_model

    model, payload = load_inference_model(args.checkpoint, device=args.device)
    learner_list = None if args.learner_list is None else args.learner_list.resolve()
    result = measure(
        model,
        load_evaluation_suite(args.suite.resolve()),
        learner_list=learner_list,
        trained=[deck.strip() for deck in args.trained.split(",") if deck.strip()],
        seed=args.seed,
        device=args.device,
        threads=args.pool_threads,
        slots=args.pool_slots,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": payload.get("step"),
            "suite": str(args.suite.resolve()),
            "learner_list": None if learner_list is None else str(learner_list),
            "seed": args.seed,
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    print(summary(result))


if __name__ == "__main__":
    main()
