"""What a network gains from cards of the list it plays: a suite's games as
they are, and again with the network barred from summoning, setting or
activating the cards of a group at the decisions of a turn (a decision that
offers nothing else is left whole), game by game on the same deals
(experiments/2026-10-19-darker-than-black.md).

    uv run python tools/forbidden_cards.py CHECKPOINT \\
        --suite eval_suites/ygo-policies-blackwing-wide.json \\
        --learner-list data/decks/darker-than-black.ydk \\
        --output analysis/darker-than-black/forbidden/half.json

A card of the list is told as in `tools/card_use.py`: of its deck's own list,
of another deck the network trained on (`--trained`), or never trained on.
What a group's cards are worth to the network is what it loses without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from card_use import (
    DECKS,
    NEVER,
    OWN,
    QUESTIONS,
    THIRTEEN,
    TRAINED,
    USES,
    Names,
    groups,
    listed,
)
from ppo_head_report import difference
from search_leads_report import percent, points

from ygo_training.environment import CARDS_DATABASE, pool
from ygo_training.evaluation_suite import load_evaluation_suite
from ygo_training.orchestration import evaluation_assignments
from ygo_training.pool import DuelPool, Round, evaluate_games

AS_IT_IS = "as it is"
GROUPS = {"never": NEVER, "trained": TRAINED, "own": OWN}


def barring(duels: DuelPool, codes: set[int]) -> Any:
    """What bars the network from summoning, setting or activating the cards
    of `codes` at the decisions of a turn."""
    uses = np.array([kind for kind, name in duels.choice_names.items() if name in USES])
    turn = np.array([kind for kind, name in duels.decision_names.items() if name in QUESTIONS])
    listed_codes = np.array(sorted(codes), dtype=np.int64)

    def forbid(round_: Round, mine: np.ndarray) -> np.ndarray:
        cards = round_.arrays["action_card"][mine]
        kinds = round_.arrays["action_kind"][mine]
        questions = round_.arrays["decision_kind"].reshape(-1)[mine]
        return (
            np.isin(cards, listed_codes) & np.isin(kinds, uses) & np.isin(questions, turn)[:, None]
        )

    return forbid


def measure(
    model: Any,
    suite: dict[str, Any],
    *,
    learner_list: Path | None,
    trained: list[str],
    barred: list[str],
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
    conditions: dict[str, Any] = {}
    with DuelPool(ROOT, threads=threads, slots=slots) as duels:
        for condition in (AS_IT_IS, *barred):
            codes = {code for code in played if cards[name(code)] == condition}
            forbid = None if condition == AS_IT_IS else barring(duels, codes)
            won, changed = [], 0
            for index, case in enumerate(suite["cases"]):
                if str(case["learner"]) != learner:
                    raise SystemExit(f"{case['name']}: another learner than {learner}")
                results = evaluate_games(
                    duels,
                    model,
                    device,
                    learner=learner,
                    assignments=evaluation_assignments(
                        [str(case["opponent"])],
                        int(case["games"]),
                        seed + int(case.get("seed_index", index)) * 10_000_000,
                    ),
                    limit=limit,
                    learner_list=None if learner_list is None else str(learner_list),
                    forbid=forbid,
                )
                won.extend(float(result["outcome"] == "win") for result in results)
                changed += sum(int(result.get("barred", 0)) for result in results)
            conditions[condition] = {
                "won": won,
                "barred": changed,
                "cards": sorted({name(code) for code in codes}),
            }
    return {"cards": cards, "conditions": conditions}


def summary(result: dict[str, Any]) -> str:
    conditions = result["conditions"]
    base = conditions[AS_IT_IS]["won"]
    lines = [
        f"Won as it is: {percent(sum(base) / len(base))} of {len(base):,} games.",
        "Barred from summoning, setting and activating the cards of a group:",
    ]
    for condition, found in conditions.items():
        if condition == AS_IT_IS:
            continue
        won = found["won"]
        gap, error = difference(won, base)
        lines.append(
            f"  {condition} ({len(found['cards'])} cards): {percent(sum(won) / len(won))}, "
            f"{points(gap, error)} on the same deals, "
            f"{found['barred'] / len(won):.2f} answers a game changed"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--learner-list", type=Path)
    parser.add_argument(
        "--trained", default=THIRTEEN, help="the decks the network trained on, by policy id"
    )
    parser.add_argument(
        "--barred",
        default="never,trained",
        help="the groups barred in turn: never, trained or own",
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
        barred=[GROUPS[group.strip()] for group in args.barred.split(",") if group.strip()],
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
