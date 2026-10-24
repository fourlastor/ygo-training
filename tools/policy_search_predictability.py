"""How much of a duel's result can be told at the decisions a search examined,
and how much of it a small network tells from a few numbers.

    uv run python tools/policy_search_predictability.py TRAIN.jsonl MEASURE.jsonl \\
        --output analysis/learner/references.json

Both files are rows of `policy-bench search --record true --log full` (or
`--log true`).  At every decision it examined, the search played the pilot's
answer out in some worlds dealt again from what the seat sees: the rate at
which it wins them estimates what the decision is worth to a seat that goes
on as its pilot does.  From the rows of MEASURE this gives

- of one result, the share a perfect predictor would explain: the spread of
  those rates over the decisions, less what so few worlds add to it, against
  the spread of a single result;
- what a small network tells of it that sees only the log's `situation` (life
  points, counts of cards, the turn and the phase), without and with the
  opponent's deck.  It is trained on the rows of TRAIN, once on the results
  of their games, as a value head is, and once on the worlds' rates.

A value head is read the same way by `evaluate-search`
(`ygo_training.evaluation`): the two stand beside each other.  Only the
decisions of the kinds that open a transaction are read (`--kinds`), which is
where a value head is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ygo_training.evaluation import VALUE_TURNS, _told
from ygo_training.records import PAYOFF

# The kinds of decision a search examines that open a transaction.
OPENING = ("Idle", "Battle", "Chain")
TARGETS = ("result", "rate")


def load(path: Path, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    """The games of a logged run: for each, its opponent, its result to the
    searching seat, and its examined decisions of ``kinds`` with the rate at
    which the pilot's answer wins the worlds of the first stage."""
    games: list[dict[str, Any]] = []
    with path.open(encoding="utf8") as rows:
        for text in rows:
            if not text.strip():
                continue
            row = json.loads(text)
            played = row["search"]
            if "examined" not in played:
                raise SystemExit(f"{path} holds no log: run policy-bench search with --log")
            winner = played.get("winner")
            games.append(
                {
                    "opponent": str(row["b"]),
                    "seat": int(row["seat"]),
                    # As the trainer rewards it: a win 1, a loss -1.
                    "result": None
                    if winner is None
                    else (1.0 if int(winner) == int(row["seat"]) else -1.0),
                    "examined": [entry for entry in played["examined"] if entry["kind"] in kinds],
                }
            )
    lengths = [len(entry["played"][0]) for game in games for entry in game["examined"]]
    if not lengths:
        raise SystemExit(f"{path} holds no examined decision of {', '.join(kinds)}")
    # The first stage is as long as the shortest run of worlds in the log.
    first = min(lengths)
    for game in games:
        decisions = []
        for entry in game.pop("examined"):
            own = [PAYOFF[mark] for mark in entry["played"][0][:first] if mark in PAYOFF]
            if len(own) > 1:
                decisions.append(
                    {
                        "situation": entry["situation"],
                        "kind": entry["kind"],
                        "turn": int(entry["turn"]),
                        "rate": sum(own) / len(own),
                        "worlds": len(own),
                    }
                )
        game["decisions"] = decisions
    return [game for game in games if game["result"] is not None]


class Features:
    """A decision as the small network sees it."""

    def __init__(self, games: list[dict[str, Any]], *, opponent: bool) -> None:
        self.opponent = opponent
        self.phases = sorted({str(d["situation"]["phase"]) for g in games for d in g["decisions"]})
        self.kinds = sorted({d["kind"] for g in games for d in g["decisions"]})
        self.opponents = sorted({g["opponent"] for g in games})

    def row(self, game: dict[str, Any], decision: dict[str, Any]) -> list[float]:
        seen = decision["situation"]
        numbers = [
            seen["life"][0] / 8000,
            seen["life"][1] / 8000,
            (seen["life"][0] - seen["life"][1]) / 8000,
            *(
                seen[name][side] / 5
                for name in ("hand", "monsters", "backrow", "set_backrow")
                for side in (0, 1)
            ),
            float(seen["own_turn"]),
            float(seen["attack"]),
            min(decision["turn"], 30) / 10,
            float(game["seat"]),
        ]
        numbers += [float(str(seen["phase"]) == phase) for phase in self.phases]
        numbers += [float(decision["kind"] == kind) for kind in self.kinds]
        if self.opponent:
            numbers += [float(game["opponent"] == name) for name in self.opponents]
        return numbers


def fit(inputs: Any, targets: Any, held_inputs: Any, held_targets: Any, *, seed: int) -> Any:
    """A network of two layers of 64 on ``inputs``, stopped where it is best
    on the games held back."""
    import torch

    torch.manual_seed(seed)
    mean, deviation = inputs.mean(0), inputs.std(0) + 1e-6
    net = torch.nn.Sequential(
        torch.nn.Linear(inputs.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Tanh(),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scaled, held = (inputs - mean) / deviation, (held_inputs - mean) / deviation
    best, kept, waited = math.inf, None, 0
    for _ in range(60):
        order = torch.randperm(len(scaled))
        for start in range(0, len(scaled), 1024):
            batch = order[start : start + 1024]
            loss = torch.nn.functional.mse_loss(net(scaled[batch]).squeeze(-1), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            error = float(torch.nn.functional.mse_loss(net(held).squeeze(-1), held_targets))
        if error < best - 1e-4:
            best, waited = error, 0
            kept = {name: value.clone() for name, value in net.state_dict().items()}
        else:
            waited += 1
            if waited >= 5:
                break
    net.load_state_dict(kept)
    weights = sum(parameter.numel() for parameter in net.parameters())

    def predict(values: Any) -> list[float]:
        with torch.no_grad():
            return net((values - mean) / deviation).squeeze(-1).tolist()

    return predict, weights


def told(decisions: list[tuple[dict[str, Any], float]]) -> dict[str, Any] | None:
    """Values against the worlds, over all the decisions and by turn."""

    def read(group: list[tuple[dict[str, Any], float]]) -> dict[str, Any] | None:
        return _told([(value, d["rate"], d["worlds"]) for d, value in group])

    summary = read(decisions)
    if summary is None:
        return None
    summary["by_turn"] = [
        {"turns": [low, high], **part}
        for low, high in VALUE_TURNS
        if (
            part := read(
                [
                    (d, value)
                    for d, value in decisions
                    if low <= d["turn"] and (high is None or d["turn"] <= high)
                ]
            )
        )
        is not None
    ]
    return summary


def share(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def by_turn(summary: dict[str, Any], key: str) -> str:
    return ", ".join(
        f"{part['turns'][0]}"
        + ("+" if part["turns"][1] is None else f"-{part['turns'][1]}")
        + f" {share(part[key])}"
        for part in summary["by_turn"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train", type=Path)
    parser.add_argument("measure", type=Path)
    parser.add_argument("--kinds", default=",".join(OPENING))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    kinds = tuple(args.kinds.split(","))
    import torch

    training, measured = load(args.train, kinds), load(args.measure, kinds)
    points = [(game, decision) for game in measured for decision in game["decisions"]]
    # The worlds themselves: every decision's rate against itself gives what
    # is real in their spread and what a result has beyond it.
    worlds = told([(decision, decision["rate"]) for _, decision in points])
    if worlds is None:
        raise SystemExit(f"{args.measure} holds too few decisions")
    won = sum(game["result"] > 0 for game in measured) / len(measured)
    print(
        f"{len(measured)} games of {args.measure.name}, {len(points)} examined decisions of "
        f"{', '.join(kinds)} ({len(points) / len(measured):.1f} a game), {share(won)} won."
    )
    print(
        f"Of one result a perfect predictor explains {share(worlds['result'])} there; "
        f"by turn: {by_turn(worlds, 'result')}."
    )
    report: dict[str, Any] = {
        "train": str(args.train),
        "measure": str(args.measure),
        "kinds": list(kinds),
        "games": len(measured),
        "decisions": len(points),
        "worlds": worlds,
        "networks": [],
    }
    # A tenth of the training games tells when to stop.
    held = {index for index in range(len(training)) if index % 10 == 0}
    print(
        f"A small network on the situation, trained on {len(training) - len(held)} games of "
        f"{args.train.name}: of what can be told it tells, and of one result that is"
    )
    for opponent in (False, True):
        features = Features(training, opponent=opponent)

        def tensors(games: list[dict[str, Any]], target: str, features: Features = features) -> Any:
            rows = [(g, d) for g in games for d in g["decisions"]]
            return (
                torch.tensor([features.row(g, d) for g, d in rows], dtype=torch.float32),
                torch.tensor(
                    [g["result"] if target == "result" else d["rate"] for g, d in rows],
                    dtype=torch.float32,
                ),
            )

        for target in TARGETS:
            inputs, targets = tensors(
                [g for index, g in enumerate(training) if index not in held], target
            )
            held_inputs, held_targets = tensors(
                [g for index, g in enumerate(training) if index in held], target
            )
            predict, weights = fit(inputs, targets, held_inputs, held_targets, seed=args.seed)
            values = predict(tensors(measured, target)[0])
            summary = told([(decision, value) for (_, decision), value in zip(points, values)])
            of_result = (
                None
                if summary is None or summary["told"] is None
                else summary["told"] * worlds["result"]
            )
            lesson = "the games' results" if target == "result" else "the worlds' rates"
            sees = "with the opponent's deck" if opponent else "without the opponent's deck"
            print(
                f"  taught {lesson}, {sees} ({weights} weights): "
                f"{share(None if summary is None else summary['told'])}, {share(of_result)}; "
                f"by turn: {'-' if summary is None else by_turn(summary, 'told')}"
            )
            report["networks"].append(
                {
                    "target": target,
                    "opponent": opponent,
                    "weights": weights,
                    "told": summary,
                    "of_one_result": of_result,
                }
            )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")


if __name__ == "__main__":
    main()
