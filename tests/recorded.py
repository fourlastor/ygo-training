"""Recorded duels for tests, made in the environment as a search run makes them."""

from __future__ import annotations

import json
from typing import Any

COMMANDS = (1, 2, 3)  # Idle, Battle, Chain


def recorded_duel(
    environment,
    *,
    seed: int = 7,
    opponent: str = "monarch",
    learner_player: int = 0,
    leave_at: int = 10,
    learner: str = "blackwing",
) -> dict[str, Any]:
    """A duel the pilots play, but for one answer of the learner's seat at its
    ``leave_at``-th command (none when zero): what a worker plays again."""
    names = [learner, opponent] if learner_player == 0 else [opponent, learner]
    seats = [{"control": "pilot"}, {"control": "pilot"}]
    seats[learner_player] = {"control": "caller"}
    setup = {"seed": seed, "decks": names, "seats": seats, "keep_record": True}
    commands, left = 0, []
    with environment.start(setup) as episode:
        while True:
            event = episode.advance()
            view = event["decision"]
            if view is not None:
                choice = view["label"]
                if view["question"]["kind"] in COMMANDS:
                    commands += 1
                    if commands == leave_at:
                        left.append(view["answer"])
                        choice = (choice + 1) % len(view["choices"])
                episode.choose(choice)
            over = event["over"]
            if over is not None:
                break
    return {
        "index": 0,
        "opponent": opponent,
        "learner_player": learner_player,
        "seed": seed,
        "record": json.dumps(over["record"]),
        "digest": over["digest"],
        "deviations": left,
        "examined": None,
        "winner": over["winner"],
        "decisions": over["decisions"],
    }


def search_row(environment, *, seed: int, opponent: str, learner_player: int) -> dict[str, Any]:
    """One deal as a row of ``policy-bench search --record true``: the pilot's
    own game, and one that leaves the pilot once."""

    def game(leave_at: int) -> dict[str, Any]:
        duel = recorded_duel(
            environment,
            seed=seed,
            opponent=opponent,
            learner_player=learner_player,
            leave_at=leave_at,
        )
        return {
            "winner": duel["winner"],
            "digest": duel["digest"],
            "decisions": duel["decisions"],
            "record": json.loads(duel["record"]),
            "deviations": [{"decision": place, "kind": "Idle"} for place in duel["deviations"]],
        }

    searched = game(10)
    return {
        "a": "blackwing",
        "b": opponent,
        "seed": seed,
        "seat": learner_player,
        "baseline": {key: value for key, value in game(0).items() if key != "deviations"},
        "search": {**searched, "examined": []},
    }
