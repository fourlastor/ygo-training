"""Aggregate the evaluation duels played by the workers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _tally(games: list[dict[str, Any]]) -> dict[str, float]:
    """Wins, losses and draws of some games, with how they ended."""
    total = len(games)
    wins = sum(game["outcome"] == "win" for game in games)
    return {
        "games": total,
        "wins": wins,
        "losses": sum(game["outcome"] == "loss" for game in games),
        "draws": sum(game["outcome"] == "draw" for game in games),
        # A duel stopped at the decision limit counts as a draw.
        "cutoff_draws": sum(game["outcome"] == "draw" and not game["terminal"] for game in games),
        "terminal_draws": sum(game["outcome"] == "draw" and game["terminal"] for game in games),
        "deckout_losses": sum(
            game["outcome"] == "loss" and game["win_reason"] == 2 for game in games
        ),
        "deckout_wins": sum(game["outcome"] == "win" and game["win_reason"] == 2 for game in games),
        "win_rate": wins / max(total, 1),
        "mean_decisions": sum(game["decisions"] for game in games) / max(total, 1),
        "mean_turns": sum(game["turns"] for game in games) / max(total, 1),
    }


def summarize_results(results: list[dict[str, Any]], *, outcomes: bool = False) -> dict[str, Any]:
    """Summarize one case's duels: overall, by opponent and by seat.  With
    ``outcomes``, how each ended as well, in the order they were given: two
    players of the same deals are then set beside each other deal by deal."""
    by_opponent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_opponent[result["opponent"]].append(result)
    answers: Counter[str] = Counter()
    for game in results:
        answers.update(game.get("learner_answers", {}))
    labels = sum(game.get("pilot_labels", 0) for game in results)
    summary: dict[str, Any] = {
        **_tally(results),
        "by_opponent": {opponent: _tally(games) for opponent, games in by_opponent.items()},
        "by_player": {
            str(player): _tally(games)
            for player in (0, 1)
            if (games := [game for game in results if game["learner_player"] == player])
        },
        # The learner's answers by question and kind, most frequent first.
        "learner_answers": dict(sorted(answers.items(), key=lambda item: (-item[1], item[0]))),
    }
    if labels:
        # How often the model's first choice was its deck's pilot's answer.
        summary["pilot_agreement"] = (
            sum(game.get("pilot_agreements", 0) for game in results) / labels
        )
    if outcomes:
        summary["outcomes"] = [str(game["outcome"]) for game in results]
    return summary
