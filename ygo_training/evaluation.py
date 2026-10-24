"""Aggregate the evaluation duels played by the workers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

# How far under the weight of the pilot's answer an alternative may stand
# and still be taken, in `summarize_tried`'s reading at several bars.
BARS = (1.0, 0.5, 0.2, 0.1, 0.05)
# The leads a player may ask of its lead head before it leaves its first
# choice, in units of win rate, and the shares of the alternatives the head
# ranks highest that `summarize_tried` gives the measured lead of.
MARGINS = (0.05, 0.1, 0.15, 0.2, 0.3)
# Where a change of the rule starts and ends: it leaves the pilot's answer,
# returns to it from another, or goes from one other answer to another.
STARTS = ("from_pilot", "to_pilot", "between_others")
RANKED_HIGHEST = (0.001, 0.005, 0.01, 0.05, 0.25)
# The turns a value head's reading is given apart for.
VALUE_TURNS = ((1, 2), (3, 4), (5, 6), (7, 10), (11, None))


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


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _answer_key(counts: Counter[str], examined_known: bool) -> dict[str, Any]:
    """One reading of the counts of ``workers.watch_worker``."""
    deviations = counts["left_search"] + counts["left_pilot"] + counts["left_other"]
    kept = counts["kept_pilot"] + counts["kept_other"]
    reading: dict[str, Any] = {
        # Where the search left its pilot: the model's first choice is the
        # search's answer, still the pilot's, or neither.
        "deviations": deviations,
        "search": counts["left_search"],
        "pilot": counts["left_pilot"],
        "other": counts["left_other"],
        "search_rate": _rate(counts["left_search"], deviations),
        "pilot_rate": _rate(counts["left_pilot"], deviations),
        # Where the played answer is the pilot's: the model's is another.
        "kept": kept,
        "kept_left": counts["kept_other"],
        "kept_left_rate": _rate(counts["kept_other"], kept),
    }
    if examined_known:
        # The same, among the decisions the search examined and kept.
        examined = counts["examined_pilot"] + counts["examined_other"]
        reading["examined_kept"] = examined
        reading["examined_kept_left"] = counts["examined_other"]
        reading["examined_kept_left_rate"] = _rate(counts["examined_other"], examined)
    return reading


def _mean(values: list[float]) -> tuple[float | None, float | None]:
    """A mean and its standard error."""
    count = len(values)
    if not count:
        return None, None
    mean = sum(values) / count
    if count < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance / count)


def _departures(picks: list[tuple[int, int | None, dict[str, Any]]], games: int) -> dict[str, Any]:
    """What some answers are worth by a search's own worlds.

    ``picks`` holds, for each decision the search examined, the game, the
    alternative answered (0 the pilot's, None one the search did not try)
    and the decision as ``workers.watch_worker`` read it.  An answer other
    than the pilot's is worth the lead the search measured for it in its
    first worlds.  Where the pilot's answer won every one of them nothing
    can be ahead, and a search tries nothing unless its log is to be full:
    without a lead such answers are counted apart, as worth nothing, which
    flatters them.
    """
    leads: list[float] = []
    by_game = [0.0] * games
    outright_by_game = [0.0] * games
    counts: Counter[str] = Counter()
    for game, place, decision in picks:
        if place == 0:
            counts["pilot"] += 1
        elif place is not None and decision["leads"][place] is not None:
            lead = float(decision["leads"][place])
            leads.append(lead)
            by_game[game] += lead
            counts["ahead" if lead > 0 else "behind" if lead < 0 else "level"] += 1
            if decision["outright"]:
                counts["tried_outright"] += 1
                outright_by_game[game] += lead
        elif decision["outright"]:
            counts["outright"] += 1
        else:
            counts["untried"] += 1
    lead, error = _mean(leads)
    per_game, per_game_error = _mean(by_game)
    return {
        "decisions": len(picks),
        # The pilot's answer, or a like card from the same pile.
        "pilot": counts["pilot"],
        # Another answer where the pilot's won every first world, and the
        # search tried nothing: no lead.
        "outright": counts["outright"],
        # The same where it did try, and what those leads add up to a game:
        # both are part of the figures below.
        "tried_outright": counts["tried_outright"],
        "lead_outright_per_game": _mean(outright_by_game)[0],
        # Another answer the search did not try: past its twelfth.
        "untried": counts["untried"],
        # Another answer the search tried: how much more often it won.
        "tried": len(leads),
        "ahead": counts["ahead"],
        "level": counts["level"],
        "behind": counts["behind"],
        "lead": lead,
        "lead_error": error,
        # The leads of a game added up, over the games.
        "lead_per_game": per_game,
        "lead_per_game_error": per_game_error,
        "lead_by_game": by_game,
    }


def _led(
    decisions: list[tuple[int, dict[str, Any]]], games: int, margin: float
) -> dict[str, Any]:
    """The answers of a model that follows its lead head: its first choice,
    unless it predicts for another alternative a lead of at least ``margin``
    over that choice (see ``searched.Reading.led``).

    Beside what those answers are worth, the gain of the rule: what the
    answers it changes lead their first choices by, added up over a game.
    The changes are also given apart by where they start (`STARTS`): a
    model whose first choice is not its pilot's answer gains by going back
    to it, which is its copy repaired and nothing the search found.
    """
    picks = []
    gains = [0.0] * games
    changed = unmeasured = 0
    split: dict[str, dict[str, Any]] = {
        start: {
            "changed": 0,
            "unmeasured": 0,
            # Against the first choice it replaces, by the measured leads.
            "ahead": 0,
            "level": 0,
            "behind": 0,
            "gain_by_game": [0.0] * games,
        }
        for start in STARTS
    }
    for game, decision in decisions:
        predicted, first = decision["predicted"], decision["model"]
        answer = first
        if first is not None and len(predicted) > 1:
            best = max(
                (place for place in range(len(predicted)) if place != first),
                key=predicted.__getitem__,
            )
            if predicted[best] - predicted[first] >= margin:
                answer = best
                changed += 1
                part = split[
                    "from_pilot" if first == 0 else "to_pilot" if best == 0 else "between_others"
                ]
                part["changed"] += 1
                measured = (decision["leads"][best], decision["leads"][first])
                if None in measured:
                    unmeasured += 1
                    part["unmeasured"] += 1
                else:
                    difference = measured[0] - measured[1]
                    gains[game] += difference
                    part["gain_by_game"][game] += difference
                    part[
                        "ahead" if difference > 0 else "behind" if difference < 0 else "level"
                    ] += 1
        picks.append((game, answer, decision))
    gain, gain_error = _mean(gains)
    for part in split.values():
        part["gain_per_game"], part["gain_per_game_error"] = _mean(part.pop("gain_by_game"))
    return {
        "margin": margin,
        # First choices left for another answer, and those of them the
        # search has no lead for on one side or the other.
        "changed": changed,
        "unmeasured": unmeasured,
        "gain_per_game": gain,
        "gain_per_game_error": gain_error,
        "gain_by_game": gains,
        "split": split,
        **_departures(picks, games),
    }


def _head(decisions: list[tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    """What a lead head knows: over every alternative a search measured,
    the lead it predicts over the pilot's answer against the one measured."""
    pairs = sorted(
        (
            (decision["predicted"][place] - decision["predicted"][0], float(lead))
            for _, decision in decisions
            for place, lead in enumerate(decision["leads"])
            if place > 0 and lead is not None
        ),
        reverse=True,
    )
    if len(pairs) < 2:
        return None
    count = len(pairs)
    mean_predicted = sum(predicted for predicted, _ in pairs) / count
    mean_measured = sum(measured for _, measured in pairs) / count
    covariance = sum((p - mean_predicted) * (m - mean_measured) for p, m in pairs)
    spread = math.sqrt(
        sum((p - mean_predicted) ** 2 for p, _ in pairs)
        * sum((m - mean_measured) ** 2 for _, m in pairs)
    )
    highest = []
    for share in RANKED_HIGHEST:
        ranked = pairs[: max(1, round(share * count))]
        lead, error = _mean([measured for _, measured in ranked])
        highest.append(
            {
                "share": share,
                "alternatives": len(ranked),
                "predicted": sum(predicted for predicted, _ in ranked) / len(ranked),
                "lead": lead,
                "lead_error": error,
            }
        )
    return {
        "alternatives": count,
        "correlation": covariance / spread if spread else None,
        "predicted": mean_predicted,
        "lead": mean_measured,
        # The alternatives it predicts the largest leads for.
        "highest": highest,
    }


def _told(pairs: list[tuple[float, float, int]]) -> dict[str, Any] | None:
    """A value head against the worlds.

    ``pairs`` holds, for a decision, the head's value, the mean result of
    the pilot's answer over the worlds a search played it out in there (a
    win 1, a loss -1), and how many those worlds were.  Over so few worlds a
    rate is not the decision's worth: of the spread of the rates, the part
    that is `real` is what remains without the spread of a mean of so many
    results.  The head is read against that part, which the noise of the
    worlds has no share in since it knows nothing of them.
    """
    pairs = [pair for pair in pairs if pair[2] > 1]
    count = len(pairs)
    if count < 2:
        return None
    mean_value = sum(value for value, _, _ in pairs) / count
    mean_rate = sum(rate for _, rate, _ in pairs) / count
    spread_value = sum((value - mean_value) ** 2 for value, _, _ in pairs) / count
    spread = sum((rate - mean_rate) ** 2 for _, rate, _ in pairs) / count
    together = sum((value - mean_value) * (rate - mean_rate) for value, rate, _ in pairs) / count
    apart = (
        sum(((value - mean_value) - (rate - mean_rate)) ** 2 for value, rate, _ in pairs) / count
    )
    # A result is 1 or -1 around the decision's worth r: its spread there is
    # 1 - r^2, which the rate of w worlds underestimates by (w - 1) / w.
    noise = sum((1.0 - rate * rate) / (worlds - 1) for _, rate, worlds in pairs) / count
    within = sum((1.0 - rate * rate) * worlds / (worlds - 1) for _, rate, worlds in pairs) / count
    real = spread - noise
    return {
        "decisions": count,
        "correlation": (
            together / math.sqrt(spread_value * spread) if spread_value * spread > 0.0 else None
        ),
        "spread": spread,
        "real": real,
        # Of what is real, the share a straight line through the head's
        # values explains, and the share its values explain as they stand.
        "told": (
            together * together / (spread_value * real)
            if spread_value > 0.0 and real > 0.0
            else None
        ),
        "explained": 1.0 - (apart - noise) / real if real > 0.0 else None,
        # Of a single result, the share the decision's worth explains: what
        # a perfect predictor would.
        "result": real / (real + within) if real + within > 0.0 else None,
    }


def _value(decisions: list[tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    """What a value head tells of the decisions a search examined, where it
    is trained: at the decisions that open a transaction."""
    measured = [
        decision
        for _, decision in decisions
        if decision.get("value") is not None and decision.get("rate") is not None
    ]
    opening = [decision for decision in measured if decision.get("start")]

    def told(group: list[dict[str, Any]]) -> dict[str, Any] | None:
        return _told(
            [
                (float(decision["value"]), float(decision["rate"]), int(decision["worlds"]))
                for decision in group
            ]
        )

    summary = told(opening)
    if summary is None:
        return None
    summary["elsewhere"] = len(measured) - len(opening)
    summary["by_turn"] = [
        {"turns": [low, high], **part}
        for low, high in VALUE_TURNS
        if (
            part := told(
                [
                    decision
                    for decision in opening
                    if low <= int(decision["turn"])
                    and (high is None or int(decision["turn"]) <= high)
                ]
            )
        )
        is not None
    ]
    return summary


def summarize_tried(watched: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A model's answers by what a search measured for them.

    At every decision it examined the search played the pilot's answer and
    some alternatives out in the same worlds.  ``watched`` holds, per duel
    played again, the model's answer set among them.  The leads are those of
    the first stage, which every alternative goes through whatever comes of
    it.  A full log has them at every decision examined, and a model that
    has not seen the games is then not flattered: its sum estimates what its
    answers gain or lose against the pilot's.  Another log has them only
    where the pilot's answer lost some world of that stage, which flatters
    every alternative alike: read a model beside another.  The search's own
    choices were taken on those very worlds, and are flattered either way.

    Where the model's lead head was asked, the same for the answers it
    would give by following the head (`led`, at each of `MARGINS`), and what
    the head knows of the leads (`head`).  Where the log holds the worlds,
    what the model's value head tells of what the pilot's answer is worth
    in them (`value`).
    """
    decisions = [
        (place, decision)
        for place, game in enumerate(watched)
        for decision in game.get("tried", [])
    ]
    if not decisions:
        return None
    games = len(watched)

    def first(group: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
        return _departures([(game, d["model"], d) for game, d in group], games)

    def bar(ratio: float) -> dict[str, Any]:
        # The alternative the model gives most, taken once its weight is
        # `ratio` of the pilot's answer's.
        picks = []
        for game, decision in decisions:
            weights = decision["weights"]
            best = max(range(1, len(weights)), key=weights.__getitem__)
            picks.append((game, best if weights[best] >= ratio * weights[0] else 0, decision))
        reading = _departures(picks, games)
        del reading["lead_by_game"]
        return {"ratio": ratio, **reading}

    by_question: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for game, decision in decisions:
        by_question[decision["question"]].append((game, decision))
    others = [
        float(lead)
        for _, decision in decisions
        for lead in decision["leads"][1:]
        if lead is not None
    ]
    any_other, any_other_error = _mean(others)
    summary = first(decisions)
    summary.update(
        {
            "games": games,
            # Answers the search examined that the seat was not asked.
            "unread": sum(game.get("tried_unread", 0) for game in watched),
            # The search's own choices, by the same reading.
            "search": _departures([(game, d["taken"], d) for game, d in decisions], games),
            # Every alternative the search tried, as if answered at random.
            "any_other": {"tried": len(others), "lead": any_other, "lead_error": any_other_error},
            "by_question": {
                question: {
                    key: value for key, value in first(group).items() if key != "lead_by_game"
                }
                for question, group in sorted(by_question.items())
            },
            "bars": [bar(ratio) for ratio in BARS],
            # Decisions the log holds that the trainer does not take a search
            # to examine, or the other way round.
            "examined_otherwise": sum(game.get("examined_otherwise", 0) for game in watched),
        }
    )
    del summary["search"]["lead_by_game"]
    if all(decision.get("predicted") is not None for _, decision in decisions):
        summary["led"] = [_led(decisions, games, margin) for margin in MARGINS]
        summary["head"] = _head(decisions)
    value = _value(decisions)
    if value is not None:
        summary["value"] = value
    return summary


def summarize_answer_key(watched: list[dict[str, Any]]) -> dict[str, Any]:
    """How a model answers the decisions of recorded searched duels.

    ``watched`` holds one report per duel played again with the model
    watching.  The key is the search: at its deviations the answer wanted is
    the one it played, everywhere else the pilot's, which it kept.
    """
    examined_known = bool(watched) and all(game["examined_known"] for game in watched)
    total: Counter[str] = Counter()
    by_question: dict[str, Counter[str]] = defaultdict(Counter)
    by_change: dict[str, Counter[str]] = defaultdict(Counter)
    deviations: list[dict[str, Any]] = []
    for game in watched:
        for question, counts in game["questions"].items():
            by_question[question].update(counts)
            total.update(counts)
        for entry in game["deviations"]:
            deviations.append(entry)
            change = by_change[f"{entry['question']}: {entry['pilot']} -> {entry['search']}"]
            change["deviations"] += 1
            change[entry["reading"]] += 1
    return {
        "games": len(watched),
        "decisions": sum(game["decisions"] for game in watched),
        # Deviations the rows list; all of them should have been asked.
        "recorded_deviations": sum(game["recorded_deviations"] for game in watched),
        # Later steps of a selection the search answered another way.
        "unread": total["unread"],
        **_answer_key(total, examined_known),
        "mean_search_probability": (
            sum(entry["search_probability"] for entry in deviations) / len(deviations)
            if deviations
            else None
        ),
        "mean_pilot_probability": (
            sum(entry["pilot_probability"] for entry in deviations) / len(deviations)
            if deviations
            else None
        ),
        "by_question": {
            question: _answer_key(counts, examined_known)
            for question, counts in sorted(by_question.items())
        },
        # By what the pilot answered and what the search played instead.
        "by_change": {
            change: {
                "deviations": counts["deviations"],
                "search": counts["search"],
                "pilot": counts["pilot"],
                "other": counts["other"],
            }
            for change, counts in sorted(
                by_change.items(), key=lambda item: (-item[1]["deviations"], item[0])
            )
        },
        "deviation_list": deviations,
        # The model's answers by what the search measured, when it logged it.
        "tried": summarize_tried(watched),
    }


def summarize_deals(
    results: list[dict[str, Any]], references: dict[str, list[str]]
) -> dict[str, Any]:
    """A model's games on the deals of a recorded run, beside what others
    made of the same deals.

    ``references`` gives, for each named player, its outcome on every deal
    in the order of ``results``.
    """
    summary = summarize_results(results)
    outcomes = [str(result["outcome"]) for result in results]
    # Deal by deal, in the order of the run's games: two models measured on
    # the same run can be compared on each deal.
    summary["outcomes"] = outcomes
    summary["references"] = {}
    summary["paired"] = {}
    for name, theirs in references.items():
        if len(theirs) != len(outcomes):
            raise ValueError(f"{name} has {len(theirs)} outcomes for {len(outcomes)} deals")
        by_opponent: dict[str, list[str]] = defaultdict(list)
        for result, outcome in zip(results, theirs):
            by_opponent[str(result["opponent"])].append(outcome)

        def tally(played: list[str]) -> dict[str, Any]:
            wins = played.count("win")
            return {
                "games": len(played),
                "wins": wins,
                "losses": played.count("loss"),
                "draws": played.count("draw"),
                "win_rate": wins / max(len(played), 1),
            }

        summary["references"][name] = {
            **tally(theirs),
            "by_opponent": {opponent: tally(played) for opponent, played in by_opponent.items()},
            # Deal by deal too, in the order of `outcomes`: a model read over
            # several checkpoints is set beside a reference on each deal.
            "outcomes": list(theirs),
        }
        # Deal by deal: who of the two won it.
        pairs = Counter((mine == "win", other == "win") for mine, other in zip(outcomes, theirs))
        summary["paired"][name] = {
            "both": pairs[(True, True)],
            "model_only": pairs[(True, False)],
            f"{name}_only": pairs[(False, True)],
            "neither": pairs[(False, False)],
        }
    return summary
