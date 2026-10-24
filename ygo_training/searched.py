"""What a search's log says of the decisions of a duel it played.

``policy-bench search --record true`` writes every deal it plays as a record
the environment plays again (``records.py``); with ``--log`` it also writes,
for every answer it examined, what it tried and how each alternative did.
Those are read against the decisions as the seat is shown them: the
alternatives are listed again from a decision as the bench lists them (the
pilot's answer first, like cards in one pile as one, twelve at most), and a
log that names others is refused, so that no lead is read for an answer it
was not measured for.  A lead head is taught and asked at the decisions a
search examines, told from the decision alone (:meth:`Reading.examined`).
"""

from __future__ import annotations

import json
import math
from typing import Any

# How `policy-bench search` lists the alternatives of a decision: like cards
# in one of these piles are one alternative, and it tries twelve at most, the
# pilot's answer among them.
PILES = ("Hand", "Deck", "Graveyard", "Banished", "Extra")
# The questions a search examines, and the positions of a monster it cannot
# deal again: see `Reading.examined`.
EXAMINED = ("Idle", "Battle", "Chain", "YesNo", "Position")
FACE_DOWN = ("FaceDownAttack", "FaceDownDefense")
TRIED = 12

# What the model's first choice is where a recorded duel left its pilot:
# the recorded answer, the pilot's, or neither.
LEFT_READINGS = ("search", "pilot", "other")
# And where the recorded answer is the pilot's own.
KEPT_READINGS = ("pilot", "other")


def probabilities(logits: list[float]) -> list[float]:
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def recorded_setup(game: dict[str, Any], names: list[str], limit: int) -> dict[str, Any]:
    """A recorded duel played again: both seats answer from the record, and
    the learner's is watched with the pilot of its deck asked alongside."""
    seats: list[dict[str, Any]] = [
        {"control": "record", "alone": True},
        {"control": "record", "alone": True},
    ]
    seats[int(game["learner_player"])] = {"control": "record", "observe": True}
    return {
        "seed": int(game["seed"]),
        "decks": names,
        "seats": seats,
        "limit": limit,
        "record": json.loads(game["record"]),
    }


def check_replay(game: dict[str, Any], over: dict[str, Any]) -> None:
    """A record counts only when it plays again as the duel it records.

    With another engine, other scripts, another card database or other
    pilots its answers may all still be accepted while the questions, and so
    what the network is shown, are no longer the ones that were answered.
    """
    if over["digest"] != game["digest"]:
        raise ValueError(
            f"the record plays again with digest {over['digest']}, not {game['digest']}: "
            "it was made with another engine, other scripts, cards or pilots"
        )
    if int(over["departures"]) != len(game["deviations"]):
        raise ValueError(
            f"the record leaves its pilot {over['departures']} times, "
            f"not the {len(game['deviations'])} it lists"
        )


class Reading:
    """The names behind the numbers of a decision, and a search's way of
    reading one: from the environment's vocabulary."""

    def __init__(self, vocabulary: dict[str, dict[str, int]]) -> None:
        self.decision_names = {number: name for name, number in vocabulary["decision"].items()}
        self.choice_names = {number: name for name, number in vocabulary["choice"].items()}
        self.piles = {vocabulary["location"][name] for name in PILES}
        self.monster_zone = vocabulary["location"]["MonsterZone"]
        self.face_down = {vocabulary["position"][name] for name in FACE_DOWN}

    def kinds(self, view: dict[str, Any], answer: int) -> tuple[str, str]:
        """The names of the question and of one of its answers, for reports."""
        return (
            self.decision_names[int(view["question"]["kind"])],
            self.choice_names[int(view["choices"][answer]["kind"])],
        )

    def examined(self, view: dict[str, Any], first: int | None = None) -> bool:
        """Whether a strict one-step search examines a decision.

        It does where the question is a command, a chain window that may be
        passed, a yes/no or a battle position, with more than one
        alternative to it, while no chain is under way and the other player
        has no face-down monster, which its worlds could not deal again.
        These are the decisions it measures leads at, so the only ones a
        lead head is taught and the only ones it is asked.
        """
        question = view["question"]
        kind = self.decision_names[int(question["kind"])]
        if kind not in EXAMINED or (kind == "Chain" and question["forced"]) or view["chain"]:
            return False
        if first is None and view["pilot"] is None:
            return False
        facing = any(
            card["controller"] == 1
            and card["location"] == self.monster_zone
            and card["position"] in self.face_down
            for card in view["cards"]
        )
        return not facing and len(self.alternatives(view, first)) > 1

    def alternatives(self, view: dict[str, Any], first: int | None = None) -> list[list[int]]:
        """The answers of a decision as a search tries them: the pilot's
        first, then the others in the order they are offered, twelve at most.

        Each alternative is the choices that play as one: the same thing done
        with a like card from the same pile.  The search plays the first of
        them.  Choices past the twelfth alternative are in none.  With
        ``first``, that answer stands where the pilot's would: a model has
        no pilot's answer to start from when it plays.
        """
        choices = view["choices"]

        def like(index: int) -> tuple[Any, ...] | None:
            choice = choices[index]
            card = choice["card"]
            if card is None or card["location"] not in self.piles:
                return None
            return (
                choice["kind"],
                card["code"],
                card["controller"],
                card["location"],
                choice["description"],
            )

        pilot = int(view["pilot"]) if first is None else first
        listed = [[pilot]]
        for index in range(len(choices)):
            if index == pilot:
                continue
            key = like(index)
            same = next(
                (group for group in listed if key is not None and like(group[0]) == key), None
            )
            if same is not None:
                same.append(index)
            elif len(listed) < TRIED:
                listed.append([index])
        return listed

    def alternative(self, view: dict[str, Any], answer: int) -> list[Any]:
        """An answer as a search's log names it: see `records.alternative`."""
        choice = view["choices"][answer]
        card = choice["card"]
        return [self.choice_names[int(choice["kind"])], int(card["code"]) if card else 0]

    def led(self, view: dict[str, Any], first: int, leads: list[float], margin: float) -> int:
        """The answer of a player that follows its lead head: its policy's
        ``first`` choice, unless at a decision a search examines it predicts
        for another alternative a lead of at least ``margin`` over that
        choice; then the alternative it predicts the largest lead for.
        """
        if not self.examined(view, first):
            return first
        best = max(
            (group[0] for group in self.alternatives(view, first)[1:]), key=leads.__getitem__
        )
        return best if leads[best] - leads[first] >= margin else first

    def listed(self, sample: dict[str, Any], entry: dict[str, Any]) -> list[list[int]]:
        """The alternatives a search tried at a decision, as the choices the
        network is offered there.

        They are listed again from the decision as the network sees it, and
        must be the ones the log names: with another engine, other scripts or
        pilots, or another way of listing them, a lead in the log would be
        read for an answer it was not measured for.
        """
        listed = self.alternatives(sample)
        names = [self.alternative(sample, group[0]) for group in listed]
        if names != entry["labels"]:
            raise ValueError(
                f"at answer {entry['answer']} the search's log tries {entry['labels']}, "
                f"where the decision offers {names}"
            )
        return listed

    def targets(self, sample: dict[str, Any], entry: dict[str, Any]) -> list[list[Any]]:
        """The lead the search measured for each answer it tried at a
        decision, the pilot's own first, with 0: a lead head's targets."""
        return [
            [group[0], float(lead)]
            for group, lead in zip(self.listed(sample, entry), entry["leads"], strict=True)
            if lead is not None
        ]

    def tried(
        self,
        sample: dict[str, Any],
        entry: dict[str, Any],
        model: int,
        logits: list[float],
        question: str,
        predicted: list[float] | None = None,
        *,
        value: float | None = None,
        start: bool = False,
    ) -> dict[str, Any]:
        """A model's answer among the alternatives a search tried at a decision."""
        listed = self.listed(sample, entry)
        weights = probabilities(logits)
        return {
            "answer": int(entry["answer"]),
            "question": question,
            "turn": int(sample["turn"]),
            # The model's value of the decision, whether the decision opens a
            # transaction, where the value head is trained, and what the
            # pilot's answer is worth by the worlds the search played it out in.
            "value": value,
            "start": start,
            "rate": entry.get("rate"),
            "worlds": entry.get("worlds"),
            "outright": bool(entry["outright"]),
            "taken": int(entry["taken"]),
            "leads": entry["leads"],
            # The alternative the model's first choice plays as; None past the twelfth.
            "model": next((place for place, group in enumerate(listed) if model in group), None),
            "weights": [sum(weights[index] for index in group) for group in listed],
            # The lead the model's head predicts for each alternative.
            "predicted": None if predicted is None else [predicted[group[0]] for group in listed],
        }
