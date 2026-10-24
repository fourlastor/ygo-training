"""Declarative held-out evaluation suites and checkpoint selection helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Who plays the opponent's deck in a case: its pilot, or a checkpoint held
# as it is, which the measurement names.
OPPONENT_POLICIES = ("ygo-policies", "checkpoint")


def against_checkpoint(case: dict[str, Any]) -> bool:
    """Whether a case's opponent is the checkpoint a measurement holds."""
    return str(case.get("opponent_policy", "ygo-policies")) == "checkpoint"


def load_evaluation_suite(path: Path) -> dict[str, Any]:
    return checked_evaluation_suite(json.loads(path.read_text(encoding="utf8")))


def joined_evaluation_suites(suites: list[dict[str, Any]]) -> dict[str, Any]:
    """Several suites as one, for a learner that plays the decks of them all:
    the cases of each on the deals it has them on when it is played alone
    (`seed_index`), and the first one's primary case."""
    return checked_evaluation_suite(
        {
            "name": "+".join(str(suite.get("name")) for suite in suites),
            "primary": suites[0]["primary"],
            "cases": [
                {"seed_index": index, **case}
                for suite in suites
                for index, case in enumerate(suite["cases"])
            ],
        }
    )


def checked_evaluation_suite(suite: dict[str, Any]) -> dict[str, Any]:
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation suite needs a non-empty cases list")
    names: set[str] = set()
    for case in cases:
        name = str(case.get("name", "")).strip()
        if not name or name in names:
            raise ValueError("Evaluation case names must be non-empty and unique")
        names.add(name)
        for field in ("learner", "opponent"):
            if not str(case.get(field, "")).strip():
                raise ValueError(f"Evaluation case {name!r} needs {field}")
        # An opponent is the pilot of its own deck, which older suites say,
        # or the checkpoint a measurement holds for the other seat.
        policy = str(case.get("opponent_policy", "ygo-policies"))
        if policy not in OPPONENT_POLICIES:
            raise ValueError(f"Evaluation case {name!r} has unsupported policy {policy!r}")
        mode = str(case.get("deck_mode", "declared"))
        if mode != "declared":
            raise ValueError(f"Evaluation case {name!r} has unsupported deck mode {mode!r}")
        games = int(case.get("games", 0))
        if games < 2 or games % 2:
            raise ValueError(f"Evaluation case {name!r} needs an even game count of at least 2")
        # The run of seeds a case's games are dealt from: its place among the
        # cases unless it names another.
        if int(case.get("seed_index", 0)) < 0:
            raise ValueError(f"Evaluation case {name!r} has a negative seed_index")
    primary = str(suite.get("primary", cases[0]["name"]))
    if primary not in names:
        raise ValueError(f"Unknown primary evaluation case {primary!r}")
    return {**suite, "primary": primary}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
