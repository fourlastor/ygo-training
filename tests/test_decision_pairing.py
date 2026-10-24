"""`tools/decision_pairing.py`: the measures of pairing at the decision, on
numbers whose answers are known, and the readings' marks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "decision_pairing", ROOT / "tools/decision_pairing.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def known(decisions: int = 4000, pairs: int = 4, seed: int = 1):
    """Two answers' results whose parts are known: what each is worth, a
    future both plays of a pair share (variance 3) and each play's own
    (variance 1); and the head's difference, what the answers are worth
    give or take its own error (variance 0.5)."""
    generator = np.random.default_rng(seed)
    worth_a = generator.normal(0.0, 1.0, (decisions, 1))
    worth_b = generator.normal(0.0, 1.0, (decisions, 1))
    shared = generator.normal(0.0, np.sqrt(3.0), (decisions, pairs))
    a = worth_a + shared + generator.normal(0.0, 1.0, (decisions, pairs))
    b = worth_b + shared + generator.normal(0.0, 1.0, (decisions, pairs))
    head = worth_a - worth_b + generator.normal(0.0, np.sqrt(0.5), (decisions, pairs))
    return a, b, head


def test_what_a_shared_future_cancels() -> None:
    # Apart: 2 * (3 + 1) = 8; paired, only the plays' own: 1 + 1 = 2.
    a, b, _ = known()
    assert tool.cancelled(a, b) == pytest.approx(0.75, abs=0.01)
    assert tool.play_noise(a, b) == pytest.approx(2.0, rel=0.03)


def test_how_far_the_head_strays_with_the_pairs_noise_allowed_for() -> None:
    a, b, head = known()
    assert tool.head_error(a, b, head) == pytest.approx(0.5, abs=0.06)
    # The head strays by 0.5, a pair by 2: the play does not judge better.
    assert tool.judged(a, b, head) == pytest.approx(-1.5, abs=0.08)


def test_how_closely_the_heads_difference_follows_the_answers_worth() -> None:
    # What the answers are worth apart spreads by a variance of 1 + 1 = 2;
    # the head's difference is that and its error of 0.5: 2 / sqrt(2 * 2.5).
    a, b, head = known()
    assert tool.worth_spread(a, b) == pytest.approx(2.0, rel=0.06)
    assert tool.head_correlation(a, b, head) == pytest.approx(2 / np.sqrt(5), abs=0.03)


def test_futures_drawn_apart_cancel_nothing() -> None:
    generator = np.random.default_rng(2)
    a = generator.normal(0.0, 1.0, (4000, 4))
    b = generator.normal(0.0, 1.0, (4000, 4))
    assert tool.cancelled(a, b) == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize(
    ("share", "error", "reading"),
    [
        (0.60, 0.04, "pairing pays"),
        (0.55, 0.04, "not told apart"),
        (0.20, 0.02, "pairing does not pay"),
        (0.22, 0.02, "not told apart"),
        (0.40, 0.05, "not told apart"),
    ],
)
def test_the_first_reading(share: float, error: float, reading: str) -> None:
    assert tool.pairing_reading(share, error) == reading


@pytest.mark.parametrize(
    ("gap", "error", "reading"),
    [
        (0.10, 0.04, "a paired play judges better"),
        (0.05, 0.04, "not shown"),
        (-0.3, 0.05, "not shown"),
    ],
)
def test_the_second_reading(gap: float, error: float, reading: str) -> None:
    assert tool.judging_reading(gap, error) == reading


@pytest.mark.parametrize(
    ("over", "seat", "result"),
    [
        ({"winner": 0, "limit": False}, 0, 1.0),
        ({"winner": 0, "limit": False}, 1, -1.0),
        ({"winner": None, "limit": False}, 0, 0.0),
        ({"winner": 1, "limit": True}, 1, 0.0),
    ],
)
def test_a_plays_result_for_the_seat_that_decided(over: dict, seat: int, result: float) -> None:
    assert tool.result_for(over, seat) == result


def test_the_report_leaves_out_decisions_a_play_did_not_come_to() -> None:
    a, b, head = known(decisions=200)
    decisions = []
    for number in range(200):
        results, after = [], []
        for pair in range(4):
            results += [a[number, pair], b[number, pair]]
            after += [head[number, pair], 0.0]
        decisions.append(
            {
                "came": [number % 50 != 0] + [True] * 7,
                "results": results,
                "after": after,
                "later": [40] * 8,
                "decisions": 100,
                "chance_a": 0.5 if number % 2 else 0.995,
                "against": "itself" if number % 3 else "the pilots",
            }
        )
    result = tool.report({"pairs": 4, "decisions": decisions}, draws=50)
    assert result["left_out"] == 4
    assert result["all"]["decisions"] == 196
    assert result["all"]["share_later"] == pytest.approx(0.4)
    assert result["reading_1"] in ("pairing pays", "not told apart")
    shown = tool.shown(result)
    assert "1. A shared future cancels the luck: " in shown
    assert "chance 0.99-1" in shown and "against the pilots" in shown
