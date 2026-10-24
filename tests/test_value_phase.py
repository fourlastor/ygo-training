"""`tools/value_phase.py`: the first reading of a value head with more
results, on numbers whose answers are known, and the parts that teach it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("value_phase", ROOT / "tools/value_phase.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


@pytest.mark.parametrize(
    ("gap", "error", "reading"),
    [
        (0.05, 0.01, "it predicts better"),
        (0.03, 0.01, "not shown to predict better"),
        (0.10, 0.05, "not shown to predict better"),
        (-0.02, 0.01, "not shown to predict better"),
    ],
)
def test_the_first_reading(gap: float, error: float, reading: str) -> None:
    assert tool.reading(gap, error) == reading


def test_the_variants_lead_and_its_error_by_the_duels_drawn_again() -> None:
    # 400 duels of 20 decisions, a result each; the variant's estimate is
    # nearer the result than the control's.
    generator = np.random.default_rng(3)
    duels = np.repeat(np.arange(400), 20)
    results = np.repeat(generator.choice([-1.0, 1.0], 400), 20)
    control = 0.3 * results + generator.normal(0.0, 0.8, len(results))
    variant = 0.4 * results + generator.normal(0.0, 0.8, len(results))
    lead, error = tool.gap(duels, results, variant, control, draws=200)
    expected = tool.explained(results, variant) - tool.explained(results, control)
    assert lead == pytest.approx(expected)
    assert lead > 0.05
    assert 0.0 < error < lead


def test_a_copys_estimates_replace_the_ones_the_games_were_played_with() -> None:
    trajectories = [
        {"winner": 0, "steps": [{"row": 1, "value": 0.2}, {"row": 2, "value": 0.1}]},
        {"winner": 1, "steps": [{"row": 3, "value": -0.4}]},
    ]
    given = tool.valued(trajectories, [0.5, 0.6, 0.7])
    assert [step["value"] for trajectory in given for step in trajectory["steps"]] == [
        0.5,
        0.6,
        0.7,
    ]
    assert [step["row"] for trajectory in given for step in trajectory["steps"]] == [1, 2, 3]
    assert trajectories[0]["steps"][0]["value"] == 0.2
    assert given[1]["winner"] == 1


def test_a_transactions_first_sample() -> None:
    samples = [
        SimpleNamespace(step={"_transaction_key": key})
        for key in [(0, 0, 1), (0, 0, 1), (0, 1, 1), (1, 0, 1), (0, 0, 2), (0, 1, 1)]
    ]
    assert tool.roots(samples) == [samples[0], samples[2], samples[3], samples[4]]
