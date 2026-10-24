"""`tools/update_signal.py`: the measures of an update's signal, on numbers
whose answers are known, and the readings' marks."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("update_signal", ROOT / "tools/update_signal.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def test_the_noise_scale_of_gradients_whose_signal_and_noise_are_known() -> None:
    # Groups of 10 duels: a signal of squared length 4 in common, and each
    # duel's own noise of variance 3 in each of 200 directions, so a group's
    # noise has squared length 200 * 3 / 10 = 60: the noise scale is
    # 10 * 60 / 4 = 150 duels.
    generator = np.random.default_rng(1)
    signal = np.zeros(200)
    signal[0] = 2.0
    groups = np.stack([signal + generator.normal(0.0, math.sqrt(3 / 10), 200) for _ in range(400)])
    scale, error = tool.left_out(groups @ groups.T, 10)
    assert scale == pytest.approx(150, rel=0.1)
    assert 0 < error < 30


def test_no_signal_is_a_noise_scale_beyond_measure() -> None:
    gram = np.array([[1.0, -0.1], [-0.1, 1.0]])
    assert tool.noise_scale(gram, 65) == math.inf


def test_the_deal_share_of_results_whose_parts_are_known() -> None:
    # A deal's part of variance 1 and a play's of variance 3: a quarter.
    generator = np.random.default_rng(2)
    deals = generator.normal(0.0, 1.0, (4000, 1))
    results = deals + generator.normal(0.0, math.sqrt(3), (4000, 16))
    assert tool.deal_share(results) == pytest.approx(0.25, abs=0.02)
    assert tool.deal_share(generator.normal(0.0, 1.0, (4000, 16))) == pytest.approx(0.0, abs=0.01)


def test_what_another_play_accounts_for_and_what_the_head_explains() -> None:
    generator = np.random.default_rng(3)
    deal = generator.normal(0.0, 1.0, 20000)
    advantage = deal + generator.normal(0.0, 1.0, 20000)
    other = deal + generator.normal(0.0, 1.0, 20000)
    # Each carries the deal at half its variance: a quarter of the one's
    # variance is the other's to account for.
    assert tool.accounted(advantage, other) == pytest.approx(0.25, abs=0.02)
    # A head that is the deal explains half of a result, one that is off by
    # as much again explains nothing.
    assert tool.explained(advantage, deal) == pytest.approx(0.5, abs=0.02)
    assert tool.explained(advantage, 2 * deal) == pytest.approx(0.0, abs=0.03)


def test_drawing_the_deals_again_errs_as_the_mean_does() -> None:
    values = np.random.default_rng(4).normal(0.0, 1.0, 2500)
    whole, error = tool.drawn_again(lambda chosen: float(values[chosen].mean()), len(values))
    assert whole == pytest.approx(float(values.mean()))
    assert error == pytest.approx(1 / 50, rel=0.15)


@pytest.mark.parametrize(
    ("scale", "error", "reading"),
    [
        (1040, 200, "limited by noise"),
        (5000, 2000, "limited by noise"),
        (1040, 300, "neither"),
        (1000, 10, "neither"),
        (260, 100, "not limited by noise"),
        (260, 140, "neither"),
        (300, 10, "neither"),
        (math.inf, math.inf, "neither"),
    ],
)
def test_the_first_reading(scale: float, error: float, reading: str) -> None:
    assert tool.noise_reading(scale, error) == reading


@pytest.mark.parametrize(
    ("share", "error", "reading"),
    [
        (0.31, 0.03, "pairing pays"),
        (0.31, 0.031, "not decided"),
        (0.06, 0.02, "pairing does not pay"),
        (0.06, 0.021, "not decided"),
        (0.18, 0.01, "not decided"),
    ],
)
def test_the_second_reading(share: float, error: float, reading: str) -> None:
    assert tool.pairing_reading(share, error) == reading


def known_groups(count: int, seed: int, signal_length: float = 2.0) -> np.ndarray:
    """Gradients of groups of 10 duels with a signal in common and each duel's
    noise of variance 3 in each of 200 directions (as above)."""
    generator = np.random.default_rng(seed)
    signal = np.zeros(200)
    signal[0] = signal_length
    return np.stack([signal + generator.normal(0.0, math.sqrt(3 / 10), 200) for _ in range(count)])


def in_blocks(groups: np.ndarray, blocks: int) -> dict[str, np.ndarray]:
    sums = np.stack([part.sum(axis=0) for part in np.split(groups, blocks)])
    squares = np.array([float((part * part).sum()) for part in np.split(groups, blocks)])
    return {
        "gram": sums @ sums.T,
        "squares": squares,
        "counts": np.full(blocks, len(groups) / blocks),
    }


def test_blocks_of_groups_tell_the_noise_scale_the_groups_tell() -> None:
    groups = known_groups(256, 5)
    kept = in_blocks(groups, 16)
    found = tool.by_blocks(kept["gram"], kept["squares"], kept["counts"], 10)
    assert found["noise_scale"] == pytest.approx(tool.noise_scale(groups @ groups.T, 10))
    assert found["noise_scale"] == pytest.approx(150, rel=0.15)
    assert found["signal_per_duel"] == pytest.approx(1 / 150, rel=0.15)
    assert 0 < found["signal_error"] < found["signal_per_duel"]
    assert found["shared_z"] > 2
    assert found["least"] < found["noise_scale"]


def test_the_gap_in_signal_of_two_measures_on_the_same_blocks() -> None:
    # Twice the signal's length is four times its squared length: a signal
    # per duel four times as large.
    weak, strong = known_groups(256, 6), known_groups(256, 7, signal_length=4.0)
    one, other = in_blocks(strong, 16), in_blocks(weak, 16)
    gap, error = tool.signal_gap(one, other, 10)
    assert gap == pytest.approx(3 / 150, rel=0.2)
    assert 0 < error < gap / 2


@pytest.mark.parametrize(
    ("new", "old", "error", "reading"),
    [
        (0.004, 0.001, 0.001, "more signal"),
        (0.004, 0.001, 0.0016, "not told apart"),
        (0.0004, 0.001, 0.0002, "less signal"),
        (0.0009, 0.001, 0.00001, "not told apart"),
        (0.0001, -0.0001, 0.00005, "more signal"),
    ],
)
def test_the_third_reading(new: float, old: float, error: float, reading: str) -> None:
    assert tool.signal_reading(new, old, error) == reading


@pytest.mark.parametrize(
    ("critic", "head", "error", "reading"),
    [
        (1.0e-4, 5.0e-5, 2.0e-5, "a stronger signal"),
        (1.0e-4, 5.0e-5, 2.6e-5, "not shown to be stronger"),
        (9.0e-5, 5.0e-5, 1.0e-6, "not shown to be stronger"),
        (2.0e-5, 5.0e-5, 1.0e-6, "not shown to be stronger"),
    ],
)
def test_the_critics_reading(critic: float, head: float, error: float, reading: str) -> None:
    assert tool.critic_reading(critic, head, error) == reading
