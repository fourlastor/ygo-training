"""`tools/pairing_weights.py`: the weights of a learner's duels against
itself, by how far behind each pairing of its decks is against `all71`."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pairing_weights", ROOT / "tools/pairing_weights.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DECKS = ["blackwing", "cyber", "lightsworn"]


def measured(tmp_path: Path, shares: dict[tuple[str, str], int], name: str = "measured") -> Path:
    """A measurement of every ordered pairing, 100 games each, the learner
    winning `shares` of them (50 where unsaid)."""
    cases = [
        {"name": f"{one}-vs-{other}", "learner": one, "opponent": other, "games": 100}
        for one in DECKS
        for other in DECKS
    ]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"name": "pairings", "cases": cases}))
    played = {
        case["name"]: {
            "outcomes": ["win"] * shares.get((case["learner"], case["opponent"]), 50)
            + ["loss"] * (100 - shares.get((case["learner"], case["opponent"]), 50))
        }
        for case in cases
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"suite": str(suite), "records": [{"evaluation_suite": played}]}))
    return path


def test_a_pairing_behind_draws_more_of_the_duels_and_none_goes_unplayed(tmp_path):
    path = measured(tmp_path, {("blackwing", "cyber"): 30, ("cyber", "blackwing"): 50})
    data = json.loads(path.read_text())
    suite = json.loads(Path(data["suite"]).read_text())
    shares = tool.pairing_shares(tool.ordered_shares(data, suite), DECKS)
    assert shares[("blackwing", "cyber")] == pytest.approx(0.4)
    assert shares[("blackwing", "blackwing")] == pytest.approx(0.5)
    weights = tool.weights_of(shares, DECKS, power=2.0, floor=0.25)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["blackwing>cyber"] == weights["cyber>blackwing"]
    # (1 - 0.4)^2 = 0.36 against (1 - 0.5)^2 = 0.25 for the others' seatings.
    assert weights["blackwing>cyber"] / weights["cyber>lightsworn"] == pytest.approx(0.36 / 0.25)
    assert weights["blackwing>blackwing"] == pytest.approx(weights["cyber>lightsworn"])
    # A pairing far ahead keeps a quarter of its share at every pair as often.
    far = tool.weights_of({**shares, ("cyber", "cyber"): 0.99}, DECKS, power=2.0, floor=0.25)
    assert far["cyber>cyber"] == pytest.approx(0.25 / 9)
    assert sum(far.values()) == pytest.approx(1.0)


def test_the_tool_writes_weights_train_reads_and_smooths_with_the_measurement_before(tmp_path):
    before = measured(tmp_path, {("lightsworn", "lightsworn"): 20}, name="before")
    now = measured(tmp_path, {("lightsworn", "lightsworn"): 40}, name="now")
    run = [sys.executable, str(ROOT / "tools/pairing_weights.py"), "--decks", ",".join(DECKS)]
    subprocess.run(
        [
            *run,
            str(before),
            "--output",
            str(tmp_path / "w0.json"),
            "--shares",
            str(tmp_path / "s0.json"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            *run,
            str(now),
            "--output",
            str(tmp_path / "w1.json"),
            "--shares",
            str(tmp_path / "s1.json"),
        ]
        + ["--before", str(tmp_path / "s0.json")],
        check=True,
        capture_output=True,
    )
    told = json.loads((tmp_path / "s1.json").read_text())
    assert told["measured"]["lightsworn|lightsworn"] == pytest.approx(0.4)
    assert told["shares"]["lightsworn|lightsworn"] == pytest.approx(0.3)
    weights = json.loads((tmp_path / "w1.json").read_text())
    assert set(weights) == {f"{one}>{other}" for one in DECKS for other in DECKS}

    from ygo_training.config import TrainerConfig

    config = TrainerConfig(
        duels=8,
        self_play_duels=4,
        pool_threads=1,
        opponent_decks=["burn"],
        learner_decks=DECKS,
        self_play_weights=weights,
    )
    assert config.pair_weights[("lightsworn", "lightsworn")] == max(config.pair_weights.values())
