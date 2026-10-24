"""Training on the recorded duels of a search run and measuring on them,
through the command line, on real duels."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.recorded import search_row
from ygo_training.environment import LIBRARY, Environment

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "native/ygo-policies/target/release"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ROOT / LIBRARY).exists(), reason="the environment is not built: cargo build --release"
    ),
]

SMALL = ["--root", ".", "--device", "cpu", "--workers", "2", "--pool-threads", "2"]


def command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ygo_training", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def search_run(path: Path, first_seed: int) -> Path:
    """Four deals as a search run records them: two opponents, both seats."""
    with Environment(ROOT) as environment:
        deals = [
            search_row(environment, seed=first_seed + seed, opponent=opponent, learner_player=seat)
            for opponent, seed, seat in (
                ("monarch", 0, 0),
                ("monarch", 1, 1),
                ("heroes", 14, 0),
                ("heroes", 15, 1),
            )
        ]
    path.write_text("".join(json.dumps(deal) + "\n" for deal in deals), encoding="utf8")
    return path


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> Path:
    """The run the models are trained on."""
    return search_run(tmp_path_factory.mktemp("search") / "rows.jsonl", 7)


@pytest.fixture(scope="module")
def heldout(tmp_path_factory) -> Path:
    """Another run, on other deals, to measure them on."""
    return search_run(tmp_path_factory.mktemp("search") / "heldout.jsonl", 107)


def train(run: Path, rows: Path, *more: str) -> list[dict]:
    result = command(
        "train",
        *SMALL,
        "--run",
        str(run),
        "--duels",
        "4",
        "--updates",
        "2",
        "--ppo-epochs",
        "1",
        "--minibatch",
        "64",
        "--seed",
        "45",
        "--eval-every",
        "0",
        "--imitation-only-updates",
        "2",
        "--imitation-value-weight",
        "0.5",
        "--imitation-records",
        str(rows),
        *more,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


def test_both_arms_train_on_the_same_deals_and_are_measured_on_a_run(tmp_path, rows, heldout):
    searched = train(tmp_path / "search", rows)
    piloted = train(tmp_path / "pilot", rows, "--imitation-game", "baseline")

    # Two updates of four duels are two passes over the four games.
    assert [record["recorded_pass"] for record in searched] == [0.0, 1.0]
    assert [record["duels"] for record in searched] == [4.0, 4.0]
    # Every searched game leaves its pilot once; the pilot's own never.
    assert [record["recorded_departures"] for record in searched] == [4.0, 4.0]
    assert [record["imitation_departures"] for record in searched] == [4.0, 4.0]
    assert [record["recorded_departures"] for record in piloted] == [0.0, 0.0]
    assert all("imitation_model_departure_agreement" in record for record in searched)
    assert all("imitation_model_departure_agreement" not in record for record in piloted)
    assert all(record["decisions"] == 0.0 and record["samples"] > 0 for record in searched)
    for arm in (searched, piloted):
        assert arm[0]["opponent_deck_duels"] == {"heroes": 2.0, "monarch": 2.0}
    protocol = json.loads((tmp_path / "search" / "run.json").read_text())
    assert set(protocol["duel_revisions"]) >= {"commit", "clean"}
    described = protocol["imitation_records"]
    assert (described["game"], described["games"]) == ("search", 4)
    other = json.loads((tmp_path / "pilot" / "run.json").read_text())["imitation_records"]
    assert other["game"] == "baseline" and other["fingerprint"] != described["fingerprint"]

    checkpoints = [
        str(tmp_path / arm / "checkpoints" / "model-00001.pt") for arm in ("search", "pilot")
    ]
    # The games a run imitated tell nothing of games it has not seen.
    refused = command("evaluate-search", *SMALL, *checkpoints, "--records", str(rows))
    assert refused.returncode != 0
    assert "4 of the run's 4 deals are among the games" in refused.stderr

    output = tmp_path / "measured.json"
    result = command(
        "evaluate-search", *SMALL, *checkpoints, "--records", str(heldout), "--output", str(output)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "on the run's deals" in result.stdout
    measured = json.loads(output.read_text())
    assert (measured["learner"], measured["games"]) == ("blackwing", 4)
    assert measured["fingerprint"] != described["fingerprint"]
    assert [item["seen_deals"] for item in measured["checkpoints"]] == [0, 0]
    assert measured["duel_revisions"] == protocol["duel_revisions"]
    assert [item["checkpoint_label"] for item in measured["checkpoints"]] == [
        "search/model-00001",
        "pilot/model-00001",
    ]
    for item in measured["checkpoints"]:
        deals, key = item["deals"], item["answer_key"]
        assert deals["games"] == 4 and set(deals["references"]) == {"pilot", "search"}
        assert sum(deals["paired"]["search"].values()) == 4
        # Every deviation the rows list was asked of the model.
        assert key["recorded_deviations"] == key["deviations"] == 4
        assert key["search"] + key["pilot"] + key["other"] == 4
        assert key["kept"] + key["deviations"] + key["unread"] == key["decisions"]
        assert len(key["deviation_list"]) == 4 and "examined_kept" not in key


@pytest.mark.skipif(
    not ((BENCH / "policy-bench").exists() and (BENCH / "libygo_policies.so").exists()),
    reason="policy-bench is not built: cargo build --release in native/ygo-policies",
)
def test_on_a_logged_run_a_models_answers_are_read_by_the_searchs_worlds(tmp_path, rows):
    train(tmp_path / "search", rows)
    logged = tmp_path / "logged.jsonl"
    subprocess.run(
        [
            str(BENCH / "policy-bench"),
            "search",
            *("--policies", "blackwing", "--opponents", "monarch", "--games", "2"),
            *("--seed", "4242", "--workers", "2"),
            *("--worlds", "2", "--confirm", "4", "--final", "8"),
            *("--log", "true", "--record", "true", "--output", str(logged)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=300,
    )

    output = tmp_path / "measured.json"
    checkpoint = str(tmp_path / "search" / "checkpoints" / "model-00001.pt")
    result = command(
        "evaluate-search", *SMALL, checkpoint, "--records", str(logged), "--output", str(output)
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "first choice" in result.stdout and "any other answer the search tried" in result.stdout
    (item,) = json.loads(output.read_text())["checkpoints"]
    key = item["answer_key"]
    tried = key["tried"]
    # Every answer the search examined is read: those it kept and those it left.
    assert tried["games"] == 2 and tried["unread"] == 0
    assert tried["decisions"] == key["examined_kept"] + key["deviations"] > 0
    read = tried["pilot"] + tried["outright"] + tried["untried"] + tried["tried"]
    assert read == tried["decisions"]
    assert tried["ahead"] + tried["level"] + tried["behind"] == tried["tried"]
    assert len(tried["lead_by_game"]) == 2
    assert tried["search"]["decisions"] - tried["search"]["pilot"] == key["deviations"]
    assert [bar["ratio"] for bar in tried["bars"]] == [1.0, 0.5, 0.2, 0.1, 0.05]
