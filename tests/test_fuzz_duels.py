"""tools/fuzz_duels.py on two decks: its counts, and the same duels again with
the same seed when each answer is drawn from its duel alone."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ygo_training.environment import library_path

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]


def _fuzzed(output: Path, *more: str) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "tools/fuzz_duels.py"), "--decks", "blackwing,monarch"]
        + ["--games", "3", "--seed", "7", "--threads", "2", "--slots", "4", "--output", str(output)]
        + list(more),
        check=True,
        capture_output=True,
    )
    return json.loads((output / "summary.json").read_text())


def test_each_pair_of_decks_is_played_and_each_duel_kept_alike(tmp_path):
    summary = _fuzzed(tmp_path / "first", "--per-duel")
    # Two decks make four pairs, three duels each.
    assert summary["counts"]["duels"] == 12
    assert summary["counts"].get("failed", 0) == 0
    assert summary["random_decisions"] > 0
    first = (tmp_path / "first" / "duels.jsonl").read_text().splitlines()
    assert len(first) == 12
    _fuzzed(tmp_path / "again", "--per-duel")
    again = (tmp_path / "again" / "duels.jsonl").read_text().splitlines()
    assert sorted(first) == sorted(again)
