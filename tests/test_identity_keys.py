"""`tools/identity_keys.py`: the identity rows of unseen cards rewritten, and read."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("identity_keys", ROOT / "tools/identity_keys.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def tables(rows: int = 64) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {name: torch.randn((rows, 4), generator=generator) for name in tool.TABLES}


def trained(start: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rows 0, 5 and 7 of the low table moved, row 3 of the high one."""
    state = {name: value.clone() for name, value in start.items()}
    low, high = tool.TABLES
    state[low][[0, 5, 7]] += 1.0
    state[high][3] += 1.0
    return state


@pytest.mark.parametrize("keys", ["blank", "mean", "redraw"])
def test_only_the_rows_no_update_moved_are_rewritten(keys):
    start = tables()
    state = trained(start)
    made, counts = tool.rewritten(state, keys, seed=1, start=start)
    low, high = tool.TABLES
    assert counts == {low: 64 - 3, high: 64 - 2}
    for name, kept in ((low, [0, 5, 7]), (high, [0, 3])):
        assert torch.equal(made[name][kept], state[name][kept])
        others = [row for row in range(64) if row not in kept]
        assert not torch.equal(made[name][others], state[name][others])
    if keys == "blank":
        assert torch.count_nonzero(made[low][[1, 2, 3]]) == 0
    if keys == "mean":
        assert torch.allclose(made[low][1], state[low][[5, 7]].mean(dim=0))
    if keys == "redraw":
        again, _ = tool.rewritten(state, keys, seed=1, start=start)
        assert torch.equal(made[low], again[low])


def test_a_network_whose_rows_mostly_moved_is_refused():
    start = tables()
    state = {name: value + 1.0 for name, value in start.items()}
    with pytest.raises(SystemExit):
        tool.rewritten(state, "blank", start=start)


def played(path: Path, wins: list[int], other: str = "old") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"other": other, "deal": deal, "seat": 0, "over": {"winner": 0 if won else 1}})
        for deal, won in enumerate(wins)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def test_an_arm_is_read_against_the_mean_of_the_random_keys(tmp_path):
    games = 400
    played(tmp_path / "as-is/new.jsonl", [1] * 200 + [0] * 200)
    for side in ("plays", "faces"):
        played(tmp_path / f"{side}/redraw-1/new.jsonl", [1] * 200 + [0] * 200)
        played(tmp_path / f"{side}/mean/new.jsonl", [1] * 200 + [0] * 200)
    played(tmp_path / "plays/blank/new.jsonl", [1] * 280 + [0] * 120)
    # Facing the decks, the arm's share is the other seat's: a network as it is that
    # wins less there is an arm that wins more.
    played(tmp_path / "faces/blank/new.jsonl", [1] * 120 + [0] * 280)
    found = tool.read(tmp_path)
    assert found["plays"]["blank"]["gap"] == pytest.approx(80 / games)
    assert found["plays"]["blank"]["verdict"] == "help"
    assert found["faces"]["blank"]["gap"] == pytest.approx(80 / games)
    assert found["plays"]["mean"]["verdict"] == "not decided"
    assert found["plays"]["draws"] == 2
    assert "1. Blank keys help, playing unseen cards" in tool.shown(found)
