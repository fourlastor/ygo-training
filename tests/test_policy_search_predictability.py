"""What can be told of a result at the decisions a search examined."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "policy_search_predictability", ROOT / "tools/policy_search_predictability.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def rows(path: Path, games: int, seed: int) -> Path:
    """A logged run in which a decision is worth what the seat's life points
    are ahead by, and nothing else tells."""
    generator = random.Random(seed)
    lines = []
    for game in range(games):
        examined = []
        for decision in range(12):
            ahead = generator.choice((-4000, -2000, 0, 2000, 4000))
            odds = 0.5 + ahead / 10000
            marks = "".join("+" if generator.random() < odds else "-" for _ in range(8))
            examined.append(
                {
                    "decision": decision,
                    "kind": "Idle" if decision % 4 else "Position",
                    "turn": 1 + decision,
                    "played": [marks, ""],
                    "situation": {
                        "phase": "Main1",
                        "own_turn": True,
                        "life": [4000 + ahead // 2, 4000 - ahead // 2],
                        "monsters": [1, 1],
                        "backrow": [0, 0],
                        "set_backrow": [0, 0],
                        "hand": [generator.randrange(6), 3],
                        "attack": False,
                    },
                }
            )
        seat = game % 2
        lines.append(
            json.dumps(
                {
                    "a": "blackwing",
                    "b": "monarch" if game % 3 else "burn",
                    "seat": seat,
                    "seed": game,
                    "search": {
                        "winner": seat if generator.random() < 0.6 else 1 - seat,
                        "examined": examined,
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return path


def test_the_log_gives_each_opening_decision_the_rate_of_its_first_worlds(tmp_path):
    games = tool.load(rows(tmp_path / "rows.jsonl", 6, 1), ("Idle", "Battle", "Chain"))

    assert len(games) == 6
    # Nine of a game's twelve decisions open a transaction.
    assert [len(game["decisions"]) for game in games] == [9] * 6
    first = games[0]["decisions"][0]
    assert first["worlds"] == 8 and -1.0 <= first["rate"] <= 1.0
    assert {game["result"] for game in games} <= {1.0, -1.0}
    with pytest.raises(SystemExit, match="no examined decision"):
        tool.load(tmp_path / "rows.jsonl", ("Counter",))


def test_a_small_network_tells_what_the_situation_holds(tmp_path, capsys, monkeypatch):
    train = rows(tmp_path / "train.jsonl", 500, 2)
    measure = rows(tmp_path / "measure.jsonl", 300, 3)
    output = tmp_path / "references.json"
    monkeypatch.setattr(
        sys, "argv", ["predictability", str(train), str(measure), "--output", str(output)]
    )

    tool.main()

    printed = capsys.readouterr().out
    assert "300 games of measure.jsonl, 2700 examined decisions of Idle, Battle, Chain" in printed
    report = json.loads(output.read_text())
    # A decision is worth -0.8 to 0.8 in five steps: a variance of 0.32, and
    # as much of one result as that.
    assert report["worlds"]["real"] == pytest.approx(0.32, abs=0.03)
    assert report["worlds"]["result"] == pytest.approx(0.32, abs=0.03)
    by_lesson = {(net["target"], net["opponent"]): net for net in report["networks"]}
    assert set(by_lesson) == {("result", False), ("rate", False), ("result", True), ("rate", True)}
    # Taught the worlds' rates it has what the life points tell, which is
    # all there is; the games' results here are drawn apart from them.
    assert by_lesson["rate", False]["told"]["told"] > 0.9
    assert by_lesson["rate", False]["of_one_result"] == pytest.approx(0.32, abs=0.04)
