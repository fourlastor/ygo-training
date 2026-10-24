"""What the benchmark's report says of the folder its exercise leaves."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_report", ROOT / "tools/benchmark_report.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

MACHINE = {
    "processor": "A processor",
    "threads": 20,
    "memory_gib": 62.4,
    "system": "Linux 6",
    "python": "3.12.3",
    "torch": "2.13.0",
    "card": "A card",
    "card_memory_gib": 12.0,
    "cuda": "12.8",
}


def update(wall: float, games: float, learning: float, decisions: int, **more) -> dict:
    return {
        "duels": 520.0,
        "wall_seconds": wall,
        "seconds": games,
        "ppo_update_seconds": learning,
        "inference_seconds": 2.0,
        "samples": float(decisions),
        **more,
    }


def run(folder: Path, name: str, rows: list[dict]) -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def exercise(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(
        json.dumps(
            {
                "stamp": "20261011-190000",
                "device": "cuda",
                "threads": 20,
                "slots": 16,
                "workers": 24,
                "duels": 520,
                "warm": 40,
                "timed": 2,
                "cargo": "cargo 1.94.1",
                "commit": "0123456789abcdef",
            }
        )
    )
    # The first update warms up: twice as long, and not counted.
    run(
        folder,
        "imitation",
        [
            update(20.0, 4.0, 10.0, 40_000),
            update(7.0, 1.8, 5.0, 41_000, pool_threads_seconds=1.4),
            update(7.4, 1.8, 5.2, 42_000, pool_threads_seconds=1.6),
        ],
    )
    # A run made before its figures told the engines' part of its games.
    run(folder, "ppo-anchor", [update(40.0, 30.0, 6.0, 44_000), update(30.0, 21.0, 5.0, 45_000)])
    run(
        folder,
        "ppo-checkpoint",
        [
            update(30.0, 15.0, 7.0, 50_000, held_decisions=48_000.0),
            # An evaluation is not part of an update.
            update(
                122.0,
                13.0,
                6.0,
                52_000,
                held_decisions=50_000.0,
                evaluation_seconds=100.0,
                pool_threads_seconds=3.5,
            ),
        ],
    )
    # A run that stopped in its first update has nothing timed.
    run(folder, "ppo-itself", [update(30.0, 20.0, 6.0, 90_000)])
    (folder / "suite.json").write_text(
        json.dumps(
            {
                "seconds": 31.2,
                "records": [
                    {
                        "evaluation_suite": {
                            "a": {"games": 128, "wins": 60},
                            "b": {"games": 128, "wins": 70},
                        }
                    }
                ],
            }
        )
    )
    (folder / "mirror.json").write_text(
        json.dumps(
            {"seconds": 5.0, "records": [{"evaluation_suite": {"a": {"error": "it failed"}}}]}
        )
    )
    (folder / "setup.tsv").write_text("submodules\t1\ntoolchain\t40\nbuild\t200\npython\t119\n")
    (folder / "exercise.tsv").write_text("imitation\t300\nall of it\t1500\n")
    return folder


def test_a_mode_is_the_mean_of_its_updates_after_the_first(tmp_path):
    folder = exercise(tmp_path)
    found = tool.figures(folder / "imitation" / "metrics.jsonl")
    assert found["updates"] == 2
    assert found["update_seconds"] == pytest.approx(7.2)
    assert found["duels_an_hour"] == pytest.approx(3600 * 520 / 7.2)
    assert found["games_seconds"] == pytest.approx(1.8)
    assert found["engines_seconds"] == pytest.approx(1.5)
    assert found["learning_seconds"] == pytest.approx(5.1)
    assert found["decisions"] == pytest.approx(41_500)
    assert found["decisions_a_second"] == pytest.approx(41_500 / 7.2)
    assert found["held_decisions"] is None
    # What a run does not tell of every timed update is not told of the mode.
    untold = tool.figures(folder / "ppo-anchor" / "metrics.jsonl")
    assert untold["engines_seconds"] is None and untold["update_seconds"] == pytest.approx(30.0)
    held = tool.figures(folder / "ppo-checkpoint" / "metrics.jsonl")
    assert held["update_seconds"] == pytest.approx(22.0)
    assert held["held_decisions"] == pytest.approx(50_000)
    # Nothing timed, or no run at all.
    assert tool.figures(folder / "ppo-itself" / "metrics.jsonl") is None
    assert tool.figures(folder / "ppo" / "metrics.jsonl") is None


def test_the_report_tells_every_mode_and_names_those_not_measured(tmp_path, capsys):
    found = tool.report(exercise(tmp_path), MACHINE)
    assert list(found["modes"]) == [name for name, _, _ in tool.MODES]
    assert found["modes"]["imitation"]["duels_an_hour"] == pytest.approx(260_000)
    assert found["modes"]["ppo"] is None and found["modes"]["ppo-itself"] is None
    assert found["measured"]["suite"] == {
        "games": 256,
        "seconds": 31.2,
        "games_an_hour": pytest.approx(3600 * 256 / 31.2),
    }
    # A suite with a case that failed is not a measure.
    assert found["measured"]["mirror"] is None
    assert found["setup"] == {"submodules": 1, "toolchain": 40, "build": 200, "python": 119}
    assert found["exercise"]["all of it"] == 1500
    told = capsys.readouterr().out
    assert "A processor, 20 threads, 62 GiB" in told and "A card, 12 GiB, CUDA 12.8" in told
    assert "520 duels an update, a pool of 20 threads with 16 duels each, 24 workers" in told
    assert "on from a copy of 40 imitating updates, 2 updates timed in each mode" in told
    line = next(row for row in told.splitlines() if row.strip().startswith("the pilot "))
    assert line.split()[2:] == ["7.2", "260,000", "1.8", "1.5", "2.0", "5.1", "41,500", "5,764"]
    held = next(row for row in told.splitlines() if "against a held checkpoint" in row)
    assert "22.0" in held and "85,091" in held
    assert "every duel against itself" in told and "not measured" in told
    assert "the suite against the pilots: 256 games in 31 s, 29,538 an hour" in told
    assert "the mirror against a held checkpoint: not measured" in told
    assert (
        "Not measured: " in told and "every duel against itself" in told.split("Not measured: ")[1]
    )
    assert "All of it 25 min 00 s, of which the setup 6 min 00 s (submodules 1 s, " in told


def test_a_container_is_told_by_the_threads_it_is_allowed(tmp_path, capsys):
    # A rented machine saw 128 threads and was held to 13.6 of them.
    tool.report(exercise(tmp_path), {**MACHINE, "threads": 128, "allowed": 13.6})
    assert "A processor, 128 threads (13.6 of them allowed at once), 62 GiB" in capsys.readouterr().out


def test_the_exercise_makes_the_runs_the_report_reads():
    """Every mode the report tells is a part of the script's exercise under
    that name, and every run the exercise trains is a mode of the report."""
    script = (ROOT / "tools/benchmark.sh").read_text(encoding="utf8")
    trained = re.findall(r"^part (\S+) train (\S+)", script, flags=re.MULTILINE)
    assert all(name == run_ for name, run_ in trained)
    assert [name for name, _ in trained] == [name for name, _, _ in tool.MODES]
    assert re.findall(r"^part (\S+) measure (\S+)", script, flags=re.MULTILINE) == [
        (name, name) for name, _ in tool.MEASURED
    ]


def test_a_folder_that_is_not_the_benchmarks_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark_report.py", str(tmp_path)])
    with pytest.raises(SystemExit, match="is not a folder of tools/benchmark.sh"):
        tool.main()
