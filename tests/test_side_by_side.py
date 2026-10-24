"""What `tools/side_by_side.py` tells of runs made at once, with a run that
only writes its figures standing in for the trainer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("side_by_side", ROOT / "tools/side_by_side.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

# A run that writes an update's figures every tenth of a second, and a checkpoint.
RUN = """
import json, sys, time
from pathlib import Path
run, updates = Path(sys.argv[1]), int(sys.argv[2])
(run / "checkpoints").mkdir(parents=True)
(run / "checkpoints" / "model-00001.pt").write_text("weights")
for update in range(updates):
    time.sleep(0.1)
    with (run / "metrics.jsonl").open("a") as kept:
        kept.write(json.dumps({"wall_seconds": 20.0, "duels": 520, "samples": 40000, "seconds": 11.0,
                               "pool_threads_seconds": 6.0, "inference_seconds": 4.0,
                               "ppo_update_seconds": 7.0, "inference_batches": 1200.0 + 100 * update,
                               **({"pool_graph_batches": 900.0, "pool_graph_failed": 0.0}
                                  if "graphs" in sys.argv else {}),
                               "rollout_decision_fingerprint": f"games-{update}"}) + "\\n")
sys.exit(int(sys.argv[3]))
"""


def settings(**more) -> argparse.Namespace:
    told = {
        "checkpoint": Path("a/checkpoint.pt"), "device": "cpu", "card": 0, "updates": 3,
        "itself": 260, "held": None, "threads": 20, "slots": 16, "duels": 520, "keep": False,
        "groups": 1, "groups_over": 0, "graphs": 0,
    }  # fmt: skip
    return argparse.Namespace(**{**told, **more})


def run(update: float) -> dict:
    return {
        "update_seconds": update, "duels": 520.0, "games_seconds": 11.0, "engines_seconds": 6.0,
        "model_seconds": 4.0, "learning_seconds": 7.0, "decisions_a_second": 40000.0 / update,
    }  # fmt: skip


def test_the_load_is_the_runs_own_threads_and_the_cards_share(monkeypatch):
    monkeypatch.setattr(tool.os, "cpu_count", lambda: 20)
    # In two seconds the runs took six seconds of processor, and a quarter of
    # the machine's time was busy.
    found = tool.load(
        [(0.0, 100, 1000, 10.0, (10.0, 1024.0)), (2.0, 150, 1200, 16.0, (30.0, 3072.0))]
    )
    assert found == {
        "runs_busy": pytest.approx(3.0),
        "machine_busy": pytest.approx(5.0),
        "card_busy": 20.0,
        "card_gib": 3.0,
    }
    assert tool.load([(0.0, 100, 1000, 0.0, None), (2.0, 150, 1200, 1.0, None)])["card_busy"] is None
    assert tool.load([(0.0, 100, 1000, 0.0, None)])["runs_busy"] is None


def test_a_card_that_cannot_be_read_is_told_with_the_reason(monkeypatch, tmp_path):
    def answering(code: int, out: str, err: str = ""):
        def run(asked, **_):
            return argparse.Namespace(returncode=code, stdout=out, stderr=err)

        return run

    monkeypatch.setattr(tool.subprocess, "run", answering(0, "23, 2048\n"))
    assert tool.card(0) == (23.0, 2048.0)
    monkeypatch.setattr(tool.subprocess, "run", answering(6, "", "No devices were found\n"))
    assert tool.card(0) == "nvidia-smi ended with 6: No devices were found"
    monkeypatch.setattr(tool.subprocess, "run", answering(0, "[N/A], 2048\n"))
    assert tool.card(0) == "nvidia-smi answered '[N/A], 2048'"

    def missing(asked, **_):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(tool.subprocess, "run", missing)
    assert tool.card(0) == "nvidia-smi: no nvidia-smi"
    # The first reason is kept, and the samples go on without the card.
    watching = tool.Load(0, every=0.01)
    watching.start()
    time.sleep(0.1)
    watching.ended.set()
    watching.join()
    assert watching.unread == "nvidia-smi: no nvidia-smi"
    assert watching.samples and all(sample[4] is None for sample in watching.samples)
    idle = {"runs_busy": 3.0, "machine_busy": 3.4, "card_busy": None, "card_gib": None}
    told = tool.report(
        settings(),
        "a stamp",
        [{"count": 1, "runs": [run(20.0)], "failed": [], "load": idle, "games": [["a"]],
          "all_of_it_seconds": 70.0, "card_unread": watching.unread}],
    )  # fmt: skip
    assert "The card's load was not read: nvidia-smi: no nvidia-smi" in told


def test_time_other_guests_of_a_host_took_is_not_the_machines(monkeypatch, tmp_path):
    stat = tmp_path / "stat"
    stat.write_text("cpu  10 1 5 100 3 2 1 50 0 0\ncpu0 1 1 1 1 1 1 1 1 0 0\n")
    monkeypatch.setattr(tool, "Path", lambda *_: stat)
    assert tool.processor() == (19, 172)


def test_a_runs_processor_time_is_its_sessions(monkeypatch):
    # This process leads no session of the stand-in's, and has spent some time.
    started = time.process_time()
    while time.process_time() - started < 0.05:
        pass
    assert tool.spent({os.getsid(0)}) >= 0.05
    assert tool.spent({0x7FFFFFF0}) == 0.0


def test_the_table_sets_runs_at_once_beside_one_alone():
    idle = {"runs_busy": 3.0, "machine_busy": 3.4, "card_busy": 25.0, "card_gib": 3.0}
    told = tool.report(
        settings(),
        "a stamp",
        [
            {"count": 1, "runs": [run(20.0)], "failed": [], "load": idle,
             "games": [["a", "b"]], "all_of_it_seconds": 70.0},
            {"count": 2, "runs": [run(24.0), run(26.0)], "failed": [], "load": idle,
             "games": [["a", "b"], ["a", "b"]], "all_of_it_seconds": 80.4},
            {"count": 3, "runs": [run(30.0), None, run(30.0)], "failed": ["runs/3-2"], "load": idle,
             "games": [["a", "b"], [], ["a", "other"]], "all_of_it_seconds": 90.0},
        ],
    )  # fmt: skip
    assert "520 duels an update, 260 of them against itself" in told[1]
    alone, two, three = (line.split() for line in told[6:9])
    assert alone[:5] == ["1", "20.0", "1.00", "93,600", "2,000"]
    # Two at once take a quarter longer each, and play 1.6 times the duels together.
    assert two[:7] == ["2", "24.0", "to", "26.0", "1.25", "150,000", "3,205"]
    assert two[-4:] == ["3.0", "25%", "3.0", "GiB"]
    assert three[:3] == ["3", "not", "measured:"] and "runs/3-2.log" in told[8]
    assert told[10] == "NOT the games of run 1-1: 3-3 from its update 2"
    assert told[11].endswith("1 at once 70 s, 2 at once 80 s, 3 at once 90 s.")
    same = tool.same_games([{"count": 1, "games": [["a"]]}, {"count": 2, "games": [["a"], ["a"]]}])
    assert same == "Every run played the games of run 1-1, update for update."
    # How a run's duels are taken forward is in the header and in its command.
    assert "duels each in every run, 3 updates" in told[2]
    assert "in every run, in 2 groups, 3 updates" in tool.report(settings(groups=2), "", [])[2]
    split = settings(groups=2, groups_over=128, workers=4, learner="blackwing", opponents="burn", rate="1e-4")
    header = tool.report(split, "", [])[2]
    assert "in 2 groups while their rounds hold more than 128 decisions, 3 updates" in header
    asked = tool.command(split, Path("a/run"))
    assert asked[asked.index("--pool-groups") :][:4] == ["--pool-groups", "2", "--pool-groups-over", "128"]
    # One group is named too: what a run takes where nothing is named is
    # another way on a card.
    whole = tool.command(settings(**{**vars(split), "groups": 1, "groups_over": 0}), Path("a/run"))
    assert whole[whole.index("--pool-groups") :][:4] == ["--pool-groups", "1", "--pool-groups-over", "0"]


@pytest.mark.parametrize("keep", [False, True])
def test_runs_are_made_at_once_and_their_checkpoints_removed(tmp_path, monkeypatch, keep):
    monkeypatch.setattr(
        tool, "command", lambda args, run: [sys.executable, "-c", RUN, str(run), str(args.updates), "0"]
    )
    found = tool.together(settings(keep=keep), 2, tmp_path)
    assert found["count"] == 2 and found["failed"] == []
    assert [made["update_seconds"] for made in found["runs"]] == [20.0, 20.0]
    assert found["games"] == [["games-0", "games-1", "games-2"]] * 2
    assert 0.3 <= found["all_of_it_seconds"] < 30
    assert (tmp_path / "2-1.log").is_file()
    assert (tmp_path / "2-2/checkpoints").is_dir() == keep


def test_a_run_that_fails_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tool, "command", lambda args, run: [sys.executable, "-c", RUN, str(run), str(args.updates), "3"]
    )
    assert tool.together(settings(), 1, tmp_path)["failed"] == [str(tmp_path / "1-1")]


def test_two_reports_are_set_beside_each_other(tmp_path):
    def kept(name: str, updates: dict[int, list[float]], games: str = "a") -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                [
                    {"count": count, "runs": [run(update) for update in each], "failed": [],
                     "games": [[games, "b"]] * count}
                    for count, each in updates.items()
                ]
            ),
            encoding="utf8",
        )  # fmt: skip
        return path

    plain = kept("plain", {1: [20.0], 2: [36.0, 36.0]})
    told = tool.beside(plain, kept("mps", {1: [20.0], 2: [24.0, 24.0], 3: [30.0, 30.0, 30.0]}))
    assert told[1].split() == ["at", "once", "plain", "mps", "plain", "mps", "plain", "mps"]
    assert told[2].split() == ["1", "93,600", "93,600", "1.00", "1.00", "20.0", "20.0"]
    # Two at once: 1.11 times the work as processes have a card, 1.67 times through MPS.
    assert told[3].split() == ["2", "104,000", "156,000", "1.11", "1.67", "36.0", "24.0"]
    assert told[4].split() == ["3", "-", "187,200", "-", "2.00", "-", "30.0"]
    assert told[-1] == "The run alone played the same games both ways, update for update."
    other = tool.beside(plain, kept("mps", {1: [20.0]}, games="other"))
    assert other[-1] == "The run alone did NOT play the same games as mps as it did as plain."
    # Two reports of one name are told apart by their folders.
    (tmp_path / "two-groups").mkdir()
    grouped = plain.rename(tmp_path / "two-groups" / "plain.json")
    (tmp_path / "one-group").mkdir()
    whole = kept("plain", {1: [22.0]})
    whole = whole.rename(tmp_path / "one-group" / "plain.json")
    named = tool.beside(whole, grouped)
    assert named[1].split()[2:4] == ["one-group", "two-groups"]
    # A long name widens its columns, and the figures stay under their names.
    (tmp_path / "two-groups-over-128").mkdir()
    longer = grouped.rename(tmp_path / "two-groups-over-128" / "plain.json")
    named = tool.beside(whole, longer)
    assert named[1].split()[2:4] == ["one-group", "two-groups-over-128"]
    assert named[2].split() == ["1", "85,091", "93,600", "1.00", "1.00", "22.0", "20.0"]
    assert named[2].index("93,600") + len("93,600") == named[1].index("two-groups-over-128") + 19


def test_the_calls_of_a_run_and_those_answered_from_graphs_are_told(tmp_path, monkeypatch):
    command = tool.command
    monkeypatch.setattr(
        tool, "command",
        lambda args, run: [sys.executable, "-c", RUN, str(run), str(args.updates), "0", *(["graphs"] if args.graphs else [])],
    )  # fmt: skip
    plain = tool.together(settings(), 1, tmp_path / "plain")
    # The updates after the first: 1,300 and 1,400 calls.
    assert plain["calls"] == [{"an_update": 1350.0, "from_graphs": None, "graphs_failed": False}]
    graphed = tool.together(settings(graphs=128), 2, tmp_path / "graphs")
    assert [calls["from_graphs"] for calls in graphed["calls"]] == [pytest.approx(1800 / 2700)] * 2
    assert tool.calls_told([plain]) == ["1 at once: 1,350 calls of the model an update"]
    assert tool.calls_told([graphed]) == [
        "2 at once: 1,350 calls of the model an update, 67% of them answered from graphs"
    ]
    failed = {"count": 3, "calls": [
        {"an_update": 1200.0, "from_graphs": 0.5, "graphs_failed": False},
        {"an_update": 1400.0, "from_graphs": 0.25, "graphs_failed": True},
    ]}  # fmt: skip
    line = tool.calls_told([failed])[0]
    assert line.startswith("3 at once: 1,200 to 1,400 calls of the model an update, 25% to 50% of them")
    assert "THE GRAPHS FAILED in a run" in line
    # A run that told nothing of its calls leaves its count out; an old report has none.
    assert tool.calls_told([{"count": 1, "calls": [None]}, {"count": 2}]) == []
    # The header and a run's command say how the calls are answered.
    asked = settings(graphs=128, groups=2, groups_over=64, workers=4, learner="blackwing", opponents="burn", rate="1e-4")
    header = tool.report(asked, "", [])[2]
    assert (
        "in 2 groups while their rounds hold more than 64 decisions, calls of up to 128 decisions "
        "from CUDA graphs, 3 updates" in header
    )
    assert "in every run, calls of up to 128 decisions from CUDA graphs, 3 updates" in (
        tool.report(settings(graphs=128), "", [])[2]
    )
    told = command(asked, Path("a/run"))
    assert told[told.index("--inference-graphs") :][:2] == ["--inference-graphs", "128"]
    plain = command(settings(**{**vars(asked), "graphs": 0}), Path("a/run"))
    assert plain[plain.index("--inference-graphs") :][:2] == ["--inference-graphs", "0"]


def test_several_reports_are_set_beside_each_other(tmp_path):
    def kept(name: str, update: float, games: str = "a") -> Path:
        (tmp_path / name).mkdir()
        path = tmp_path / name / "plain.json"
        path.write_text(
            json.dumps([{"count": 1, "runs": [run(update)], "failed": [], "games": [[games, "b"]]}]),
            encoding="utf8",
        )
        return path

    three = [kept("one-group", 20.0), kept("graphs", 16.0), kept("graphs-two-groups", 15.0, games="other")]
    told = tool.beside(*three)
    assert told[1].split() == ["at", "once", *["one-group", "graphs", "graphs-two-groups"] * 3]
    assert told[2].split() == ["1", "93,600", "117,000", "124,800", "1.00", "1.00", "1.00", "20.0", "16.0", "15.0"]
    # The figures stay under their names.
    assert told[2].index("124,800") + len("124,800") == told[1].index("graphs-two-groups") + len("graphs-two-groups")
    assert told[-1] == "The run alone did NOT play the same games as graphs-two-groups as it did as one-group."
    assert tool.beside(*three[:2])[-1] == "The run alone played the same games both ways, update for update."
    same = tool.beside(three[0], three[1], three[1])
    assert same[-1] == "The run alone played the same games every way, update for update."
