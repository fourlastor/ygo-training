"""`tools/queue/luna_queue.py`: a task of luna's queue, as it is queued and
as the worker runs it, on a checkout of its own."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("luna_queue", ROOT / "tools/queue/luna_queue.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def git(where: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(where), *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def place(tmp_path, monkeypatch):
    """A checkout of two commits, and where logs go: as luna's, for a task."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    git(checkout, "config", "user.email", "queue@test")
    git(checkout, "config", "user.name", "queue")
    commits = []
    for told in ("first", "second"):
        (checkout / "told.txt").write_text(told)
        git(checkout, "add", "told.txt")
        git(checkout, "commit", "--quiet", "-m", told)
        commits.append(git(checkout, "rev-parse", "HEAD"))
    monkeypatch.setenv("YGO_QUEUE_CHECKOUT", str(checkout))
    monkeypatch.setenv("YGO_QUEUE_LOGS", str(tmp_path / "logs"))
    monkeypatch.setenv("RQ_JOB_ID", "try-1")
    return checkout, commits, tmp_path / "logs"


def task(commit: str, command: str, **more) -> dict:
    return tool.task_from(
        more.pop("name", "try"), commit, more.pop("hours", 1.0), "low", more.pop("env", []), command
    )


def test_a_task_runs_in_the_checkout_at_its_commit_with_its_variables(place):
    checkout, (first, second), logs = place
    done = tool.run(task(first, 'cat told.txt > seen.txt; echo "said $SAID"', env=["SAID=it"]))
    assert done == {"exit": 0, "log": str(logs / "try-1.log"), "commit": first}
    assert git(checkout, "rev-parse", "HEAD") == first
    assert (checkout / "seen.txt").read_text() == "first"
    log = (logs / "try-1.log").read_text()
    assert f"try at {first}, low priority, 1 hours" in log
    assert "SAID=it" in log and "said it" in log and log.rstrip().endswith("exited 0")
    # The next task moves the checkout to its own commit.
    tool.run(task(second, "cat told.txt > seen.txt"))
    assert (checkout / "seen.txt").read_text() == "second"


def test_a_task_that_fails_fails_its_job_and_says_where_its_log_is(place):
    _, (first, _), logs = place
    with pytest.raises(RuntimeError, match="exited 3: .*try-1.log"):
        tool.run(task(first, "echo before; exit 3"))
    assert "before" in (logs / "try-1.log").read_text()


def test_no_task_runs_over_changes_to_what_the_checkout_tracks(place):
    checkout, (first, second), _ = place
    (checkout / "told.txt").write_text("changed by hand")
    with pytest.raises(RuntimeError, match="changes of its own"):
        tool.run(task(first, "true"))
    assert git(checkout, "rev-parse", "HEAD") == second
    # What it does not track does not stop a task.
    git(checkout, "checkout", "--", "told.txt")
    (checkout / "untracked.txt").write_text("results")
    assert tool.run(task(first, "true"))["exit"] == 0


@pytest.mark.parametrize(
    ("hours", "priority", "env", "told"),
    [
        (4.0, "low", [], "3 hours at most: split it"),
        (0.0, "low", [], "how many hours"),
        (1.0, "urgent", [], "priority is one of"),
        (1.0, "low", ["lower=case"], "NAME=VALUE"),
        (1.0, "low", ["NOSIGN"], "NAME=VALUE"),
    ],
)
def test_a_task_that_cannot_be_queued_says_why(hours, priority, env, told):
    with pytest.raises(SystemExit, match=told):
        tool.task_from("try", "abc", hours, priority, env, "true")


def test_a_task_of_high_priority_may_be_long_and_its_timeout_follows_its_hours():
    long = tool.task_from("long-run", "abc", 8.0, "high", ["UPDATES=240"], "bash run.sh")
    assert long["env"] == {"UPDATES": "240"} and long["priority"] == "high"
    assert tool.timeout_of(long) == int(8 * 3600 * 1.5 + 1800)
    with pytest.raises(SystemExit, match="lowercase words"):
        tool.task_from("Long Run", "abc", 1.0, "high", [], "true")


def test_a_stopped_task_takes_what_its_command_started_with_it(tmp_path):
    """As in RQ's work horse, the leader of a process group of its own: the
    command's own children are stopped with it, and the horse goes on."""
    started = tmp_path / "started.pid"
    child = os.fork()
    if child == 0:
        try:
            os.setpgrp()
            process = subprocess.Popen(["bash", "-c", f"sleep 60 & echo $! > {started}; sleep 60"])
            while not started.exists() or not started.read_text().strip():
                time.sleep(0.05)
            tool.stopped(process)
            os._exit(0 if process.returncode is not None else 2)
        except BaseException:  # noqa: BLE001 - a forked child never returns into the tests
            os._exit(3)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    pid = int(started.read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        raise AssertionError("the command's own child went on")


def test_a_tasks_command_is_what_comes_after_the_double_dash(monkeypatch):
    seen = []
    monkeypatch.setattr(tool, "add", seen.append)
    tool.main(
        ["add", "try", "--commit", "abc", "--hours", "1", "--env", "A=b", "--"]
        + ["bash", "-c", "echo --hours 2; exit 0"]
    )
    (args,) = seen
    assert (args.name, args.commit, args.hours, args.env) == ("try", "abc", 1.0, ["A=b"])
    assert args.command == ["bash", "-c", "echo --hours 2; exit 0"]
    with pytest.raises(SystemExit):
        tool.main(["add", "try", "--commit", "abc", "--hours", "1"])
