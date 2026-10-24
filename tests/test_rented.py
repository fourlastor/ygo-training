"""`tools/rented.sh` against a machine that is this one: an `ssh` that skips
the address and runs the command here stands in for the connection, so every
command of the script is run as it is, with its quoting, and nothing leaves
the machine."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SSH = """#!/usr/bin/env bash
# ssh's options and the address are passed over, and the command is run here
# as the machine's shell would run it.
while (($#)); do
  case "$1" in
    -o | -p | -l) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
shift
exec bash -c "$*"
"""


@pytest.fixture
def rented(tmp_path):
    """A small repository that holds the script, and a place that is the
    machine's. Returns what runs the script, and the two folders."""
    stand_in = tmp_path / "bin"
    stand_in.mkdir()
    (stand_in / "ssh").write_text(SSH, encoding="utf8")
    (stand_in / "ssh").chmod(0o755)
    here = tmp_path / "here"
    (here / "tools").mkdir(parents=True)
    shutil.copy(ROOT / "tools/rented.sh", here / "tools/rented.sh")
    (here / "word.txt").write_text("first\n", encoding="utf8")
    there = tmp_path / "a place/ygo"
    environment = {
        **os.environ,
        "PATH": f"{stand_in}{os.pathsep}{os.environ['PATH']}",
        "RENTED": "someone@nowhere:2222",
        "AT": str(there),
        "KNOWN": str(tmp_path / "hosts"),
        "GIT_AUTHOR_NAME": "A test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "A test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    environment.pop("INTO", None)

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(here), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()

    def script(*arguments: str, **more: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "tools/rented.sh", *arguments],
            cwd=here,
            check=False,
            capture_output=True,
            text=True,
            env={**environment, **more},
            timeout=60,
        )

    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-q", "-m", "First")
    return script, git, here, there


def test_the_checkout_is_sent_and_sent_again(rented):
    script, git, here, there = rented
    sent = script("send")
    assert sent.returncode == 0, sent.stderr
    assert f"there: {git('rev-parse', '--short', 'HEAD')}" in sent.stdout
    assert (there / "word.txt").read_text(encoding="utf8") == "first\n"
    # What is not committed stays here, and is told.
    (here / "word.txt").write_text("second\n", encoding="utf8")
    again = script("send")
    assert "1 changed files here are not committed" in again.stderr
    assert (there / "word.txt").read_text(encoding="utf8") == "first\n"
    git("commit", "-q", "-am", "Second")
    assert script("send").returncode == 0
    assert (there / "word.txt").read_text(encoding="utf8") == "second\n"


def test_a_command_is_left_running_watched_and_waited_for(rented):
    script, _, _, there = rented
    assert script("send").returncode == 0
    started = script("start", "job", "bash", "-c", "echo 'a word; and $HOME'; sleep 1; exit 3")
    assert started.returncode == 0, started.stderr
    # A name that runs is not started twice.
    assert "job is still running there" in script("start", "job", "true").stderr
    assert "-- job is running" in script("watch", "job").stdout
    assert script("wait", "job").returncode == 3
    watched = script("watch", "job", "5").stdout
    assert "a word; and $HOME" in watched and "-- job ended with status 3" in watched
    assert (there / "runs/rented/job.log").is_file()
    # Ended, the name can be started again.
    assert script("start", "job", "true").returncode == 0
    assert script("wait", "job").returncode == 0


def test_what_was_written_there_is_brought_back(rented):
    script, _, here, there = rented
    assert script("send").returncode == 0
    made = script("run", "bash", "-c", "mkdir -p out/deep && echo found > 'out/deep/a file.txt' && pwd")
    assert made.returncode == 0 and made.stdout.strip() == str(there)
    assert script("fetch", "out/").returncode == 0
    assert (here / "out/deep/a file.txt").read_text(encoding="utf8") == "found\n"
    assert script("fetch", "out/deep/a file.txt", INTO="kept apart").returncode == 0
    assert (here / "kept apart/out/deep/a file.txt").is_file()
    assert script("fetch", "../word.txt").returncode == 2


def test_what_git_does_not_hold_is_put_there(rented):
    script, _, here, there = rented
    assert script("send").returncode == 0
    (here / "runs/kept").mkdir(parents=True)
    (here / "runs/kept/a checkpoint.pt").write_bytes(b"\x00weights\n")
    put = script("put", "runs/kept/a checkpoint.pt")
    assert put.returncode == 0, put.stderr
    assert put.stdout.strip() == f"there: {there}/runs/kept/a checkpoint.pt"
    assert (there / "runs/kept/a checkpoint.pt").read_bytes() == b"\x00weights\n"
    # A folder is merged with what is there.
    (there / "runs/kept/theirs.txt").write_text("stays\n", encoding="utf8")
    (here / "runs/kept/ours.txt").write_text("comes\n", encoding="utf8")
    assert script("put", "runs/").returncode == 0
    assert (there / "runs/kept/theirs.txt").is_file() and (there / "runs/kept/ours.txt").is_file()
    assert "Nothing at runs/nothing.pt here" in script("put", "runs/nothing.pt").stderr
    assert script("put", "../word.txt").returncode == 2
    assert script("put").returncode == 2


def test_the_machine_is_told_and_a_wrong_call_is_refused(rented):
    script, _, _, _ = rented
    told = script("machine")
    assert told.returncode == 0, told.stderr
    assert "threads: " in told.stdout and " allowed at once" in told.stdout
    assert "tools: " in told.stdout and " git" in told.stdout and told.stderr == ""
    assert script("start", "two words", "true").returncode == 2
    assert script("nothing").returncode == 2
    assert script("send", RENTED="").returncode == 2
    assert script("send", AT="not/from/the/root").returncode == 2


def test_a_commands_state_is_one_word(rented):
    import signal
    import time

    script, _, _, there = rented
    assert script("send").returncode == 0
    assert script("state", "job").stdout.strip() == "none"
    assert script("start", "job", "bash", "-c", "sleep 1; exit 4").returncode == 0
    assert script("state", "job").stdout.strip() == "running"
    assert script("wait", "job").returncode == 4
    assert script("state", "job").stdout.strip() == "ended 4"
    # Stopped from outside, it leaves no status.
    before = (there / "runs/rented/job.pid").read_text()
    assert script("start", "job", "sleep", "30").returncode == 0
    for _ in range(50):
        if (there / "runs/rented/job.pid").read_text() not in ("", before):
            break
        time.sleep(0.1)
    os.killpg(int((there / "runs/rented/job.pid").read_text()), signal.SIGKILL)
    assert script("state", "job").stdout.strip() == "gone"
    assert script("state", "two words").returncode == 2


def test_a_run_is_followed_and_what_it_names_is_here_as_it_is_there(rented):
    """While a command runs there the files a second command names are
    brought here, and once more when it has ended, whatever came of it; what
    that command does not name stays there."""
    script, _, here, there = rented
    assert script("send").returncode == 0
    job = (
        "mkdir -p 'out/a run'; for n in 1 2 3; do echo $n > 'out/a run'/kept-$n.txt; "
        "echo $n > out/passing.tmp; sleep 1; done; echo last > 'out/a run/kept-4.txt'; exit 5"
    )
    assert script("start", "job", "bash", "-c", job).returncode == 0
    followed = script("follow", "job", "find", "out", "-name", "kept-*", EVERY="1", AGAIN="0")
    assert followed.returncode == 0, followed.stderr
    assert "job: ended 5" in followed.stdout
    assert "Every file named there is here as it is there: 4 files under ." in followed.stdout
    for number in (1, 2, 3):
        assert (here / f"out/a run/kept-{number}.txt").read_text(encoding="utf8") == f"{number}\n"
    assert (here / "out/a run/kept-4.txt").read_text(encoding="utf8") == "last\n"
    assert (there / "out/passing.tmp").is_file() and not (here / "out/passing.tmp").exists()
    assert script("state", "job").stdout.strip() == "ended 5"
    # Again, with nothing new to bring, and to another place.
    again = script("follow", "job", "find", "out", "-name", "kept-*", EVERY="1", INTO="kept apart")
    assert again.returncode == 0 and (here / "kept apart/out/a run/kept-4.txt").is_file()
    # A file named that cannot be brought: not all is here, and it says so.
    missing = script("follow", "job", "echo", "out/nothing.txt", EVERY="1", AGAIN="0")
    assert missing.returncode == 3
    assert "is NOT all here: the machine still holds it" in missing.stderr
    assert script("follow", "job").returncode == 2
    assert script("follow", "job", "true", EVERY="soon").returncode == 2


def test_files_are_brought_through_a_folder_that_is_a_link_there(rented, tmp_path):
    """What a command writes may be kept on another disk there, its folder in
    the checkout a link to it: the files are brought, and the folder here
    stays a folder."""
    script, _, here, there = rented
    assert script("send").returncode == 0
    kept = tmp_path / "a volume" / "kept"
    kept.mkdir(parents=True)
    (there / "out").symlink_to(kept, target_is_directory=True)
    (here / "out").mkdir()
    (here / "out/ours.txt").write_text("stays\n", encoding="utf8")
    job = "mkdir -p out/deep; echo one > out/deep/first.txt; sleep 1; echo two > 'out/deep/the second.txt'"
    assert script("start", "job", "bash", "-c", job).returncode == 0
    followed = script("follow", "job", "find", "-L", "out", "-type", "f", EVERY="1", AGAIN="0")
    assert followed.returncode == 0, followed.stderr
    assert "Every file named there is here as it is there: 2 files" in followed.stdout
    assert (kept / "deep/first.txt").is_file()
    assert not (here / "out").is_symlink() and (here / "out/ours.txt").is_file()
    assert (here / "out/deep/first.txt").read_text(encoding="utf8") == "one\n"
    assert (here / "out/deep/the second.txt").read_text(encoding="utf8") == "two\n"
