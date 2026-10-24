"""`experiments/commands/more-decks-pod.sh` against a pod that is this
machine: an `ssh` that runs the command here stands in for the connection, a
`runpodctl` that writes down what it is asked for the one that stops and
deletes pods, and a script that writes a run's files for the arms.  Nothing
leaves the machine, and no pod is touched."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_rented import SSH

ROOT = Path(__file__).resolve().parents[1]

RUNPODCTL = """#!/usr/bin/env bash
# What it is asked is written down. It names a pod until it has deleted it,
# tells it as stopped once it has stopped it, and does not do what the test
# has it refuse.
echo "$*" >> "$ASKED"
case " ${REFUSED:-} " in
  *" $2 "*) echo '{"error": "refused"}'; exit 1 ;;
esac
case "$2" in
  get)
    if [ -e "$ASKED.deleted" ] && [ -n "${LINGERS:-}" ]; then
      printf '{"id": "%s", "desiredStatus": "EXITED", "runtimeStatus": "terminated"}\\n' "$3"
    elif [ -e "$ASKED.deleted" ]; then
      printf '{"error": "pod %s not found"}\\n' "$3"
      exit 1
    elif [ -e "$ASKED.stopped" ]; then
      printf '{"id": "%s", "desiredStatus": "EXITED", "runtimeStatus": "stopped"}\\n' "$3"
    else
      printf '{"id": "%s", "desiredStatus": "RUNNING", "runtimeStatus": "running"}\\n' "$3"
    fi
    ;;
  stop) touch "$ASKED.stopped" ;;
  delete) touch "$ASKED.deleted" ;;
esac
"""

# rsync itself, but for when the test has it fail.
RSYNC = """#!/usr/bin/env bash
[ -z "${BROKEN:-}" ] || exit 12
exec {rsync} "$@"
"""

# The arms: a run that passes through checkpoints and keeps one in two.
ARMS = """#!/usr/bin/env bash
run=$RUNS/decks-01
mkdir -p "$run/checkpoints"
for update in 0 1 2 3; do
  echo "weights $update" > "$run/checkpoints/model-0000$update.pt"
  echo "row $update" >> "$run/metrics.jsonl"
  sleep 0.4
done
echo "the arms say: at once $AT_ONCE, sizes ${SIZES:-all}"
exit "${ENDS:-0}"
"""


@pytest.fixture
def pod(tmp_path):
    """A small repository with the scripts, a place that is the pod's, and
    what runs the driver there: returns that, the two folders and the file
    of what runpodctl was asked."""
    if shutil.which("rsync") is None:
        pytest.skip("no rsync on this machine")
    stand_ins = tmp_path / "bin"
    stand_ins.mkdir()
    for name, text in (
        ("ssh", SSH),
        ("runpodctl", RUNPODCTL),
        ("rsync", RSYNC.replace("{rsync}", shutil.which("rsync"))),
    ):
        (stand_ins / name).write_text(text, encoding="utf8")
        (stand_ins / name).chmod(0o755)
    here = tmp_path / "here"
    for folder in ("tools", "experiments/commands"):
        (here / folder).mkdir(parents=True)
    shutil.copy(ROOT / "tools/rented.sh", here / "tools/rented.sh")
    for script in ("more-decks-pod.sh", "more-decks.sh"):
        shutil.copy(ROOT / "experiments/commands" / script, here / "experiments/commands" / script)
    (here / "experiments/commands/arms.sh").write_text(ARMS, encoding="utf8")
    (here / ".gitignore").write_text("runs/\n", encoding="utf8")
    there = tmp_path / "the pod/ygo"
    (tmp_path / "the volume").mkdir()
    asked = tmp_path / "asked.txt"
    # No runpodctl but the stand-in: the real one would ask Runpod.
    path = [str(stand_ins)] + [
        folder
        for folder in os.environ["PATH"].split(os.pathsep)
        if not (Path(folder) / "runpodctl").exists()
    ]
    environment = {
        **os.environ,
        "PATH": os.pathsep.join(path),
        "RENTED": "root@nowhere:2222",
        "POD": "pod123",
        "AT": str(there),
        "VOLUME": str(tmp_path / "the volume"),
        "KNOWN": str(tmp_path / "hosts"),
        "ASKED": str(asked),
        "COMMAND": "bash experiments/commands/arms.sh",
        "ON_THE_PROCESSOR": "1",
        "EVERY": "1",
        "AGAIN": "0",
        "STEP": "2",
        "GIT_AUTHOR_NAME": "A test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "A test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    for name in (
        "INTO", "LEAVE", "RUNS", "NAME", "AT_ONCE", "SIZES", "BROKEN", "REFUSED", "ENDS", "LINGERS"
    ):  # fmt: skip
        environment.pop(name, None)
    for arguments in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "First"]):
        subprocess.run(["git", "-C", str(here), *arguments], check=True, env=environment)

    def driver(**more: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "experiments/commands/more-decks-pod.sh"],
            cwd=here,
            check=False,
            capture_output=True,
            text=True,
            env={**environment, **{k: v for k, v in more.items() if v is not None}},
            timeout=120,
        )

    def calls() -> list[str]:
        return asked.read_text(encoding="utf8").splitlines() if asked.exists() else []

    return driver, here, there, calls


def kept_here(here: Path) -> list[str]:
    return sorted(
        str(path.relative_to(here / "runs")) for path in (here / "runs").rglob("*") if path.is_file()
    )


def test_the_arms_are_made_on_the_pod_brought_here_and_the_pod_deleted(pod):
    driver, here, there, calls = pod
    made = driver(SIZES="1 2")
    assert made.returncode == 0, made.stderr
    # What a run keeps is here as it is there, and what it passed through is not.
    assert kept_here(here) == [
        "more-decks/decks-01/checkpoints/model-00001.pt",
        "more-decks/decks-01/checkpoints/model-00003.pt",
        "more-decks/decks-01/metrics.jsonl",
        "rented/more-decks.log",
        "rented/more-decks.volume",
    ]
    assert (here / "runs/rented/more-decks.volume").read_text().strip().endswith("the volume/ygo-runs")
    assert (here / "runs/more-decks/decks-01/checkpoints/model-00003.pt").read_text() == "weights 3\n"
    assert (here / "runs/more-decks/decks-01/metrics.jsonl").read_text().count("row") == 4
    assert (there / "runs/more-decks/decks-01/checkpoints/model-00002.pt").is_file()
    # The pod's runs are on its volume, which a stop leaves.
    assert (there / "runs").is_symlink()
    assert "The pod keeps its runs on its volume, which a stop leaves" in made.stdout
    # The arms were told what was named here, and six at once where nothing is.
    assert "the arms say: at once 6, sizes 1 2" in (here / "runs/rented/more-decks.log").read_text()
    # The pod was looked up before anything, stopped and deleted after
    # everything, and looked up again: it is no longer named.
    assert calls() == [f"pod {verb} pod123" for verb in ("get", "stop", "delete", "get")]
    assert "more-decks has ended 0, and what it kept is here under runs/more-decks" in made.stdout
    assert "The pod pod123 is deleted: runpodctl no longer names it" in made.stdout
    # A pod that is gone is not driven: nothing can stop it at the end.
    gone = driver()
    assert gone.returncode == 1 and "runpodctl does not name the pod pod123" in gone.stderr
    assert (there / "runs/rented/more-decks.log").read_text().count("the arms say") == 1


def test_started_again_it_goes_on_from_where_things_stand(pod):
    driver, here, there, calls = pod
    # Left as it is the first time: the arms are made and brought.
    assert driver(LEAVE="1", POD="").returncode == 0
    assert calls() == []
    # Again, with the pod named: nothing is started there a second time, what
    # is there is brought, and the pod is deleted.
    again = driver()
    assert again.returncode == 0, again.stderr
    assert "more-decks has ended 0 there: what it kept is brought" in again.stdout
    assert calls() == [f"pod {verb} pod123" for verb in ("get", "stop", "delete", "get")]
    assert (there / "runs/rented/more-decks.log").read_text().count("the arms say") == 1
    assert "more-decks/decks-01/checkpoints/model-00003.pt" in kept_here(here)


def test_arms_that_fail_are_brought_and_the_pod_deleted_all_the_same(pod):
    driver, here, _, calls = pod
    failed = driver(ENDS="5", AT_ONCE="3")
    assert failed.returncode == 1
    assert "more-decks has ended 5, and what it kept is here" in failed.stdout
    assert "more-decks/decks-01/checkpoints/model-00003.pt" in kept_here(here)
    assert "at once 3" in (here / "runs/rented/more-decks.log").read_text()
    assert calls() == [f"pod {verb} pod123" for verb in ("get", "stop", "delete", "get")]
    assert "The pod pod123 is deleted: runpodctl no longer names it" in failed.stdout


def test_a_pod_that_runpodctl_still_names_is_told(pod):
    driver, _, _, calls = pod
    # It could not be deleted: that is what the driver ends with, whatever the arms did.
    kept = driver(REFUSED="delete")
    assert kept.returncode == 4
    assert "runpodctl did not delete the pod pod123" in kept.stderr
    assert "runpodctl STILL NAMES the pod pod123: delete it where it was rented" in kept.stderr
    assert calls() == [f"pod {verb} pod123" for verb in ("get", "stop", "delete", "get")]
    # It could not be stopped and was deleted all the same. A pod that is
    # still told of, as one that is terminated, is gone.
    unstopped = driver(REFUSED="stop", LINGERS="1")
    assert unstopped.returncode == 0, unstopped.stderr
    assert "runpodctl did not stop the pod pod123" in unstopped.stderr
    assert "The pod pod123 is deleted: runpodctl no longer names it" in unstopped.stdout
    assert "does not name the pod pod123" in driver(LINGERS="1").stderr


def test_a_pod_whose_files_are_not_all_here_is_stopped_and_not_deleted(pod, tmp_path):
    """What the arms wrote is on the pod's volume, which a stop leaves: the
    pod is stopped, and when it is started again, with a new disk of its
    own, what is on the volume is brought and the pod deleted."""
    driver, here, there, calls = pod
    volume = tmp_path / "the volume"
    broken = driver(BROKEN="1")
    assert broken.returncode == 3
    assert "The pod keeps its runs on its volume, which a stop leaves" in broken.stdout
    assert "The pod pod123 is STOPPED and not deleted: what the arms wrote is on its volume" in broken.stderr
    assert calls() == ["pod get pod123", "pod stop pod123", "pod get pod123"]
    assert (there / "runs").is_symlink()
    assert (volume / "ygo-runs/more-decks/decks-01/checkpoints/model-00003.pt").is_file()
    assert not (here / "runs/more-decks").exists()
    # Started again, the pod has a new disk of its own and its volume as it was.
    shutil.rmtree(there)
    mended = driver()
    assert mended.returncode == 0, mended.stderr
    assert "more-decks has ended 0 there: what it kept is brought" in mended.stdout
    assert "more-decks/decks-01/checkpoints/model-00003.pt" in kept_here(here)
    assert calls()[3:] == [f"pod {verb} pod123" for verb in ("get", "stop", "delete", "get")]
    # The arms were made once: nothing was started over what the volume held.
    assert (here / "runs/rented/more-decks.log").read_text().count("the arms say") == 1


def test_a_pod_that_is_not_told_as_stopped_is_said_so(pod):
    driver, _, _, calls = pod
    unstopped = driver(BROKEN="1", REFUSED="stop")
    assert unstopped.returncode == 3
    assert "runpodctl does not tell the pod pod123 as stopped" in unstopped.stderr
    assert "may be RUNNING at its price" in unstopped.stderr
    assert calls()[:2] == ["pod get pod123", "pod stop pod123"] and "pod delete pod123" not in calls()


def test_a_pod_without_a_volume_is_left_running_where_the_copy_is_not_whole(pod):
    driver, here, there, calls = pod
    broken = driver(BROKEN="1", VOLUME="")
    assert broken.returncode == 3
    assert "The pod keeps its runs on its own disk, which a stop clears" in broken.stdout
    assert "is left as it is, RUNNING at its price: it has no volume" in broken.stderr
    assert calls() == ["pod get pod123"]
    assert (there / "runs").is_dir() and not (there / "runs").is_symlink()
    assert not (here / "runs/more-decks").exists()
    # Started again once the way is back, it brings what is there and deletes the pod.
    mended = driver(VOLUME="")
    assert mended.returncode == 0, mended.stderr
    assert "more-decks/decks-01/checkpoints/model-00003.pt" in kept_here(here)
    assert calls()[1:] == [f"pod {verb} pod123" for verb in ("get", "stop", "delete", "get")]


def test_arms_that_stopped_without_a_word_are_started_again(pod, tmp_path):
    """A pod that restarted in the middle: its volume holds what the arms
    had written and that they were running, and they are started again."""
    driver, here, _, _ = pod
    kept = tmp_path / "the volume/ygo-runs"
    (kept / "rented").mkdir(parents=True)
    (kept / "rented/more-decks.pid").write_text("999999\n", encoding="utf8")
    (kept / "more-decks/decks-01/checkpoints").mkdir(parents=True)
    (kept / "more-decks/decks-01/metrics.jsonl").write_text("row before\n", encoding="utf8")
    again = driver(LEAVE="1", POD="")
    assert again.returncode == 0, again.stderr
    assert "more-decks stopped there and left no status: it is started again and goes on" in again.stdout
    rows = (here / "runs/more-decks/decks-01/metrics.jsonl").read_text(encoding="utf8")
    assert rows.startswith("row before\n") and rows.count("row") == 5


def test_the_driver_refuses_what_it_could_not_end(pod, tmp_path):
    driver, _, there, calls = pod
    assert driver(POD="").returncode == 2
    unknown = driver(REFUSED="get")
    assert unknown.returncode == 1 and "runpodctl does not name the pod pod123" in unknown.stderr
    assert driver(RENTED="").returncode == 2
    assert not there.exists()
    # Left as it is where that is asked for: no pod is named, and none is touched.
    left = driver(LEAVE="1", POD="")
    assert left.returncode == 0, left.stderr
    assert "The pod is left as it is (LEAVE)" in left.stdout
    assert calls() == ["pod get pod123"]  # the refused look above, and nothing since
    # A pod that no longer holds what was brought from it, on its own disk or
    # on its volume, has lost it: nothing is started over what is here.
    shutil.rmtree(there)
    shutil.rmtree(tmp_path / "the volume/ygo-runs")
    lost = driver()
    assert lost.returncode == 1
    assert "is here and not on the pod, which has lost what it held" in lost.stderr
    assert not (tmp_path / "the volume/ygo-runs/rented/more-decks.pid").exists()
