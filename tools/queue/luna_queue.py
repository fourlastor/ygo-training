"""Luna's queue of tasks (docs/luna-queue.md): RQ over Redis, the two in
Docker (tools/queue/compose.yaml), and a worker on the host that runs each
task in luna's checkout at the task's commit
(tools/queue/ygo-queue-worker.service).

A task names a commit, a priority, the hours it takes and what to run: a
shell command, run from the checkout's root with the variables it is given.
The worker takes the oldest task of `high` before any of `low`, so a task of
high priority waits at most for the one running. A task of low priority
runs 3 hours at most: longer work is split at its checkpoints into tasks
that each go on from the one before (`--after`).

    luna-queue add NAME --commit COMMIT --hours H [--priority low|high] \\
        [--env NAME=VALUE ...] [--after JOB] -- COMMAND...
    luna-queue list
    luna-queue log JOB
    luna-queue cancel JOB
    luna-queue requeue JOB

The worker imports `run` from this file where it is put in place
(tools/queue/deploy.sh), outside the checkout, which the tasks move from
commit to commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

PRIORITIES = ("high", "low")
# A task of low priority runs this long at most: longer work is split.
LOW_HOURS = 3.0
# How long finished tasks stay listed, and failed ones.
FINISHED_DAYS = 30
FAILED_DAYS = 365
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def checkout() -> Path:
    """Where tasks run: luna's checkout, unless the environment names another
    (a try)."""
    return Path(os.environ.get("YGO_QUEUE_CHECKOUT", Path.home() / "ygo-training/ygo"))


def logs() -> Path:
    return Path(os.environ.get("YGO_QUEUE_LOGS", Path.home() / "ygo-training/queue/logs"))


def redis_url() -> str:
    return os.environ.get("YGO_QUEUE_REDIS", "redis://127.0.0.1:6379/0")


def queues() -> dict[str, str]:
    """The queue of each priority, by its name in Redis: a try's have a
    prefix of their own."""
    prefix = os.environ.get("YGO_QUEUE_PREFIX", "")
    return {priority: f"{prefix}{priority}" for priority in PRIORITIES}


def git(where: Path, *arguments: str) -> str:
    found = subprocess.run(
        ["git", "-C", str(where), *arguments], capture_output=True, text=True, check=False
    )
    if found.returncode:
        raise RuntimeError(f"git {' '.join(arguments)}: {found.stderr.strip()}")
    return found.stdout.strip()


def task_from(
    name: str, commit: str, hours: float, priority: str, env: list[str], command: str
) -> dict[str, Any]:
    """A task as it is queued, or why it cannot be."""
    if not NAME.fullmatch(name):
        raise SystemExit(f"A task's name is lowercase words joined by dashes, not {name!r}")
    if priority not in PRIORITIES:
        raise SystemExit(f"A task's priority is one of {', '.join(PRIORITIES)}")
    if not hours > 0:
        raise SystemExit("A task says how many hours it takes")
    if priority == "low" and hours > LOW_HOURS:
        raise SystemExit(
            f"A task of low priority runs {LOW_HOURS:g} hours at most: split it at its "
            "checkpoints into tasks that each go on from the one before (--after)"
        )
    if not command.strip():
        raise SystemExit("A task says what to run")
    named = {}
    for given in env:
        variable, sign, value = given.partition("=")
        if not sign or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", variable):
            raise SystemExit(f"A variable is NAME=VALUE, not {given!r}")
        named[variable] = value
    return {
        "name": name,
        "commit": commit,
        "hours": float(hours),
        "priority": priority,
        "env": named,
        "command": command,
    }


def timeout_of(task: dict[str, Any]) -> int:
    """How long a task may run before it is stopped: half again its hours,
    and half an hour more."""
    return int(task["hours"] * 3600 * 1.5 + 1800)


def stopped(process: subprocess.Popen) -> None:
    """The task's command stopped, and whatever it started. RQ makes a job's
    work horse the leader of a process group of its own, which the command
    stays in, and stops a job by stopping the group: here the horse stops the
    others of its group itself. Anywhere else, the command alone."""
    if os.getpgrp() != os.getpid():
        process.terminate()
        process.wait()
        return
    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
        process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def run(task: dict[str, Any]) -> dict[str, Any]:
    """A task, run: the checkout at its commit, its submodules at their pins,
    its command from the checkout's root with its variables beside the
    worker's, everything it prints in its log. A checkout with changes of
    its own to what it tracks is left as it is, and the task fails."""
    where = checkout()
    if git(where, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{where} has changes of its own: no task runs over them")
    git(where, "checkout", "--quiet", "--detach", task["commit"])
    git(where, "submodule", "update", "--init", "--recursive", "--quiet")
    job = os.environ.get("RQ_JOB_ID") or f"{task['name']}-{uuid.uuid4().hex[:6]}"
    log = logs() / f"{job}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf8") as told:
        told.write(
            f"{now()} {task['name']} at {task['commit']}, {task['priority']} priority, "
            f"{task['hours']:g} hours\n"
            + "".join(f"  {name}={value}\n" for name, value in task["env"].items())
            + f"  {task['command']}\n"
        )
        told.flush()
        process = subprocess.Popen(
            ["bash", "-c", task["command"]],
            cwd=where,
            env={**os.environ, **task["env"]},
            stdout=told,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait()
        except BaseException:
            # A timeout, or the horse asked to stop: the command goes with it.
            stopped(process)
            told.write(f"{now()} stopped\n")
            raise
        told.write(f"{now()} exited {code}\n")
    if code:
        raise RuntimeError(f"{task['name']} exited {code}: {log}")
    return {"exit": code, "log": str(log), "commit": git(where, "rev-parse", "HEAD")}


def connection() -> Any:
    from redis import Redis

    return Redis.from_url(redis_url())


def add(args: argparse.Namespace) -> None:
    from rq import Queue

    command = shlex.join(args.command) if len(args.command) > 1 else "".join(args.command)
    commit = git(checkout(), "rev-parse", "--verify", f"{args.commit}^{{commit}}")
    task = task_from(args.name, commit, args.hours, args.priority, args.env, command)
    queue = Queue(queues()[task["priority"]], connection=connection())
    job = queue.enqueue(
        "luna_queue.run",
        task,
        job_id=f"{task['name']}-{uuid.uuid4().hex[:6]}",
        description=f"{task['name']} at {commit[:8]}",
        job_timeout=timeout_of(task),
        result_ttl=FINISHED_DAYS * 86400,
        failure_ttl=FAILED_DAYS * 86400,
        depends_on=args.after or None,
    )
    print(job.id)


def listed(_: argparse.Namespace) -> None:
    from rq import Queue
    from rq.job import Job
    from rq.registry import (
        DeferredJobRegistry,
        FailedJobRegistry,
        FinishedJobRegistry,
        StartedJobRegistry,
    )

    redis = connection()
    for priority, name in queues().items():
        queue = Queue(name, connection=redis)
        kinds = (
            ("running", StartedJobRegistry(queue=queue).get_job_ids()),
            ("queued", queue.get_job_ids()),
            ("waiting", DeferredJobRegistry(queue=queue).get_job_ids()),
            ("failed", FailedJobRegistry(queue=queue).get_job_ids()),
            ("finished", FinishedJobRegistry(queue=queue).get_job_ids()[-10:]),
        )
        print(f"{priority}:")
        for kind, ids in kinds:
            for job in Job.fetch_many(ids, connection=redis):
                if job is None:
                    continue
                task = job.args[0] if job.args else {}
                when = job.ended_at or job.started_at or job.enqueued_at
                told = f"{when.astimezone():%Y-%m-%d %H:%M}" if when else ""
                print(
                    f"  {kind:<9} {job.id:<28} {task.get('hours', 0):>4g} h  "
                    f"{job.description}  {told}"
                )


def log(args: argparse.Namespace) -> None:
    path = logs() / f"{args.job}.log"
    if not path.is_file():
        raise SystemExit(f"No log at {path}")
    lines = path.read_text(encoding="utf8", errors="replace").splitlines()
    print(path)
    print("\n".join(lines[-args.lines :]))


def cancel(args: argparse.Namespace) -> None:
    from rq.command import send_stop_job_command
    from rq.job import Job

    redis = connection()
    job = Job.fetch(args.job, connection=redis)
    if job.get_status() == "started":
        # The worker stops the job's process group: its command with it.
        send_stop_job_command(redis, job.id)
        print(f"{job.id}: stopped")
    else:
        job.cancel()
        print(f"{job.id}: cancelled")


def requeue(args: argparse.Namespace) -> None:
    from rq.job import Job

    job = Job.fetch(args.job, connection=connection())
    job.requeue()
    print(f"{job.id}: queued again")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="what", required=True)
    adding = commands.add_parser("add", help="queue a task")
    adding.add_argument("name")
    adding.add_argument("--commit", required=True, help="the commit to run it at, on luna")
    adding.add_argument("--hours", type=float, required=True, help="how long it takes")
    adding.add_argument("--priority", choices=PRIORITIES, default="low")
    adding.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    adding.add_argument(
        "--after",
        action="append",
        default=[],
        metavar="JOB",
        help="a task to finish first, as the part of the same work before it",
    )
    adding.set_defaults(handled=add)
    commands.add_parser("list", help="the tasks of each priority").set_defaults(handled=listed)
    logged = commands.add_parser("log", help="the end of a task's log")
    logged.add_argument("job")
    logged.add_argument("--lines", type=int, default=40)
    logged.set_defaults(handled=log)
    for name, handled, told in (
        ("cancel", cancel, "take a task out of the queue, or stop it running"),
        ("requeue", requeue, "queue a failed or stopped task again"),
    ):
        named = commands.add_parser(name, help=told)
        named.add_argument("job")
        named.set_defaults(handled=handled)
    given = list(sys.argv[1:] if argv is None else argv)
    # What a task runs comes after --, its options before.
    command = given[given.index("--") + 1 :] if "--" in given else []
    args = parser.parse_args(given[: given.index("--")] if "--" in given else given)
    if args.what == "add":
        if not command:
            parser.error("add says what to run, after --")
        args.command = command
    args.handled(args)


if __name__ == "__main__":
    sys.exit(main())
