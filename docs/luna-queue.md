# Luna's queue of tasks

Luna, the CUDA machine, runs one task at a time from a queue. When nothing
more pressing is asked for, it takes work given to it beforehand: a champion
of another deck, the next part of a long run.

- **What it is.** [RQ](https://python-rq.org) over Redis. Redis and RQ's
  dashboard run in Docker (`tools/queue/compose.yaml`), and come back with
  Docker after a restart, the queue with them. The tasks run on the host, by
  RQ's worker under a user service (`tools/queue/ygo-queue-worker.service`;
  linger is on, so it starts with luna): the GPU, the toolchain and the
  Python environment are the ones every run before used.
- **Where a task runs.** In luna's checkout, `~/ygo-training/ygo`, at the
  task's commit: the worker checks the commit out, updates the submodules to
  its pins, and runs the task's command from the checkout's root, with its
  variables beside the worker's (`tools/queue/luna_queue.py`, `run`). What it
  writes goes where a run's results always go, to `runs/` and `analysis/`,
  which no commit tracks. A checkout with changes of its own to what it
  tracks is left alone, and the task fails. The checkout belongs to the
  worker: nothing is run there by hand while it works, and a commit reaches
  luna by a push to `critic-build`.
- **Priorities.** `high` and `low`. The worker takes the oldest task of
  `high` before any of `low`, so a task of high priority waits at most for
  the one running. A task of low priority runs 3 hours at most: longer work
  is split at its checkpoints into tasks that each go on from the one
  before, chained with `--after`, and a task of high priority can come
  between any two of them. The scripts of `experiments/commands/` go on
  from a run's last checkpoint, which is what a part needs.
- **Timeouts.** A task is stopped once it has run half again its hours and
  half an hour more, and fails. Stopping a task stops everything it started.
  Queued again, a run goes on from its last checkpoint.
- **Restarts.** A task running when luna stops fails, as abandoned, once
  the worker is back; queued again, it goes on from its last checkpoint.

## From here

The queue is used over ssh, with `luna-queue` where it was put in place:

```sh
ssh daniele@luna ygo-training/queue/luna-queue add champions-gishki \
  --commit 541482c7 --hours 2.5 --env RUNNING=gishki \
  -- bash experiments/commands/champions.sh
ssh daniele@luna ygo-training/queue/luna-queue list
ssh daniele@luna ygo-training/queue/luna-queue log champions-gishki-1a2b3c
ssh daniele@luna ygo-training/queue/luna-queue cancel champions-gishki-1a2b3c
ssh daniele@luna ygo-training/queue/luna-queue requeue champions-gishki-1a2b3c
```

- `add NAME --commit COMMIT --hours H [--priority low|high] [--env NAME=VALUE]
  [--after JOB] -- COMMAND...`: the commit must be on luna already; the
  command, after `--`, is run by `bash -c`. It prints the task's job, the
  name and a few letters more.
- `list`: by priority, the tasks running, queued (in their order), waiting
  for a task before them, failed, and the last ten finished.
- `log JOB`: the end of a task's log (`~/ygo-training/queue/logs/JOB.log`),
  which holds what it was told and everything it printed.
- `cancel JOB`: out of the queue, or stopped where it runs.
- `requeue JOB`: a failed or stopped task, queued again.

The dashboard, from here: `ssh -L 9181:127.0.0.1:9181 daniele@luna`, then
<http://localhost:9181>. Redis and the dashboard are reached from luna
alone.

## Putting it in place

```sh
bash tools/queue/deploy.sh            # Redis and the dashboard up, RQ, the service
START=1 bash tools/queue/deploy.sh    # and the worker started
```

It copies the queue's files to `~/ygo-training/queue` on luna, outside the
checkout the tasks move from commit to commit, starts Redis and the dashboard
in Docker, installs RQ in a virtual environment of its own there, and
installs the worker's user service. Its versions are pinned: `redis:7.4.11-alpine`,
`cjlapao/rq-dashboard:0.9.0` and `rq` 2.12.0. To update the worker, put it in
place again and restart the service (`systemctl --user restart
ygo-queue-worker`) while no task runs: a restart stops the one running.
