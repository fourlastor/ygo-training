#!/usr/bin/env bash
# Puts luna's queue of tasks in place (docs/luna-queue.md), from this
# checkout: the queue's files beside each other in PLACE on HOST, Redis and
# the dashboard up in Docker, RQ in a virtual environment of its own, and the
# worker's user service installed. The worker is started only with
# START=1: it takes the checkout over, moving it from commit to commit.
#   bash tools/queue/deploy.sh
set -euo pipefail
HOST=${HOST:-daniele@luna}
PLACE=${PLACE:-ygo-training/queue}
RQ_VERSION=${RQ_VERSION:-2.12.0}
START=${START:-0}
here=$(cd "$(dirname "$0")" && pwd)
ssh "$HOST" "mkdir -p $PLACE/logs .config/systemd/user"
scp -q "$here/compose.yaml" "$here/luna_queue.py" "$here/luna-queue" "$HOST:$PLACE/"
scp -q "$here/ygo-queue-worker.service" "$HOST:.config/systemd/user/"
ssh "$HOST" "cd $PLACE && chmod +x luna-queue && docker compose up -d \
  && { test -x venv/bin/python || ~/.local/bin/uv venv --quiet venv; } \
  && ~/.local/bin/uv pip install --quiet --python venv/bin/python rq==$RQ_VERSION \
  && systemctl --user daemon-reload"
if [ "$START" = 1 ]; then
  ssh "$HOST" "systemctl --user enable --now ygo-queue-worker"
  echo "The queue is in place on $HOST, and its worker runs."
else
  echo "The queue is in place on $HOST; its worker starts with START=1 (or:" \
    "ssh $HOST systemctl --user enable --now ygo-queue-worker)."
fi
