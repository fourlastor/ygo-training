"""What a machine does with several runs of the trainer at once.

    uv run python tools/side_by_side.py CHECKPOINT --itself 260

The same short run of PPO, on from CHECKPOINT, is made alone and then two
and three times side by side (``--counts``).  For each count it tells what
an update took in every run and where its seconds went, how much longer
that is than alone, how many duels an hour and decisions a second the
machine made in all, and the load while the timed updates ran: the
processor threads the runs themselves kept busy, the card's share of the
time and the memory in use on it.  It also tells whether every run played
the games the run plays alone, and what all of a count took from its first
start to its last end.

A run alone leaves most of a machine idle (its threads wait while the model
answers, and the card while the engines play), and this says how much of
that other runs can use.  A measure of speed, not an experiment: nothing it
writes is a result, and the runs' checkpoints are removed at the end
(``--keep`` keeps them).  A run keeps the pool threads it has alone, so its
games should be the games it plays alone, and the copies are one run: the
same seeds, the same games.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPPONENTS = (
    "blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,"
    "monarch,morphtronic,rock-block,spellcaster"
)
TICKS = os.sysconf("SC_CLK_TCK")

_spec = importlib.util.spec_from_file_location("benchmark_report", ROOT / "tools/benchmark_report.py")
assert _spec is not None and _spec.loader is not None
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)
figures = _report.figures

# A look at the load: when, the machine's processor time busy and in all, the
# runs' own processor seconds, and the card's share and memory.
Sample = tuple[float, int, int, float, tuple[float, float] | None]


def command(args: argparse.Namespace, run: Path) -> list[str]:
    """The run that is measured: an experiment's, short, with no evaluation."""
    told = [
        sys.executable, "-m", "ygo_training", "train",
        "--device", args.device, "--workers", str(args.workers),
        "--pool-threads", str(args.threads), "--pool-slots", str(args.slots),
        "--run", str(run), "--learner", args.learner, "--opponents", args.opponents,
        "--duels", str(args.duels), "--eval-every", "0", "--keep-checkpoints", "1",
        "--seed", "2012", "--ppo-epochs", "1", "--resume", str(args.checkpoint),
        "--resume-learning-rate", args.rate, "--updates", str(args.updates),
        # How the model is asked is what is measured here: named whatever a
        # run takes where it is not.
        "--pool-groups", str(args.groups), "--pool-groups-over", str(args.groups_over),
        "--inference-graphs", str(args.graphs),
    ]  # fmt: skip
    if args.itself:
        told += ["--self-play-duels", str(args.itself)]
    if args.held:
        told += ["--opponent-checkpoint", str(args.held)]
        told += ["--opponent-checkpoint-duels", str(args.duels)]
    return told


def processor() -> tuple[int, int]:
    """The machine's processor time so far, busy and in all, as the system
    counts it.  What other guests of the host took (steal) is not its work."""
    fields = [int(field) for field in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    user, nice, system, _idle, _waiting, interrupts, soft, _stolen = fields
    return user + nice + system + interrupts + soft, sum(fields)


def spent(sessions: set[int]) -> float:
    """The processor seconds so far of every process of those sessions: a
    run and whatever it started."""
    ticks = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            text = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        # After the name: state, parent, group, session, ..., user and system time.
        fields = text.rsplit(")", 1)[1].split()
        if int(fields[3]) in sessions:
            ticks += int(fields[11]) + int(fields[12])
    return ticks / TICKS


def card(index: int) -> tuple[float, float] | str:
    """The card's share of the time just now and its memory in use, in MiB;
    or, in words, why it could not be read."""
    asked = ["nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"]  # fmt: skip
    try:
        answered = subprocess.run(asked, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return f"{asked[0]}: {error}"
    told = (answered.stdout.strip().splitlines() or [""])[0]
    if answered.returncode:
        said = (answered.stderr.strip() or told).splitlines()
        return f"{asked[0]} ended with {answered.returncode}: {said[0] if said else 'no word'}"
    try:
        busy, memory = told.split(",")
        return float(busy), float(memory)
    except ValueError:
        return f"{asked[0]} answered {told!r}"


class Load(threading.Thread):
    """The load, looked at every other second while runs go."""

    def __init__(self, card_index: int | None, every: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.card_index, self.every = card_index, every
        # A run is started as a session of its own, which its process leads.
        self.sessions: set[int] = set()
        self.samples: list[Sample] = []
        # Why the card was not read, the first time it was not.
        self.unread: str | None = None
        self.ended = threading.Event()

    def run(self) -> None:
        while not self.ended.is_set():
            busy, total = processor()
            seen = None if self.card_index is None else card(self.card_index)
            if isinstance(seen, str):
                self.unread = self.unread or seen
                seen = None
            self.samples.append(
                (time.monotonic(), busy, total, spent(set(self.sessions)), seen)
            )
            self.ended.wait(self.every)

    def between(self, opened: float, closed: float) -> dict[str, float | None]:
        return load([sample for sample in self.samples if opened <= sample[0] <= closed])


def load(samples: list[Sample]) -> dict[str, Any]:
    """Over the samples: the threads the runs kept busy on average, those
    busy on the whole machine, and the card's part."""
    found: dict[str, Any] = dict.fromkeys(("runs_busy", "machine_busy", "card_busy", "card_gib"))
    if len(samples) < 2:
        return found
    (at, busy, total, runs, _), (now, busy_now, total_now, runs_now, _) = samples[0], samples[-1]
    if now > at:
        found["runs_busy"] = (runs_now - runs) / (now - at)
    if total_now > total:
        found["machine_busy"] = (os.cpu_count() or 1) * (busy_now - busy) / (total_now - total)
    cards = [sample[4] for sample in samples if sample[4] is not None]
    if cards:
        found["card_busy"] = sum(share for share, _ in cards) / len(cards)
        found["card_gib"] = max(memory for _, memory in cards) / 1024
    return found


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def together(args: argparse.Namespace, count: int, folder: Path) -> dict[str, Any]:
    """`count` runs at once: each one's figures and games, what all of it
    took, and the load while every one of them was past its first update and
    none had ended."""
    runs = [folder / f"{count}-{place + 1}" for place in range(count)]
    watching = Load(args.card if args.device == "cuda" else None)
    started: list[tuple[subprocess.Popen, Any]] = []
    began = time.monotonic()
    try:
        for run in runs:
            run.parent.mkdir(parents=True, exist_ok=True)
            log = Path(f"{run}.log").open("w", encoding="utf8")  # noqa: SIM115 - closed below
            process = subprocess.Popen(
                command(args, run), cwd=ROOT, stdout=log, stderr=log, start_new_session=True
            )
            watching.sessions.add(process.pid)
            started.append((process, log))
        watching.start()
        opened = closed = None
        while any(process.poll() is None for process, _ in started):
            made = [len(rows(run / "metrics.jsonl")) for run in runs]
            now = time.monotonic()
            if opened is None and min(made) >= 1:
                opened = now
            if closed is None and (
                max(made) >= args.updates or any(p.poll() is not None for p, _ in started)
            ):
                closed = now
            time.sleep(1.0)
        ended = time.monotonic()
    finally:
        # A run is a session of its own: an interrupt here does not reach it.
        for process, log in started:
            if process.poll() is None:
                process.terminate()
            log.close()
        watching.ended.set()
        if watching.is_alive():
            watching.join()
    failed = [str(run) for run, (process, _) in zip(runs, started, strict=True) if process.returncode]
    kept = [rows(run / "metrics.jsonl") for run in runs]

    def calls(made: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Of a run's updates after the first: the calls of the model an
        update, the share of them answered from graphs, and whether the
        graphs failed."""
        timed = made[1:]
        if not timed or any("inference_batches" not in row for row in timed):
            return None
        asked = sum(float(row["inference_batches"]) for row in timed)
        graphed = None
        if all("pool_graph_batches" in row for row in timed):
            graphed = sum(float(row["pool_graph_batches"]) for row in timed) / max(asked, 1.0)
        return {
            "an_update": asked / len(timed),
            "from_graphs": graphed,
            "graphs_failed": any(row.get("pool_graph_failed") for row in made),
        }

    if not args.keep:
        for run in runs:
            shutil.rmtree(run / "checkpoints", ignore_errors=True)
    return {
        "count": count,
        "runs": [figures(run / "metrics.jsonl") for run in runs],
        "calls": [calls(made) for made in kept],
        "failed": failed,
        # What tells a run's games from another's, an update at a time.
        "games": [[row.get("rollout_decision_fingerprint") for row in made] for made in kept],
        "all_of_it_seconds": ended - began,
        "load": watching.between(opened, closed or ended) if opened else load([]),
        "card_unread": watching.unread,
    }


def mean(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept and len(kept) == len(values) else None


def told(value: float | None, form: str = "{:.1f}") -> str:
    return "-" if value is None else form.format(value)


def same_games(measured: list[dict[str, Any]]) -> str:
    """Whether every run played the games of the first run measured."""
    played = [
        (f"{found['count']}-{place + 1}", games)
        for found in measured
        for place, games in enumerate(found.get("games", []))
        if games and all(games)
    ]
    if len(played) < 2:
        return "The runs' games are not told: nothing to set beside each other."
    (first, alone), others = played[0], played[1:]
    apart = [
        f"{name} from its update {next(i for i, (a, b) in enumerate(zip(games, alone, strict=False), start=1) if a != b)}"
        for name, games in others
        if games[: len(alone)] != alone[: len(games)]
    ]
    if apart:
        return f"NOT the games of run {first}: " + ", ".join(apart)
    return f"Every run played the games of run {first}, update for update."


def grouped(args: argparse.Namespace) -> str:
    """How a run's duels are taken forward, where not all together, and how
    its calls are answered, where not all as they are."""
    graphs = f", calls of up to {args.graphs} decisions from CUDA graphs" if args.graphs else ""
    if args.groups == 1:
        return graphs
    if not args.groups_over:
        return f", in {args.groups} groups{graphs}"
    return (
        f", in {args.groups} groups while their rounds hold more than {args.groups_over} decisions"
        f"{graphs}"
    )


def calls_told(measured: list[dict[str, Any]]) -> list[str]:
    """The calls of the model an update for each count of runs, and where
    graphs were asked for, the share of the calls they answered."""
    out = []
    for found in measured:
        made = [calls for calls in found.get("calls", []) if calls]
        if not made or len(made) != len(found.get("calls", [])):
            continue
        asked = [calls["an_update"] for calls in made]
        line = f"{found['count']} at once: {min(asked):,.0f}"
        line += f" to {max(asked):,.0f}" if max(asked) - min(asked) >= 0.5 else ""
        line += " calls of the model an update"
        shares = [calls["from_graphs"] for calls in made if calls["from_graphs"] is not None]
        if shares:
            line += f", {100 * min(shares):.0f}%"
            line += f" to {100 * max(shares):.0f}%" if max(shares) - min(shares) >= 0.005 else ""
            line += " of them answered from graphs"
        if any(calls["graphs_failed"] for calls in made):
            line += "; THE GRAPHS FAILED in a run, which answered its calls as they are from there (its log says why)"
        out.append(line)
    return out


def report(args: argparse.Namespace, stamp: str, measured: list[dict[str, Any]]) -> list[str]:
    """The table, a line a count."""
    against = f", {args.itself} of them against itself" if args.itself else ""
    against += f", every one against {args.held} held" if args.held else ""
    out = [
        f"Runs side by side, {stamp}: {args.checkpoint}",
        f"  PPO in one pass, {args.duels} duels an update{against}",
        (
            f"  device {args.device}, a pool of {args.threads} threads with {args.slots} duels "
            f"each in every run{grouped(args)}, "
            f"{args.updates} updates a run of which the first is not timed"
        ),
        "",
        (
            "   Runs   An update in each   Times as   Duels an hour   Decisions a   Its games  engines"
            "   model      The   Threads the   The card   The card's"
        ),
        (
            "at once                   s       long          in all   second, all           s        s"
            "       s update s   runs keep busy     busy       memory"
        ),
    ]
    alone = None
    for found in measured:
        runs = found["runs"]
        if found["failed"] or any(run is None for run in runs):
            out.append(
                f"{found['count']:>7}   not measured: see "
                f"{', '.join(found['failed']) or 'its runs'}.log"
            )
            continue
        updates = [run["update_seconds"] for run in runs]
        each = sum(updates) / len(updates)
        if found["count"] == 1:
            alone = each
        span = f"{each:.1f}" if len(runs) == 1 else f"{min(updates):.1f} to {max(updates):.1f}"
        loaded = found["load"]
        out.append(
            f"{found['count']:>7} {span:>19} {told(each / alone if alone else None, '{:.2f}'):>10} "
            f"{sum(3600.0 * run['duels'] / run['update_seconds'] for run in runs):>15,.0f} "
            f"{sum(run['decisions_a_second'] for run in runs):>13,.0f} "
            f"{told(mean([run['games_seconds'] for run in runs])):>11} "
            f"{told(mean([run['engines_seconds'] for run in runs])):>8} "
            f"{told(mean([run['model_seconds'] for run in runs])):>7} "
            f"{told(mean([run['learning_seconds'] for run in runs])):>8} "
            f"{told(loaded['runs_busy']):>16} {told(loaded['card_busy'], '{:.0f}%'):>8} "
            f"{told(loaded['card_gib'], '{:.1f} GiB'):>12}"
        )
    unread = next((found["card_unread"] for found in measured if found.get("card_unread")), None)
    if unread:
        out += ["", f"The card's load was not read: {unread}"]
    out += ["", same_games(measured)]
    out += calls_told(measured)
    spans = [
        f"{found['count']} at once {found['all_of_it_seconds']:.0f} s"
        for found in measured
        if "all_of_it_seconds" in found
    ]
    if spans:
        out.append(
            "All of it, from the first start to the last end, with every run's start and "
            f"its first update: {', '.join(spans)}."
        )
    return out


def beside(*reports: Path) -> list[str]:
    """Reports of the same runs (`report.json`), each made another way,
    through MPS say: for each count the duels an hour in all, how many times
    the run alone that is, and what an update took in each run; and whether
    the run alone played the same games every way."""
    ways = [
        {found["count"]: found for found in json.loads(path.read_text(encoding="utf8"))}
        for path in reports
    ]

    def hourly(found: dict[str, Any] | None) -> float | None:
        if found is None or found["failed"] or any(run is None for run in found["runs"]):
            return None
        return sum(3600.0 * run["duels"] / run["update_seconds"] for run in found["runs"])

    def each(found: dict[str, Any] | None) -> float | None:
        if hourly(found) is None:
            return None
        return sum(run["update_seconds"] for run in found["runs"]) / len(found["runs"])

    names = [path.stem for path in reports]
    if len(set(names)) < len(names):
        # Reports of one name are told apart by their folders.
        names = [path.parent.name for path in reports]
    # A column is as wide as the longest name.
    wide = max(12, *(len(name) for name in names))
    block = len(names) * (wide + 1) - 1
    out = [
        (
            f"{'Runs':>7}   {'Duels an hour in all':>{block}}   {'Times the run alone':>{block}}   "
            f"{'An update in each, s':>{block}}"
        ),
        f"{'at once':>7}   " + "   ".join(" ".join(f"{name:>{wide}}" for name in names) for _ in range(3)),
    ]
    alone = [hourly(way.get(1)) for way in ways]
    for count in sorted(set().union(*ways)):
        hours = [hourly(way.get(count)) for way in ways]
        times = [hour / first if hour and first else None for hour, first in zip(hours, alone, strict=True)]
        updates = [each(way.get(count)) for way in ways]
        out.append(
            f"{count:>7}   "
            + "   ".join(
                " ".join(f"{told(value, form):>{wide}}" for value in values)
                for values, form in ((hours, "{:,.0f}"), (times, "{:.2f}"), (updates, "{:.1f}"))
            )
        )
    games = [way[1]["games"][0] if 1 in way and way[1]["games"] else None for way in ways]
    out.append("")
    every = "both ways" if len(names) == 2 else "every way"
    others = [name for name, played in zip(names[1:], games[1:], strict=True) if played != games[0]]
    if None in games or not games[0]:
        out.append(f"The run alone was not made {every}.")
    elif not others:
        out.append(f"The run alone played the same games {every}, update for update.")
    else:
        out.append(
            f"The run alone did NOT play the same games as {' or as '.join(others)} as it did as {names[0]}."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="the runs go on from it")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--updates", type=int, default=5, help="the first is not timed")
    parser.add_argument("--itself", type=int, default=0, help="duels of an update against itself")
    parser.add_argument("--held", type=Path, help="a checkpoint held in the other seat, every duel")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--card", type=int, default=0, help="the card whose load is told")
    parser.add_argument("--threads", type=int, default=20, help="a run's pool threads")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument(
        "--groups", type=int, default=1,
        help="groups a thread's duels are shared out among (--pool-groups of a run)",
    )  # fmt: skip
    parser.add_argument(
        "--groups-over", type=int, default=0,
        help="the groups go on together once their rounds hold so many decisions or fewer "
        "(--pool-groups-over of a run; 0: by groups to the last duel)",
    )  # fmt: skip
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--duels", type=int, default=520)
    parser.add_argument("--rate", default="1e-4")
    parser.add_argument("--learner", default="blackwing")
    parser.add_argument("--opponents", default=OPPONENTS)
    parser.add_argument("--folder", type=Path, default=Path("runs/side-by-side"))
    parser.add_argument("--keep", action="store_true", help="keep the runs' checkpoints")
    parser.add_argument(
        "--graphs", type=int, default=0,
        help="calls of up to so many decisions are answered from CUDA graphs "
        "(--inference-graphs of a run; 0: none)",
    )  # fmt: skip
    parser.add_argument(
        "--beside", type=Path, nargs="+",
        help="reports this made (report.json), two or more: set beside each other, nothing made",
    )  # fmt: skip
    args = parser.parse_args()
    if args.beside:
        if len(args.beside) < 2:
            raise SystemExit("Two reports at least are set beside each other")
        print("\n".join(beside(*args.beside)))
        return
    if not args.checkpoint.is_file():
        raise SystemExit(f"No checkpoint at {args.checkpoint}")
    if args.updates < 2 or min(args.counts) < 1:
        raise SystemExit("Two updates at least, the first not being timed, and one run at least")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = (ROOT / args.folder / stamp).resolve()
    measured = []
    for count in args.counts:
        print(f"{count} at once ...", flush=True)
        measured.append(together(args, count, folder))
    out = report(args, stamp, measured)
    print("\n".join(out))
    (folder / "report.txt").write_text("\n".join(out) + "\n", encoding="utf8")
    (folder / "report.json").write_text(json.dumps(measured, indent=2), encoding="utf8")
    print(f"\nKept in {folder}")
    raise SystemExit(1 if any(found["failed"] for found in measured) else 0)


if __name__ == "__main__":
    main()
