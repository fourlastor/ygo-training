"""What `tools/benchmark.sh` measured: a machine's speed in every mode of the
trainer.

    python tools/benchmark_report.py benchmark/STAMP --output benchmark/STAMP/report.json

The folder holds a run for each mode (`NAME/metrics.jsonl`), the copy
measured on two suites (`suite.json`, `mirror.json`), how the exercise was
set (`settings.json`) and the seconds of its steps (`setup.tsv`,
`exercise.tsv`).  Speed only: nothing here says how a model learns.

A mode is told by the mean of its updates after the first, which warms up:

- an update, all of it: its games, its estimates worked out again where the
  model played them, the update and the checkpoint written;
- duels an hour, at that;
- the games, with the engines' and the model's parts of them where the run
  tells them, and the update;
- the decisions it learns from, and how many of those a second of an update
  comes to.

A mode with no figures failed or was not run; the exercise's log of that
name says why.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

IMITATING = "Updates that imitate (the one imitated plays the learner's seat)"
PLAYING = "Updates whose games the model plays (PPO, one pass)"
# The run of each mode, the kind of update it makes and what it is.
MODES = (
    ("imitation", IMITATING, "the pilot"),
    ("imitation-decks", IMITATING, "the pilots of two decks, one learner"),
    ("imitation-checkpoint", IMITATING, "a checkpoint"),
    ("imitation-records", IMITATING, "recorded duels"),
    ("imitation-leads", IMITATING, "recorded duels, the lead head taught too"),
    ("ppo", PLAYING, "against the pilots"),
    ("ppo-anchor", PLAYING, "against the pilots, their answers an anchor"),
    ("ppo-half", PLAYING, "half of the duels against itself"),
    ("ppo-itself", PLAYING, "every duel against itself"),
    ("ppo-checkpoint", PLAYING, "every duel against a held checkpoint"),
)
MEASURED = (
    ("suite", "the suite against the pilots"),
    ("mirror", "the mirror against a held checkpoint"),
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def figures(metrics: Path) -> dict[str, Any] | None:
    """A mode's updates after the first: what one takes and where it goes."""
    if not metrics.is_file():
        return None
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf8").splitlines() if line]
    timed = rows[1:]
    if not timed:
        return None

    def of(key: str) -> float | None:
        found = [float(row[key]) for row in timed if key in row]
        return mean(found) if len(found) == len(timed) else None

    update = mean(
        [float(row["wall_seconds"]) - float(row.get("evaluation_seconds", 0.0)) for row in timed]
    )
    duels, decisions = of("duels"), of("samples")
    assert duels is not None and decisions is not None
    return {
        "updates": len(timed),
        "duels": duels,
        "update_seconds": update,
        "duels_an_hour": 3600.0 * duels / update,
        "games_seconds": of("seconds"),
        "engines_seconds": of("pool_threads_seconds"),
        "model_seconds": of("inference_seconds"),
        "learning_seconds": of("ppo_update_seconds"),
        "decisions": decisions,
        "held_decisions": of("held_decisions"),
        "decisions_a_second": decisions / update,
    }


def measured(path: Path) -> dict[str, Any] | None:
    """A suite the copy was measured on: its games and its seconds."""
    if not path.is_file():
        return None
    kept = json.loads(path.read_text(encoding="utf8"))
    cases = [case for record in kept["records"] for case in record["evaluation_suite"].values()]
    if any("error" in case for case in cases):
        return None
    games = sum(int(case["games"]) for case in cases)
    seconds = float(kept["seconds"])
    return {"games": games, "seconds": seconds, "games_an_hour": 3600.0 * games / seconds}


def steps(path: Path) -> dict[str, int]:
    """The seconds of the steps a file of `NAME<tab>SECONDS` lines tells."""
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf8").splitlines():
        name, _, seconds = line.rpartition("\t")
        found[name] = int(seconds)
    return found


def processor() -> str:
    """The processor's name, where the system tells it."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def memory() -> float | None:
    """The machine's memory in GiB, where the system tells it."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf8").splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    return None


def allowed() -> float | None:
    """The threads a container may run at once, where it is held to fewer
    than the system counts."""
    try:
        control = Path("/sys/fs/cgroup")
        if (control / "cpu.max").is_file():
            quota, period = (control / "cpu.max").read_text(encoding="utf8").split()
        else:
            quota = (control / "cpu/cpu.cfs_quota_us").read_text(encoding="utf8")
            period = (control / "cpu/cpu.cfs_period_us").read_text(encoding="utf8")
        share = int(quota) / int(period)
    except (OSError, ValueError):
        return None
    return share if 0 < share < (os.cpu_count() or 0) else None


def machine(device: str | None) -> dict[str, Any]:
    """The machine the report is made on, which is the one that was measured."""
    found: dict[str, Any] = {
        "processor": processor(),
        "threads": os.cpu_count(),
        "allowed": allowed(),
        "memory_gib": memory(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import torch

        found["torch"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            found["card"] = torch.cuda.get_device_name(0)
            found["card_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
            found["cuda"] = torch.version.cuda
    except Exception as error:  # noqa: BLE001 - a report is still worth making without it
        found["torch"] = f"not told ({type(error).__name__})"
    return found


def pinned(root: Path) -> str | None:
    """The revision of the pilots the duels ran on."""
    try:
        return subprocess.run(
            ["git", "-C", str(root / "native/ygo-policies"), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def span(seconds: float | None, digits: int = 1) -> str:
    return "-" if seconds is None else f"{seconds:.{digits}f}"


def minutes(seconds: int) -> str:
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


def report(folder: Path, told: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = json.loads((folder / "settings.json").read_text(encoding="utf8"))
    result: dict[str, Any] = {
        "settings": settings,
        "machine": machine(settings.get("device")) if told is None else told,
        "policies": pinned(Path(__file__).resolve().parents[1]),
        "modes": {},
        "measured": {},
        "setup": steps(folder / "setup.tsv"),
        "exercise": steps(folder / "exercise.tsv"),
    }
    on = result["machine"]
    print(f"Training speed, {settings.get('stamp')}")
    line = f"  {on.get('processor')}, {on.get('threads')} threads"
    if on.get("allowed"):
        line += f" ({on['allowed']:.1f} of them allowed at once)"
    if on.get("memory_gib"):
        line += f", {on['memory_gib']:.0f} GiB"
    print(line)
    if on.get("card"):
        print(f"  {on['card']}, {on['card_memory_gib']:.0f} GiB, CUDA {on.get('cuda')}")
    print(
        f"  {on.get('system')}, Python {on.get('python')}, torch {on.get('torch')}, "
        f"{settings.get('cargo')}"
    )
    print(
        f"  the trainer at {str(settings.get('commit'))[:8]}, "
        f"the pilots at {str(result['policies'])[:8]}"
    )
    print(
        f"  device {settings['device']}, {settings['duels']} duels an update, "
        f"a pool of {settings['threads']} threads with {settings['slots']} duels each, "
        f"{settings['workers']} workers"
    )
    print(
        f"  on from a copy of {settings['warm']} imitating updates, {settings['timed']} updates "
        "timed in each mode after one that warms up"
    )
    print()
    width = max(len(title) for _, _, title in MODES) + 2
    head = (
        f"  {'':<{width}} {'An update':>10} {'Duels':>10} {'Its games':>10} {'engines':>8} "
        f"{'model':>7} {'The':>8} {'Decisions':>10} {'Decisions':>10}"
    )
    units = (
        f"  {'':<{width}} {'s':>10} {'an hour':>10} {'s':>10} {'s':>8} {'s':>7} "
        f"{'update s':>8} {'':>10} {'a second':>10}"
    )
    group = None
    for name, kind, title in MODES:
        found = figures(folder / name / "metrics.jsonl")
        result["modes"][name] = found
        if kind != group:
            group = kind
            print(kind)
            print(head)
            print(units)
        if found is None:
            print(f"  {title:<{width}} {'not measured':>10}")
            continue
        print(
            f"  {title:<{width}} {found['update_seconds']:>10.1f} {found['duels_an_hour']:>10,.0f} "
            f"{span(found['games_seconds']):>10} {span(found['engines_seconds']):>8} "
            f"{span(found['model_seconds']):>7} {span(found['learning_seconds']):>8} "
            f"{found['decisions']:>10,.0f} {found['decisions_a_second']:>10,.0f}"
        )
    print("The copy measured by its first choice")
    for name, title in MEASURED:
        found = measured(folder / f"{name}.json")
        result["measured"][name] = found
        if found is None:
            print(f"  {title}: not measured")
        else:
            print(
                f"  {title}: {found['games']:,} games in {found['seconds']:.0f} s, "
                f"{found['games_an_hour']:,.0f} an hour"
            )
    missing = [title for name, _, title in MODES if result["modes"][name] is None]
    missing += [title for name, title in MEASURED if result["measured"][name] is None]
    if missing:
        print(f"Not measured: {'; '.join(missing)}. The logs of the exercise say why.")
    if result["setup"] or result["exercise"]:
        setup = sum(result["setup"].values())
        whole = result["exercise"].get("all of it")
        told_steps = ", ".join(f"{name} {minutes(took)}" for name, took in result["setup"].items())
        print(
            "All of it "
            + ("-" if whole is None else minutes(whole))
            + f", of which the setup {minutes(setup)}"
            + (f" ({told_steps})" if told_steps else "")
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (args.folder / "settings.json").is_file():
        raise SystemExit(f"{args.folder} is not a folder of tools/benchmark.sh")
    found = report(args.folder)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    sys.exit(main())
