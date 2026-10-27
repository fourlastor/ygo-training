"""How many measurements at a time pay: the groups that `LEVELS` of
`experiments/commands/fourteen-decks.sh` plays so many at a time, each
group's games in the time it took, and the card's and the processor's load
meanwhile.

    uv run python tools/at_once_report.py analysis/fourteen-decks

Reads OUT/logs/levels.jsonl, each group's files and how many went at a time,
from when to when in seconds since the epoch, and what part it is: the
decks' own measurements in their mirrors, one deck's played again alone, or
the rest of the measurements; and OUT/logs/load.txt, every ten seconds the time, the card's busy
share in percent and its memory in MiB, and the processor's counters from
/proc/stat. Tells for each count at a time: the files, games and turns, the
seconds, the games and turns a second, how many times the turns a second of
one at a time those are, the card's mean busy share and most memory, and the
share of the processor's time busy; for the mirrors the counts compared, a
group of each beside one alone, and the rest after them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def played(path: Path) -> tuple[int, float]:
    """A measurement's games and their turns, every case of its suite."""
    games, turns = 0, 0.0
    for record in json.loads(path.read_text(encoding="utf8"))["records"]:
        for told in record["evaluation_suite"].values():
            games += int(told["games"])
            turns += int(told["games"]) * float(told["mean_turns"])
    return games, turns


def samples(path: Path) -> list[tuple[int, float | None, float | None, list[int]]]:
    """Every sample of the load: its time, the card's busy share and memory,
    and the processor's counters."""
    found = []
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        busy, memory = (None if part == "-" else float(part) for part in parts[1:3])
        found.append((int(parts[0]), busy, memory, [int(part) for part in parts[3:]]))
    return found


def load(found: list[Any], start: int, end: int) -> dict[str, float | None]:
    """The card's mean busy share and most memory, and the share of the
    processor's time busy, between ``start`` and ``end``."""
    inside = [sample for sample in found if start <= sample[0] <= end]
    cards = [sample[1] for sample in inside if sample[1] is not None]
    memory = [sample[2] for sample in inside if sample[2] is not None]
    processor = None
    if len(inside) >= 2:
        # user, nice, system, idle, iowait, irq, softirq, steal: guests are
        # counted in user already.
        first, last = (sample[3][:8] for sample in (inside[0], inside[-1]))
        total = sum(last) - sum(first)
        idle = (last[3] + last[4]) - (first[3] + first[4])
        processor = None if total <= 0 else 1 - idle / total
    return {
        "card": sum(cards) / len(cards) if cards else None,
        "memory": max(memory) if memory else None,
        "processor": processor,
    }


def report(directory: Path) -> list[dict[str, Any]]:
    logs = directory / "logs"
    groups = [
        json.loads(line)
        for line in (logs / "levels.jsonl").read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    found = samples(logs / "load.txt")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    places = {"alone": 0, "mirrors": 1, "rest": 2}
    for group in groups:
        part = str(group.get("part", "mirrors"))
        row = rows.setdefault(
            (places.get(part, 1), int(group["at_once"])),
            {
                "part": part,
                "at once": int(group["at_once"]),
                "files": 0,
                "games": 0,
                "turns": 0.0,
                "seconds": 0,
            },
        )
        for file in group["files"]:
            games, turns = played(Path(file))
            row["games"] += games
            row["turns"] += turns
            row["files"] += 1
        row["seconds"] += int(group["to"]) - int(group["from"])
        row.setdefault("windows", []).append((int(group["from"]), int(group["to"])))
    ordered = [rows[key] for key in sorted(rows)]
    alone = next((row for row in ordered if row["part"] == "alone"), None)
    for row in ordered:
        seconds = max(row["seconds"], 1)
        row["games a second"] = row["games"] / seconds
        row["turns a second"] = row["turns"] / seconds
        # The rest are other suites than a mirror's: not held to one alone.
        row["times alone"] = (
            row["turns a second"] / alone["turns a second"]
            if alone and alone["turns"] and row["part"] != "rest"
            else None
        )
        loads = [load(found, start, end) for start, end in row.pop("windows")]
        for name in ("card", "processor"):
            values = [each[name] for each in loads if each[name] is not None]
            row[name] = sum(values) / len(values) if values else None
        memory = [each["memory"] for each in loads if each["memory"] is not None]
        row["memory"] = max(memory) if memory else None
    return ordered


def shown(rows: list[dict[str, Any]]) -> str:
    def share(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    lines = [
        (
            "Part     At a time  files    games  seconds  games/s  turns/s  times alone  "
            "card busy  card MiB  processor busy"
        )
    ]
    for row in rows:
        times = "-" if row["times alone"] is None else f"{row['times alone']:.2f}"
        memory = "-" if row["memory"] is None else f"{row['memory']:.0f}"
        card = "-" if row["card"] is None else f"{row['card']:.0f}%"
        lines.append(
            f"{row['part']:<7}  {row['at once']:>9}  {row['files']:>5}  {row['games']:>7}  {row['seconds']:>7}  "
            f"{row['games a second']:>7.1f}  {row['turns a second']:>7.0f}  {times:>11}  "
            f"{card:>9}  {memory:>8}  {share(row['processor']):>14}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(shown(report(args.directory)))


if __name__ == "__main__":
    main()
