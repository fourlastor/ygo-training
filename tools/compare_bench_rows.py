"""Compare two runs of ``policy-bench`` game by game.

Usage: compare_bench_rows.py BEFORE.jsonl AFTER.jsonl

For the same command run at two sets of revisions (the pilots, the engine,
the card scripts, the card database): which games ended otherwise, were
answered otherwise, or only asked otherwise.  Rows of ``matchup`` and of
``search`` are both read; a search row holds two games.  The exit status is 1
when anything differs, so a change meant to leave the duels alone can be
checked by it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

GAMES = ("game", "baseline", "search")


def games(path: str) -> dict[tuple, dict]:
    """Every game of a run, by the deal and which of its row's games it is."""
    found: dict[tuple, dict] = {}
    with Path(path).open(encoding="utf8") as rows:
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            for name in GAMES:
                if isinstance(row.get(name), dict):
                    found[(row["a"], row["b"], row["seed"], row["seat"], name)] = row[name]
    return found


def compare(before: dict[tuple, dict], after: dict[tuple, dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["only before"] = len(before.keys() - after.keys())
    counts["only after"] = len(after.keys() - before.keys())
    for key in sorted(before.keys() & after.keys()):
        old, new = before[key], after[key]
        counts["games"] += 1
        ended = tuple(old.get(field) for field in ("winner", "reason", "turns")) != tuple(
            new.get(field) for field in ("winner", "reason", "turns")
        )
        counts["another winner, reason or turn count"] += ended
        counts["another digest"] += old.get("digest") != new.get("digest")
        if "record" in old and "record" in new:
            counts["with records"] += 1
            counts["other answers"] += old["record"]["responses"] != new["record"]["responses"]
        if key[-1] == "search":
            left = [
                [(item["decision"], item.get("to")) for item in played.get("deviations", [])]
                for played in (old, new)
            ]
            counts["other deviations"] += left[0] != left[1]
    return counts


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    counts = compare(games(sys.argv[1]), games(sys.argv[2]))
    print(f"{counts['games']} games in both runs ({counts['with records']} with records)")
    differences = 0
    for name in (
        "only before",
        "only after",
        "another winner, reason or turn count",
        "other answers",
        "other deviations",
        "another digest",
    ):
        print(f"  {name}: {counts[name]}")
        differences += counts[name]
    print("identical" if not differences else "the runs differ")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
