"""Summarize `policy-bench search` rows: a pilot alone against the pilot with search.

Each row of a search run holds two games on the same seed: `baseline`, played
by the deck's policy, and `search`, played by the policy with a one-step
search at each of its decisions. The gain is their paired difference.

Usage, from the repository root::

    uv run python tools/policy_search_summary.py --cards native/ygo-policies/vendor/BabelCdb/cards.cdb \\
        analysis/policy-search/*.jsonl --output analysis/policy-search/summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

COUNTS = ("searched", "in_chain", "facing_set", "playouts", "failed_playouts", "seconds")


def card_names(database: Path) -> dict[int, str]:
    with sqlite3.connect(database) as connection:
        return dict(connection.execute("select id, name from texts"))


def describe(choice: dict[str, Any], names: dict[int, str]) -> str:
    kind = choice["kind"] if isinstance(choice["kind"], str) else json.dumps(choice["kind"])
    code = choice.get("code")
    return kind if not code else f"{kind} {names.get(code, code)}"


def summarize(path: Path, names: dict[int, str]) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    games = [r for r in rows if "error" not in r["baseline"] and "error" not in r["search"]]
    count = len(games)
    difference = [r["search"]["score"] - r["baseline"]["score"] for r in games]
    gain = sum(difference) / count
    error = math.sqrt(sum((d - gain) ** 2 for d in difference) / (count - 1) / count)
    per_game = {key: sum(r["search"].get(key, 0) for r in games) / count for key in COUNTS}
    per_game["deviations"] = sum(len(r["search"]["deviations"]) for r in games) / count
    by_opponent: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for r in games:
        by_opponent[r["b"]].append((r["baseline"]["score"], r["search"]["score"]))
    decisions: collections.Counter[str] = collections.Counter()
    moves: collections.Counter[tuple[str, str, str]] = collections.Counter()
    for r in games:
        for deviation in r["search"]["deviations"]:
            decisions[deviation["kind"]] += 1
            moves[
                (
                    deviation["kind"],
                    describe(deviation["from"], names),
                    describe(deviation["to"], names),
                )
            ] += 1
    return {
        "file": path.name,
        "pilot_deck": games[0]["a"],
        "games": count,
        "failed_games": len(rows) - count,
        "pilot": sum(r["baseline"]["score"] for r in games) / count,
        "search": sum(r["search"]["score"] for r in games) / count,
        "gain": gain,
        "gain_standard_error": error,
        "turned_into_wins": sum(d > 0 for d in difference),
        "turned_into_losses": sum(d < 0 for d in difference),
        "per_game": per_game,
        "by_opponent": {
            opponent: {
                "games": len(pairs),
                "pilot": sum(a for a, _ in pairs) / len(pairs),
                "search": sum(b for _, b in pairs) / len(pairs),
            }
            for opponent, pairs in sorted(by_opponent.items())
        },
        "deviations_by_decision": dict(decisions),
        "deviations": [
            {"decision": kind, "from": before, "to": after, "count": number}
            for (kind, before, after), number in moves.most_common()
        ],
    }


def report(summary: dict[str, Any], *, moves: int) -> str:
    lines = [
        f"== {summary['file']}: {summary['games']} games ({summary['failed_games']} failed)",
        (
            f"   pilot {100 * summary['pilot']:.1f}%   with search {100 * summary['search']:.1f}%   "
            f"gain {100 * summary['gain']:+.1f} ± {100 * summary['gain_standard_error']:.1f} points (paired)"
        ),
        (
            f"   games turned into wins {summary['turned_into_wins']}, "
            f"into losses {summary['turned_into_losses']}"
        ),
        (
            "   per game: {searched:.1f} decisions searched, {in_chain:.1f} left in a chain, "
            "{facing_set:.1f} left facing a set monster, {deviations:.2f} deviations, "
            "{playouts:.0f} playouts ({failed_playouts:.2f} failed), {seconds:.0f} s"
        ).format(**summary["per_game"]),
        "   opponent        games   pilot   search    gain",
    ]
    for opponent, row in summary["by_opponent"].items():
        lines.append(
            f"   {opponent:14} {row['games']:5d}  {100 * row['pilot']:5.1f}%  "
            f"{100 * row['search']:5.1f}%  {100 * (row['search'] - row['pilot']):+6.1f}"
        )
    lines.append(f"   deviations by decision: {summary['deviations_by_decision']}")
    for move in summary["deviations"][:moves]:
        lines.append(
            f"   {move['count']:5d}  {move['decision']}: {move['from']}  ->  {move['to']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", type=Path, nargs="+", help="search JSONL files")
    parser.add_argument("--cards", type=Path, required=True, help="cards.cdb, for card names")
    parser.add_argument("--output", type=Path, help="write the summaries as JSON")
    parser.add_argument("--moves", type=int, default=15, help="deviations listed per file")
    args = parser.parse_args()
    names = card_names(args.cards)
    summaries = [summarize(path, names) for path in args.rows]
    for summary in summaries:
        print(report(summary, moves=args.moves))
    if args.output is not None:
        args.output.write_text(json.dumps(summaries, indent=1) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
