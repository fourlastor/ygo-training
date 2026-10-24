"""An arm's run against `all71`'s, from its copy to R1: the third phase of
a critic that sees the hidden cards (experiments/2026-10-22-privileged-
critic.md), and the run of a head that learns how cards work
(experiments/2026-09-08-engine-transition-prediction.md), whose second
reading is the same.

    uv run python tools/privileged_critic_run_report.py analysis/privileged-critic/run \\
        --base analysis/seventy-one-decks --output analysis/privileged-critic/run/report.json

The directory is what `PARTS=measure` of
`experiments/commands/privileged-critic.sh` (or `card-head.sh`) leaves: the
arm (`--arm`, `critic` unless told, by its
checkpoints after the last three `--step` of its `--point` updates of PPO)
on the screen of each of the thirteen decks (`screens/ARM.DECK.json`), on
the eighteen other decks of thirty-one (`old/ARM.json`) and on the forty
decks new to the pool (`new/ARM.json`), and both runs' own figures
(`ARM.metrics.jsonl`, and `all71.metrics.jsonl`, the first `--point`
updates of `all71`'s). `--base` is where seventy-one decks measured `all71`
there, on the same games (`screens/all71-POINT.DECK.json`,
`old/all71-POINT.json`, `new/all71-POINT.json`).  With `--cards`, the first
reading of the head that learns how cards work beside it, as
`tools/card_head.py read` took it.

A game is read by the share of a run's three checkpoints that win it, a won
game a win (a drawn one is not).

The reading, fixed on 2026-10-22 before the run:

3. It learns faster: the arm against `all71` at R1, both by their
   checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by game
   over the three suites together: faster when the gap less two standard
   errors is at least 2 points, slower when the gap and two standard errors
   are at most -2 points, otherwise not told apart (and then the second
   setting, `half` from its copy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_head_report import difference
from search_leads_report import percent, points
from thirteen_decks_report import DECKS, ERRORS, MARK, Games, figures, mean, points_of

BASE = "all71"
KINDS = {
    "screens": "the thirteen screens",
    "old": "the eighteen other decks of thirty-one",
    "new": "the forty new decks",
}


def faster(gap: float, error: float) -> str:
    """The reading: faster, slower, or neither."""
    if gap - ERRORS * error >= MARK:
        return "faster"
    if gap + ERRORS * error <= -MARK:
        return "slower"
    return "not told apart"


def along(path: Path, width: int) -> list[dict[str, float]]:
    """A run's figures by blocks of ``width`` updates: what it won against
    the pilots in training and, where it has a critic, what the critic and
    the value head explained of the results of each update's games before
    the critic was taught them; where it has a head that learns how cards
    work, its log-loss of the cards' places."""
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    blocks = []
    for start in range(0, len(rows), width):
        part = rows[start : start + width]
        block: dict[str, float] = {"first": start + 1, "last": start + len(part)}
        against = [
            row["other_seat_win_rate"]["pilot"]
            for row in part
            if "pilot" in (row.get("other_seat_win_rate") or {})
        ]
        if against:
            block["pilots"] = mean(against)
        if all("critic_outcome_explained_variance" in row for row in part):
            block["critic"] = mean([row["critic_outcome_explained_variance"] for row in part])
            block["head"] = mean([row["critic_head_outcome_explained_variance"] for row in part])
        if all("card_head_places_loss" in row for row in part):
            block["places"] = mean([row["card_head_places_loss"] for row in part])
        blocks.append(block)
    return blocks


def report(
    directory: Path,
    base: Path,
    thirteen: list[str],
    copied: int = 17_040,
    point: int = 1_920,
    step: int = 120,
    arm: str = "critic",
    base_name: str = BASE,
    base_run: str | None = None,
) -> dict[str, Any]:
    """The arm against the run it is set beside: `all71` unless told
    (``base_name``), its measurements under ``base`` named
    ``base_run`` (``all71-POINT`` unless told)."""
    at = points_of(point, step)
    if base_run is None:
        base_run = f"{BASE}-{point}"
    wanted = {
        arm: {f"{arm}@{each}": copied + each for each in at},
        base_run: {f"{base_run}@{each}": copied + each for each in at},
    }
    files = {
        kind: [
            (
                (directory / kind / f"{arm}.{deck}.json", base / kind / f"{base_run}.{deck}.json")
                if kind == "screens"
                else (directory / kind / f"{arm}.json", base / kind / f"{base_run}.json")
            )
            for deck in (thirteen if kind == "screens" else [None])
        ]
        for kind in KINDS
    }
    result: dict[str, Any] = {"arm": arm, "point": point, "by_kind": {}, "by_screen": {}}
    pooled: dict[str, list[float]] = {arm: [], base_run: []}
    for kind, what in KINDS.items():
        played: dict[str, list[float]] = {arm: [], base_run: []}
        for index, (ours, theirs) in enumerate(files[kind]):
            found = Games(f"{what}, {thirteen[index]}" if kind == "screens" else what)
            found.add(ours, wanted[arm])
            found.add(theirs, wanted[base_run])
            told = {run: found.three(run, at) for run in played}
            missing = [run for run, games in told.items() if games is None]
            if missing:
                raise SystemExit(f"{what}: {', '.join(missing)} not measured")
            for run, games in told.items():
                played[run] += games
            if kind == "screens":
                result["by_screen"][thirteen[index]] = {run: mean(told[run]) for run in told}
        gap, error = difference(played[arm], played[base_run])
        result["by_kind"][kind] = {
            arm: mean(played[arm]),
            base_name: mean(played[base_run]),
            "gap": gap,
            "error": error,
            "games": len(played[arm]),
        }
        for run in pooled:
            pooled[run] += played[run]
    gap, error = difference(pooled[arm], pooled[base_run])
    result["reading"] = {
        arm: mean(pooled[arm]),
        base_name: mean(pooled[base_run]),
        "gap": gap,
        "error": error,
        "games": len(pooled[arm]),
        "verdict": faster(gap, error),
    }
    result["figures"] = {}
    result["along"] = {}
    for run in (arm, base_name):
        path = directory / f"{run}.metrics.jsonl"
        told = figures(path, step)
        if told is not None:
            result["figures"][run] = told
        result["along"][run] = along(path, 2 * step)
    return result


def show(result: dict[str, Any], cards: dict[str, Any] | None = None) -> str:
    arm = result["arm"]
    reading = result["reading"]
    # The critic's third reading; the second of the head that learns how
    # cards work, its first beside it.
    number = 3 if cards is None else 2
    lines = [
        (
            f"The run {arm} against all71 at {result['point']:,} updates of PPO, each by "
            "its last three checkpoints, game by game:"
        ),
        (
            f"{number}. It learns faster: {reading['verdict']}. Over the three suites "
            f"{percent(reading[arm])} against {percent(reading[BASE])}: "
            f"{points(reading['gap'], reading['error'])} over {reading['games']:,} games."
        ),
        "Beside it:",
    ]
    if cards is not None:
        from card_head import verdict

        held = cards["held"]
        lines.insert(
            1,
            (
                f"1. It learns how cards move: {verdict(held['share'], held['error'])}. On "
                f"{held['cards']:,} cards of held games the head takes away {held['share']:.3f} ± "
                f"{held['error']:.3f} of the frequencies' log-loss ({held['head_log_loss']:.4f} "
                f"against {held['frequencies_log_loss']:.4f})."
            ),
        )
    for kind, what in KINDS.items():
        told = result["by_kind"][kind]
        lines.append(
            f"  {what}: {percent(told[arm])} against {percent(told[BASE])}, "
            f"{points(told['gap'], told['error'])} ({told['games']:,} games)"
        )
    lines.append(f"  each screen, {arm} against all71:")
    for deck, told in result["by_screen"].items():
        ours, theirs = told[arm], told[f"{BASE}-{result['point']}"]
        lines.append(
            f"    {deck}: {percent(ours)} against {percent(theirs)} ({points(ours - theirs)})"
        )
    for run, told in result["figures"].items():
        lines.append(
            f"  {run}'s last updates: {told['update_seconds']:.1f} s an update, "
            f"{told['duels_an_hour']:,.0f} duels an hour"
        )
    blocks = dict(result["along"])
    if blocks.get(arm):
        lines.append(
            f"  along the run, by blocks: against the pilots in training ({arm}, all71), and of "
            "each update's games what the critic and the value head explain before the critic "
            "is taught them, or the head's log-loss of the cards' places:"
        )
        theirs = {block["first"]: block for block in blocks.get(BASE, [])}
        for block in blocks[arm]:
            other = theirs.get(block["first"], {})
            pilots = (
                f"{percent(block['pilots'])}, {percent(other['pilots'])}"
                if "pilots" in block and "pilots" in other
                else "-"
            )
            explained = (
                f"the critic {block['critic']:+.3f}, the head {block['head']:+.3f}"
                if "critic" in block
                else f"places {block['places']:.4f}"
                if "places" in block
                else "-"
            )
            lines.append(
                f"    updates {block['first']:5,}-{block['last']:5,}: {pilots}; {explained}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base", type=Path, required=True, help="seventy-one decks' measurements")
    parser.add_argument("--thirteen", nargs="+", default=list(DECKS))
    parser.add_argument("--copied", type=int, default=17_040, help="updates of all71's copy")
    parser.add_argument("--point", type=int, default=1_920, help="updates of PPO read at")
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--arm", default="critic", help="what the arm's files are named")
    parser.add_argument("--cards", type=Path, help="tools/card_head.py read's first reading")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.directory, args.base, args.thirteen, args.copied, args.point, args.step, args.arm
    )
    cards = None if args.cards is None else json.loads(args.cards.read_text())
    if cards is not None:
        result["cards"] = cards["held"]
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(show(result, cards))


if __name__ == "__main__":
    main()
